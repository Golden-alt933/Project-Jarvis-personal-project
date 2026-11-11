"""Native Ollama client (/api/chat, NDJSON). Used instead of the /v1 shim because
the shim ignores num_ctx, which silently truncates long prompts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from jarvis import platform as plat
from jarvis.llm.openai_compat import ThinkStripper, new_call_id
from jarvis.llm.types import (
    LLMBadRequest,
    LLMRateLimited,
    LLMResponse,
    LLMUnavailable,
    Message,
    ResponseDone,
    StreamEvent,
    TextDelta,
    ToolCall,
    Usage,
)

log = logging.getLogger(__name__)

_NOT_RUNNING = "Ollama is not running (or not reachable at {url}). Start it with `ollama serve`."
_START_FAILED = "Ollama is not running at {url} and could not be started automatically ({why}). Start it with `ollama serve`."
# A cold `ollama serve` took 15-25 s to answer /api/tags on this machine (measured 2026-11-11).
_STARTUP_WAIT = 30.0  # seconds to wait for a freshly launched server to answer /api/tags
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class _AutostartState:
    """Process-wide: the server is launched at most once, however many clients exist."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.attempted = False
        self.launched = False
        self.failure = ""


_autostart = _AutostartState()


class _Refused(LLMUnavailable):
    """The connection was refused: nothing was sent or received."""


def is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def find_ollama_exe() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in plat.current().ollama_extra_paths():
        if candidate.is_file():
            return str(candidate)
    return None


def launch_ollama_serve(exe: str) -> None:
    """Start `ollama serve` detached (no console window on Windows, its own session
    elsewhere). Never killed by us."""
    subprocess.Popen(
        [exe, "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        **plat.current().detached_popen_kwargs(),
    )


async def server_responds(base_url: str, timeout: float = 1.5) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return (await client.get(f"{base_url}/api/tags")).status_code == 200
    except httpx.HTTPError:
        return False


async def _wait_until_up(base_url: str, deadline: float) -> bool:
    while time.monotonic() < deadline:
        if await server_responds(base_url):
            return True
        await asyncio.sleep(0.5)
    return False


async def ensure_ollama_running(base_url: str) -> bool:
    """Try (once per process) to start a loopback Ollama server; True when it answers.

    Returns False without side effects for a non-loopback URL. After the one attempt,
    `_autostart.failure` says why it did not work.
    """
    if not is_loopback_url(base_url):
        return False
    with _autostart.lock:
        first = not _autostart.attempted
        _autostart.attempted = True
        if first:
            exe = find_ollama_exe()
            if exe is None:
                places = ", ".join(str(c.parent) for c in plat.current().ollama_extra_paths())
                _autostart.failure = "ollama was not found on PATH" + (f" or in {places}" if places else "")
            else:
                try:
                    launch_ollama_serve(exe)
                    _autostart.launched = True
                except OSError as exc:
                    _autostart.failure = f"launching it failed: {exc.__class__.__name__}"
    if not _autostart.launched:
        return False
    if await _wait_until_up(base_url, time.monotonic() + _STARTUP_WAIT):
        log.info("Started the Ollama server")
        return True
    _autostart.failure = f"it did not answer within {_STARTUP_WAIT:g}s"
    return False


def message_to_ollama(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
        ]
    if m.role == "tool" and m.name:
        out["tool_name"] = m.name
    return out


class NDJSONAssembler:
    """Accumulates Ollama /api/chat stream objects (one decoded JSON object each)."""

    def __init__(self, model: str = "") -> None:
        self._model = model
        self._think = ThinkStripper()
        self._text: list[str] = []
        self._calls: list[ToolCall] = []
        self._reason = "stop"
        self._usage: Usage | None = None

    def feed(self, obj: dict[str, Any]) -> list[str]:
        if obj.get("error"):
            raise LLMBadRequest(f"Ollama error: {str(obj['error'])[:300]}")
        out: list[str] = []
        message = obj.get("message") or {}
        # message["thinking"] (reasoning models) is intentionally ignored.
        content = message.get("content")
        if content:
            visible = self._think.feed(content)
            if visible:
                self._text.append(visible)
                out.append(visible)
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if fn.get("name"):
                self._calls.append(
                    ToolCall(id=tc.get("id") or new_call_id(), name=fn["name"], arguments=args or {})
                )
        if obj.get("done"):
            self._reason = obj.get("done_reason") or "stop"
            if "prompt_eval_count" in obj or "eval_count" in obj:
                self._usage = Usage(
                    prompt_tokens=obj.get("prompt_eval_count") or 0,
                    completion_tokens=obj.get("eval_count") or 0,
                )
            self._model = obj.get("model") or self._model
        return out

    def finish(self) -> tuple[list[str], LLMResponse]:
        tail = self._think.flush()
        tail_parts = [tail] if tail else []
        self._text.extend(tail_parts)
        reason = "tool_calls" if self._calls else self._reason
        return tail_parts, LLMResponse(
            content="".join(self._text),
            tool_calls=self._calls,
            finish_reason=reason,
            usage=self._usage,
            model=self._model,
        )


class OllamaClient:
    name = "ollama"
    is_local = True

    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int,
        request_timeout: float = 90.0,
        keep_alive: str = "30m",
        autostart: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.context_window = num_ctx
        self._keep_alive = keep_alive
        self._autostart = autostart
        self._transport: httpx.AsyncBaseTransport | None = None  # tests inject a mock
        self._timeout = httpx.Timeout(request_timeout, connect=5.0)

    def _unreachable(self) -> LLMUnavailable:
        if self._autostart and is_loopback_url(self.base_url) and _autostart.failure:
            return LLMUnavailable(_START_FAILED.format(url=self.base_url, why=_autostart.failure))
        return LLMUnavailable(_NOT_RUNNING.format(url=self.base_url))

    async def _revive(self) -> bool:
        return self._autostart and await ensure_ollama_running(self.base_url)

    def _body(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"num_ctx": self.context_window}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [message_to_ollama(m) for m in messages],
            "stream": True,
            "options": options,
            "keep_alive": self._keep_alive,
        }
        if tools:
            body["tools"] = tools
        return body

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        body = self._body(messages, tools, max_tokens, temperature)
        try:
            async for event in self._attempt(body):
                yield event
            return
        except _Refused:
            # Connection refused happens before any output, so retrying cannot duplicate text.
            if not await self._revive():
                raise self._unreachable() from None
        async for event in self._attempt(body):
            yield event

    async def _attempt(self, body: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        assembler = NDJSONAssembler(self.model)
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=body) as resp:
                    if resp.status_code >= 400:
                        await self._raise_status(resp)
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            log.warning("Skipping non-JSON line from Ollama")
                            continue
                        for text in assembler.feed(obj):
                            yield TextDelta(text)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise _Refused(_NOT_RUNNING.format(url=self.base_url)) from None
        except httpx.TimeoutException:
            raise LLMUnavailable("Ollama timed out (model may still be loading).") from None
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama connection failed ({type(exc).__name__}).") from None
        tail, final = assembler.finish()
        for text in tail:
            yield TextDelta(text)
        yield ResponseDone(final)

    async def _raise_status(self, resp: httpx.Response) -> None:
        detail = (await resp.aread()).decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        detail = str(detail).strip()[:300]
        code = resp.status_code
        if code == 429:
            raise LLMRateLimited(f"Ollama busy: {detail}")
        if code >= 500:
            raise LLMUnavailable(f"Ollama server error {code}: {detail}")
        if code == 404:
            raise LLMBadRequest(
                f"{detail} (try `ollama pull {self.model}`)"
            )
        raise LLMBadRequest(f"Ollama rejected the request ({code}): {detail}")

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        async for event in self.stream(
            messages, tools, max_tokens=max_tokens, temperature=temperature
        ):
            if isinstance(event, ResponseDone):
                return event.response
        raise LLMUnavailable("Stream ended without a response.")

    async def _pulled_models(self) -> set[str]:
        async with httpx.AsyncClient(timeout=2.0, transport=self._transport) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return {m.get("name", "") for m in resp.json().get("models", [])}

    async def is_available(self) -> bool:
        """True when Ollama answers and the configured model is pulled."""
        try:
            names = await self._pulled_models()
        except httpx.ConnectError:
            if not await self._revive():
                return False
            try:
                names = await self._pulled_models()
            except (httpx.HTTPError, ValueError):
                return False
        except (httpx.HTTPError, ValueError):
            return False
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names
