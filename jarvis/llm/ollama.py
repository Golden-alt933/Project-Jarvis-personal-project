"""Native Ollama client (/api/chat, NDJSON). Used instead of the /v1 shim because
the shim ignores num_ctx, which silently truncates long prompts."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.context_window = num_ctx
        self._keep_alive = keep_alive
        self._timeout = httpx.Timeout(request_timeout, connect=5.0)

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
        assembler = NDJSONAssembler(self.model)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
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
            raise LLMUnavailable(_NOT_RUNNING.format(url=self.base_url)) from None
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

    async def is_available(self) -> bool:
        """True when Ollama answers and the configured model is pulled."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return False
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names
