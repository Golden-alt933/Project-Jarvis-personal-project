"""One streaming client for every OpenAI-compatible endpoint.

Stream assembly lives in `ChunkAssembler`, which has no I/O so it can be tested
by feeding it fake chunks (SDK objects or plain dicts).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator

import openai

from jarvis.llm.types import (
    LLMBadRequest,
    LLMError,
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

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


class ThinkStripper:
    """Removes <think>...</think> blocks from streamed text, across chunk boundaries."""

    def __init__(self) -> None:
        self._inside = False
        self._pending = ""

    def feed(self, text: str) -> str:
        buf = self._pending + text
        self._pending = ""
        out: list[str] = []
        while buf:
            tag = _THINK_CLOSE if self._inside else _THINK_OPEN
            idx = buf.find(tag)
            if idx >= 0:
                if not self._inside:
                    out.append(buf[:idx])
                buf = buf[idx + len(tag):]
                self._inside = not self._inside
                continue
            # Hold back a tail that could be the start of the tag.
            keep = _partial_suffix(buf, tag)
            if not self._inside:
                out.append(buf[: len(buf) - keep])
            self._pending = buf[len(buf) - keep:]
            break
        return "".join(out)

    def flush(self) -> str:
        """End of stream: release held-back text unless it sits inside a think block."""
        rest, self._pending = self._pending, ""
        return "" if self._inside else rest


def _partial_suffix(text: str, tag: str) -> int:
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def parse_arguments(raw: str) -> dict[str, Any]:
    """JSON-decode tool arguments; repair lightly; never raise."""
    text = raw.strip()
    if not text:
        return {}
    for candidate in (text, _repair_json(text)):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
        break
    return {"_raw": raw}


def _repair_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return re.sub(r",\s*([}\]])", r"\1", text)


class ChunkAssembler:
    """Accumulates OpenAI-style streaming chunks into text deltas and a final response."""

    def __init__(self, model: str = "") -> None:
        self._model = model
        self._think = ThinkStripper()
        self._text: list[str] = []
        self._calls: dict[int, dict[str, str]] = {}
        self._finish: str | None = None
        self._usage: Usage | None = None
        self.reasoning_chars = 0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self._calls)

    def feed(self, chunk: Any) -> list[str]:
        """Consume one chunk; return visible text deltas (possibly empty)."""
        usage = _get(chunk, "usage")
        if usage is not None:
            self._usage = Usage(
                prompt_tokens=_get(usage, "prompt_tokens") or 0,
                completion_tokens=_get(usage, "completion_tokens") or 0,
            )
        model = _get(chunk, "model")
        if model:
            self._model = model
        out: list[str] = []
        for choice in _get(chunk, "choices") or []:
            if _get(choice, "finish_reason"):
                self._finish = _get(choice, "finish_reason")
            delta = _get(choice, "delta")
            # Reasoning models stream their chain of thought in a separate field. It is
            # counted (diagnostics only) and never becomes text or part of the reply.
            self.reasoning_chars += len(
                _get(delta, "reasoning_content") or _get(delta, "reasoning") or ""
            )
            content = _get(delta, "content")
            if content:
                visible = self._think.feed(content)
                if visible:
                    self._text.append(visible)
                    out.append(visible)
            for tc in _get(delta, "tool_calls") or []:
                self._add_tool_delta(tc)
        return out

    def _slot_for(self, tc: Any) -> int:
        index = _get(tc, "index")
        if index is not None:
            return index
        call_id = _get(tc, "id")
        if call_id:
            for i, slot in self._calls.items():
                if slot["id"] == call_id:
                    return i
            return len(self._calls)
        return max(self._calls) if self._calls else 0

    def _add_tool_delta(self, tc: Any) -> None:
        slot = self._calls.setdefault(
            self._slot_for(tc), {"id": "", "name": "", "args": ""}
        )
        if _get(tc, "id"):
            slot["id"] = _get(tc, "id")
        fn = _get(tc, "function")
        name = _get(fn, "name")
        if name and name != slot["name"]:
            slot["name"] += name
        args = _get(fn, "arguments")
        if args:
            slot["args"] += args if isinstance(args, str) else json.dumps(args)

    def finish(self) -> tuple[list[str], LLMResponse]:
        """Flush held-back text; return (final deltas, assembled response)."""
        tail = self._think.flush()
        tail_parts = [tail] if tail else []
        self._text.extend(tail_parts)
        calls = [
            ToolCall(
                id=slot["id"] or new_call_id(),
                name=slot["name"],
                arguments=parse_arguments(slot["args"]),
            )
            for _, slot in sorted(self._calls.items())
            if slot["name"]
        ]
        reason = self._finish or "stop"
        if calls and reason == "stop":
            reason = "tool_calls"
        return tail_parts, LLMResponse(
            content="".join(self._text),
            tool_calls=calls,
            finish_reason=reason,
            usage=self._usage,
            model=self._model,
        )


def build_payload_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [m.to_openai() for m in messages]


def map_openai_error(exc: Exception, secret: str = "") -> LLMError:
    """Translate an SDK exception into the provider-neutral hierarchy."""

    def clean(text: str) -> str:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):  # SDK message is often just "Error code: 404"
            detail = body.get("detail") or body.get("message") or (body.get("error") or {})
            text = f"{text} {detail if isinstance(detail, str) else detail.get('message', '')}"
        text = text.replace(secret, "***") if secret else text
        return text.strip()[:300]

    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimited(f"Rate limited by provider: {clean(exc.message)}")
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return LLMUnavailable(f"Could not reach the provider ({type(exc).__name__}).")
    if isinstance(exc, openai.InternalServerError):
        return LLMUnavailable(f"Provider server error {exc.status_code}: {clean(exc.message)}")
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return LLMUnavailable(f"Provider server error {exc.status_code}: {clean(exc.message)}")
        if exc.status_code == 429:
            return LLMRateLimited(f"Rate limited by provider: {clean(exc.message)}")
        return LLMBadRequest(f"Request rejected ({exc.status_code}): {clean(exc.message)}")
    return LLMUnavailable(f"Unexpected provider failure: {type(exc).__name__}")


_END = object()


class _FirstByteTimeout(Exception):
    """Internal: nothing arrived from the provider within the first-byte window."""


def completion_to_chunk(resp: Any) -> dict[str, Any]:
    """Re-express a non-streaming completion as one streaming chunk so the same
    assembler (think-stripping, tool-call repair, reasoning handling) serves both modes."""
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    choices = []
    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        calls = [
            {"index": i, "id": c.get("id"), "function": c.get("function") or {}}
            for i, c in enumerate(message.get("tool_calls") or [])
        ]
        delta = {
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content") or message.get("reasoning"),
            "tool_calls": calls,
        }
        choices.append({"delta": delta, "finish_reason": choice.get("finish_reason")})
    return {"choices": choices, "usage": data.get("usage"), "model": data.get("model")}


class OpenAICompatClient:
    is_local = False

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        context_window: int,
        request_timeout: float = 90.0,
        *,
        first_byte_timeout: float = 0.0,
        extra_body: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
        streaming: bool = True,
    ) -> None:
        """`extra_body` is merged into the JSON body (provider-specific fields such as
        reasoning_effort); `extra_params` are SDK keyword arguments (top_p, ...).
        `first_byte_timeout` (seconds, 0 = off): a streaming request that has produced no
        data by then is abandoned and retried once. `streaming=False` makes stream() do one
        plain request and emit the content as a single delta.

        The same timeout bounds the rest of the stream: a gap that long between chunks, or
        three times that long with reasoning but no reply text and no tool call, ends the
        request as unavailable so the caller can move to another model."""
        self.name = name
        self.model = model
        self.context_window = context_window
        self.streaming = streaming
        self._api_key = api_key
        self._first_byte_timeout = first_byte_timeout
        self._extra_body = dict(extra_body or {})
        self._extra_params = dict(extra_params or {})
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=request_timeout,
            max_retries=1,
        )

    def _request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self.model,
            "messages": build_payload_messages(messages),
            "stream": self.streaming,
        }
        if tools:
            req["tools"] = tools
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        if temperature is not None:
            req["temperature"] = temperature
        req.update(self._extra_params)
        if self._extra_body:
            req["extra_body"] = dict(self._extra_body)
        return req

    async def _open(self, req: dict[str, Any]) -> Any:
        try:
            return await self._client.chat.completions.create(
                **req, stream_options={"include_usage": True}
            )
        except (openai.BadRequestError, openai.UnprocessableEntityError) as exc:
            log.info("%s rejected stream_options; retrying without (%s)", self.name, exc.status_code)
        return await self._client.chat.completions.create(**req)

    async def _open_and_first(self, req: dict[str, Any], opened: list[Any]) -> tuple[Any, Any]:
        response = await self._open(req)
        opened.append(response)
        iterator = response.__aiter__()
        return iterator, await anext(iterator, _END)

    async def _first_chunk(self, req: dict[str, Any]) -> tuple[Any, Any]:
        """Open the stream and read its first chunk, bounded by the first-byte timeout."""
        opened: list[Any] = []
        work = self._open_and_first(req, opened)
        if self._first_byte_timeout <= 0:
            return await work
        try:
            return await asyncio.wait_for(work, self._first_byte_timeout)
        except asyncio.TimeoutError:
            for response in opened:
                with contextlib.suppress(Exception):
                    await response.close()
            raise _FirstByteTimeout from None

    async def _next_chunk(self, iterator: Any, started: float, spoke: bool) -> Any:
        """The next chunk, or LLMUnavailable when the stream has stalled or only thinks."""
        limit = self._first_byte_timeout
        if limit <= 0:
            return await anext(iterator, _END)
        if not spoke and time.monotonic() - started > 3 * limit:
            await self._abandon(iterator)
            raise LLMUnavailable(f"{self.model} produced no reply within {3 * limit:g}s.")
        try:
            return await asyncio.wait_for(anext(iterator, _END), limit)
        except asyncio.TimeoutError:
            await self._abandon(iterator)
            raise LLMUnavailable(f"{self.model} stopped sending mid-reply for {limit:g}s.") from None

    @staticmethod
    async def _abandon(iterator: Any) -> None:
        for name in ("aclose", "close"):
            closer = getattr(iterator, name, None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer()
                return

    async def _stream_once(self, req: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        assembler = ChunkAssembler(self.model)
        try:
            iterator, chunk = await self._first_chunk(req)
            started, spoke = time.monotonic(), False
            while chunk is not _END:
                for text in assembler.feed(chunk):
                    spoke = True
                    yield TextDelta(text)
                spoke = spoke or assembler.has_tool_calls
                chunk = await self._next_chunk(iterator, started, spoke)
        except openai.OpenAIError as exc:
            raise map_openai_error(exc, self._api_key) from None
        tail, final = assembler.finish()
        for text in tail:
            yield TextDelta(text)
        self._note_empty(final, assembler)
        yield ResponseDone(final)

    async def _single_request(self, req: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        assembler = ChunkAssembler(self.model)
        try:
            resp = await self._client.chat.completions.create(**req)
        except openai.OpenAIError as exc:
            raise map_openai_error(exc, self._api_key) from None
        deltas = assembler.feed(completion_to_chunk(resp))
        tail, final = assembler.finish()
        text = "".join([*deltas, *tail])
        if text:
            yield TextDelta(text)
        self._note_empty(final, assembler)
        yield ResponseDone(final)

    def _note_empty(self, final: LLMResponse, assembler: ChunkAssembler) -> None:
        if not final.content and not final.tool_calls and assembler.reasoning_chars:
            log.warning(
                "%s produced only reasoning (%d chars, finish=%s); the reply is empty",
                self.model, assembler.reasoning_chars, final.finish_reason,
            )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        req = self._request(messages, tools, max_tokens, temperature)
        if not self.streaming:
            async for event in self._single_request(req):
                yield event
            return
        for attempt in (1, 2):
            yielded = False
            try:
                async for event in self._stream_once(req):
                    yielded = True
                    yield event
                return
            except _FirstByteTimeout:
                # Raised only before the first chunk, so `yielded` is always False here;
                # the guard keeps a retry from ever duplicating output.
                if yielded or attempt == 2:
                    raise LLMUnavailable(
                        f"{self.name} did not respond within {self._first_byte_timeout:g}s."
                    ) from None
                log.warning("%s sent nothing for %gs; retrying once", self.name, self._first_byte_timeout)

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
