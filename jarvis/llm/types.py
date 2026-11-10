"""Provider-neutral LLM types. Everything above the client speaks only these."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role == "tool"
    name: str | None = None  # tool name on role == "tool"

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "assistant" and self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in self.tool_calls
            ]
        if self.role == "tool":
            out["tool_call_id"] = self.tool_call_id or ""
            if self.name:
                out["name"] = self.name
        return out

    def to_dict(self) -> dict[str, Any]:
        """Lossless form for persistence."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content") or "",
            tool_calls=[ToolCall(**c) for c in d.get("tool_calls") or []],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str = "stop"
    usage: Usage | None = None
    model: str = ""


@dataclass
class TextDelta:
    """A chunk of assistant text, emitted as it is generated."""

    text: str


@dataclass
class ResponseDone:
    """Always the last stream event; carries the fully assembled response."""

    response: LLMResponse


StreamEvent = TextDelta | ResponseDone


class LLMError(Exception):
    """Base class. `retryable` tells the caller whether a fallback is worth trying."""

    retryable = False


class LLMUnavailable(LLMError):
    """Network failure, timeout, 5xx, or the provider is not configured."""

    retryable = True


class LLMRateLimited(LLMUnavailable):
    """HTTP 429 or quota exhausted."""


class LLMBadRequest(LLMError):
    """The provider rejected the request (4xx other than 429), e.g. context too long."""


@runtime_checkable
class LLMClient(Protocol):
    name: str  # provider profile name, e.g. "nvidia"
    model: str
    context_window: int
    is_local: bool

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield TextDelta events, then exactly one ResponseDone."""
        ...

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Non-streaming convenience wrapper around stream()."""
        ...
