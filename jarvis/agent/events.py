"""Events the agent emits during a turn. Frontends (CLI, voice, HUD) render
these; the agent never prints or speaks itself."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.llm.types import ToolCall, Usage
from jarvis.tools.base import ToolResult


@dataclass
class TextChunk:
    """Assistant text as it streams. Concatenate chunks for the full reply."""

    text: str


@dataclass
class ToolStarted:
    call: ToolCall


@dataclass
class ToolFinished:
    call: ToolCall
    result: ToolResult
    seconds: float


@dataclass
class Notice:
    """Out-of-band information for the user, e.g. 'cloud unreachable, using local model'."""

    text: str


@dataclass
class TurnDone:
    text: str  # the final assistant reply
    steps: int  # model calls made this turn
    tool_calls: int
    seconds: float
    model: str
    usage: Usage | None = None
    stopped: str = "done"  # done | max_steps | time_budget | cancelled | error


AgentEvent = TextChunk | ToolStarted | ToolFinished | Notice | TurnDone
