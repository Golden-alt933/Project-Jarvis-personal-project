"""UI-facing session events and the session interface.

This module imports nothing heavy (no numpy, no audio), so any frontend can depend
on it: the terminal status display, the HUD, tests. A session reports everything
through one callback, `on_event(event)`; a frontend never reaches into the session.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


class VoiceState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class StateChanged:
    state: VoiceState


@dataclass
class Heard:
    """Something was transcribed. `addressed` is False when it was not for Jarvis."""

    text: str
    addressed: bool
    confidence: float


@dataclass
class UserText:
    """The user typed a request (HUD text box) instead of speaking it."""

    text: str


@dataclass
class Spoke:
    """A sentence of the reply, emitted as it starts to be spoken (or shown, in a
    text-only session). Concatenate with spaces for the full reply."""

    text: str


@dataclass
class ToolActivity:
    name: str
    done: bool
    ok: bool | None = None
    summary: str = ""  # compact one-line view of the arguments, safe to display
    seconds: float = 0.0  # duration, set when done


@dataclass
class Levels:
    """Audio levels 0..1, sent about 15 times a second while audio is running."""

    mic: float
    speaker: float


@dataclass
class Info:
    text: str


@dataclass
class ConfirmRequested:
    """A risky action is waiting for the user's yes or no (voice or UI button)."""

    description: str


@dataclass
class ConfirmResolved:
    approved: bool


@dataclass
class MicMuted:
    muted: bool


@dataclass
class TurnFinished:
    text: str
    seconds: float
    steps: int
    tool_calls: int
    model: str
    stopped: str  # done | max_steps | time_budget | cancelled | error


VoiceEvent = (
    StateChanged
    | Heard
    | UserText
    | Spoke
    | ToolActivity
    | Levels
    | Info
    | ConfirmRequested
    | ConfirmResolved
    | MicMuted
    | TurnFinished
)
EventCallback = Callable[[VoiceEvent], None]


@runtime_checkable
class Session(Protocol):
    """What a frontend may call. Implemented by VoiceSession and TextSession."""

    has_audio: bool  # False for a text-only session
    muted: bool

    async def run(self, greeting: str | None = None) -> None:
        """Run until stop() is called."""
        ...

    def stop(self) -> None: ...

    def is_idle(self) -> bool: ...

    async def say(self, text: str) -> None:
        """Speak (or show) a line that is not part of a turn, e.g. a reminder."""
        ...

    def submit_text(self, text: str) -> None:
        """Start a turn from typed text. Interrupts whatever Jarvis is doing, exactly
        like a spoken barge-in. Safe to call from the UI thread of the same loop."""
        ...

    def cancel(self) -> None:
        """Stop the current turn and any speech. No-op when idle."""
        ...

    def answer_confirm(self, approved: bool) -> None:
        """Answer the pending ConfirmRequested from the UI. No-op if none is pending."""
        ...

    def set_muted(self, muted: bool) -> None:
        """Ignore the microphone (it stays open). Emits MicMuted."""
        ...
