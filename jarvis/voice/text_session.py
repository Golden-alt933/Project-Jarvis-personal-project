"""A session with no audio: typed requests in, reply text out, same events as VoiceSession.

Deliberately imports nothing from the speech stack (no numpy, no sounddevice), so
the HUD can run on a machine without the voice extra.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

from jarvis.agent.events import Notice, TextChunk, ToolFinished, ToolStarted, TurnDone
from jarvis.config import Settings
from jarvis.voice.events import (
    ConfirmRequested,
    ConfirmResolved,
    EventCallback,
    Info,
    MicMuted,
    Spoke,
    StateChanged,
    ToolActivity,
    TurnFinished,
    UserText,
    VoiceEvent,
    VoiceState,
)
from jarvis.voice.summary import summarize_args

log = logging.getLogger(__name__)

APOLOGY_TEXT = "Sorry, something went wrong."
CONFIRM_TIMEOUT = 60.0
CANCEL_GRACE = 5.0  # how long a cancelled turn gets to wind down before it is forced
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


class SentenceSplitter:
    """Cuts streamed text into sentence-sized pieces; a dependency-free stand-in for
    the TTS chunker. Pieces are stripped; join them with spaces."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        parts = _SENTENCE_END.split(self._buf)
        self._buf = parts.pop()  # the unfinished tail
        return [p.strip() for p in parts if p.strip()]

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


@dataclass
class _Turn:
    text: str
    task: asyncio.Task[None] | None = None
    cancelled: bool = False
    got_text: bool = False
    splitter: SentenceSplitter = field(default_factory=SentenceSplitter)


class TextSession:
    has_audio = False

    def __init__(self, agent, settings: Settings, on_event: EventCallback | None = None) -> None:
        self.agent = agent
        self.settings = settings
        self.on_event = on_event
        self.confirm_timeout = CONFIRM_TIMEOUT
        self.state = VoiceState.IDLE
        self._muted = False
        self._turn: _Turn | None = None
        self._stop = asyncio.Event()
        self._confirm_future: asyncio.Future[bool] | None = None

    # -- the Session protocol --------------------------------------------------------
    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        self._emit(MicMuted(muted))

    async def run(self, greeting: str | None = None) -> None:
        if greeting:
            self._emit(Spoke(greeting))
        try:
            await self._stop.wait()
        finally:
            self.cancel()
            turn = self._turn
            if turn is not None and turn.task is not None:
                await asyncio.gather(turn.task, return_exceptions=True)

    def stop(self) -> None:
        self._stop.set()

    def is_idle(self) -> bool:
        return self._turn is None

    async def say(self, text: str) -> None:
        self._emit(Spoke(text))

    def submit_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._emit(UserText(text))
        previous = self._turn
        self.cancel()
        turn = _Turn(text)
        self._turn = turn
        turn.task = asyncio.ensure_future(self._run_turn(turn, previous))

    def cancel(self) -> None:
        turn = self._turn
        if turn is None or turn.cancelled:
            return
        turn.cancelled = True
        self.agent.cancel()
        self._resolve_confirm(False)

    def answer_confirm(self, approved: bool) -> None:
        self._resolve_confirm(approved)

    # -- the turn ---------------------------------------------------------------------
    async def _run_turn(self, turn: _Turn, previous: _Turn | None) -> None:
        started = time.monotonic()
        finished = False
        try:
            await self._wait_for(previous)
            if turn.cancelled:
                return
            self._set_state(VoiceState.THINKING)
            async for event in self.agent.run_turn(turn.text, output_mode="text", confirm=self._confirm):
                finished |= self._handle(turn, event)
        except asyncio.CancelledError:
            turn.cancelled = True
        except Exception:
            log.exception("text turn failed")
            if not turn.cancelled:
                self._emit(Info(APOLOGY_TEXT))
            self._emit(TurnFinished("", time.monotonic() - started, 0, 0, "", "error"))
            finished = True
        finally:
            if not finished and not turn.cancelled:
                self._emit(TurnFinished("", time.monotonic() - started, 0, 0, "", "done"))
            elif not finished:
                self._emit(TurnFinished("", time.monotonic() - started, 0, 0, "", "cancelled"))
            if self._turn is turn:
                self._turn = None
                self._set_state(VoiceState.IDLE)

    @staticmethod
    async def _wait_for(previous: _Turn | None) -> None:
        task = previous.task if previous is not None else None
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), CANCEL_GRACE)
        except asyncio.TimeoutError:
            log.warning("previous turn did not stop after cancel; forcing")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except Exception:
            log.exception("previous turn ended with an error")

    def _handle(self, turn: _Turn, event) -> bool:
        """Emit what one agent event means. True when it was the turn's TurnDone."""
        if isinstance(event, TextChunk):
            turn.got_text = True
            if event.text.strip() and not turn.cancelled:
                self._set_state(VoiceState.SPEAKING)
            self._speak(turn, turn.splitter.feed(event.text))
        elif isinstance(event, ToolStarted):
            self._emit(ToolActivity(event.call.name, False, summary=summarize_args(event.call.arguments)))
        elif isinstance(event, ToolFinished):
            self._emit(ToolActivity(
                event.call.name, True, event.result.ok,
                summary=summarize_args(event.call.arguments), seconds=event.seconds,
            ))
        elif isinstance(event, Notice):
            self._emit(Info(event.text))
        elif isinstance(event, TurnDone):
            if not turn.got_text and event.stopped != "cancelled":
                self._speak(turn, turn.splitter.feed(event.text))  # e.g. an error apology
            self._speak(turn, turn.splitter.flush())
            self._emit(TurnFinished(
                event.text, event.seconds, event.steps, event.tool_calls, event.model, event.stopped,
            ))
            return True
        return False

    def _speak(self, turn: _Turn, pieces: list[str]) -> None:
        for piece in pieces:
            if turn.cancelled:
                return
            self._set_state(VoiceState.SPEAKING)
            self._emit(Spoke(piece))

    # -- confirmation -------------------------------------------------------------------
    async def _confirm(self, description: str) -> bool:
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._confirm_future = future
        self._emit(ConfirmRequested(description))
        approved = False
        try:
            approved = await asyncio.wait_for(future, self.confirm_timeout)
        except asyncio.TimeoutError:
            approved = False
        finally:
            self._confirm_future = None
            self._emit(ConfirmResolved(approved))
        return approved

    def _resolve_confirm(self, approved: bool) -> None:
        future = self._confirm_future
        if future is not None and not future.done():
            future.set_result(approved)

    # -- plumbing -----------------------------------------------------------------------
    def _set_state(self, state: VoiceState) -> None:
        if state is not self.state:
            self.state = state
            self._emit(StateChanged(state))

    def _emit(self, event: VoiceEvent) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            log.exception("event callback failed")
