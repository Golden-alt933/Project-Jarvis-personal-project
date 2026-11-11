"""A scripted fake Session for `jarvis --hud-demo`: no mic, no model, a realistic loop of events."""

from __future__ import annotations

import asyncio
import math
import random

from jarvis.voice.events import (
    CodeShown,
    ConfirmRequested,
    ConfirmResolved,
    EventCallback,
    Heard,
    Info,
    Levels,
    MicMuted,
    Spoke,
    StateChanged,
    ToolActivity,
    TurnFinished,
    UserText,
    VoiceEvent,
    VoiceState,
)

LEVEL_HZ = 15.0
DEMO_CODE = '''def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
'''


class DemoSession:
    has_audio = True

    def __init__(self, on_event: EventCallback, *, speed: float = 1.0, loop: bool = True) -> None:
        self.on_event = on_event
        self.speed = speed
        self.loop = loop
        self.muted = False
        self._stop = asyncio.Event()
        self._confirm: asyncio.Future[bool] | None = None
        self._rng = random.Random(3)
        self._busy = False

    # -- Session protocol ------------------------------------------------------------
    def is_idle(self) -> bool:
        return not self._busy

    def stop(self) -> None:
        self._stop.set()

    def cancel(self) -> None:
        self.on_event(StateChanged(VoiceState.IDLE))

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        self.on_event(MicMuted(muted))

    def answer_confirm(self, approved: bool) -> None:
        if self._confirm is not None and not self._confirm.done():
            self._confirm.set_result(approved)

    def submit_text(self, text: str) -> None:
        asyncio.ensure_future(self._typed(text))

    async def say(self, text: str) -> None:
        self.on_event(Spoke(text))

    async def run(self, greeting: str | None = None) -> None:
        if greeting:
            await self._speak([greeting], 0.5)
        while not self._stop.is_set():
            await self.script()
            if not self.loop:
                return
            await self._sleep(2.5)

    # -- the script ------------------------------------------------------------------
    async def script(self) -> None:
        emit = self.on_event
        self._busy = True
        await self._sleep(1.2)
        emit(StateChanged(VoiceState.LISTENING))
        await self._levels(1.1, mic=0.55, speaker=0.0)
        emit(Heard("jarvis, what is on my calendar tomorrow", True, 0.93))
        await self._levels(1.2, mic=0.8, speaker=0.0)
        emit(StateChanged(VoiceState.THINKING))
        emit(ToolActivity("calendar_events", False, summary="day=tomorrow"))
        await self._sleep(1.4)
        emit(ToolActivity("calendar_events", True, True, "day=tomorrow", 1.3))
        emit(ToolActivity("web_search", False, summary="query=weather in Pune tomorrow"))
        await self._sleep(1.1)
        emit(ToolActivity("web_search", True, True, "query=weather in Pune tomorrow", 1.0))
        await self._speak(
            ["You have three events tomorrow.", "Standup at nine thirty, a design review at one,",
             "and dinner with Priya at eight.", "It should stay clear and warm all day."],
            0.7,
        )
        emit(TurnFinished("", 6.2, 3, 2, "nemotron-3-ultra", "done"))
        emit(StateChanged(VoiceState.IDLE))
        await self._sleep(2.0)
        emit(Heard("jarvis, write a python function that returns the nth fibonacci number", True, 0.92))
        emit(StateChanged(VoiceState.THINKING))
        await self._sleep(1.0)
        await self._speak(["Here is an iterative version that runs in linear time."], 0.7)
        emit(CodeShown(DEMO_CODE, "python"))
        emit(TurnFinished("", 2.4, 0, 0, "nemotron-3-ultra", "done"))
        emit(StateChanged(VoiceState.IDLE))
        await self._sleep(7.0)
        emit(Heard("remind me to call the bank", True, 0.9))
        emit(StateChanged(VoiceState.THINKING))
        emit(ToolActivity("send_email", False, summary="to=priya@example.com"))
        emit(ConfirmRequested("Send an email to priya@example.com with the subject \"Dinner tonight\"?"))
        self._confirm = asyncio.get_running_loop().create_future()
        try:
            approved = await asyncio.wait_for(self._confirm, 6.0 / self.speed)
        except asyncio.TimeoutError:
            approved = False
        emit(ConfirmResolved(approved))
        emit(ToolActivity("send_email", True, False, "to=priya@example.com", 0.4))
        emit(Info("Something went wrong with the mail service."))
        emit(TurnFinished("", 3.1, 1, 1, "nemotron-3-ultra", "error"))
        emit(StateChanged(VoiceState.IDLE))
        self._busy = False
        await self._sleep(3.0)

    async def _typed(self, text: str) -> None:
        self._busy = True
        self.on_event(UserText(text))
        self.on_event(StateChanged(VoiceState.THINKING))
        await self._sleep(1.0)
        await self._speak([f"You said: {text}.", "This is the demo session, so that is all I can do."], 0.6)
        self.on_event(TurnFinished("", 2.0, 1, 0, "demo", "done"))
        self.on_event(StateChanged(VoiceState.IDLE))
        self._busy = False

    async def _speak(self, sentences: list[str], level: float) -> None:
        self.on_event(StateChanged(VoiceState.SPEAKING))
        for sentence in sentences:
            self.on_event(Spoke(sentence))
            await self._levels(0.5 + 0.09 * len(sentence.split()), mic=0.0, speaker=level)

    async def _levels(self, seconds: float, *, mic: float, speaker: float) -> None:
        """Speech-like envelope: syllable-rate modulation with random emphasis."""
        steps = max(1, int(seconds * LEVEL_HZ))
        for i in range(steps):
            t = i / LEVEL_HZ
            env = 0.35 + 0.65 * abs(math.sin(t * 7.3 + self._rng.random()))
            gate = min(1.0, i / 3.0) * min(1.0, (steps - i) / 3.0)
            self.on_event(Levels(mic * env * gate * (0.0 if self.muted else 1.0), speaker * env * gate))
            await self._sleep(1.0 / LEVEL_HZ)
        self.on_event(Levels(0.0, 0.0))

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), seconds / self.speed)
        except asyncio.TimeoutError:
            pass
