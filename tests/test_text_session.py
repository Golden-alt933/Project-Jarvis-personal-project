"""TextSession tests: a scripted fake agent, no audio, no models, no network."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import time
from typing import Any, AsyncIterator, Callable

import pytest

from jarvis.agent.events import Notice, TextChunk, ToolFinished, ToolStarted, TurnDone
from jarvis.config import Settings
from jarvis.llm.types import ToolCall
from jarvis.tools.base import ToolResult
from jarvis.voice.events import (
    ConfirmRequested,
    ConfirmResolved,
    Info,
    MicMuted,
    Session,
    Spoke,
    StateChanged,
    ToolActivity,
    TurnFinished,
    UserText,
    VoiceState,
)
from jarvis.voice.text_session import SentenceSplitter, TextSession

IDLE, THINKING, SPEAKING = VoiceState.IDLE, VoiceState.THINKING, VoiceState.SPEAKING


def done(text: str = "", stopped: str = "done") -> TurnDone:
    return TurnDone(text, 1, 0, 0.5, "fake", stopped=stopped)


class FakeAgent:
    def __init__(self, scripts: dict[str, Callable[..., AsyncIterator[Any]]] | None = None) -> None:
        self.scripts = scripts or {}
        self.texts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []
        self.cancel_count = 0
        self._cancel = asyncio.Event()

    def cancel(self) -> None:
        self.cancel_count += 1
        self._cancel.set()

    async def wait(self, seconds: float) -> bool:
        try:
            await asyncio.wait_for(self._cancel.wait(), seconds)
            return True
        except asyncio.TimeoutError:
            return False

    async def run_turn(self, text: str, *, confirm: Any = None, notify: Any = None, output_mode: str = "text") -> AsyncIterator[Any]:
        self._cancel = asyncio.Event()
        self.texts.append(text)
        self.kwargs.append({"output_mode": output_mode, "confirm": confirm})
        script = self.scripts.get(text, self._default)
        async for event in script(self, text, confirm):
            yield event

    @staticmethod
    async def _default(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        yield TextChunk(f"Reply to {text}.")
        yield done(f"Reply to {text}.")


def slow(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
    async def gen() -> AsyncIterator[Any]:
        yield TextChunk("Working on it. ")
        if await agent.wait(5.0):
            yield done("Working on it.", stopped="cancelled")
            return
        yield done("Working on it.")

    return gen()


async def until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


class Rig:
    def __init__(self, scripts: dict[str, Callable[..., AsyncIterator[Any]]] | None = None) -> None:
        self.agent = FakeAgent(scripts)
        self.events: list[Any] = []
        self.session = TextSession(self.agent, Settings(provider="ollama"), self.events.append)
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Rig:
        self.task = asyncio.ensure_future(self.session.run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.session.stop()
        assert self.task is not None
        await asyncio.wait_for(self.task, 3.0)

    async def idle(self) -> None:
        await until(self.session.is_idle)

    def of(self, cls: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, cls)]

    def states(self) -> list[VoiceState]:
        return [e.state for e in self.of(StateChanged)]

    def spoke(self) -> list[str]:
        return [e.text for e in self.of(Spoke)]


def test_both_sessions_satisfy_the_session_protocol() -> None:
    from jarvis.voice.session import VoiceSession

    assert isinstance(TextSession(FakeAgent(), Settings(provider="ollama")), Session)
    assert issubclass(VoiceSession, object) and all(
        hasattr(VoiceSession, name) for name in ("submit_text", "cancel", "answer_confirm", "set_muted", "muted", "has_audio")
    )


def test_importing_text_session_does_not_import_the_speech_stack() -> None:
    code = textwrap.dedent(
        """
        import sys
        sys.modules["numpy"] = None
        sys.modules["sounddevice"] = None
        import jarvis.voice.text_session
        bad = [m for m in sys.modules if m.startswith(("jarvis.voice.tts", "jarvis.voice.stt", "jarvis.voice.audio", "jarvis.voice.session"))]
        assert not bad, bad
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        (["Hello there. How are", " you? Fine!"], ["Hello there.", "How are you?", "Fine!"]),
        (["One", ".", " Two"], ["One.", "Two"]),
        (["Line one\nLine two"], ["Line one", "Line two"]),
        ([""], []),
    ],
)
def test_sentence_splitter(deltas: list[str], expected: list[str]) -> None:
    splitter = SentenceSplitter()
    out = [piece for d in deltas for piece in splitter.feed(d)] + splitter.flush()
    assert out == expected


async def test_plain_reply_is_streamed_into_sentence_pieces() -> None:
    reply = "It is noon. The sun is high! Anything else?"

    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        for i in range(0, len(reply), 7):
            yield TextChunk(reply[i : i + 7])
        yield done(reply)

    async with Rig({"time": script}) as rig:
        rig.session.submit_text("time")
        await rig.idle()
        assert rig.spoke() == ["It is noon.", "The sun is high!", "Anything else?"]
        assert " ".join(rig.spoke()) == reply
        assert rig.agent.kwargs[0]["output_mode"] == "text"
        assert rig.of(UserText) == [UserText("time")]
        assert rig.states() == [THINKING, SPEAKING, IDLE]
        finished = rig.of(TurnFinished)
        assert [(f.stopped, f.model, f.steps) for f in finished] == [("done", "fake", 1)]


async def test_blank_text_is_ignored_and_state_is_idle_without_a_turn() -> None:
    async with Rig() as rig:
        rig.session.submit_text("   ")
        assert rig.events == [] and rig.session.is_idle()


async def test_tool_events_and_notices() -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        call = ToolCall("1", "write_file", {"path": "a.txt", "api_key": "sk-123"})
        yield Notice("cloud unreachable, using local model")
        yield ToolStarted(call)
        yield ToolFinished(call, ToolResult.success("ok"), 2.5)
        yield TextChunk("Done.")
        yield done("Done.")

    async with Rig({"go": script}) as rig:
        rig.session.submit_text("go")
        await rig.idle()
        assert rig.of(Info) == [Info("cloud unreachable, using local model")]
        started, finished = rig.of(ToolActivity)
        assert started.summary == finished.summary == "path=a.txt, api_key=***"
        assert (started.done, finished.done, finished.ok, finished.seconds) == (False, True, True, 2.5)


async def test_say_emits_spoke_and_mute_is_recorded() -> None:
    async with Rig() as rig:
        await rig.session.say("Reminder: stand up.")
        rig.session.set_muted(True)
        assert rig.session.muted and rig.session.has_audio is False
        assert rig.events == [Spoke("Reminder: stand up."), MicMuted(True)]


def confirm_script(asked: list[bool]) -> Callable[..., AsyncIterator[Any]]:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        asked.append(await confirm("delete the file"))
        yield TextChunk("Deleted." if asked[-1] else "Left it alone.")
        yield done()

    return script


@pytest.mark.parametrize("approved", [True, False])
async def test_confirmation_is_answered_from_the_ui(approved: bool) -> None:
    asked: list[bool] = []
    async with Rig({"rm": confirm_script(asked)}) as rig:
        rig.session.submit_text("rm")
        await until(lambda: bool(rig.of(ConfirmRequested)))
        rig.session.answer_confirm(approved)
        rig.session.answer_confirm(not approved)  # ignored
        await rig.idle()
        assert asked == [approved]
        assert rig.of(ConfirmRequested) == [ConfirmRequested("delete the file")]
        assert rig.of(ConfirmResolved) == [ConfirmResolved(approved)]
        assert rig.spoke() == ["Deleted." if approved else "Left it alone."]


async def test_confirmation_timeout_is_a_no() -> None:
    asked: list[bool] = []
    async with Rig({"rm": confirm_script(asked)}) as rig:
        rig.session.confirm_timeout = 0.05
        rig.session.submit_text("rm")
        await rig.idle()
        assert asked == [False]
        assert rig.of(ConfirmResolved) == [ConfirmResolved(False)]


async def test_answer_confirm_with_nothing_pending_is_a_no_op() -> None:
    async with Rig() as rig:
        rig.session.answer_confirm(True)
        assert rig.events == []


async def test_cancel_during_confirmation_resolves_false() -> None:
    asked: list[bool] = []

    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        asked.append(await confirm("delete the file"))
        yield done(stopped="cancelled")

    async with Rig({"rm": script}) as rig:
        rig.session.submit_text("rm")
        await until(lambda: bool(rig.of(ConfirmRequested)))
        rig.session.cancel()
        await rig.idle()
        assert asked == [False]
        assert rig.of(ConfirmResolved) == [ConfirmResolved(False)]
        assert [f.stopped for f in rig.of(TurnFinished)] == ["cancelled"]


async def test_submit_while_running_cancels_the_first_turn() -> None:
    async with Rig({"first": slow}) as rig:
        rig.session.submit_text("first")
        await until(lambda: rig.session.state is SPEAKING)
        rig.session.submit_text("second")
        await rig.idle()
        assert rig.agent.cancel_count >= 1
        assert rig.agent.texts == ["first", "second"]
        assert [f.stopped for f in rig.of(TurnFinished)] == ["cancelled", "done"]
        assert rig.spoke() == ["Working on it.", "Reply to second."]
        assert rig.states()[-1] is IDLE and rig.states().count(IDLE) == 1


async def test_cancel_and_repeated_cancel() -> None:
    async with Rig({"first": slow}) as rig:
        rig.session.cancel()  # idle: no-op
        assert rig.agent.cancel_count == 0
        rig.session.submit_text("first")
        await until(lambda: rig.session.state is SPEAKING)
        rig.session.cancel()
        rig.session.cancel()
        await rig.idle()
        assert rig.agent.cancel_count == 1
        assert [f.stopped for f in rig.of(TurnFinished)] == ["cancelled"]
        assert rig.states() == [THINKING, SPEAKING, IDLE]


async def test_agent_exception_does_not_kill_the_session() -> None:
    async def boom(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        raise RuntimeError("agent exploded")
        yield  # pragma: no cover

    async with Rig({"bad": boom}) as rig:
        rig.session.submit_text("bad")
        await rig.idle()
        assert [i.text for i in rig.of(Info)] == ["Sorry, something went wrong."]
        assert [f.stopped for f in rig.of(TurnFinished)] == ["error"]
        assert rig.session.state is IDLE
        rig.session.submit_text("hello")
        await rig.idle()
        assert rig.spoke() == ["Reply to hello."]


async def test_event_callback_failure_is_contained() -> None:
    agent = FakeAgent()

    def bad(event: Any) -> None:
        raise ValueError("ui bug")

    session = TextSession(agent, Settings(provider="ollama"), bad)
    runner = asyncio.ensure_future(session.run())
    session.submit_text("hi")
    await until(session.is_idle)
    session.stop()
    await asyncio.wait_for(runner, 3.0)
    assert agent.texts == ["hi"]
