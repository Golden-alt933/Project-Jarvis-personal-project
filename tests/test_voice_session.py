"""Voice session tests: fakes only, no audio hardware, no models, no network."""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable

import numpy as np
import pytest

from jarvis.agent.events import Notice, TextChunk, ToolFinished, ToolStarted, TurnDone
from jarvis.config import Settings
from jarvis.llm.types import ToolCall
from jarvis.proactive import scheduler
from jarvis.proactive.scheduler import ReminderWatcher
from jarvis.tools.base import ToolResult
from jarvis.voice import session as vs
from jarvis.voice.session import (
    Heard,
    Info,
    Spoke,
    StateChanged,
    ToolActivity,
    VoiceSession,
    VoiceState,
    is_affirmative,
    split_wake_word,
)
from jarvis.voice.types import SpeechStarted, Transcript, Utterance

IDLE, LISTENING, THINKING, SPEAKING = VoiceState.IDLE, VoiceState.LISTENING, VoiceState.THINKING, VoiceState.SPEAKING


# -- fakes ---------------------------------------------------------------------------
class FakeChunker:
    """Splits on sentence ends; stands in for tts.SentenceChunker."""

    def __init__(self) -> None:
        self.buf = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while (m := re.search(r"[.!?]\s", self.buf)):
            out.append(self.buf[: m.end()].strip())
            self.buf = self.buf[m.end():]
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


def done(text: str = "", stopped: str = "done") -> TurnDone:
    return TurnDone(text, 1, 0, 0.0, "fake", stopped=stopped)


class FakeAgent:
    """Scripted per user text: script(agent, text, confirm) yields AgentEvents."""

    def __init__(self, scripts: dict[str, Callable[..., AsyncIterator[Any]]] | None = None) -> None:
        self.scripts = scripts or {}
        self.texts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []
        self.cancel_count = 0
        self.finished_at: list[float] = []
        self._cancel = asyncio.Event()

    def cancel(self) -> None:
        self.cancel_count += 1
        self._cancel.set()

    async def wait(self, seconds: float) -> bool:
        """Sleep; True if the turn was cancelled meanwhile."""
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
            if self._cancel.is_set():
                self.finished_at.append(time.monotonic())
                yield done(stopped="cancelled")
                return
        self.finished_at.append(time.monotonic())

    @staticmethod
    async def _default(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        yield TextChunk(f"Reply to {text}. ")
        yield done(f"Reply to {text}.")


def slow_reply(seconds: float = 0.4) -> Callable[..., AsyncIterator[Any]]:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        yield TextChunk("First sentence. ")
        if await agent.wait(seconds):
            return
        yield TextChunk("Second sentence.")
        yield done("First sentence. Second sentence.")

    return script


class FakeSynth:
    name = "fake"
    sample_rate = 22050

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.delays: dict[str, float] = {}
        self.fail_on: set[str] = set()

    async def warm_up(self) -> None: ...

    async def synthesize(self, text: str) -> np.ndarray:
        self.calls.append(text)
        await asyncio.sleep(self.delays.get(text, 0.0))
        if text in self.fail_on:
            raise RuntimeError("synth boom")
        return np.full(100, 0.1, dtype=np.float32)


class FakeSpeaker:
    def __init__(self, play_time: float = 0.02) -> None:
        self.is_playing = False
        self.level = 0.0
        self.play_time = play_time
        self.played: list[float] = []  # time of each play() call
        self.interrupts = 0
        self.started = self.stopped = False
        self._gen = 0

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        self.played.append(time.monotonic())
        self.is_playing = True

    def interrupt(self) -> None:
        self.interrupts += 1
        self._gen += 1
        self.is_playing = False

    async def drain(self) -> None:
        gen, end = self._gen, time.monotonic() + self.play_time
        while time.monotonic() < end and gen == self._gen:
            await asyncio.sleep(0.005)
        if gen == self._gen:
            self.is_playing = False


class FakeMic:
    def __init__(self, frames: list[np.ndarray] | None = None, forever: bool = False) -> None:
        self.level = 0.0
        self.started = self.stopped = False
        self._frames = frames or []
        self._forever = forever

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def frames(self) -> AsyncIterator[np.ndarray]:
        for frame in self._frames:
            yield frame
            await asyncio.sleep(0)
        while self._forever:
            await asyncio.sleep(0.01)


class FakeTranscriber:
    name = "fake"

    def __init__(self, scripts: dict[int, Transcript | Exception]) -> None:
        self.scripts = scripts
        self.hints: list[str | None] = []

    async def warm_up(self) -> None: ...

    async def transcribe(self, pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
        self.hints.append(hint)
        result = self.scripts.get(int(pcm[0]), Transcript("", 0.0, 0.0))
        if isinstance(result, Exception):
            raise result
        return result


def say(text: str, confidence: float = 0.9) -> Transcript:
    return Transcript(text, 0.01, confidence)


def utt(i: int) -> Utterance:
    return Utterance(np.full(160, i, dtype=np.float32), 1.0)


def source(*steps: Any) -> Callable[[], AsyncIterator[Any]]:
    """Steps: a number sleeps, a callable runs, anything else is yielded to the session."""

    async def gen() -> AsyncIterator[Any]:
        for step in steps:
            if isinstance(step, (int, float)):
                await asyncio.sleep(step)
            elif callable(step):
                result = step()
                if inspect.isawaitable(result):
                    await result
            else:
                yield step

    return gen


async def until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


class Rig:
    def __init__(
        self,
        transcripts: dict[int, Transcript | Exception],
        steps: tuple[Any, ...] | list[Any] = (),
        *,
        agent: FakeAgent | None = None,
        play_time: float = 0.02,
        **settings: Any,
    ) -> None:
        base = dict(
            provider="ollama", wake_word="jarvis", require_wake_word=True,
            followup_seconds=10.0, barge_in="vad",
        )
        self.settings = Settings(**{**base, **settings})
        self.agent = agent or FakeAgent()
        self.mic = FakeMic()
        self.speaker = FakeSpeaker(play_time)
        self.synth = FakeSynth()
        self.transcriber = FakeTranscriber(transcripts)
        self.events: list[Any] = []
        self.now = 1000.0
        self.session = VoiceSession(
            self.agent, self.mic, self.speaker, SimpleNamespace(start_threshold=0.5, in_speech=False),
            self.transcriber, self.synth, self.settings, self.events.append,
            events=source(*steps), chunker_factory=FakeChunker, clock=lambda: self.now,
        )
        self.session.filler_after = 0.05

    async def run(self, timeout: float = 5.0) -> None:
        await asyncio.wait_for(self.session.run(), timeout)

    def states(self) -> list[VoiceState]:
        return [e.state for e in self.events if isinstance(e, StateChanged)]

    def heard(self) -> list[Heard]:
        return [e for e in self.events if isinstance(e, Heard)]

    def spoke(self) -> list[str]:
        return [e.text for e in self.events if isinstance(e, Spoke)]


# -- pure gating helpers ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "found", "rest"),
    [
        ("Jarvis, what time is it?", True, "what time is it?"),
        ("jarvis what time is it", True, "what time is it"),
        ("Hey Jarvis: open notepad", True, "open notepad"),
        ("Jervis, hello", True, "hello"),
        ("Travis what's up", True, "what's up"),
        ("Jarvus.", True, ""),
        ("JARVIS", True, ""),
        ("what time is it, jarvis", False, "what time is it, jarvis"),
        ("tell Travis hello", False, "tell Travis hello"),
        ("service please", False, "service please"),
        ("hello there", False, "hello there"),
    ],
)
def test_split_wake_word(text: str, found: bool, rest: str) -> None:
    assert split_wake_word(text, "jarvis") == (found, rest)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", True), ("Yeah, go ahead.", True), ("do it", True), ("Sure!", True), ("please do", True),
        ("confirm", True), ("no", False), ("yes no wait", False), ("maybe", False), ("", False),
        ("don't do it", False), ("yes... actually cancel", False),
    ],
)
def test_is_affirmative(text: str, expected: bool) -> None:
    assert is_affirmative(text) is expected


def test_echo_coupling_adapts_to_observed_ratio() -> None:
    rig = Rig({})
    rig.speaker.level, rig.mic.level = 0.5, 0.05  # clean headphones-like coupling
    for _ in range(3000):
        rig.session._adapt_coupling()
    assert rig.session.coupling < 0.15
    rig.mic.level = 0.5  # loud speakers leak a lot
    rig.session._adapt_coupling()
    assert rig.session.coupling == pytest.approx(1.0)
    rig.speaker.level = 0.0
    rig.mic.level = 5.0
    rig.session._adapt_coupling()  # silence from the speaker must not poison the estimate
    assert rig.session.coupling == pytest.approx(1.0)


# -- wake gating -------------------------------------------------------------------------
async def test_wake_word_is_stripped_and_hinted() -> None:
    rig = Rig({1: say("Jarvis, what time is it?")}, [SpeechStarted(), utt(1)])
    await rig.run()
    assert rig.agent.texts == ["what time is it?"]
    assert rig.agent.kwargs[0]["output_mode"] == "voice"
    assert rig.agent.kwargs[0]["confirm"] == rig.session._voice_confirm
    assert rig.transcriber.hints == ["jarvis"]
    assert rig.synth.calls == ["Reply to what time is it?."]


async def test_unaddressed_utterance_is_ignored_silently() -> None:
    rig = Rig({1: say("what time is it")}, [SpeechStarted(), utt(1)])
    await rig.run()
    assert rig.agent.texts == []
    assert rig.synth.calls == []
    assert [(h.text, h.addressed) for h in rig.heard()] == [("what time is it", False)]
    assert rig.states() == [LISTENING, IDLE]


@pytest.mark.parametrize(
    ("heard", "sent"),
    [
        ("Travis, open notepad", "open notepad"),
        ("Jervis what's the weather", "what's the weather"),
        ("what's the weather jarvis", "what's the weather jarvis"),
    ],
)
async def test_misheard_wake_word_at_start_and_wake_word_anywhere(heard: str, sent: str) -> None:
    rig = Rig({1: say(heard)}, [utt(1)])
    await rig.run()
    assert rig.agent.texts == [sent]


async def test_misheard_wake_word_mid_sentence_is_not_addressed() -> None:
    rig = Rig({1: say("tell Travis I said hello")}, [utt(1)])
    await rig.run()
    assert rig.agent.texts == []


async def test_followup_window_open_and_expired() -> None:
    expire = lambda: setattr(rig, "now", rig.now + 11.0)  # noqa: E731
    rig = Rig(
        {1: say("Jarvis, hello"), 2: say("and what about tomorrow"), 3: say("is anyone there")},
        [utt(1), 0.2, utt(2), 0.2, expire, utt(3)],
    )
    await rig.run()
    assert rig.agent.texts == ["hello", "and what about tomorrow"]  # 3 arrived after the window closed
    assert [h.addressed for h in rig.heard()] == [True, True, False]


async def test_bare_wake_word_acknowledges_and_opens_followup() -> None:
    rig = Rig({1: say("Jarvis."), 2: say("what time is it")}, [utt(1), 0.2, utt(2)])
    await rig.run()
    assert rig.synth.calls[0] == "Yes?"
    assert rig.agent.texts == ["what time is it"]


async def test_low_confidence_and_empty_are_ignored() -> None:
    rig = Rig({1: say("Jarvis, delete everything", 0.2), 2: Transcript("", 0.0, 0.0)}, [utt(1), utt(2)])
    await rig.run()
    assert rig.agent.texts == []
    assert rig.synth.calls == []


async def test_no_wake_mode_treats_everything_as_addressed() -> None:
    rig = Rig({1: say("what time is it")}, [utt(1)], require_wake_word=False)
    await rig.run()
    assert rig.agent.texts == ["what time is it"]


# -- speaking pipeline -----------------------------------------------------------------------
async def test_sentences_synthesized_in_order_and_playback_starts_before_turn_ends() -> None:
    rig = Rig({1: say("Jarvis, go")}, [utt(1)], agent=FakeAgent({"go": slow_reply(0.3)}))
    rig.synth.delays = {"First sentence.": 0.05}
    await rig.run()
    assert rig.synth.calls == ["First sentence.", "Second sentence."]
    assert rig.spoke() == ["First sentence.", "Second sentence."]
    assert len(rig.speaker.played) == 2
    assert rig.speaker.played[0] < rig.agent.finished_at[0] - 0.1  # first audio queued while the model was still going
    assert rig.states() == [THINKING, SPEAKING, IDLE]


async def test_slow_tool_gets_one_filler_when_nothing_was_said() -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        call = ToolCall("1", "web_search", {})
        for _ in range(2):  # two slow tools in a row: still a single filler
            yield ToolStarted(call)
            await asyncio.sleep(0.2)
            yield ToolFinished(call, ToolResult.success("ok"), 0.2)
        yield TextChunk("All done.")
        yield done("All done.")

    rig = Rig({1: say("Jarvis, search")}, [utt(1)], agent=FakeAgent({"search": script}))
    await rig.run()
    assert rig.synth.calls == ["One moment.", "All done."]
    tool_events = [e for e in rig.events if isinstance(e, ToolActivity)]
    assert [(e.name, e.done, e.ok) for e in tool_events] == [("web_search", False, None), ("web_search", True, True)] * 2


async def test_slow_model_without_tools_gets_filler_and_fillers_rotate() -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        await asyncio.sleep(0.2)  # the model itself is slow; no tool involved
        yield TextChunk("Here you go.")
        yield done("Here you go.")

    rig = Rig({1: say("Jarvis, think"), 2: say("Jarvis, think")}, [utt(1), 0.6, utt(2)], agent=FakeAgent({"think": script}))
    await rig.run()
    assert rig.synth.calls == ["One moment.", "Here you go.", "Let me check.", "Here you go."]


async def test_stock_phrases_are_synthesized_once_by_prepare() -> None:
    rig = Rig({1: say("Jarvis")}, [utt(1)])
    await rig.session.prepare()
    prepared = list(rig.synth.calls)
    assert "Yes?" in prepared and "One moment." in prepared
    await rig.run()
    assert rig.spoke() == ["Yes?"]
    assert rig.synth.calls == prepared  # played from the cache, not synthesized again


async def test_no_filler_when_fast_tool_or_already_speaking() -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        call = ToolCall("1", "t", {})
        yield TextChunk("Checking now. ")
        yield ToolStarted(call)
        await asyncio.sleep(0.2)
        yield ToolFinished(call, ToolResult.success("ok"), 0.2)
        yield done("Checking now.")

    rig = Rig({1: say("Jarvis, check")}, [utt(1)], agent=FakeAgent({"check": script}))
    await rig.run()
    assert rig.synth.calls == ["Checking now."]


async def test_notice_is_spoken_and_error_text_is_read_out() -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        yield Notice("Cloud is down, using the local model.")
        yield done("I couldn't reach the language model.", stopped="error")

    rig = Rig({1: say("Jarvis, hi")}, [utt(1)], agent=FakeAgent({"hi": script}))
    await rig.run()
    assert rig.synth.calls == ["Cloud is down, using the local model.", "I couldn't reach the language model."]


# -- barge-in ----------------------------------------------------------------------------------
def levels(rig: Rig, mic: float, speaker: float = 0.5) -> Callable[[], None]:
    def apply() -> None:
        rig.mic.level, rig.speaker.level = mic, speaker

    return apply


async def test_barge_in_vad_passes_echo_gate_and_next_utterance_needs_no_wake_word() -> None:
    rig = Rig({1: say("Jarvis, start"), 2: say("what's the weather")}, [], agent=FakeAgent({"start": slow_reply(0.6)}))
    rig.session._events = source(utt(1), 0.15, levels(rig, mic=0.9), SpeechStarted(), 0.05, utt(2))
    await rig.run()
    assert rig.agent.cancel_count == 1
    assert rig.speaker.interrupts >= 1
    assert rig.agent.texts == ["start", "what's the weather"]
    assert rig.synth.calls == ["First sentence.", "Reply to what's the weather."]  # old reply is not resumed
    assert LISTENING in rig.states()


async def test_barge_in_vad_blocked_by_echo_gate() -> None:
    rig = Rig({1: say("Jarvis, start"), 2: say("some echo of the speaker")}, [], agent=FakeAgent({"start": slow_reply(0.3)}))
    rig.session._events = source(utt(1), 0.15, levels(rig, mic=0.1), SpeechStarted(), utt(2))
    await rig.run()
    assert rig.agent.cancel_count == 0
    assert rig.agent.texts == ["start"]
    assert rig.synth.calls == ["First sentence.", "Second sentence."]
    assert LISTENING not in rig.states()


async def test_barge_in_vad_while_thinking_in_silence_always_passes() -> None:
    async def silent_thinker(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        await agent.wait(1.0)
        yield done("", stopped="cancelled")

    rig = Rig({1: say("Jarvis, think"), 2: say("never mind that, open notepad")}, [], agent=FakeAgent({"think": silent_thinker}))
    rig.session.filler_after = 60  # stay silent: this test is about barging in with nothing playing
    rig.session._events = source(utt(1), 0.1, levels(rig, mic=0.0, speaker=0.0), SpeechStarted(), utt(2))
    await rig.run()
    assert rig.agent.cancel_count == 1
    assert rig.agent.texts == ["think", "never mind that, open notepad"]


async def test_false_trigger_barge_in_returns_to_idle_with_followup_open() -> None:
    rig = Rig({1: say("Jarvis, start"), 2: Transcript("", 0.0, 0.0)}, [], agent=FakeAgent({"start": slow_reply(0.6)}))
    rig.session._events = source(utt(1), 0.15, levels(rig, mic=0.9), SpeechStarted(), 0.05, utt(2))
    await rig.run()
    assert rig.agent.texts == ["start"]
    assert rig.synth.calls == ["First sentence."]
    assert rig.session.state is IDLE
    assert rig.states()[-1] is IDLE
    assert rig.session._followup_until > rig.now


async def test_wake_mode_only_wake_word_interrupts() -> None:
    rig = Rig(
        {1: say("Jarvis, start"), 2: say("blah blah echo"), 3: say("Jarvis, what's the weather")},
        [], agent=FakeAgent({"start": slow_reply(0.8)}), barge_in="wake",
    )
    rig.session._events = source(utt(1), 0.15, levels(rig, mic=0.9), SpeechStarted(), utt(2), 0.1, utt(3))
    await rig.run()
    assert rig.agent.cancel_count == 1  # the speech start did nothing; only utterance 3 interrupted
    assert rig.agent.texts == ["start", "what's the weather"]


async def test_wake_mode_bare_wake_word_or_stop_only_stops() -> None:
    rig = Rig({1: say("Jarvis, start"), 2: say("Jarvis")}, [], agent=FakeAgent({"start": slow_reply(0.8)}), barge_in="wake")
    rig.session._events = source(utt(1), 0.15, utt(2))
    await rig.run()
    assert rig.agent.cancel_count == 1
    assert rig.agent.texts == ["start"]
    assert rig.synth.calls == ["First sentence."]  # no "Yes?" while stopping


async def test_off_mode_ignores_the_mic_while_busy() -> None:
    rig = Rig(
        {1: say("Jarvis, start"), 2: say("Jarvis, what's the weather")},
        [], agent=FakeAgent({"start": slow_reply(0.3)}), barge_in="off",
    )
    rig.session._events = source(utt(1), 0.15, levels(rig, mic=0.9), SpeechStarted(), utt(2))
    await rig.run()
    assert rig.agent.cancel_count == 0
    assert rig.agent.texts == ["start"]
    assert rig.synth.calls == ["First sentence.", "Second sentence."]


@pytest.mark.parametrize("mode", ["vad", "wake", "off"])
@pytest.mark.parametrize("phrase", ["stop", "Jarvis, never mind", "be quiet", "Shut up."])
async def test_stop_phrase_cancels_the_turn_silently_in_every_mode(mode: str, phrase: str) -> None:
    rig = Rig(
        {1: say("Jarvis, start"), 2: say(phrase)},
        [], agent=FakeAgent({"start": slow_reply(0.8)}), barge_in=mode,
    )
    rig.session._events = source(utt(1), 0.15, utt(2))
    await rig.run()
    assert rig.agent.cancel_count == 1
    assert rig.agent.texts == ["start"]
    assert rig.synth.calls == ["First sentence."]
    assert rig.session.state is IDLE


async def test_stop_phrase_when_idle_does_nothing() -> None:
    rig = Rig({1: say("Jarvis, stop")}, [utt(1)])
    await rig.run()
    assert rig.agent.texts == [] and rig.synth.calls == []


# -- voice confirmation -----------------------------------------------------------------------
def confirm_script(asked: list[bool]) -> Callable[..., AsyncIterator[Any]]:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        asked.append(await confirm("delete the file"))
        yield TextChunk("Deleted." if asked[-1] else "Left it alone.")
        yield done()

    return script


@pytest.mark.parametrize(
    ("answer", "approved"),
    [
        (say("Yes, go ahead"), True),
        (say("no"), False),
        (say("maybe later"), False),
        (say("yes", 0.2), False),  # low confidence is a no
    ],
)
async def test_voice_confirm_answers(answer: Transcript, approved: bool) -> None:
    asked: list[bool] = []
    rig = Rig({1: say("Jarvis, remove it"), 2: answer}, [], agent=FakeAgent({"remove it": confirm_script(asked)}))
    rig.session._events = source(utt(1), lambda: until(lambda: rig.session._confirming == "listening"), utt(2))
    await rig.run()
    assert asked == [approved]
    assert rig.synth.calls == [
        "Shall I go ahead and delete the file? Say yes or no.",
        "On it." if approved else "Okay, skipping.",
        "Deleted." if approved else "Left it alone.",
    ]
    assert rig.agent.texts == ["remove it"]  # the answer was not treated as a new command


async def test_voice_confirm_timeout_is_a_no() -> None:
    asked: list[bool] = []
    rig = Rig({1: say("Jarvis, remove it")}, [utt(1)], agent=FakeAgent({"remove it": confirm_script(asked)}))
    rig.session.confirm_timeout = 0.1
    await rig.run()
    assert asked == [False]
    assert rig.synth.calls[-2:] == ["Okay, skipping.", "Left it alone."]


async def test_utterance_while_question_is_playing_is_not_taken_as_the_answer() -> None:
    asked: list[bool] = []
    rig = Rig({1: say("Jarvis, remove it"), 2: say("yes")}, [], agent=FakeAgent({"remove it": confirm_script(asked)}), play_time=0.3)
    rig.session.confirm_timeout = 0.1
    rig.session._events = source(utt(1), lambda: until(lambda: rig.session._confirming == "asking"), utt(2))
    await rig.run()
    assert asked == [False]  # "yes" arrived during the question (echo-like) and was dropped


# -- robustness ------------------------------------------------------------------------------------
async def test_agent_exception_apologises_and_the_loop_survives() -> None:
    async def boom(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        yield TextChunk("Partial. ")
        raise RuntimeError("agent exploded")

    rig = Rig({1: say("Jarvis, explode"), 2: say("Jarvis, hello")}, [utt(1), 0.3, utt(2)], agent=FakeAgent({"explode": boom}))
    await rig.run()
    assert rig.synth.calls == ["Partial.", "Sorry, something went wrong.", "Reply to hello."]
    assert rig.session.state is IDLE


async def test_synthesis_failure_skips_that_sentence_only() -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        yield TextChunk("Bad one. Good one.")
        yield done()

    rig = Rig({1: say("Jarvis, go")}, [utt(1)], agent=FakeAgent({"go": script}))
    rig.synth.fail_on = {"Bad one."}
    await rig.run()
    assert rig.synth.calls == ["Bad one.", "Good one."]
    assert rig.spoke() == ["Good one."]


async def test_transcriber_exception_does_not_kill_the_loop() -> None:
    rig = Rig({1: RuntimeError("stt boom"), 2: say("Jarvis, hello")}, [utt(1), utt(2)])
    await rig.run()
    assert rig.agent.texts == ["hello"]


async def test_clean_shutdown_on_stop() -> None:
    rig = Rig({})
    rig.session._events = source(lambda: asyncio.sleep(10))
    runner = asyncio.ensure_future(rig.session.run())
    await asyncio.sleep(0.05)
    rig.session.stop()
    await asyncio.wait_for(runner, 2)
    assert rig.mic.started and rig.mic.stopped and rig.speaker.started and rig.speaker.stopped


async def test_cancelling_run_stops_audio_and_the_running_turn() -> None:
    rig = Rig({1: say("Jarvis, start")}, [utt(1), lambda: asyncio.sleep(10)], agent=FakeAgent({"start": slow_reply(5)}))
    runner = asyncio.ensure_future(rig.session.run())
    await until(lambda: rig.session._turn is not None and bool(rig.speaker.played))
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    assert rig.mic.stopped and rig.speaker.stopped
    assert rig.agent.cancel_count == 1


async def test_default_source_raises_vad_threshold_while_speaking() -> None:
    seen: list[float] = []

    class ScriptedEndpointer:
        start_threshold = 0.5
        in_speech = False

        def feed(self, frame: np.ndarray) -> Any:
            seen.append(self.start_threshold)
            return None

    rig = Rig({})
    rig.mic = FakeMic([np.zeros(512, np.float32)] * 3)
    session = VoiceSession(
        rig.agent, rig.mic, rig.speaker, ScriptedEndpointer(), rig.transcriber, rig.synth, rig.settings, rig.events.append,
    )
    rig.speaker.is_playing = True
    await asyncio.wait_for(session.run(), 2)
    rig.speaker.is_playing = False
    assert seen == [0.8, 0.8, 0.8]
    assert any(e.__class__.__name__ == "Levels" for e in rig.events)


# -- events -------------------------------------------------------------------------------------------
async def test_state_sequence_and_events_for_a_normal_turn() -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        call = ToolCall("1", "get_time", {})
        yield ToolStarted(call)
        yield ToolFinished(call, ToolResult.success("12:00"), 0.0)
        yield TextChunk("It is noon.")
        yield done("It is noon.")

    rig = Rig({1: say("Jarvis, what time is it")}, [SpeechStarted(), utt(1)], agent=FakeAgent({"what time is it": script}))
    await rig.run()
    assert rig.states() == [LISTENING, THINKING, SPEAKING, IDLE]
    kinds = [type(e).__name__ for e in rig.events if not isinstance(e, StateChanged)]
    assert kinds == ["Heard", "ToolActivity", "ToolActivity", "Spoke"]
    heard = rig.heard()[0]
    assert (heard.text, heard.addressed) == ("Jarvis, what time is it", True)
    assert rig.spoke() == ["It is noon."]


async def test_say_speaks_and_reports() -> None:
    rig = Rig({})
    await rig.session.say("Online and ready.")
    assert rig.synth.calls == ["Online and ready."]
    assert rig.states() == [SPEAKING, IDLE]
    assert rig.session.is_idle()
    assert rig.session._followup_until > rig.now


async def test_info_event_type_is_exported() -> None:
    assert Info("x").text == "x"
    assert vs.VoiceState.IDLE.value == "idle"


# -- reminders ----------------------------------------------------------------------------------------
class FakeReminderStore:
    def __init__(self, due: list[dict[str, Any]]) -> None:
        self.due = due
        self.fired: list[int] = []


@pytest.fixture
def reminder_store(monkeypatch: pytest.MonkeyPatch) -> FakeReminderStore:
    store = FakeReminderStore([{"id": 7, "message": "call mum"}])
    monkeypatch.setattr(scheduler, "due_reminders", lambda s, now: [r for r in s.due if r["id"] not in s.fired])
    monkeypatch.setattr(scheduler, "mark_reminder_fired", lambda s, rid: s.fired.append(rid))
    return store


async def test_reminder_is_spoken_when_idle(reminder_store: FakeReminderStore) -> None:
    rig = Rig({})
    watcher = ReminderWatcher(reminder_store, rig.session.say, rig.session.is_idle)
    assert await watcher.check() == 1
    assert rig.synth.calls == ["Reminder: call mum"]
    assert reminder_store.fired == [7]
    assert await watcher.check() == 0


async def test_reminder_waits_while_a_turn_is_running(reminder_store: FakeReminderStore) -> None:
    rig = Rig({1: say("Jarvis, go")}, [utt(1)], agent=FakeAgent({"go": slow_reply(0.4)}))
    runner = asyncio.ensure_future(rig.session.run())
    await until(lambda: rig.session._turn is not None)
    watcher = ReminderWatcher(reminder_store, rig.session.say, rig.session.is_idle, idle_recheck=0.01)
    check = asyncio.ensure_future(watcher.check())
    await asyncio.sleep(0.2)
    assert not check.done() and reminder_store.fired == []  # deferred while busy
    assert await asyncio.wait_for(check, 3) == 1
    await asyncio.wait_for(runner, 3)
    assert rig.synth.calls[-1] == "Reminder: call mum"
    assert reminder_store.fired == [7]


async def test_reminder_watcher_with_real_store(tmp_path: Path) -> None:
    from datetime import datetime, timedelta

    from jarvis.memory.store import MemoryStore
    from jarvis.tools.builtin.productivity import _iso, due_reminders

    store = MemoryStore(tmp_path / "t.db")
    try:
        due_reminders(store, datetime.now())  # creates the table
        store.execute(
            "INSERT INTO reminders (message, due_at, created_at) VALUES (?, ?, ?)",
            ("stretch", _iso(datetime.now() - timedelta(minutes=1)), _iso(datetime.now())),
        )
        spoken: list[str] = []

        async def deliver(text: str) -> None:
            spoken.append(text)

        watcher = ReminderWatcher(store, deliver)
        assert await watcher.check() == 1
        assert await watcher.check() == 0
        assert spoken == ["Reminder: stretch"]
    finally:
        store.close()


async def test_reminder_watcher_survives_errors(reminder_store: FakeReminderStore) -> None:
    calls: list[int] = []

    async def flaky(text: str) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("no speaker")

    watcher = ReminderWatcher(reminder_store, flaky, interval=0.02)
    watcher.start()
    await until(lambda: reminder_store.fired == [7])
    await watcher.stop()
    assert len(calls) == 2


# -- cli flags and display -----------------------------------------------------------------------------
def test_no_wake_flag_overrides_setting() -> None:
    from jarvis.interfaces import cli

    base = Settings(provider="ollama", require_wake_word=True)
    assert cli.apply_overrides(base, None, None, True).require_wake_word is False
    assert cli.apply_overrides(base, None, None).require_wake_word is True


def test_voice_and_once_conflict() -> None:
    from jarvis.interfaces import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--voice", "--once", "hi"])
    assert exc.value.code == 2


def test_install_message_names_the_command() -> None:
    from jarvis.interfaces import cli

    assert "uv sync --extra voice" in cli.voice_install_message("sounddevice")


def test_display_tracks_events() -> None:
    from rich.console import Console

    from jarvis.interfaces.voice import VoiceDisplay
    from jarvis.voice.session import Levels

    display = VoiceDisplay("jarvis")
    for event in (
        Heard("hello there", False, 0.9), StateChanged(THINKING), ToolActivity("web_search", False),
        ToolActivity("web_search", True, True), Levels(0.2, 0.0), Spoke("Hi."),
    ):
        display.handle(event)
    console = Console(record=True, width=100, color_system=None)
    console.print(display)
    out = console.export_text()
    assert "THINKING" in out and "not for jarvis" in out and "web_search" in out and "jarvis: Hi." in out
