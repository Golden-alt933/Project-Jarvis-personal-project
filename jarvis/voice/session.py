"""The voice session: one always-open microphone, one agent, one speaker.

    mic -> Endpointer -> (SpeechStarted | Utterance) -> gating -> Agent turn
                                                                    |
    speaker <- synthesis worker (in order) <- SentenceChunker <- TextChunk

The session is the only place that looks at a transcript, and only to decide
whether it is addressed to Jarvis, whether it is a stop phrase, or whether it
answers a confirmation. It never routes or answers by content: that is the
agent's job.

Everything slow (VAD, STT, TTS, the agent) is awaited; the event loop is never
blocked, so the microphone keeps being processed while Jarvis thinks and speaks.
"""

from __future__ import annotations

import asyncio
import difflib
import enum
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Protocol

import numpy as np

from jarvis.agent.events import Notice, TextChunk, ToolFinished, ToolStarted, TurnDone
from jarvis.config import Settings
from jarvis.voice.summary import summarize_args
from jarvis.voice.types import (
    AudioInput,
    AudioOutput,
    SpeechStarted,
    Synthesizer,
    Transcriber,
    Transcript,
    Utterance,
)

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.35
FILLER_TEXT = "One moment."
# Rotated so a slow model does not make Jarvis repeat himself every turn.
FILLERS = (FILLER_TEXT, "Let me check.", "Right away.", "Working on it.")
ACK_TEXT = "Yes?"
APOLOGY_TEXT = "Sorry, something went wrong."
CONFIRM_ACK_YES = "On it."
CONFIRM_ACK_NO = "Okay, skipping."
STOP_PHRASES = {"stop", "cancel", "never mind", "nevermind", "be quiet", "shut up", "stop it", "stop talking"}
# Wake word mis-hearings seen from Whisper-class models; matched at the START only.
WAKE_ALIASES = {"jarvus", "jervis", "travis", "jarvas", "jarvice", "garvis", "gervis", "charvis", "jarvish"}
_NEGATIVE = re.compile(r"\b(no|nope|nah|don'?t|not|stop|cancel|never)\b")
_AFFIRMATIVE = re.compile(r"\b(yes|yeah|yep|yup|sure|confirm|confirmed|affirmative|go ahead|do it|please do)\b")
_WORD = re.compile(r"[\w']+")
_LEADING = re.compile(r"^\W*(?:(?:hey|hi|hello|ok|okay|yo)\W+)?([\w']+)", re.IGNORECASE)
# Echo gate: mic must exceed speaker level * coupling * margin to count as the user.
ECHO_MARGIN = 1.3
INITIAL_COUPLING = 0.5
MAX_COUPLING = 3.0
VAD_PLAYING_THRESHOLD = 0.8
LEVEL_EVENT_INTERVAL = 1 / 15


# Events and the frontend-facing interface live in jarvis.voice.events (no heavy
# imports there); re-exported here so existing imports keep working.
from jarvis.voice.events import (  # noqa: E402,F401
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
SourceFactory = Callable[[], AsyncIterator[SpeechStarted | Utterance]]


class Chunker(Protocol):
    def feed(self, delta: str) -> list[str]: ...
    def flush(self) -> list[str]: ...


def _default_chunker() -> Chunker:
    from jarvis.voice.tts import SentenceChunker

    return SentenceChunker()


# -- transcript gating (pure functions) -----------------------------------------------
def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def is_wake_like(word: str, wake_word: str) -> bool:
    word, wake = word.lower().strip("'"), wake_word.lower()
    if word == wake:
        return True
    if wake == "jarvis" and word in WAKE_ALIASES:
        return True
    return len(wake) >= 4 and difflib.SequenceMatcher(None, word, wake).ratio() >= 0.8


def split_wake_word(text: str, wake_word: str) -> tuple[bool, str]:
    """(leading wake word found, text with it and its punctuation removed)."""
    m = _LEADING.match(text)
    if m and is_wake_like(m.group(1), wake_word):
        return True, text[m.end():].lstrip(" \t,.:;!?-–—").strip()
    return False, text.strip()


def contains_wake_word(text: str, wake_word: str) -> bool:
    return wake_word.lower() in words(text)


def is_stop_phrase(text: str) -> bool:
    return " ".join(words(text)) in STOP_PHRASES


def is_affirmative(text: str) -> bool:
    lowered = " ".join(words(text))
    return not _NEGATIVE.search(lowered) and bool(_AFFIRMATIVE.search(lowered))


@dataclass
class _Turn:
    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None
    synth_task: asyncio.Task[None] | None = None
    filler_task: asyncio.Task[None] | None = None
    cancelled: bool = False
    said: bool = False  # something has been queued for speech this turn
    filler_said: bool = False
    running_tools: int = 0
    skip: str | None = None  # a queued sentence not to speak (an already answered question)


@dataclass
class _Heard:
    utterance: Utterance
    barge: bool  # captured after a barge-in: addressed without the wake word


@dataclass
class _Typed:
    text: str


class VoiceSession:
    def __init__(
        self,
        agent,
        mic: AudioInput,
        speaker: AudioOutput,
        endpointer,
        transcriber: Transcriber,
        synthesizer: Synthesizer,
        settings: Settings,
        on_event: EventCallback | None = None,
        *,
        events: SourceFactory | None = None,
        chunker_factory: Callable[[], Chunker] = _default_chunker,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.agent = agent
        self.mic = mic
        self.speaker = speaker
        self.endpointer = endpointer
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.settings = settings
        self.on_event = on_event
        self._events = events or self._mic_events
        self._chunker_factory = chunker_factory
        self.clock = clock

        # Seconds of silence after the user stops before Jarvis acknowledges. Most of
        # the wait is the model itself, so the timer starts with the turn, not a tool.
        self.filler_after = 2.5
        self._filler_index = 0
        self._stock: dict[str, np.ndarray] = {}
        self.confirm_timeout = 15.0
        self.state = VoiceState.IDLE
        self.coupling = INITIAL_COUPLING

        self._base_threshold = getattr(endpointer, "start_threshold", 0.5)
        self._turn: _Turn | None = None
        self._saying = 0
        self._capturing = False
        self._barge = False
        self._followup_until = 0.0
        self._stopping = False
        self._epoch = 0  # bumped on every interrupt so late synthesis is discarded
        self._confirming: str | None = None  # None | "asking" | "listening"
        self._confirm_future: asyncio.Future[Transcript | None] | None = None
        self._question: str | None = None
        self._muted = False
        self._utt_q: asyncio.Queue[_Heard | _Typed] = asyncio.Queue()
        self._pump_task: asyncio.Task[None] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._last_level_event = 0.0

    # -- public API ------------------------------------------------------------------
    has_audio = True

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        """Keep the mic open but ignore it: no wake, no barge-in, no utterance."""
        self._muted = muted
        if muted:
            reset = getattr(self.endpointer, "reset", None)
            if reset is not None:
                reset()
            self._capturing = False
            self._barge = False
            self._settle()
        self._emit(MicMuted(muted))

    def submit_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._emit(UserText(text))
        self._interrupt_output()  # also answers a pending confirmation with "no"
        self._utt_q.put_nowait(_Typed(text))

    def cancel(self) -> None:
        if self._busy():
            self._interrupt_output()

    def answer_confirm(self, approved: bool) -> None:
        future = self._confirm_future
        if future is None or future.done():
            return
        turn = self._turn
        if self._confirming == "asking" and turn is not None:
            turn.skip = self._question
        self.speaker.interrupt()
        future.set_result(Transcript("yes" if approved else "no", 0.0, 1.0))

    def is_idle(self) -> bool:
        return self.state is VoiceState.IDLE and not self._busy()

    def stop(self) -> None:
        self._stopping = True
        if self._pump_task is not None:
            self._pump_task.cancel()

    async def say(self, text: str) -> None:
        """Speak `text` now (greeting, acknowledgement, reminder) and wait until done."""
        self._saying += 1
        epoch = self._epoch
        try:
            audio = await self._synthesize(text)
            if audio is None or epoch != self._epoch:
                return
            self._set_state(VoiceState.SPEAKING)
            self.speaker.play(audio, self.synthesizer.sample_rate)
            self._emit(Spoke(text))
            await self.speaker.drain()
        finally:
            self._saying -= 1
            self._settle()
            if self._turn is None and self._saying == 0:
                self._open_followup()

    async def run(self, greeting: str | None = None) -> None:
        """Run until stop(), cancellation, or the audio source ends."""
        await self.mic.start()
        await self.speaker.start()
        self._worker = asyncio.ensure_future(self._utterance_worker())
        try:
            if greeting:
                await self._guarded(self.say(greeting), "greeting")
            self._pump_task = asyncio.ensure_future(self._pump())
            try:
                await self._pump_task
            except asyncio.CancelledError:
                if not self._stopping:
                    raise
            if not self._stopping:
                await self._utt_q.join()
                turn = self._turn
                if turn is not None and turn.task is not None:
                    await asyncio.wait([turn.task])
        finally:
            await self._shutdown()

    # -- source of speech events -----------------------------------------------------
    async def _mic_events(self) -> AsyncIterator[SpeechStarted | Utterance]:
        async for frame in self.mic.frames():
            self._on_frame()
            if self._muted:
                continue
            item = self.endpointer.feed(frame)
            if item is not None:
                yield item

    def _on_frame(self) -> None:
        playing = self.speaker.is_playing
        self.endpointer.start_threshold = max(self._base_threshold, VAD_PLAYING_THRESHOLD) if playing else self._base_threshold
        if playing and not self.endpointer.in_speech:
            self._adapt_coupling()
        now = time.monotonic()
        if now - self._last_level_event >= LEVEL_EVENT_INTERVAL:
            self._last_level_event = now
            self._emit(Levels(self.mic.level, self.speaker.level))

    def _adapt_coupling(self) -> None:
        """Running maximum (with slow decay) of mic/speaker level while only Jarvis is heard."""
        out = self.speaker.level
        if out < 0.02:
            return
        self.coupling = min(MAX_COUPLING, max(self.mic.level / out, self.coupling * 0.998, 0.05))

    def echo_estimate(self) -> float:
        return self.speaker.level * self.coupling * ECHO_MARGIN

    async def _pump(self) -> None:
        failures = 0
        while not self._stopping:
            try:
                async for item in self._events():
                    failures = 0
                    self._dispatch(item)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                log.exception("audio source failed (%d)", failures)
                if failures >= 10:
                    self._emit(Info("The microphone keeps failing; stopping voice mode."))
                    return
                await asyncio.sleep(0.5)

    def _dispatch(self, item: SpeechStarted | Utterance) -> None:
        if isinstance(item, SpeechStarted):
            self._on_speech_started()
        elif isinstance(item, Utterance):
            self._capturing = False
            barge, self._barge = self._barge, False
            self._utt_q.put_nowait(_Heard(item, barge))

    # -- barge-in ----------------------------------------------------------------------
    def _busy(self) -> bool:
        return self._turn is not None or self._saying > 0

    def _passes_echo_gate(self) -> bool:
        if not self.speaker.is_playing:
            return True  # thinking in silence: nothing to echo
        return self.mic.level > self.echo_estimate()

    def _on_speech_started(self) -> None:
        mode = self.settings.barge_in
        if self._confirming is not None:
            self._capturing = True
            if self._confirming == "asking" and mode == "vad" and self._passes_echo_gate():
                self.speaker.interrupt()  # the user is answering over the question
                self._confirming = "listening"
            return
        if not self._busy():
            self._capturing = True
            self._set_state(VoiceState.LISTENING)
            return
        if mode != "vad" or not self._passes_echo_gate():
            return
        log.info("barge-in")
        self._interrupt_output()
        self._barge = True
        self._capturing = True
        self._set_state(VoiceState.LISTENING)

    def _interrupt_output(self) -> None:
        """Silence everything and tell the agent to stop. Does not wait for anything."""
        self._epoch += 1
        turn = self._turn
        if turn is not None and not turn.cancelled:
            turn.cancelled = True
            self.agent.cancel()
        self.speaker.interrupt()
        if self._confirm_future is not None and not self._confirm_future.done():
            self._confirm_future.set_result(None)

    async def _cancel_turn(self) -> None:
        turn = self._turn
        self._interrupt_output()
        if turn is None or turn.task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(turn.task), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning("turn did not stop after cancel; forcing")
            turn.task.cancel()
            await asyncio.gather(turn.task, return_exceptions=True)
        except Exception:
            log.exception("turn ended with an error while cancelling")

    # -- utterance handling -----------------------------------------------------------
    async def _utterance_worker(self) -> None:
        while True:
            item = await self._utt_q.get()
            try:
                await self._handle(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("handling an utterance failed")
                self._emit(Info("Something went wrong handling that."))
                self._settle()
            finally:
                self._utt_q.task_done()

    async def _transcribe(self, utterance: Utterance) -> Transcript | None:
        try:
            return await self.transcriber.transcribe(utterance.pcm, hint=self.settings.wake_word)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("transcription failed")
            return None

    async def _handle_typed(self, item: _Typed) -> None:
        self._open_followup()
        await self._start_turn(item.text)
        self._settle()

    async def _handle(self, item: _Heard | _Typed) -> None:
        if isinstance(item, _Typed):
            await self._handle_typed(item)
            return
        if self._confirming == "asking":
            return  # probably the question itself coming back through the mic
        transcript = await self._transcribe(item.utterance)
        text = transcript.text.strip() if transcript is not None else ""
        confidence = transcript.confidence if transcript is not None else 0.0
        if self._confirming == "listening":
            self._resolve_confirm(text, confidence)
            return
        if not text or confidence < MIN_CONFIDENCE:
            if text:
                self._emit(Heard(text, False, confidence))
            log.debug("ignored empty/low-confidence transcript %r (%.2f)", text, confidence)
            if item.barge:
                self._open_followup()  # false trigger: do not resume the old reply
            self._settle()
            return
        await self._route(text, confidence, item.barge)
        self._settle()

    def _resolve_confirm(self, text: str, confidence: float) -> None:
        if text:
            self._emit(Heard(text, True, confidence))
        future = self._confirm_future
        if future is not None and not future.done():
            future.set_result(Transcript(text, 0.0, confidence))

    async def _route(self, text: str, confidence: float, barge: bool) -> None:
        wake = self.settings.wake_word
        leading, remainder = split_wake_word(text, wake)
        in_window = self.clock() < self._followup_until
        addressed = barge or not self.settings.require_wake_word or leading or in_window or contains_wake_word(text, wake)
        busy = self._busy()
        stop = is_stop_phrase(remainder)

        if busy and not barge and not (addressed and stop):
            # While Jarvis is active only an explicit wake word interrupts; the
            # follow-up window must not let his own echo through.
            addressed = self.settings.barge_in != "off" and (leading or contains_wake_word(text, wake))

        self._emit(Heard(text, bool(addressed), confidence))
        if not addressed:
            log.debug("not addressed to Jarvis: %r", text)
            return

        if stop:
            if busy:
                await self._cancel_turn()
            return
        if not remainder:
            if busy and not barge:
                await self._cancel_turn()  # just the wake word during playback: only stop
                return
            await self._cancel_turn()
            await self.say(ACK_TEXT)
            return
        self._open_followup()
        await self._start_turn(remainder)

    # -- follow-up window -----------------------------------------------------------------
    def _open_followup(self) -> None:
        self._followup_until = self.clock() + self.settings.followup_seconds

    # -- the turn ------------------------------------------------------------------------
    async def _start_turn(self, text: str) -> None:
        await self._cancel_turn()
        turn = _Turn()
        self._turn = turn
        turn.task = asyncio.ensure_future(self._run_turn(turn, text))

    def _enqueue(self, turn: _Turn, sentences: list[str]) -> None:
        for sentence in sentences:
            if sentence.strip() and not turn.cancelled:
                turn.said = True
                turn.queue.put_nowait(sentence)

    async def _run_turn(self, turn: _Turn, text: str) -> None:
        self._set_state(VoiceState.THINKING)
        turn.synth_task = asyncio.ensure_future(self._synth_worker(turn))
        turn.filler_task = asyncio.ensure_future(self._filler(turn))
        chunker = self._chunker_factory()
        got_text = flushed = finished = failed = False
        started = time.monotonic()
        try:
            async for event in self.agent.run_turn(text, output_mode="voice", confirm=self._voice_confirm):
                if isinstance(event, TextChunk):
                    got_text = True
                    self._enqueue(turn, chunker.feed(event.text))
                elif isinstance(event, ToolStarted):
                    self._emit(ToolActivity(event.call.name, False, summary=summarize_args(event.call.arguments)))
                    self._tool_started(turn)
                elif isinstance(event, ToolFinished):
                    self._emit(ToolActivity(
                        event.call.name, True, event.result.ok,
                        summary=summarize_args(event.call.arguments), seconds=event.seconds,
                    ))
                    self._tool_finished(turn)
                elif isinstance(event, Notice):
                    self._enqueue(turn, [event.text])
                elif isinstance(event, TurnDone):
                    finished = True
                    self._emit(TurnFinished(
                        event.text, event.seconds, event.steps, event.tool_calls, event.model, event.stopped,
                    ))
                    if not got_text and event.stopped != "cancelled":
                        self._enqueue(turn, chunker.feed(event.text))  # e.g. an error apology
                    self._enqueue(turn, chunker.flush())
                    flushed = True
        except asyncio.CancelledError:
            turn.cancelled = True
            raise
        except Exception:
            log.exception("turn failed")
            self._enqueue(turn, [APOLOGY_TEXT])
            failed = True
        finally:
            if not finished:  # the agent died or stopped without a TurnDone
                stopped = "error" if failed else "cancelled" if turn.cancelled else "done"
                self._emit(TurnFinished("", time.monotonic() - started, 0, 0, "", stopped))
            if not flushed and not turn.cancelled:
                self._enqueue(turn, chunker.flush())
            await self._finish_turn(turn)

    async def _finish_turn(self, turn: _Turn) -> None:
        if turn.filler_task is not None:
            turn.filler_task.cancel()
        turn.queue.put_nowait(None)
        try:
            if turn.synth_task is not None:
                if turn.cancelled:
                    turn.synth_task.cancel()  # pending sentences are discarded
                    await asyncio.gather(turn.synth_task, return_exceptions=True)
                else:
                    await turn.synth_task
            if not turn.cancelled:
                await self.speaker.drain()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("finishing the turn failed")
        finally:
            if self._turn is turn:
                self._turn = None
            if not turn.cancelled:
                self._open_followup()
            self._settle()

    async def _synth_worker(self, turn: _Turn) -> None:
        epoch = self._epoch
        while True:
            sentence = await turn.queue.get()
            try:
                if sentence is None:
                    return
                if turn.skip is not None and sentence == turn.skip:
                    turn.skip = None
                    continue
                if turn.cancelled or epoch != self._epoch:
                    continue
                audio = await self._synthesize(sentence)
                if audio is None or turn.cancelled or epoch != self._epoch:
                    continue
                self._set_state(VoiceState.SPEAKING)
                self.speaker.play(audio, self.synthesizer.sample_rate)
                self._emit(Spoke(sentence))
            finally:
                turn.queue.task_done()

    async def prepare(self) -> None:
        """Synthesize the stock phrases once, so acknowledgements play with no delay."""
        for phrase in (*FILLERS, ACK_TEXT, APOLOGY_TEXT, CONFIRM_ACK_YES, CONFIRM_ACK_NO):
            audio = await self._synthesize(phrase)
            if audio is not None:
                self._stock[phrase] = audio

    async def _synthesize(self, text: str) -> np.ndarray | None:
        cached = self._stock.get(text)
        if cached is not None:
            return cached
        try:
            audio = await self.synthesizer.synthesize(text)
            return audio if audio is not None and len(audio) else None  # nothing speakable
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("synthesis failed for %r", text[:60])
            return None

    # -- filler while nothing has been said ----------------------------------------------
    def _tool_started(self, turn: _Turn) -> None:
        turn.running_tools += 1

    def _tool_finished(self, turn: _Turn) -> None:
        turn.running_tools = max(0, turn.running_tools - 1)

    async def _filler(self, turn: _Turn) -> None:
        await asyncio.sleep(self.filler_after)
        if not turn.said and not turn.cancelled:
            turn.filler_said = True
            phrase = FILLERS[self._filler_index % len(FILLERS)]
            self._filler_index += 1
            self._enqueue(turn, [phrase])

    # -- voice confirmation -------------------------------------------------------------------
    async def _voice_confirm(self, description: str) -> bool:
        turn = self._turn
        loop = asyncio.get_running_loop()
        self._confirm_future = future = loop.create_future()
        self._confirming = "asking"
        self._question = question = f"Shall I go ahead and {description}? Say yes or no."
        self._emit(ConfirmRequested(description))
        approved = False
        try:
            if turn is not None:
                turn.said = True
                turn.queue.put_nowait(question)
                await turn.queue.join()
                if not future.done():
                    await self.speaker.drain()
            else:
                await self.say(question)
            if not future.done():
                if self._confirming == "asking":
                    self._confirming = "listening"
                self._set_state(VoiceState.LISTENING)
            try:
                answer = await asyncio.wait_for(future, self.confirm_timeout)
            except asyncio.TimeoutError:
                answer = None
            approved = (
                answer is not None and answer.confidence >= MIN_CONFIDENCE and is_affirmative(answer.text)
            )
        finally:
            self._confirming = None
            self._confirm_future = None
            self._question = None
            self._capturing = False
            if turn is not None:
                turn.skip = None
            self._emit(ConfirmResolved(approved))
        if turn is not None and not turn.cancelled:
            self._enqueue(turn, [CONFIRM_ACK_YES if approved else CONFIRM_ACK_NO])
            self._set_state(VoiceState.THINKING)
        return approved

    # -- state, events, plumbing ---------------------------------------------------------------
    def _set_state(self, state: VoiceState) -> None:
        if state is not self.state:
            self.state = state
            self._emit(StateChanged(state))

    def _settle(self) -> None:
        if not self._busy() and not self._capturing and self._utt_q.empty():
            self._set_state(VoiceState.IDLE)

    def _emit(self, event: VoiceEvent) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            log.exception("event callback failed")

    async def _guarded(self, coro, what: str) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s failed", what)

    async def _shutdown(self) -> None:
        turn = self._turn
        if turn is not None:
            turn.cancelled = True
            self.agent.cancel()
        try:
            self.speaker.interrupt()
        except Exception:
            log.exception("interrupt on shutdown failed")
        tasks = [t for t in (self._worker, turn.task if turn else None, turn.synth_task if turn else None) if t]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stopper in (self.mic.stop, self.speaker.stop):
            try:
                await stopper()
            except Exception:
                log.exception("stopping audio failed")
