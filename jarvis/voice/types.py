"""Voice pipeline contracts.

One microphone stream stays open for the whole session and feeds everything
(wake gating, endpointing, barge-in). Nothing ever closes and reopens the device:
Friday's hand-off of the mic between wake word and recorder was the source of
its crashes and its inability to hear the user while speaking.

    mic frames -> VAD -> Endpointer -> Utterance -> Transcriber -> text
    text -> Agent (output_mode="voice") -> TextChunk stream -> SentenceChunker
         -> Synthesizer -> AudioOutput        (sentence 1 plays while 2 is generated)

Audio everywhere is float32 mono in [-1, 1]. Input is always 16 kHz in frames of
exactly FRAME_SAMPLES samples (the window Silero VAD requires).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE  # 32 ms


@runtime_checkable
class AudioInput(Protocol):
    """Always-open microphone."""

    level: float  # smoothed RMS of recent input, 0..1 (for meters and echo gating)

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield consecutive frames of FRAME_SAMPLES float32 samples at 16 kHz.
        Must never block the event loop; if the consumer falls behind, old frames
        are dropped rather than queued without bound."""
        ...


@runtime_checkable
class AudioOutput(Protocol):
    """Gapless, instantly interruptible speaker output."""

    is_playing: bool  # True while audio is queued or sounding
    level: float  # smoothed RMS of what is being played right now, 0..1

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Queue audio and return immediately. Chunks play back to back."""
        ...

    def interrupt(self) -> None:
        """Silence now and drop everything queued. Safe from any thread."""
        ...

    async def drain(self) -> None:
        """Wait until everything queued has been played (or interrupt() was called)."""
        ...


@runtime_checkable
class VAD(Protocol):
    def probability(self, frame: np.ndarray) -> float:
        """Speech probability 0..1 for one frame. Stateful across frames."""
        ...

    def reset(self) -> None: ...


@dataclass
class SpeechStarted:
    """The user began speaking (emitted once min_speech_ms of speech is confirmed)."""


@dataclass
class Utterance:
    """A complete utterance, including pre-roll, with trailing silence trimmed."""

    pcm: np.ndarray
    seconds: float


@dataclass
class Transcript:
    text: str
    seconds: float  # time spent transcribing
    confidence: float = 1.0  # 0..1; low values mean probable noise/non-speech


@runtime_checkable
class Transcriber(Protocol):
    name: str  # e.g. "faster-whisper small.en (cuda)"

    async def warm_up(self) -> None: ...

    async def transcribe(self, pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
        """Transcribe 16 kHz mono audio. `hint` biases spelling (e.g. the wake word).
        Returns empty text for silence/noise instead of a hallucinated phrase."""
        ...


@runtime_checkable
class Synthesizer(Protocol):
    name: str
    sample_rate: int

    async def warm_up(self) -> None: ...

    async def synthesize(self, text: str) -> np.ndarray:
        """One sentence or short passage -> float32 mono at `sample_rate`."""
        ...
