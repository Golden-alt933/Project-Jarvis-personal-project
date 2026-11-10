"""Voice activity detection and utterance endpointing.

SileroVAD runs faster-whisper's bundled silero_vad_v6.onnx directly, one 512-sample
frame at a time. That export takes `input` [N, 64 + 512] (64 samples of context from
the previous frame, then the new frame) plus LSTM state `h` and `c` ([1, 1, 128]
each) and returns `speech_probs`, `hn`, `cn`. We carry h/c and the context between
calls. No batch re-scan: cost per frame is constant.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from collections import deque
from pathlib import Path

import numpy as np

from jarvis.voice.types import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, SpeechStarted, Utterance, VAD

log = logging.getLogger(__name__)

_CONTEXT = 64
_MODEL_FILE = "silero_vad_v6.onnx"


def _model_path() -> Path:
    spec = importlib.util.find_spec("faster_whisper")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("faster_whisper is not installed")
    path = Path(next(iter(spec.submodule_search_locations))) / "assets" / _MODEL_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _fit(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32).reshape(-1)
    if len(frame) == FRAME_SAMPLES:
        return frame
    return np.pad(frame[:FRAME_SAMPLES], (0, max(0, FRAME_SAMPLES - len(frame))))


class SileroVAD:
    def __init__(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(_model_path()), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        frame = _fit(frame)
        x = np.concatenate([self._context, frame])[None, :]
        probs, self._h, self._c = self._session.run(
            None, {"input": x, "h": self._h, "c": self._c}
        )
        self._context = frame[-_CONTEXT:]
        return float(probs.reshape(-1)[0])


class EnergyVAD:
    """Fallback: speech = energy well above an adaptive noise floor."""

    def __init__(self, snr_db: float = 12.0) -> None:
        self._snr = snr_db
        self.reset()

    def reset(self) -> None:
        self._floor = 1e-3

    def probability(self, frame: np.ndarray) -> float:
        frame = _fit(frame)
        rms = float(np.sqrt(np.mean(np.square(frame)))) + 1e-9
        # Floor follows quiet passages quickly and loud ones very slowly.
        rate = 0.05 if rms < self._floor * 1.5 else 0.002
        self._floor = max(1e-4, self._floor + (rms - self._floor) * rate)
        snr = 20 * math.log10(rms / self._floor)
        return float(1.0 / (1.0 + math.exp(-(snr - self._snr) / 2.5)))


def build_vad() -> VAD:
    try:
        return SileroVAD()
    except Exception as exc:
        log.warning("Silero VAD unavailable (%s); using the energy fallback.", exc)
        return EnergyVAD()


class Endpointer:
    """Turns per-frame speech probabilities into SpeechStarted / Utterance events."""

    _HYSTERESIS = 0.15
    _TRAILING_KEEP_MS = 200

    def __init__(
        self,
        vad: VAD,
        *,
        start_threshold: float = 0.5,
        end_silence_ms: int = 650,
        min_speech_ms: int = 200,
        preroll_ms: int = 300,
        max_seconds: float = 30.0,
    ) -> None:
        self.vad = vad
        self.start_threshold = start_threshold  # mutable: raised while Jarvis speaks
        self._end_frames = max(1, math.ceil(end_silence_ms / FRAME_MS))
        self._min_frames = max(1, math.ceil(min_speech_ms / FRAME_MS))
        self._preroll_frames = math.ceil(preroll_ms / FRAME_MS)
        self._keep_tail = math.ceil(self._TRAILING_KEEP_MS / FRAME_MS)
        self._max_frames = max(1, int(max_seconds * 1000 / FRAME_MS))
        self._pre: deque[np.ndarray] = deque(
            maxlen=self._preroll_frames + 3 * self._min_frames + 1
        )
        self.reset(reset_vad=False)

    @property
    def in_speech(self) -> bool:
        return self._speech is not None

    def reset(self, *, reset_vad: bool = True) -> None:
        if reset_vad:
            self.vad.reset()
        self._pre.clear()
        self._speech: list[np.ndarray] | None = None
        self._run = 0  # frames since the current candidate onset (holds included)
        self._hits = 0  # frames above the start threshold within the candidate
        self._silence = 0

    def feed(self, frame: np.ndarray) -> SpeechStarted | Utterance | None:
        frame = _fit(frame)
        p = self.vad.probability(frame)
        end_threshold = max(0.05, self.start_threshold - self._HYSTERESIS)
        if self._speech is None:
            return self._feed_idle(frame, p, end_threshold)
        return self._feed_speech(frame, p, end_threshold)

    def _feed_idle(self, frame: np.ndarray, p: float, end_threshold: float) -> SpeechStarted | None:
        self._pre.append(frame)
        if p >= self.start_threshold:
            self._run += 1
            self._hits += 1
        elif p < end_threshold or self._run > 3 * self._min_frames:
            self._run = self._hits = 0
        elif self._run:
            self._run += 1  # in-between: hold the candidate, don't count it
        if self._hits < self._min_frames:
            return None
        take = min(len(self._pre), self._preroll_frames + self._run)
        self._speech = list(self._pre)[-take:]
        self._pre.clear()
        self._silence = 0
        return SpeechStarted()

    def _feed_speech(self, frame: np.ndarray, p: float, end_threshold: float) -> Utterance | None:
        assert self._speech is not None
        self._speech.append(frame)
        self._silence = 0 if p >= end_threshold else self._silence + 1
        if self._silence >= self._end_frames or len(self._speech) >= self._max_frames:
            return self._finish()
        return None

    def _finish(self) -> Utterance:
        assert self._speech is not None
        frames = self._speech
        if self._silence > self._keep_tail:
            frames = frames[: len(frames) - (self._silence - self._keep_tail)]
        pcm = np.concatenate(frames)
        self.reset(reset_vad=False)
        return Utterance(pcm=pcm, seconds=len(pcm) / SAMPLE_RATE)
