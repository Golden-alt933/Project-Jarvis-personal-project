"""Speech-to-text with faster-whisper.

Device/model selection ("auto") is encoded from measurements on the target machine
(i5-12450H, RTX 3050 Laptop 4 GB); see AUTO_BENCHMARK below.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from jarvis import platform as plat
from jarvis.voice.types import SAMPLE_RATE, Transcript

if TYPE_CHECKING:
    from jarvis.config import Settings

log = logging.getLogger(__name__)

# Measured 2026-11-11 on i5-12450H + RTX 3050 4 GB, beam 1, Piper-synthesized speech
# (clean audio; a real microphone is noisier), seconds for a 3 s / 9 s utterance:
#   cpu int8          tiny.en 0.7/0.8   base.en 1.3/1.4   small.en 4.3/4.8   distil-small.en 3.9/4.4
#   cuda int8_float16                   base.en 0.2/0.35  small.en 0.3/0.6   distil-small.en 0.18/0.3
# distil-small.en was the only model with zero errors with and without a hint, and it
# needs about 0.4 GB VRAM. On CPU only tiny.en meets the 0.7 s target, but it is too
# weak on real microphone audio, so base.en is the CPU default.
AUTO_BENCHMARK = "measured 2026-11-11"
AUTO_MODEL_CPU = "base.en"
AUTO_MODEL_CUDA = "distil-small.en"
MIN_FREE_VRAM_MB = 1200

# Whisper invents these on near-silent input. Real replies like "yes"/"stop" are not listed.
_HALLUCINATIONS = frozenset(
    {
        "thank you", "thanks", "thank you very much", "thanks for watching",
        "thank you for watching", "thanks for watching!", "you", "bye", "bye bye",
        "see you next time", "see you in the next video", "subscribe",
        "please subscribe", "like and subscribe", "the end", "amara.org",
        "subtitles by the amara.org community", "so", "uh", "um", "hmm", "mm",
    }
)
SILENCE_RMS = 0.003  # below this the audio is effectively silent: skip inference
LOW_ENERGY_RMS = 0.012  # hallucination phrases are dropped below this level...
POOR_CONFIDENCE = 0.5  # ...or below this confidence
NO_SPEECH_PROB = 0.6
NO_SPEECH_LOGPROB = -1.0
MAX_COMPRESSION = 2.4  # repetition loops compress too well


# ------------------------------------------------------------------ CUDA discovery


def _add_nvidia_dll_dirs() -> None:
    """Make the pip-installed cuBLAS/cuDNN (nvidia-*-cu12) libraries findable by ctranslate2.
    The OS-specific loading lives in the platform backend."""
    try:
        plat.current().prepare_gpu_libs()
    except Exception:  # the CPU path works without it
        log.debug("could not prepare GPU libraries", exc_info=True)


_add_nvidia_dll_dirs()


def _cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _free_vram_mb() -> int | None:
    """Free VRAM of GPU 0 in MB via nvidia-smi, or None if it cannot be determined."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def resolve_device(device: str) -> str:
    """"auto" -> cuda only if a device exists and enough VRAM is free right now."""
    if device != "auto":
        return device
    if _cuda_device_count() == 0:
        return "cpu"
    free = _free_vram_mb()
    if free is None or free < MIN_FREE_VRAM_MB:
        log.info("STT: GPU has %s MB free (< %d), using CPU", free, MIN_FREE_VRAM_MB)
        return "cpu"
    return "cuda"


def resolve_model(model: str, device: str) -> str:
    if model != "auto":
        return model
    return AUTO_MODEL_CUDA if device == "cuda" else AUTO_MODEL_CPU


# ------------------------------------------------------------------ confidence logic


@dataclass
class _Scored:
    text: str
    confidence: float
    non_speech: bool


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()


def score_segments(segments: Iterable[Any]) -> _Scored:
    """Join segment text and derive a 0..1 confidence from Whisper's own statistics."""
    texts: list[str] = []
    weights: list[float] = []
    confs: list[float] = []
    non_speech = True
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        no_speech = float(getattr(seg, "no_speech_prob", 0.0))
        logprob = float(getattr(seg, "avg_logprob", 0.0))
        compression = float(getattr(seg, "compression_ratio", 1.0))
        conf = min(1.0, math.exp(min(logprob, 0.0))) * (1.0 - no_speech)
        looks_silent = no_speech > NO_SPEECH_PROB and logprob < NO_SPEECH_LOGPROB
        looped = compression > MAX_COMPRESSION
        if looped:
            conf *= 0.3
        if not (looks_silent or looped):
            non_speech = False
        texts.append(text)
        confs.append(conf)
        weights.append(max(len(text), 1))
    if not texts:
        return _Scored("", 0.0, True)
    confidence = sum(c * w for c, w in zip(confs, weights)) / sum(weights)
    return _Scored(" ".join(texts), confidence, non_speech)


def filter_hallucination(scored: _Scored, rms: float) -> str:
    """Return the final text, or "" when the output is clearly not real speech."""
    if scored.non_speech or not scored.text:
        return ""
    if _normalise(scored.text) in _HALLUCINATIONS and (
        rms < LOW_ENERGY_RMS or scored.confidence < POOR_CONFIDENCE
    ):
        return ""
    return scored.text


def initial_prompt(hint: str | None) -> str | None:
    """Turn a bare name into a natural sentence.

    Measured: with the prompt "Jarvis" alone, base.en dropped the word and small.en
    answered just "Jarvis"; "Hey Jarvis, can you help me?" kept it in every case.
    """
    hint = (hint or "").strip()
    if not hint:
        return None
    if " " not in hint and len(hint) <= 24:
        return f"Hey {hint.capitalize()}, can you help me?"
    return hint


def _rms(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0


# ----------------------------------------------------------------------- transcriber


class WhisperTranscriber:
    """faster-whisper behind the Transcriber protocol. Falls back to CPU if CUDA fails."""

    def __init__(self, model: str = "auto", device: str = "auto", beam_size: int = 1):
        self._requested_device = device
        self.device = resolve_device(device)
        self.model_name = resolve_model(model, self.device)
        self._model_setting = model
        self.compute_type = "int8_float16" if self.device == "cuda" else "int8"
        self.beam_size = beam_size
        self._model: Any = None
        self.name = self._describe()

    def _describe(self) -> str:
        return f"faster-whisper {self.model_name} ({self.device})"

    def _build(self, device: str, compute_type: str) -> Any:
        from faster_whisper import WhisperModel

        kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
        if device == "cpu":
            kwargs["cpu_threads"] = min(8, os.cpu_count() or 4)
        return WhisperModel(self.model_name, **kwargs)

    def _fall_back_to_cpu(self, reason: object) -> None:
        log.warning("STT: CUDA failed (%s); falling back to CPU", reason)
        self.device, self.compute_type = "cpu", "int8"
        self.model_name = resolve_model(self._model_setting, "cpu")
        self._model = self._build("cpu", "int8")
        self.name = self._describe()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        start = time.perf_counter()
        try:
            self._model = self._build(self.device, self.compute_type)
            if self.device == "cuda":
                self._infer(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), None)  # surfaces missing DLLs now
        except Exception as exc:
            if self.device != "cuda":
                raise
            self._model = None
            self._fall_back_to_cpu(exc)
        log.info("STT: loaded %s in %.1fs", self.name, time.perf_counter() - start)
        return self._model

    def _infer(self, pcm: np.ndarray, hint: str | None) -> list[Any]:
        segments, _ = self._model.transcribe(
            pcm,
            language="en",
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt(hint),
            vad_filter=False,  # the audio is already endpointed
            without_timestamps=True,
        )
        return list(segments)  # the generator does the real work; consume it here

    def _transcribe_blocking(self, pcm: np.ndarray, hint: str | None) -> Transcript:
        start = time.perf_counter()
        pcm = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
        rms = _rms(pcm)
        if rms < SILENCE_RMS or pcm.size < SAMPLE_RATE // 10:
            return Transcript("", time.perf_counter() - start, 0.0)
        self._load()
        try:
            segments = self._infer(pcm, hint)
        except Exception as exc:
            if self.device != "cuda":
                raise
            self._fall_back_to_cpu(exc)
            segments = self._infer(pcm, hint)
        scored = score_segments(segments)
        text = filter_hallucination(scored, rms)
        return Transcript(text, time.perf_counter() - start, scored.confidence if text else 0.0)

    async def warm_up(self) -> None:
        def work() -> None:
            self._load()
            self._infer(np.zeros(SAMPLE_RATE, dtype=np.float32), None)

        await asyncio.to_thread(work)

    async def transcribe(self, pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
        return await asyncio.to_thread(self._transcribe_blocking, pcm, hint)


def build_transcriber(settings: Settings) -> WhisperTranscriber:
    return WhisperTranscriber(settings.stt_model, settings.stt_device)
