"""Online speech synthesis through Microsoft Edge's neural voices (the ``edge-tts`` package).

This is an unofficial client of a Microsoft service: it needs no key and no account, but
Microsoft can change or block it at any time, which is why it always sits behind the Piper
fallback (``jarvis.voice.tts.FallbackSynthesizer``). The text of each spoken sentence is
sent to Microsoft; nothing else is.

The service streams MP3. ``Mp3StreamDecoder`` turns the chunks into float32 mono with PyAV
as they arrive, so playback can start before the sentence has been fully synthesized.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Callable

import numpy as np

from jarvis.voice.tts import clean_for_speech

log = logging.getLogger(__name__)

EDGE_SAMPLE_RATE = 24000
DEFAULT_ONLINE_VOICE = "en-GB-RyanNeural"


class Mp3StreamDecoder:
    """Incremental MP3 -> float32 mono decoder (PyAV). Feed bytes, get audio as it is decoded."""

    def __init__(self, sample_rate: int = EDGE_SAMPLE_RATE) -> None:
        import av

        self.sample_rate = sample_rate
        self._av = av
        self._codec = av.codec.CodecContext.create("mp3", "r")
        self._resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)

    def _frames_to_array(self, frames) -> np.ndarray | None:
        parts = []
        for frame in frames:
            for out in self._resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
        if not parts:
            return None
        return np.concatenate(parts).astype(np.float32, copy=False)

    def feed(self, data: bytes) -> np.ndarray | None:
        frames = []
        for packet in self._codec.parse(data):
            try:
                frames.extend(self._codec.decode(packet))
            except self._av.error.InvalidDataError:
                continue  # an ID3 tag or a Xing header frame: not audio
        return self._frames_to_array(frames)

    def flush(self) -> np.ndarray | None:
        frames = list(self._codec.decode(None)) if self._codec.is_open else []
        parts = []
        out = self._frames_to_array(frames)
        if out is not None:
            parts.append(out)
        try:
            for tail in self._resampler.resample(None):
                parts.append(tail.to_ndarray().reshape(-1))
        except Exception:  # an already-flushed resampler
            pass
        return np.concatenate(parts).astype(np.float32, copy=False) if parts else None


def decode_mp3(data: bytes, sample_rate: int = EDGE_SAMPLE_RATE) -> np.ndarray:
    """Decode a complete MP3 in one go (used by tests and the stock-phrase cache)."""
    decoder = Mp3StreamDecoder(sample_rate)
    parts = [p for p in (decoder.feed(data), decoder.flush()) if p is not None]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


ChunkSource = Callable[[str], AsyncIterator[bytes]]


class EdgeSynthesizer:
    """Edge neural voice behind the Synthesizer protocol, with ``synthesize_stream``."""

    sample_rate = EDGE_SAMPLE_RATE

    def __init__(
        self,
        voice: str = DEFAULT_ONLINE_VOICE,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        *,
        chunk_source: ChunkSource | None = None,
    ) -> None:
        self.voice = voice
        self.rate = rate or "+0%"
        self.pitch = pitch or "+0Hz"
        self.name = f"edge {voice}"
        self.cache_key = f"edge:{voice}:{self.rate}:{self.pitch}"
        self._chunk_source = chunk_source or self._edge_chunks

    async def _edge_chunks(self, text: str) -> AsyncIterator[bytes]:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        async for item in communicate.stream():
            if item["type"] == "audio":
                yield item["data"]

    async def warm_up(self) -> None:
        # A real round trip: proves the service answers, and warms DNS/TLS.
        await self.synthesize("Ready.")

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        text = clean_for_speech(text)
        if not text:
            return
        decoder = Mp3StreamDecoder(self.sample_rate)
        got_audio = False
        async for data in self._chunk_source(text):
            audio = await asyncio.to_thread(decoder.feed, data) if len(data) > 8192 else decoder.feed(data)
            if audio is not None and len(audio):
                got_audio = True
                yield audio
        tail = decoder.flush()
        if tail is not None and len(tail):
            got_audio = True
            yield tail
        if not got_audio:
            raise RuntimeError("the speech service returned no audio")

    async def synthesize(self, text: str) -> np.ndarray:
        parts = [part async for part in self.synthesize_stream(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
