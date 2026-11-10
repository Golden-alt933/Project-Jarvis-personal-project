"""Audio devices: an always-open microphone and a gapless, interruptible speaker.

The microphone opens ONE input stream at start() and keeps it until stop(); the
PortAudio callback only copies the block into a bounded buffer. Re-framing to
FRAME_SAMPLES (and resampling to 16 kHz when the device refuses that rate) happens
on the asyncio side. The speaker keeps one output stream open and feeds it from a
queue inside the callback, so consecutive chunks are gapless and interrupt() takes
effect on the very next audio block (an epoch counter discards stale chunks).
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from typing import Any, AsyncIterator

import numpy as np

from jarvis.voice.types import FRAME_SAMPLES, SAMPLE_RATE

try:  # the voice extra may be missing; NullOutput/ArrayInput must still import
    import sounddevice as sd
except Exception:  # pragma: no cover - depends on the environment
    sd = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_FADE_SECONDS = 0.005
_LEVEL_ATTACK = 0.4  # weight of the newest block in the smoothed level


# --------------------------------------------------------------------- resampling


class StreamResampler:
    """Streaming resampler: Kaiser low-pass (only when downsampling) + linear interpolation.

    For integer decimation (48k -> 16k) the interpolation points land exactly on
    samples, so this is a plain FIR decimator. Cheap enough for the audio thread.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.step = src_rate / dst_rate
        if self.step > 1.0:
            taps = 16 * math.ceil(self.step) + 1
            cutoff = 0.5 / self.step * 0.92
            n = np.arange(taps) - (taps - 1) / 2
            h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(taps, 8.0)
            self._h: np.ndarray | None = (h / h.sum()).astype(np.float32)
            self._tail = np.zeros(taps - 1, dtype=np.float32)
            self.delay = (taps - 1) // 2  # group delay in source samples
        else:
            self._h = None
            self._tail = np.zeros(0, dtype=np.float32)
            self.delay = 0
        self._pend = np.zeros(0, dtype=np.float32)
        self._pos = 0.0  # next output position, relative to _pend (may exceed its length)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._h is not None:
            x = np.concatenate([self._tail, x])
            y = np.convolve(x, self._h, mode="valid").astype(np.float32)
            self._tail = x[-(len(self._h) - 1):]
        else:
            y = x
        buf = np.concatenate([self._pend, y])
        n = len(buf)
        count = math.ceil((n - 1 - self._pos) / self.step) if n - 1 > self._pos else 0
        if count <= 0:
            self._pend = buf
            return np.zeros(0, dtype=np.float32)
        idx = self._pos + np.arange(count) * self.step
        i = idx.astype(np.int64)
        frac = (idx - i).astype(np.float32)
        out = buf[i] * (1.0 - frac) + buf[i + 1] * frac
        nxt = self._pos + count * self.step
        k = min(int(nxt), n)
        self._pend = buf[k:]
        self._pos = nxt - k
        return out.astype(np.float32)


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """One-shot resample that preserves duration and removes the filter delay."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if src_rate == dst_rate or len(pcm) == 0:
        return pcm
    r = StreamResampler(src_rate, dst_rate)
    out = r.process(np.concatenate([pcm, np.zeros(r.delay + 4, dtype=np.float32)]))
    skip = int(round(r.delay / r.step))
    want = int(round(len(pcm) * dst_rate / src_rate))
    out = out[skip:skip + want]
    if len(out) < want:
        out = np.pad(out, (0, want - len(out)))
    return out


def _to_mono_f32(pcm: np.ndarray) -> np.ndarray:
    pcm = np.asarray(pcm)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


# ----------------------------------------------------------------------- devices


def resolve_device(spec: str | int | None, *, output: bool = False) -> int | None:
    """Map a name substring / index / empty to a PortAudio device index (None = default)."""
    if spec is None or sd is None:
        return None
    if isinstance(spec, str):
        spec = spec.strip()
        if not spec or spec.lower() == "default":
            return None
        if spec.lstrip("-").isdigit():
            spec = int(spec)
    if isinstance(spec, int):
        return spec
    key = "max_output_channels" if output else "max_input_channels"
    try:
        devices = sd.query_devices()
        default_api = sd.default.hostapi
    except Exception as exc:
        log.warning("Cannot list audio devices (%s); using the default.", exc)
        return None
    matches = [
        (i, d) for i, d in enumerate(devices)
        if spec.lower() in str(d["name"]).lower() and d[key] > 0
    ]
    if not matches:
        log.warning("No %s device matches %r; using the default.", "output" if output else "input", spec)
        return None
    # Windows lists each device once per host API; prefer the default one.
    matches.sort(key=lambda m: m[1].get("hostapi") != default_api)
    return matches[0][0]


def list_devices() -> str:
    """A readable device table for the `--devices` flag."""
    if sd is None:
        return "sounddevice is not installed (install the 'voice' extra)."
    try:
        devices = sd.query_devices()
        apis = sd.query_hostapis()
        default_in, default_out = sd.default.device
    except Exception as exc:
        return f"Cannot query audio devices: {exc}"
    lines = [f"{'idx':>3}  {'name':<44} {'api':<14} {'in':>3} {'out':>3} {'rate':>6}  default"]
    for i, d in enumerate(devices):
        mark = ("in " if i == default_in else "") + ("out" if i == default_out else "")
        api = apis[d["hostapi"]]["name"] if d["hostapi"] < len(apis) else "?"
        lines.append(
            f"{i:>3}  {str(d['name'])[:44]:<44} {api[:14]:<14} "
            f"{d['max_input_channels']:>3} {d['max_output_channels']:>3} "
            f"{int(d['default_samplerate']):>6}  {mark}"
        )
    return "\n".join(lines)


# -------------------------------------------------------------------- microphone


class MicInput:
    """Always-open mono microphone delivering exact FRAME_SAMPLES frames at 16 kHz."""

    def __init__(
        self,
        device: str | int | None = None,
        *,
        retry_seconds: float = 2.0,
        max_backlog_seconds: float = 2.0,
    ) -> None:
        self.device = device
        self.level = 0.0
        self.available = False
        self.sample_rate = SAMPLE_RATE  # rate the stream was actually opened at
        self._retry = retry_seconds
        self._max_backlog_seconds = max_backlog_seconds
        self._lock = threading.Lock()
        self._blocks: deque[np.ndarray] = deque()
        self._backlog = 0
        self._max_backlog = int(max_backlog_seconds * SAMPLE_RATE)
        self._dropped = 0
        self._drop_logged = False
        self._fail_logged = False
        self._pending = np.zeros(0, dtype=np.float32)
        self._resampler: StreamResampler | None = None
        self._stream: Any = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake = asyncio.Event()
        self._supervisor: asyncio.Task[None] | None = None

    # lifecycle

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        await asyncio.to_thread(self._try_open)
        self._supervisor = asyncio.create_task(self._supervise())

    async def stop(self) -> None:
        self._running = False
        if self._supervisor is not None:
            self._supervisor.cancel()
            await asyncio.gather(self._supervisor, return_exceptions=True)
            self._supervisor = None
        await asyncio.to_thread(self._close)
        self._wake.set()

    # PortAudio side: do as little as possible here.

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        block = np.array(indata[:, 0] if indata.ndim == 2 else indata, dtype=np.float32)
        self.level += _LEVEL_ATTACK * (_rms(block) - self.level)
        with self._lock:
            self._blocks.append(block)
            self._backlog += len(block)
            cap = int(self._max_backlog_seconds * self.sample_rate)
            while self._backlog > cap and len(self._blocks) > 1:
                gone = len(self._blocks.popleft())
                self._backlog -= gone
                self._dropped += gone
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._wake.set)
            except RuntimeError:  # loop already closed
                pass

    # device management

    def _open_stream(self, rate: int, device: int | None) -> Any:
        stream = sd.InputStream(
            samplerate=rate, channels=1, dtype="float32", device=device,
            blocksize=int(rate * 0.02), callback=self._callback,
        )
        stream.start()
        return stream

    def _try_open(self) -> bool:
        if sd is None:
            log.error("sounddevice is not installed; microphone unavailable.")
            return False
        try:
            device = resolve_device(self.device)
            try:
                stream, rate = self._open_stream(SAMPLE_RATE, device), SAMPLE_RATE
            except Exception as exc:
                native = int(sd.query_devices(device, "input")["default_samplerate"])
                log.info("Mic refused %d Hz (%s); opening at native %d Hz.", SAMPLE_RATE, exc, native)
                stream, rate = self._open_stream(native, device), native
        except Exception as exc:
            if not self._fail_logged:
                log.warning("Microphone unavailable (%s); retrying every %.0fs.", exc, self._retry)
                self._fail_logged = True
            self.available = False
            return False
        with self._lock:
            self._blocks.clear()
            self._backlog = 0
            self._pending = np.zeros(0, dtype=np.float32)
            self._resampler = StreamResampler(rate, SAMPLE_RATE) if rate != SAMPLE_RATE else None
        self.sample_rate = rate
        self._stream = stream
        self.available = True
        self._fail_logged = False
        log.info("Microphone open at %d Hz.", rate)
        return True

    def _close(self) -> None:
        stream, self._stream = self._stream, None
        self.available = False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                log.debug("Closing microphone stream: %s", exc)

    async def _supervise(self) -> None:
        while self._running:
            stream = self._stream
            healthy = stream is not None and bool(getattr(stream, "active", True))
            if healthy:
                await asyncio.sleep(0.25)
                continue
            if stream is not None:
                log.warning("Microphone stream stopped; reopening.")
                await asyncio.to_thread(self._close)
            if not await asyncio.to_thread(self._try_open):
                await asyncio.sleep(self._retry)

    # asyncio side

    def _take(self) -> list[np.ndarray]:
        with self._lock:
            blocks = list(self._blocks)
            self._blocks.clear()
            self._backlog = 0
            resampler = self._resampler
        if resampler is not None:
            blocks = [resampler.process(b) for b in blocks]
        return blocks

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while self._running:
            blocks = self._take()
            if self._dropped and not self._drop_logged:
                self._drop_logged = True
                log.warning("Audio consumer is behind; dropping the oldest audio (once-only notice).")
            if not blocks:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass
                continue
            data = np.concatenate([self._pending, *blocks])
            n = len(data) // FRAME_SAMPLES
            self._pending = data[n * FRAME_SAMPLES:]
            for i in range(n):
                yield data[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]


# ----------------------------------------------------------------------- speaker


class SpeakerOutput:
    """One open output stream fed from a queue; interrupt() is immediate and thread-safe."""

    def __init__(self, device: str | int | None = None, sample_rate: int = 22050) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.level = 0.0
        self.available = False
        self._lock = threading.Lock()
        self._queue: deque[np.ndarray] = deque()
        self._offset = 0
        self._queued = 0
        self._epoch = 0
        self._last = 0.0
        self._fade_left = 0
        self._fade_amp = 0.0
        self._fade_total = 1
        self._play_end = 0.0  # monotonic time the last written audio finishes sounding
        self._latency = 0.0
        self._stream: Any = None

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def is_playing(self) -> bool:
        return self._queued > 0 or time.monotonic() < self._play_end

    # lifecycle

    async def start(self) -> None:
        if self._stream is None:
            await asyncio.to_thread(self._open)

    async def stop(self) -> None:
        self.interrupt()
        await asyncio.to_thread(self._close)

    def _open_stream(self, rate: int, device: int | None) -> Any:
        stream = sd.OutputStream(
            samplerate=rate, channels=1, dtype="float32", device=device,
            latency="low", callback=self._callback,
        )
        stream.start()
        return stream

    def _open(self) -> None:
        if sd is None:
            log.error("sounddevice is not installed; speaker unavailable.")
            return
        try:
            device = resolve_device(self.device, output=True)
            try:
                stream, rate = self._open_stream(self.sample_rate, device), self.sample_rate
            except Exception as exc:
                native = int(sd.query_devices(device, "output")["default_samplerate"])
                log.info("Speaker refused %d Hz (%s); opening at %d Hz.", self.sample_rate, exc, native)
                stream, rate = self._open_stream(native, device), native
        except Exception as exc:
            log.warning("Speaker unavailable (%s); audio will be discarded.", exc)
            return
        latency = getattr(stream, "latency", 0.0)
        self._latency = float(latency[-1] if isinstance(latency, (tuple, list)) else latency or 0.0)
        self.sample_rate = rate
        self._fade_total = max(1, int(_FADE_SECONDS * rate))
        self._stream = stream
        self.available = True

    def _close(self) -> None:
        stream, self._stream = self._stream, None
        self.available = False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                log.debug("Closing speaker stream: %s", exc)

    # audio thread

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        out = outdata[:, 0] if outdata.ndim == 2 else outdata
        out[:] = 0.0
        with self._lock:
            pos = 0
            if self._fade_left > 0:
                pos = min(frames, self._fade_left)
                ramp = np.arange(self._fade_left, self._fade_left - pos, -1) / self._fade_total
                out[:pos] = self._fade_amp * ramp
                self._fade_left -= pos
            start = pos
            while pos < frames and self._queue:
                chunk = self._queue[0]
                take = min(len(chunk) - self._offset, frames - pos)
                out[pos:pos + take] = chunk[self._offset:self._offset + take]
                pos += take
                self._offset += take
                self._queued -= take
                if self._offset >= len(chunk):
                    self._queue.popleft()
                    self._offset = 0
            if pos > start:
                self._last = float(out[pos - 1])
                self._play_end = time.monotonic() + self._latency + frames / self.sample_rate
        self.level += _LEVEL_ATTACK * (_rms(out) - self.level)

    # public API

    def play(self, pcm: np.ndarray, sample_rate: int, *, epoch: int | None = None) -> None:
        """Queue audio. A stale `epoch` (captured before an interrupt) is silently ignored."""
        if self._stream is None:
            log.debug("play() ignored: speaker not open.")
            return
        chunk = _to_mono_f32(pcm)
        if sample_rate != self.sample_rate:
            chunk = resample(chunk, sample_rate, self.sample_rate)
        if len(chunk) == 0:
            return
        with self._lock:
            if epoch is not None and epoch != self._epoch:
                return
            self._queue.append(chunk)
            self._queued += len(chunk)

    def interrupt(self) -> None:
        with self._lock:
            self._epoch += 1
            self._queue.clear()
            self._offset = 0
            self._queued = 0
            if time.monotonic() < self._play_end and self._last != 0.0:
                self._fade_left = self._fade_total
                self._fade_amp = self._last
            self._last = 0.0
            self._play_end = 0.0

    async def drain(self) -> None:
        epoch = self._epoch
        while self._epoch == epoch and self.is_playing:
            await asyncio.sleep(0.01)


# ------------------------------------------------------------- test / offline I/O


class NullOutput:
    """Discards audio but keeps real-time pacing so drain() lasts the true duration."""

    def __init__(self, speed: float = 1.0) -> None:
        self.played: list[np.ndarray] = []
        self.level = 0.0
        self._speed = speed
        self._epoch = 0
        self._end = 0.0
        self._segments: deque[tuple[float, float, float]] = deque()  # start, end, rms

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def is_playing(self) -> bool:
        return time.monotonic() < self._end

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.interrupt()

    def play(self, pcm: np.ndarray, sample_rate: int, *, epoch: int | None = None) -> None:
        if epoch is not None and epoch != self._epoch:
            return
        chunk = _to_mono_f32(pcm)
        self.played.append(chunk.copy())
        now = time.monotonic()
        start = max(now, self._end)
        self._end = start + len(chunk) / sample_rate / self._speed
        self._segments.append((start, self._end, _rms(chunk)))

    def interrupt(self) -> None:
        self._epoch += 1
        self._end = 0.0
        self._segments.clear()

    async def drain(self) -> None:
        epoch = self._epoch
        while self._epoch == epoch and self.is_playing:
            await asyncio.sleep(0.005)

    @property
    def level(self) -> float:  # type: ignore[override]
        now = time.monotonic()
        while self._segments and self._segments[0][1] <= now:
            self._segments.popleft()
        if self._segments and self._segments[0][0] <= now:
            return self._segments[0][2]
        return 0.0

    @level.setter
    def level(self, _value: float) -> None:
        return None


class ArrayInput:
    """Yields frames from a prepared 16 kHz array.

    After the array: ends (default), or yields `silence_after` seconds of silence
    (math.inf = until stop()). A trailing partial frame is zero-padded.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        realtime: bool = False,
        *,
        silence_after: float | None = None,
    ) -> None:
        data = _to_mono_f32(pcm)
        silent = 0 if not silence_after else (
            math.inf if math.isinf(silence_after)
            else int(silence_after * SAMPLE_RATE)
        )
        self._data = data
        self._silence = silent
        self._realtime = realtime
        self._pos = 0  # frame index
        self._stopped = False
        self.level = 0.0

    async def start(self) -> None:
        self._stopped = False

    async def stop(self) -> None:
        self._stopped = True

    def _frame_at(self, index: int) -> np.ndarray | None:
        lo = index * FRAME_SAMPLES
        if lo < len(self._data):
            frame = self._data[lo:lo + FRAME_SAMPLES]
            if len(frame) < FRAME_SAMPLES:
                frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
            return frame
        past = lo - len(self._data)
        if math.isinf(self._silence) or past < self._silence:
            return np.zeros(FRAME_SAMPLES, dtype=np.float32)
        return None

    async def frames(self) -> AsyncIterator[np.ndarray]:
        period = FRAME_SAMPLES / SAMPLE_RATE
        next_at = time.monotonic()
        while not self._stopped:
            frame = self._frame_at(self._pos)
            if frame is None:
                return
            self._pos += 1
            self.level += _LEVEL_ATTACK * (_rms(frame) - self.level)
            if self._realtime:
                next_at += period
                await asyncio.sleep(max(0.0, next_at - time.monotonic()))
            else:
                await asyncio.sleep(0)
            yield frame


def open_audio(settings: Any) -> tuple[MicInput, SpeakerOutput]:
    """Build (not start) the mic and speaker for the configured devices."""
    return (
        MicInput(settings.input_device or None),
        SpeakerOutput(settings.output_device or None),
    )
