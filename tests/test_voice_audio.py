"""Audio layer tests. None needs a real audio device."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.voice import audio
from jarvis.voice.audio import (
    ArrayInput, MicInput, NullOutput, SpeakerOutput, StreamResampler, resample,
)
from jarvis.voice.types import FRAME_SAMPLES, SAMPLE_RATE, AudioInput, AudioOutput, SpeechStarted, Utterance
from jarvis.voice.vad import EnergyVAD, Endpointer, SileroVAD, build_vad


# ------------------------------------------------------------------ fake PortAudio


class FakePortAudioError(Exception):
    pass


class FakeStream:
    def __init__(self, kind, owner, **kw):
        self.kind, self.owner = kind, owner
        self.samplerate = kw["samplerate"]
        self.callback = kw["callback"]
        self.latency = 0.0
        self.active = False
        self.closed = False
        if self.samplerate in owner.reject_rates:
            raise FakePortAudioError(f"rate {self.samplerate} rejected")
        if owner.fail_open > 0:
            owner.fail_open -= 1
            raise FakePortAudioError("device busy")
        owner.streams.append(self)

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False
        self.closed = True

    # test helpers
    def push(self, block):
        self.callback(np.asarray(block, dtype=np.float32).reshape(-1, 1), len(block), None, None)

    def pull(self, frames):
        out = np.zeros((frames, 1), dtype=np.float32)
        self.callback(out, frames, None, None)
        return out[:, 0].copy()


class FakeSD:
    PortAudioError = FakePortAudioError

    def __init__(self, native_rate=48000):
        self.streams: list[FakeStream] = []
        self.reject_rates: set[int] = set()
        self.fail_open = 0
        self.native_rate = native_rate
        self.default = SimpleNamespace(hostapi=0, device=(0, 1))

    def InputStream(self, **kw):
        return FakeStream("in", self, **kw)

    def OutputStream(self, **kw):
        return FakeStream("out", self, **kw)

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            return [
                {"name": "Microphone (USB Mic)", "hostapi": 1, "max_input_channels": 1,
                 "max_output_channels": 0, "default_samplerate": 44100.0},
                {"name": "Microphone (USB Mic)", "hostapi": 0, "max_input_channels": 1,
                 "max_output_channels": 0, "default_samplerate": 48000.0},
                {"name": "Speakers (Realtek)", "hostapi": 0, "max_input_channels": 0,
                 "max_output_channels": 2, "default_samplerate": 48000.0},
            ]
        return {"default_samplerate": float(self.native_rate)}

    def query_hostapis(self):
        return [{"name": "WASAPI"}, {"name": "MME"}]


@pytest.fixture
def fake_sd(monkeypatch):
    fake = FakeSD()
    monkeypatch.setattr(audio, "sd", fake)
    return fake


def tone(freq, seconds, rate, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


async def collect(mic, count, timeout=3.0):
    got = []

    async def run():
        async for frame in mic.frames():
            got.append(frame)
            if len(got) >= count:
                return

    await asyncio.wait_for(run(), timeout)
    return got


# --------------------------------------------------------------------- resampling


def test_resample_preserves_tone_and_length():
    src = tone(440, 1.0, 48000)
    out = resample(src, 48000, 16000)
    assert len(out) == 16000
    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak_hz = np.argmax(spectrum) * 16000 / len(out)
    assert abs(peak_hz - 440) < 2
    assert abs(np.sqrt(np.mean(out**2)) - 0.5 / np.sqrt(2)) < 0.02


def test_resample_rejects_alias():
    # 10 kHz would fold to 6 kHz at 16 kHz if not low-passed.
    out = resample(tone(10000, 0.5, 48000), 48000, 16000)
    assert np.sqrt(np.mean(out[200:-200] ** 2)) < 0.02


def test_stream_resampler_matches_across_odd_blocks():
    src = tone(440, 1.0, 48000)
    r = StreamResampler(48000, 16000)
    pieces, i = [], 0
    for n in (137, 480, 999, 1, 2048):
        while i < len(src):
            pieces.append(r.process(src[i:i + n]))
            i += n
            break
    pieces.append(r.process(src[i:]))
    out = np.concatenate(pieces)
    ref = StreamResampler(48000, 16000).process(src)
    assert abs(len(out) - len(ref)) <= 1
    m = min(len(out), len(ref))
    assert np.allclose(out[:m], ref[:m], atol=1e-5)


def test_upsample_length():
    out = resample(tone(440, 0.5, 22050), 22050, 48000)
    assert len(out) == 24000


# -------------------------------------------------------------------- MicInput


async def test_mic_reframes_odd_blocks_exactly(fake_sd):
    mic = MicInput()
    await mic.start()
    assert mic.available and mic.sample_rate == SAMPLE_RATE
    stream = fake_sd.streams[0]
    data = np.arange(5000, dtype=np.float32) / 5000
    i = 0
    for n in (100, 333, 1, 777, 2000, 1789):
        stream.push(data[i:i + n])
        i += n
    frames = await collect(mic, 9)  # 5000 // 512 = 9
    assert all(f.shape == (FRAME_SAMPLES,) and f.dtype == np.float32 for f in frames)
    assert np.array_equal(np.concatenate(frames), data[:9 * FRAME_SAMPLES])
    await mic.stop()


async def test_mic_native_rate_fallback_resamples(fake_sd):
    fake_sd.reject_rates = {16000}
    mic = MicInput()
    await mic.start()
    assert mic.sample_rate == 48000
    stream = fake_sd.streams[0]
    sig = tone(440, 1.0, 48000)
    for i in range(0, len(sig), 960):
        stream.push(sig[i:i + 960])
    frames = await collect(mic, 28)
    out = np.concatenate(frames)
    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    assert abs(np.argmax(spectrum) * 16000 / len(out) - 440) < 5
    await mic.stop()


async def test_mic_drops_oldest_when_consumer_lags(fake_sd, caplog):
    mic = MicInput(max_backlog_seconds=2.0)
    await mic.start()
    stream = fake_sd.streams[0]
    for k in range(10):  # 10 s of audio, value = second index, in 0.5 s blocks
        for _ in range(2):
            stream.push(np.full(8000, k, dtype=np.float32))
    with caplog.at_level("WARNING"):
        frames = await collect(mic, 10)
    assert frames[0][0] >= 8  # only the newest ~2 s survive
    assert sum("dropping" in r.message for r in caplog.records) == 1
    await mic.stop()


async def test_mic_level_tracks_input(fake_sd):
    mic = MicInput()
    await mic.start()
    for _ in range(20):
        fake_sd.streams[0].push(np.full(320, 0.3, dtype=np.float32))
    assert 0.25 < mic.level < 0.31
    await mic.stop()


async def test_mic_recovers_after_device_error(fake_sd):
    mic = MicInput(retry_seconds=0.05)
    await mic.start()
    first = fake_sd.streams[0]
    first.active = False  # unplug
    fake_sd.fail_open = 4  # first two reopen rounds (16 kHz + native each) fail
    for _ in range(100):
        await asyncio.sleep(0.05)
        if len(fake_sd.streams) > 1 and mic.available:
            break
    assert first.closed and mic.available and len(fake_sd.streams) == 2
    fake_sd.streams[1].push(np.zeros(1024, dtype=np.float32))
    assert len(await collect(mic, 2)) == 2
    await mic.stop()


async def test_mic_start_without_device_does_not_crash(fake_sd):
    fake_sd.fail_open = 2  # both the 16 kHz and the native attempt fail
    mic = MicInput(retry_seconds=0.05)
    await mic.start()
    assert not mic.available
    for _ in range(60):
        await asyncio.sleep(0.05)
        if mic.available:
            break
    assert mic.available
    await mic.stop()


def test_resolve_device_by_name_and_index(fake_sd):
    assert audio.resolve_device("usb mic") == 1  # prefers the default host API
    assert audio.resolve_device("2") == 2
    assert audio.resolve_device("") is None
    assert audio.resolve_device("realtek", output=True) == 2
    assert audio.resolve_device("nonexistent") is None
    assert "Realtek" in audio.list_devices()


def test_open_audio_uses_settings():
    mic, spk = audio.open_audio(SimpleNamespace(input_device="usb", output_device=""))
    assert mic.device == "usb" and spk.device is None


def test_protocols():
    assert isinstance(MicInput(), AudioInput)
    assert isinstance(SpeakerOutput(), AudioOutput)
    assert isinstance(NullOutput(), AudioOutput)
    assert isinstance(ArrayInput(np.zeros(10)), AudioInput)


# ----------------------------------------------------------------- SpeakerOutput


async def make_speaker(fake_sd, **kw):
    spk = SpeakerOutput(**kw)
    await spk.start()
    return spk, fake_sd.streams[-1]


async def test_speaker_gapless_concatenation(fake_sd):
    spk, stream = await make_speaker(fake_sd, sample_rate=22050)
    a, b, c = tone(300, 0.05, 22050), tone(500, 0.033, 22050), tone(700, 0.021, 22050)
    for chunk in (a, b, c):
        spk.play(chunk, 22050)
    total = len(a) + len(b) + len(c)
    out = np.concatenate([stream.pull(441) for _ in range(total // 441 + 1)])
    expected = np.concatenate([a, b, c])
    assert np.array_equal(out[:total], expected)
    assert np.all(out[total:] == 0)
    await spk.stop()


async def test_speaker_resamples_other_rates(fake_sd):
    spk, stream = await make_speaker(fake_sd, sample_rate=22050)
    spk.play(tone(440, 1.0, 16000), 16000)
    out = np.concatenate([stream.pull(2205) for _ in range(10)])
    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    assert abs(np.argmax(spectrum) * 22050 / len(out) - 440) < 5
    await spk.stop()


async def test_speaker_interrupt_discards_and_fades(fake_sd):
    spk, stream = await make_speaker(fake_sd, sample_rate=22050)
    spk.play(tone(440, 2.0, 22050, amp=0.5), 22050)
    stream.pull(1000)
    assert spk.is_playing
    spk.interrupt()
    assert not spk.is_playing
    out = stream.pull(1000)
    fade = int(0.005 * 22050)
    assert np.all(out[fade:] == 0)  # queued audio is gone
    assert np.abs(out[0]) < 0.51 and np.abs(out[:fade]).max() > 0  # ramp, not a hard cut
    assert np.all(np.abs(np.diff(out[:fade + 1])) < 0.02)  # no step discontinuity
    spk.play(tone(440, 0.1, 22050), 22050)  # works again afterwards
    assert np.abs(stream.pull(500)).max() > 0.1
    await spk.stop()


async def test_speaker_stale_epoch_ignored(fake_sd):
    spk, stream = await make_speaker(fake_sd)
    epoch = spk.epoch
    spk.interrupt()
    spk.play(tone(440, 0.1, 22050), 22050, epoch=epoch)
    assert not spk.is_playing
    await spk.stop()


async def test_speaker_interrupt_from_other_thread(fake_sd):
    spk, stream = await make_speaker(fake_sd)
    spk.play(tone(440, 2.0, 22050), 22050)
    t = threading.Thread(target=spk.interrupt)
    t.start()
    t.join()
    assert not spk.is_playing
    await spk.stop()


async def test_speaker_drain_semantics(fake_sd):
    spk, stream = await make_speaker(fake_sd)
    spk.play(tone(440, 0.2, 22050), 22050)
    drained = asyncio.create_task(spk.drain())
    await asyncio.sleep(0.05)
    assert not drained.done()
    for _ in range(12):
        stream.pull(441)
    await asyncio.wait_for(drained, 1)

    spk.play(tone(440, 2.0, 22050), 22050)  # interrupt releases a pending drain
    drained = asyncio.create_task(spk.drain())
    await asyncio.sleep(0.03)
    spk.interrupt()
    await asyncio.wait_for(drained, 1)
    await spk.stop()


async def test_speaker_level_tracks_actual_playback(fake_sd):
    spk, stream = await make_speaker(fake_sd)
    spk.play(tone(440, 0.5, 22050, amp=0.5), 22050)
    assert spk.level == 0.0  # enqueued, not yet sounding
    for _ in range(10):
        stream.pull(441)
    assert 0.3 < spk.level < 0.37  # ~0.5/sqrt(2)
    spk.interrupt()
    for _ in range(10):
        stream.pull(441)
    assert spk.level < 0.01
    await spk.stop()


async def test_speaker_falls_back_to_native_rate(fake_sd):
    fake_sd.reject_rates = {22050}
    spk, stream = await make_speaker(fake_sd, sample_rate=22050)
    assert spk.sample_rate == 48000
    spk.play(tone(440, 0.1, 22050), 22050)
    assert np.abs(stream.pull(2000)).max() > 0.1
    await spk.stop()


# -------------------------------------------------------------- NullOutput/ArrayInput


async def test_null_output_takes_real_duration():
    out = NullOutput()
    out.play(np.zeros(2205, dtype=np.float32), 22050)  # 0.1 s
    out.play(np.zeros(2205, dtype=np.float32), 22050)
    assert out.is_playing
    t0 = time.monotonic()
    await out.drain()
    assert 0.17 < time.monotonic() - t0 < 0.4
    assert len(out.played) == 2 and not out.is_playing


async def test_null_output_interrupt():
    out = NullOutput()
    out.play(np.full(22050 * 5, 0.2, dtype=np.float32), 22050)
    assert 0.19 < out.level < 0.21
    task = asyncio.create_task(out.drain())
    await asyncio.sleep(0.02)
    out.interrupt()
    await asyncio.wait_for(task, 0.5)
    assert not out.is_playing and out.level == 0.0
    assert len(out.played) == 1  # record is kept for assertions


async def test_array_input_frames_and_end():
    pcm = np.ones(FRAME_SAMPLES * 3 + 10, dtype=np.float32)
    frames = [f async for f in ArrayInput(pcm).frames()]
    assert len(frames) == 4
    assert all(len(f) == FRAME_SAMPLES for f in frames)
    assert frames[-1][10] == 0  # zero padded


async def test_array_input_silence_tail_and_stop():
    mic = ArrayInput(np.ones(FRAME_SAMPLES, dtype=np.float32), silence_after=0.1)
    frames = [f async for f in mic.frames()]
    assert len(frames) == 1 + int(0.1 * SAMPLE_RATE) // FRAME_SAMPLES + 1 - 1 or len(frames) > 2
    assert np.all(frames[-1] == 0)

    forever = ArrayInput(np.zeros(10), silence_after=float("inf"))
    n = 0
    async for _ in forever.frames():
        n += 1
        if n == 50:
            await forever.stop()
    assert n == 50


async def test_array_input_realtime_pacing():
    mic = ArrayInput(np.zeros(FRAME_SAMPLES * 5, dtype=np.float32), realtime=True)
    t0 = time.monotonic()
    frames = [f async for f in mic.frames()]
    assert len(frames) == 5 and time.monotonic() - t0 > 0.12


# ---------------------------------------------------------------------- Endpointer


class ScriptedVAD:
    def __init__(self):
        self.script: list[float] = []
        self.i = 0

    def probability(self, frame):
        p = self.script[self.i]
        self.i += 1
        return p

    def reset(self):
        pass


FRAME = np.full(FRAME_SAMPLES, 0.1, dtype=np.float32)


def run_script(script, **kw):
    vad = ScriptedVAD()
    vad.script = script
    ep = Endpointer(vad, **kw)
    return [(i, e) for i, f in enumerate(script) if (e := ep.feed(FRAME)) is not None], ep


def kinds(events):
    return [type(e).__name__ for _, e in events]


def test_endpointer_ignores_blip():
    events, _ = run_script([0.0] * 10 + [0.9] * 4 + [0.0] * 40)  # 128 ms < 200 ms
    assert events == []


def test_endpointer_normal_utterance_and_trimmed_tail():
    script = [0.0] * 20 + [0.9] * 30 + [0.0] * 40
    events, ep = run_script(script)
    assert kinds(events) == ["SpeechStarted", "Utterance"]
    started_at, _ = events[0]
    assert started_at == 20 + 6  # 7 frames (>=200 ms) of speech confirm
    utt = events[1][1]
    assert not ep.in_speech
    # preroll(10) + 30 speech frames + 7 trailing frames (~200 ms)
    assert len(utt.pcm) == (10 + 30 + 7) * FRAME_SAMPLES
    assert utt.seconds == pytest.approx(len(utt.pcm) / SAMPLE_RATE)
    assert events[1][0] == 20 + 30 + 21 - 1  # 650 ms = 21 silent frames


def test_endpointer_short_pause_does_not_split():
    script = [0.9] * 20 + [0.0] * 10 + [0.9] * 20 + [0.0] * 40  # 320 ms pause
    events, _ = run_script(script)
    assert kinds(events) == ["SpeechStarted", "Utterance"]


def test_endpointer_hysteresis_keeps_weak_tail():
    # 0.4 is below start (0.5) but above end (0.35): still speech, no split.
    script = [0.9] * 20 + [0.4] * 40 + [0.9] * 10 + [0.0] * 40
    events, _ = run_script(script)
    assert kinds(events) == ["SpeechStarted", "Utterance"]
    assert len(events[1][1].pcm) > 60 * FRAME_SAMPLES


def test_endpointer_max_seconds_cut():
    events, _ = run_script([0.9] * 100, max_seconds=1.0)
    assert kinds(events)[:2] == ["SpeechStarted", "Utterance"]  # continuous speech re-triggers after the cut
    assert events[1][1].seconds == pytest.approx(1.0, abs=0.05)


def test_endpointer_preroll_included_and_bounded():
    vad = ScriptedVAD()
    vad.script = [0.0] * 30 + [0.9] * 20 + [0.0] * 30
    ep = Endpointer(vad, preroll_ms=300)
    marker = 0
    utt = None
    for i in range(len(vad.script)):
        frame = np.full(FRAME_SAMPLES, i + 1, dtype=np.float32)
        out = ep.feed(frame)
        if isinstance(out, Utterance):
            utt = out
    assert utt is not None
    first_value = utt.pcm[0]
    # speech starts at frame index 30 (value 31); 10 pre-roll frames precede it
    assert first_value == 31 - 10 and marker == 0


def test_endpointer_threshold_change_takes_effect():
    vad = ScriptedVAD()
    vad.script = [0.6] * 20 + [0.0] * 40 + [0.6] * 20 + [0.0] * 40
    ep = Endpointer(vad)
    ep.start_threshold = 0.8  # Jarvis is speaking
    first = [ep.feed(FRAME) for _ in range(20)]
    assert all(e is None for e in first)
    ep.start_threshold = 0.5
    for _ in range(40):
        ep.feed(FRAME)
    got = [ep.feed(FRAME) for _ in range(20 + 40)]
    assert any(isinstance(e, SpeechStarted) for e in got)
    assert any(isinstance(e, Utterance) for e in got)


def test_endpointer_reset():
    vad = ScriptedVAD()
    vad.script = [0.9] * 20
    ep = Endpointer(vad)
    for _ in range(20):
        ep.feed(FRAME)
    assert ep.in_speech
    ep.reset()
    assert not ep.in_speech


# ------------------------------------------------------------------------- VAD


def test_energy_vad_adapts():
    vad = EnergyVAD()
    rng = np.random.default_rng(0)
    quiet = [(rng.standard_normal(FRAME_SAMPLES) * 0.002).astype(np.float32) for _ in range(60)]
    assert max(vad.probability(f) for f in quiet[30:]) < 0.2
    loud = (rng.standard_normal(FRAME_SAMPLES) * 0.1).astype(np.float32)
    assert vad.probability(loud) > 0.9


def test_build_vad_falls_back(monkeypatch):
    from jarvis.voice import vad as vad_mod

    monkeypatch.setattr(vad_mod, "_model_path", lambda: (_ for _ in ()).throw(FileNotFoundError("x")))
    assert isinstance(build_vad(), EnergyVAD)
    monkeypatch.undo()
    assert isinstance(build_vad(), SileroVAD)


@pytest.fixture(scope="module")
def silero():
    return SileroVAD()


def probs(vad, pcm):
    vad.reset()
    n = len(pcm) // FRAME_SAMPLES
    return np.array([vad.probability(pcm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]) for i in range(n)])


def test_silero_silence_and_noise_score_low(silero):
    assert probs(silero, np.zeros(SAMPLE_RATE * 2, dtype=np.float32)).max() < 0.1
    noise = (np.random.default_rng(1).standard_normal(SAMPLE_RATE * 2) * 0.05).astype(np.float32)
    assert np.percentile(probs(silero, noise), 90) < 0.3


def test_silero_frame_is_fast(silero):
    frame = np.random.default_rng(2).standard_normal(FRAME_SAMPLES).astype(np.float32) * 0.1
    for _ in range(20):
        silero.probability(frame)
    t0 = time.perf_counter()
    for _ in range(200):
        silero.probability(frame)
    assert (time.perf_counter() - t0) / 200 < 0.005


def test_silero_state_matters_and_reset_restores(silero):
    rng = np.random.default_rng(3)
    frame = (rng.standard_normal(FRAME_SAMPLES) * 0.1).astype(np.float32)
    silero.reset()
    first = silero.probability(frame)
    silero.probability(frame)
    silero.reset()
    assert silero.probability(frame) == pytest.approx(first)


@pytest.fixture(scope="module")
def sapi_speech(tmp_path_factory):
    if sys.platform != "win32":
        pytest.skip("SAPI is Windows-only")
    path = tmp_path_factory.mktemp("sapi") / "speech.wav"
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SetOutputToWaveFile('{path}');"
        "$s.Speak('Jarvis, what is the weather like today?');"
        "$s.Dispose()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True, capture_output=True, timeout=60,
        )
        with wave.open(str(path), "rb") as w:
            rate, width, chans = w.getframerate(), w.getsampwidth(), w.getnchannels()
            raw = w.readframes(w.getnframes())
    except Exception as exc:  # no SAPI voice, no powershell, ...
        pytest.skip(f"SAPI synthesis unavailable: {exc}")
    assert width == 2
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if chans > 1:
        pcm = pcm.reshape(-1, chans).mean(axis=1)
    pcm = resample(pcm, rate, SAMPLE_RATE)
    if len(pcm) < SAMPLE_RATE:
        pytest.skip("SAPI produced no audio")
    return pcm


def test_silero_scores_real_speech_high(silero, sapi_speech):
    p = probs(silero, sapi_speech)
    n = len(p)
    rms = np.array([
        np.sqrt(np.mean(sapi_speech[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] ** 2)) for i in range(n)
    ])
    active = rms > 0.2 * rms.max()
    assert active.sum() >= 10
    assert (p[active] > 0.5).mean() > 0.7
    assert p.max() > 0.9


def test_endpointer_single_utterance_from_real_speech(silero, sapi_speech):
    pcm = np.concatenate([sapi_speech, np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float32)])
    ep = Endpointer(silero, end_silence_ms=650)
    silero.reset()
    events = []
    for i in range(len(pcm) // FRAME_SAMPLES):
        out = ep.feed(pcm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
        if out is not None:
            events.append(out)
    assert [type(e).__name__ for e in events] == ["SpeechStarted", "Utterance"]
    speech_seconds = len(sapi_speech) / SAMPLE_RATE
    utt = events[1]
    assert 1.0 < utt.seconds <= speech_seconds + 0.3
    assert utt.seconds > speech_seconds - 1.0
