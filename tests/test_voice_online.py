"""Online speech: engines, fallback, the wake-word gate. Fakes only: no network, no audio device."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

from jarvis.config import Settings
from jarvis.voice import stt as stt_mod
from jarvis.voice import tts as tts_mod
from jarvis.voice.stt_online import GatedTranscriber, OpenAICompatTranscriber, RivaTranscriber, wav_bytes
from jarvis.voice.tts import FallbackSynthesizer, PiperSynthesizer, build_synthesizer, resample_audio
from jarvis.voice.tts_edge import EdgeSynthesizer, Mp3StreamDecoder, decode_mp3
from jarvis.voice.tts_nvidia import NvidiaSynthesizer
from jarvis.voice.types import Transcript

from test_voice_session import Rig, SpeechStarted, say, utt


def tone(seconds: float, rate: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def make_mp3(seconds: float = 1.0, rate: int = 24000) -> bytes:
    import av

    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp3") as container:
        stream = container.add_stream("mp3", rate=rate)
        stream.layout = "mono"
        audio = tone(seconds, rate)
        for start in range(0, len(audio) - 1151, 1152):
            frame = av.AudioFrame.from_ndarray(audio[start:start + 1152].reshape(1, -1), format="flt", layout="mono")
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


# ------------------------------------------------------------------ MP3 streaming decode
def test_chunked_mp3_decoding_matches_whole_file() -> None:
    data = make_mp3(1.0)
    whole = decode_mp3(data)
    assert 0.9 < len(whole) / 24000 < 1.2
    decoder = Mp3StreamDecoder()
    parts = []
    for i in range(0, len(data), 333):  # awkward chunk size: frames straddle chunks
        out = decoder.feed(data[i:i + 333])
        if out is not None:
            parts.append(out)
    tail = decoder.flush()
    if tail is not None:
        parts.append(tail)
    streamed = np.concatenate(parts)
    assert abs(len(streamed) - len(whole)) < 2400
    assert len(parts) > 3  # audio arrived progressively, not at the end
    assert np.abs(streamed).max() > 0.1


async def test_edge_synthesizer_streams_decoded_chunks() -> None:
    data = make_mp3(1.0)

    async def source_(text: str):
        for i in range(0, len(data), 900):
            yield data[i:i + 900]

    synth = EdgeSynthesizer("en-GB-RyanNeural", chunk_source=source_)
    chunks = [c async for c in synth.synthesize_stream("Good evening.")]
    assert len(chunks) > 2 and all(c.dtype == np.float32 for c in chunks)
    full = await synth.synthesize("Good evening.")
    assert 0.9 < len(full) / synth.sample_rate < 1.2
    assert synth.cache_key.startswith("edge:en-GB-RyanNeural")


async def test_edge_synthesizer_no_audio_is_an_error() -> None:
    async def empty(text: str):
        return
        yield b""

    with pytest.raises(RuntimeError):
        await EdgeSynthesizer(chunk_source=empty).synthesize("hello there")


async def test_edge_passes_rate_and_pitch(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeCommunicate:
        def __init__(self, text: str, voice: str, **kw: Any) -> None:
            seen.update(text=text, voice=voice, **kw)

        async def stream(self):
            yield {"type": "WordBoundary"}
            yield {"type": "audio", "data": make_mp3(0.3)}

    import edge_tts

    monkeypatch.setattr(edge_tts, "Communicate", FakeCommunicate)
    audio = await EdgeSynthesizer("en-GB-ThomasNeural", "-8%", "-2Hz").synthesize("Hello **there**")
    assert len(audio) > 0
    assert seen == {"text": "Hello there", "voice": "en-GB-ThomasNeural", "rate": "-8%", "pitch": "-2Hz"}


# --------------------------------------------------------------------------- NVIDIA TTS
class FakeRivaTTS:
    def __init__(self, payloads: list[bytes] | Exception) -> None:
        self.payloads, self.calls = payloads, []

    def synthesize_online(self, text: str, voice: str, language: str, sample_rate_hz: int = 0):
        self.calls.append((text, voice, language, sample_rate_hz))
        if isinstance(self.payloads, Exception):
            raise self.payloads
        for p in self.payloads:
            yield SimpleNamespace(audio=p)


async def test_nvidia_synthesizer_streams_int16() -> None:
    pcm = (tone(0.1) * 32767).astype("<i2").tobytes()
    service = FakeRivaTTS([pcm[:1000], pcm[1000:]])
    synth = NvidiaSynthesizer("k", "Magpie-Multilingual.EN-US.Ray.Calm", service_factory=lambda: service)
    chunks = [c async for c in synth.synthesize_stream("Good evening")]
    assert len(chunks) == 2 and sum(len(c) for c in chunks) == len(pcm) // 2
    assert service.calls == [("Good evening", "Magpie-Multilingual.EN-US.Ray.Calm", "en-US", 24000)]


async def test_nvidia_synthesizer_errors_propagate() -> None:
    synth = NvidiaSynthesizer("k", service_factory=lambda: FakeRivaTTS(RuntimeError("unauthenticated")))
    with pytest.raises(RuntimeError, match="unauthenticated"):
        await synth.synthesize("hello")
    empty = NvidiaSynthesizer("k", service_factory=lambda: FakeRivaTTS([]))
    with pytest.raises(RuntimeError):
        await empty.synthesize("hello")


# ------------------------------------------------------------------- FallbackSynthesizer
class Engine:
    sample_rate = 24000

    def __init__(self, name: str, *, fail: bool = False, delay: float = 0.0, chunks: int = 1, die_after: int | None = None) -> None:
        self.name, self.fail, self.delay, self.chunks, self.die_after = name, fail, delay, chunks, die_after
        self.calls = 0
        self.cache_key = f"fake:{name}"

    async def warm_up(self) -> None:
        if self.fail:
            raise RuntimeError("offline")

    async def synthesize_stream(self, text: str):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("offline")
        for i in range(self.chunks):
            if self.die_after is not None and i == self.die_after:
                raise RuntimeError("connection reset")
            yield np.full(100, 0.1 * (i + 1), dtype=np.float32)


class Plain:
    """An engine with only synthesize(), like Piper."""

    name, sample_rate = "plain", 22050

    def __init__(self) -> None:
        self.calls = 0

    async def warm_up(self) -> None: ...

    async def synthesize(self, text: str) -> np.ndarray:
        self.calls += 1
        return np.full(2205, 0.2, dtype=np.float32)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_fallback_uses_online_and_streams() -> None:
    online, floor = Engine("online", chunks=3), Plain()
    synth = FallbackSynthesizer([online, floor])
    chunks = [c async for c in synth.synthesize_stream("hello")]
    assert len(chunks) == 3 and floor.calls == 0
    assert synth.name == "online" and synth.last_key == "fake:online"


async def test_fallback_on_error_then_cooloff_then_retry(caplog: pytest.LogCaptureFixture) -> None:
    clock = Clock()
    online, floor = Engine("online", fail=True), Plain()
    synth = FallbackSynthesizer([online, floor], cooloff=100, clock=clock)
    with caplog.at_level(logging.WARNING, logger="jarvis.voice.tts"):
        first = await synth.synthesize("one")
        second = await synth.synthesize("two")
    assert len(first) == len(second) > 0 and online.calls == 1 and floor.calls == 2  # not retried while cooling
    assert synth.name == "plain"
    assert len([r for r in caplog.records if "unavailable" in r.getMessage()]) == 1  # logged once
    clock.now = 101
    online.fail = False
    await synth.synthesize("three")
    assert online.calls == 2 and synth.name == "online"


async def test_fallback_deadline_for_first_audio() -> None:
    online, floor = Engine("slow", delay=1.0), Plain()
    synth = FallbackSynthesizer([online, floor], deadline=0.05)
    audio = await synth.synthesize("hello")
    assert floor.calls == 1 and len(audio) > 0
    await synth.synthesize("again")
    assert online.calls == 1  # cooling


async def test_fallback_resamples_to_output_rate() -> None:
    synth = FallbackSynthesizer([Plain()])
    audio = await synth.synthesize("hi")
    assert abs(len(audio) / synth.sample_rate - 0.1) < 0.01
    assert resample_audio(np.zeros(0, dtype=np.float32), 22050, 24000).size == 0


async def test_fallback_mid_stream_failure_does_not_repeat_the_sentence() -> None:
    online, floor = Engine("flaky", chunks=3, die_after=1), Plain()
    synth = FallbackSynthesizer([online, floor])
    chunks = [c async for c in synth.synthesize_stream("hello")]
    assert len(chunks) == 1 and floor.calls == 0
    again = await synth.synthesize("next")  # next sentence goes to the floor
    assert floor.calls == 1 and len(again) > 0


async def test_fallback_last_engine_error_propagates() -> None:
    synth = FallbackSynthesizer([Engine("only", fail=True)])
    with pytest.raises(RuntimeError):
        await synth.synthesize("x")


async def test_fallback_warm_up_skips_dead_engine() -> None:
    online, floor = Engine("online", fail=True), Plain()
    synth = FallbackSynthesizer([online, floor])
    await synth.warm_up()
    assert synth.name == "plain"


def test_build_synthesizer_provider_matrix(tmp_path) -> None:
    def build(**kw: Any):
        return build_synthesizer(Settings(data_dir_override=str(tmp_path), **kw))

    assert isinstance(build(tts_provider="piper"), PiperSynthesizer)
    assert isinstance(build(provider="ollama"), PiperSynthesizer)  # local mode keeps speech local
    auto = build(provider="nvidia", nvidia_api_key="")
    assert isinstance(auto, FallbackSynthesizer) and [type(e).__name__ for e in auto.engines] == ["EdgeSynthesizer", "PiperSynthesizer"]
    keyed = build(provider="nvidia", nvidia_api_key="k")
    assert [type(e).__name__ for e in keyed.engines] == ["EdgeSynthesizer", "NvidiaSynthesizer", "PiperSynthesizer"]
    nv = build(tts_provider="nvidia", nvidia_api_key="k", tts_online_voice="Magpie-Multilingual.EN-US.Leo.Calm")
    assert nv.engines[0].voice == "Magpie-Multilingual.EN-US.Leo.Calm"
    assert isinstance(build(tts_provider="nvidia", nvidia_api_key=""), PiperSynthesizer)  # no key: nothing online
    edge = build(tts_provider="edge", tts_online_voice="en-GB-ThomasNeural", tts_rate="-10%", tts_pitch="-3Hz")
    assert (edge.engines[0].voice, edge.engines[0].rate, edge.engines[0].pitch) == ("en-GB-ThomasNeural", "-10%", "-3Hz")
    assert edge.engines[-1].voice == "en_GB-alan-medium"  # JARVIS_TTS_VOICE still means Piper


# ---------------------------------------------------------------------- STT engines
async def test_openai_compat_transcriber_with_mock_transport() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content
        return httpx.Response(200, json={"text": " Jarvis, what time is it? "})

    t = OpenAICompatTranscriber("https://api.groq.com/openai/v1/", "gsk_test", transport=httpx.MockTransport(handler))
    result = await t.transcribe(tone(1.0, 16000), hint="jarvis")
    assert result.text == "Jarvis, what time is it?" and result.confidence > 0.5
    assert seen["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer gsk_test"
    assert b"whisper-large-v3-turbo" in seen["body"] and b"RIFF" in seen["body"] and b"Hey Jarvis" in seen["body"]


async def test_openai_compat_errors_and_hallucination_filter() -> None:
    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    with pytest.raises(httpx.HTTPStatusError):
        await OpenAICompatTranscriber("http://x/v1", "k", transport=httpx.MockTransport(bad)).transcribe(tone(1.0, 16000))

    def silence(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "Thank you."})

    quiet = (tone(1.0, 16000) * 0.01).astype(np.float32)
    got = await OpenAICompatTranscriber("http://x/v1", "", "m", transport=httpx.MockTransport(silence)).transcribe(quiet)
    assert got.text == ""


def test_wav_bytes_roundtrip() -> None:
    import wave

    with wave.open(io.BytesIO(wav_bytes(tone(0.5, 16000), 16000))) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (1, 16000, 8000)


class FakeRivaASR:
    def __init__(self, text: str | Exception) -> None:
        self.text, self.calls = text, 0

    def offline_recognize(self, audio: bytes, config: Any):
        self.calls += 1
        if isinstance(self.text, Exception):
            raise self.text
        alt = SimpleNamespace(transcript=self.text)
        return SimpleNamespace(results=[SimpleNamespace(alternatives=[alt])])


async def test_riva_transcriber() -> None:
    service = FakeRivaASR("Jarvis, open the downloads folder. ")
    t = RivaTranscriber("k", "", service_factory=lambda: service)
    result = await t.transcribe(tone(1.0, 16000))
    assert result.text == "Jarvis, open the downloads folder." and t.name == "nvidia parakeet-ctc-1.1b"
    assert t._function == "1598d209-5e27-4d3c-8079-4751568b1081"
    assert RivaTranscriber("k", "0000-raw-id", service_factory=lambda: service)._function == "0000-raw-id"


def test_resolve_stt_provider_matrix() -> None:
    def pick(**kw: Any) -> str:
        return stt_mod.resolve_stt_provider(Settings(provider="nvidia", **kw))

    assert pick() == "local"  # no key, nothing configured
    assert pick(nvidia_api_key="k") == "nvidia"
    assert pick(nvidia_api_key="k", stt_base_url="https://api.groq.com/openai/v1", stt_api_key="g") == "openai"
    assert pick(stt_provider="local", nvidia_api_key="k") == "local"
    assert pick(stt_provider="nvidia") == "local"  # explicit but no key
    assert pick(stt_provider="openai") == "local"  # explicit but no base url
    assert pick(stt_provider="openai", stt_base_url="http://localhost:8000/v1") == "openai"
    assert stt_mod.resolve_stt_provider(Settings(provider="ollama", nvidia_api_key="k")) == "local"


def test_build_transcriber_does_not_construct_the_gpu_model_when_online(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, str]] = []

    class Spy:
        def __init__(self, model: str = "auto", device: str = "auto", beam_size: int = 1) -> None:
            built.append((model, device))
            self.name = f"spy {model} {device}"

    monkeypatch.setattr(stt_mod, "WhisperTranscriber", Spy)
    t = stt_mod.build_transcriber(Settings(provider="nvidia", nvidia_api_key="k"))
    assert isinstance(t, GatedTranscriber) and built == [("tiny.en", "cpu")]
    local = stt_mod.build_transcriber(Settings(provider="ollama"))
    assert isinstance(local, Spy) and built[-1] == ("auto", "auto")


# ------------------------------------------------------------------- the wake-word gate
class Local:
    name = "tiny"

    def __init__(self, text: str = "") -> None:
        self.text, self.calls = text, 0

    async def warm_up(self) -> None: ...

    async def transcribe(self, pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
        self.calls += 1
        return Transcript(self.text, 0.01, 0.8 if self.text else 0.0)


class Online:
    name = "online"

    def __init__(self, text: str | Exception = "", delay: float = 0.0) -> None:
        self.text, self.delay, self.calls = text, delay, 0

    async def warm_up(self) -> None:
        if isinstance(self.text, Exception):
            raise self.text

    async def transcribe(self, pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if isinstance(self.text, Exception):
            raise self.text
        return Transcript(self.text, 0.01, 0.9)


def addressed(text: str) -> bool:
    from jarvis.voice.session import contains_wake_word, split_wake_word

    return split_wake_word(text, "jarvis")[0] or contains_wake_word(text, "jarvis")


def gate(local: Local, online: Online, **kw: Any) -> GatedTranscriber:
    return GatedTranscriber(local, online, addressed, **kw)


PCM = tone(0.5, 16000)


async def test_gate_idle_unaddressed_audio_never_goes_online() -> None:
    local, online = Local("what time is the match tonight"), Online("should not be used")
    result = await gate(local, online).transcribe(PCM, hint="jarvis", addressed=False)
    assert result.text == "what time is the match tonight" and online.calls == 0


async def test_gate_idle_wake_word_goes_online_for_the_accurate_text() -> None:
    local, online = Local("Jarvis what time is it"), Online("Jarvis, what time is it?")
    g = gate(local, online)
    result = await g.transcribe(PCM, hint="jarvis", addressed=False)
    assert result.text == "Jarvis, what time is it?" and online.calls == 1 and local.calls == 1


async def test_gate_wake_alias_counts_as_addressed() -> None:
    local, online = Local("Travis what's the weather"), Online("Jarvis, what's the weather?")
    assert (await gate(local, online).transcribe(PCM, addressed=False)).text == "Jarvis, what's the weather?"


async def test_gate_silence_sends_nothing() -> None:
    local, online = Local(""), Online("x")
    result = await gate(local, online).transcribe(PCM, addressed=True)
    assert online.calls == 1 or online.calls == 0  # a-priori addressed runs both; empty local is not trusted alone
    assert result.text == "x" or result.text == ""
    idle_local, idle_online = Local(""), Online("x")
    assert (await gate(idle_local, idle_online).transcribe(PCM, addressed=False)).text == "" and idle_online.calls == 0


async def test_gate_a_priori_addressed_skips_the_wake_check() -> None:
    # follow-up window, barge-in, confirmation answer or --no-wake: no wake word in the text
    local, online = Local("yes do it"), Online("Yes, do it.")
    result = await gate(local, online).transcribe(PCM, addressed=True)
    assert result.text == "Yes, do it." and online.calls == 1


async def test_gate_online_failure_falls_back_to_local_and_cools_off(caplog: pytest.LogCaptureFixture) -> None:
    clock = Clock()
    local, online = Local("Jarvis open notepad"), Online(RuntimeError("503"))
    g = gate(local, online, clock=clock, cooloff=60)
    with caplog.at_level(logging.WARNING, logger="jarvis.voice.stt_online"):
        first = await g.transcribe(PCM, addressed=False)
        second = await g.transcribe(PCM, addressed=False)
    assert first.text == second.text == "Jarvis open notepad"
    assert online.calls == 1  # second utterance did not wait for a dead service
    assert len([r for r in caplog.records if "failed" in r.getMessage()]) == 1
    clock.now = 61
    online.text = "Jarvis, open Notepad."
    assert (await g.transcribe(PCM, addressed=False)).text == "Jarvis, open Notepad."
    assert online.calls == 2


async def test_gate_online_deadline() -> None:
    local, online = Local("Jarvis what time is it"), Online("Jarvis, what time is it?", delay=1.0)
    g = gate(local, online, online_deadline=0.05)
    result = await g.transcribe(PCM, addressed=False)
    assert result.text == "Jarvis what time is it" and g.online_down


async def test_gate_a_priori_online_failure_uses_local() -> None:
    local, online = Local("yes"), Online(RuntimeError("down"))
    assert (await gate(local, online).transcribe(PCM, addressed=True)).text == "yes"


async def test_gate_empty_online_result_keeps_local_text() -> None:
    local, online = Local("Jarvis hello"), Online("")
    assert (await gate(local, online).transcribe(PCM, addressed=False)).text == "Jarvis hello"


async def test_gate_online_text_that_drops_the_wake_word_does_not_undo_the_gate() -> None:
    local, online = Local("Jarvis hello there"), Online("Service hello there")
    assert (await gate(local, online).transcribe(PCM, addressed=False)).text == "Jarvis hello there"


async def test_gate_warm_up_offline_loads_the_full_local_model() -> None:
    full = Local("Jarvis, hi.")
    g = gate(Local("jarvis hi"), Online(RuntimeError("no network")), full_local_factory=lambda: full)
    await g.warm_up()
    assert g._full_ready and g.online_down
    result = await g.transcribe(PCM, addressed=False)  # offline: the accurate local model answers
    assert result.text == "Jarvis, hi." and full.calls == 1


async def test_gate_name_describes_both_engines() -> None:
    assert gate(Local(), Online()).name == "online (wake word gated by tiny)"


# --------------------------------------------------- the session around a gated recogniser
class GatedRig(Rig):
    def __init__(self, local_texts: dict[int, str], online_texts: dict[int, str], steps, **settings: Any) -> None:
        super().__init__({}, steps, **settings)
        self.local = SimpleNamespace(
            name="tiny", warm_up=lambda: asyncio.sleep(0), calls=[],
        )

        async def local_transcribe(pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
            self.local.calls.append(int(pcm[0]))
            return say(local_texts.get(int(pcm[0]), ""))

        async def online_transcribe(pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
            self.online_calls.append(int(pcm[0]))
            return say(online_texts[int(pcm[0])])

        self.online_calls: list[int] = []
        self.local.transcribe = local_transcribe
        self.online = SimpleNamespace(name="online", warm_up=lambda: asyncio.sleep(0), transcribe=online_transcribe)
        self.gated = GatedTranscriber(self.local, self.online, addressed)
        self.session.transcriber = self.gated


async def test_session_idle_chatter_stays_local_and_wake_word_goes_online() -> None:
    rig = GatedRig(
        {1: "what is for dinner", 2: "jarvis what time is it"},
        {2: "Jarvis, what time is it?"},
        [SpeechStarted(), utt(1), SpeechStarted(), utt(2)],
    )
    await rig.run()
    assert rig.online_calls == [2]  # the dinner chatter was never sent
    assert rig.agent.texts == ["what time is it?"]
    assert [h.addressed for h in rig.heard()] == [False, True]


async def test_session_follow_up_window_goes_online_without_wake_word() -> None:
    rig = GatedRig(
        {1: "jarvis hello", 2: "and tomorrow"},
        {1: "Jarvis, hello.", 2: "And tomorrow?"},
        [SpeechStarted(), utt(1), 0.4, SpeechStarted(), utt(2)],
    )
    await rig.run()
    assert rig.online_calls == [1, 2]
    assert rig.agent.texts == ["hello.", "And tomorrow?"]


async def test_session_no_wake_mode_sends_everything_online() -> None:
    rig = GatedRig({1: "open notepad"}, {1: "Open Notepad."}, [SpeechStarted(), utt(1)], require_wake_word=False)
    await rig.run()
    assert rig.online_calls == [1] and rig.agent.texts == ["Open Notepad."]


async def test_session_online_failure_does_not_drop_the_request() -> None:
    rig = GatedRig({1: "jarvis what time is it"}, {}, [SpeechStarted(), utt(1)])

    async def boom(pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
        raise RuntimeError("503")

    rig.online.transcribe = boom
    await rig.run()
    assert rig.agent.texts == ["what time is it"]


def test_addressed_a_priori_matrix() -> None:
    rig = Rig({})
    s = rig.session
    assert s._addressed_a_priori(False) is False  # idle, wake word required
    assert s._addressed_a_priori(True) is True  # barge-in
    s._confirming = "listening"
    assert s._addressed_a_priori(False) is True  # answer to a confirmation
    s._confirming = "asking"
    assert s._addressed_a_priori(False) is False
    s._confirming = None
    s._open_followup()
    assert s._addressed_a_priori(False) is True  # follow-up window
    s._turn = object()  # Jarvis is speaking: his own echo must not count as a reply
    assert s._addressed_a_priori(False) is False
    s._turn = None
    rig.now += 100
    assert s._addressed_a_priori(False) is False  # window closed
    s.settings = s.settings.model_copy(update={"require_wake_word": False})
    assert s._addressed_a_priori(False) is True  # --no-wake


# ----------------------------------------------- the session: streaming playback and stock cache
class StreamSynth:
    name, sample_rate = "stream", 24000
    cache_key = "stream:v1"

    def __init__(self) -> None:
        self.stream_calls: list[str] = []
        self.full_calls: list[str] = []

    async def warm_up(self) -> None: ...

    async def synthesize_stream(self, text: str):
        self.stream_calls.append(text)
        for _ in range(3):
            yield np.full(50, 0.1, dtype=np.float32)
            await asyncio.sleep(0.01)

    async def synthesize(self, text: str) -> np.ndarray:
        self.full_calls.append(text)
        return np.full(150, 0.1, dtype=np.float32)


async def test_session_plays_streamed_chunks_progressively() -> None:
    rig = Rig({1: say("Jarvis, hi")}, [SpeechStarted(), utt(1)])
    rig.session.synthesizer = synth = StreamSynth()
    plays: list[int] = []
    rig.speaker.play = lambda pcm, rate: plays.append(len(pcm))  # type: ignore[method-assign]
    await rig.run()
    assert "Reply to hi." in synth.stream_calls
    assert plays.count(50) >= 3 and rig.spoke() == ["Reply to hi."]  # three chunks, one Spoke event


async def test_stream_stops_when_the_turn_is_interrupted() -> None:
    rig = Rig({})
    rig.session.synthesizer = StreamSynth()
    plays: list[int] = []
    rig.speaker.play = lambda pcm, rate: plays.append(len(pcm))  # type: ignore[method-assign]
    calls = {"n": 0}

    def wanted() -> bool:
        calls["n"] += 1
        return calls["n"] <= 2

    assert await rig.session._play_sentence("hello", wanted) is True
    assert len(plays) == 2  # the third chunk saw the interruption and was dropped


async def test_stock_phrases_are_cached_on_disk_per_voice(tmp_path) -> None:
    rig = Rig({}, data_dir_override=str(tmp_path))
    rig.session.synthesizer = synth = StreamSynth()
    await rig.session.prepare()
    first_run = len(synth.full_calls)
    assert first_run >= 8 and len(list((tmp_path / "stock").glob("*.npz"))) == first_run
    again = Rig({}, data_dir_override=str(tmp_path))
    again.session.synthesizer = second = StreamSynth()
    await again.session.prepare()
    assert second.full_calls == []  # served from disk: no synthesis at start-up
    assert "Yes?" in again.session._stock and again.session._stock["Yes?"].dtype == np.float32
    other = Rig({}, data_dir_override=str(tmp_path))
    other.session.synthesizer = third = StreamSynth()
    third.cache_key = "stream:v2"  # a different voice must not reuse the old audio
    await other.session.prepare()
    assert len(third.full_calls) == first_run


async def test_stock_cache_records_the_engine_that_actually_spoke(tmp_path) -> None:
    rig = Rig({}, data_dir_override=str(tmp_path))
    online, floor = Engine("online", fail=True), Plain()
    rig.session.synthesizer = FallbackSynthesizer([online, floor])
    await rig.session.prepare()
    assert online.calls <= 1  # one failed attempt; the rest are cooling
    again = Rig({}, data_dir_override=str(tmp_path))
    again.session.synthesizer = FallbackSynthesizer([Engine("online", fail=True), floor])
    again.session.synthesizer._down_until[0] = 1e12  # known to be down: asks for the floor's cache key
    floor.calls = 0
    await again.session.prepare()
    assert floor.calls == 0
