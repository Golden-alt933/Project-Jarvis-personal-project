"""Online speech recognition, used only for speech that is addressed to Jarvis.

    idle, wake word required:   audio -> local tiny.en (CPU) -> wake word? -> no: stays local
                                                                           -> yes: online recogniser
    follow-up window, barge-in, answer to a confirmation, --no-wake: audio -> online directly
                                                              (a local transcript runs alongside as the fallback)

``GatedTranscriber`` implements that rule. The online engines are ``RivaTranscriber`` (NVIDIA's
hosted Parakeet, verified with a real key) and ``OpenAICompatTranscriber`` (any
``/audio/transcriptions`` endpoint such as Groq's free Whisper; verified only against a
mocked transport). If the online call fails or is slower than ``online_deadline`` the local
transcript is used, and online is skipped for ``cooloff`` seconds.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from typing import Any, Callable

import httpx
import numpy as np

from jarvis.voice.stt import _Scored, _rms, filter_hallucination, initial_prompt
from jarvis.voice.types import SAMPLE_RATE, Transcript

log = logging.getLogger(__name__)

ONLINE_CONFIDENCE = 0.9  # hosted recognisers do not report one we can compare with Whisper's


def wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    data = (np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)
    return buffer.getvalue()


def _finish(text: str, pcm: np.ndarray, started: float) -> Transcript:
    text = filter_hallucination(_Scored(text.strip(), ONLINE_CONFIDENCE, False), _rms(pcm))
    return Transcript(text, time.perf_counter() - started, ONLINE_CONFIDENCE if text else 0.0)


# ---------------------------------------------------------------------------- engines


class OpenAICompatTranscriber:
    """POST {base_url}/audio/transcriptions (OpenAI, Groq, and compatible servers)."""

    def __init__(
        self, base_url: str, api_key: str = "", model: str = "", *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model or ("whisper-large-v3-turbo" if "groq" in self.base_url else "whisper-1")
        self.name = f"{self.model} @ {httpx.URL(self.base_url).host}"
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, timeout=10.0)
        return self._client

    async def warm_up(self) -> None:
        # No free probe exists for this API; the first real call proves it works.
        self._http()

    async def transcribe(self, pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = {"model": self.model, "language": "en", "response_format": "json", "temperature": "0"}
        prompt = initial_prompt(hint)
        if prompt:
            data["prompt"] = prompt
        response = await self._http().post(
            f"{self.base_url}/audio/transcriptions",
            headers=headers, data=data, files={"file": ("speech.wav", wav_bytes(pcm), "audio/wav")},
        )
        response.raise_for_status()
        return _finish(str(response.json().get("text", "")), pcm, started)


class RivaTranscriber:
    """NVIDIA hosted Riva ASR (Parakeet) over gRPC."""

    def __init__(self, api_key: str, model: str = "", *, service_factory: Callable[[], Any] | None = None) -> None:
        from jarvis.voice.nvidia_riva import ASR_FUNCTIONS, DEFAULT_ASR_MODEL, function_id

        self.api_key = api_key
        self.model = model or DEFAULT_ASR_MODEL
        self._function = function_id(self.model, ASR_FUNCTIONS, DEFAULT_ASR_MODEL)
        self.name = f"nvidia {self.model}"
        self._factory = service_factory or self._default_service
        self._service: Any = None

    def _default_service(self):
        import riva.client

        from jarvis.voice.nvidia_riva import riva_auth

        return riva.client.ASRService(riva_auth(self.api_key, self._function))

    def _config(self):
        import riva.client

        return riva.client.RecognitionConfig(
            language_code="en-US", max_alternatives=1, enable_automatic_punctuation=True,
            sample_rate_hertz=SAMPLE_RATE, audio_channel_count=1, encoding=riva.client.AudioEncoding.LINEAR_PCM,
        )

    def _recognize(self, pcm: np.ndarray) -> str:
        if self._service is None:
            self._service = self._factory()
        audio = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        response = self._service.offline_recognize(audio, self._config())
        return " ".join(r.alternatives[0].transcript.strip() for r in response.results if r.alternatives).strip()

    async def warm_up(self) -> None:
        # One short silent request: opens the channel and proves the key and function id work.
        await asyncio.to_thread(self._recognize, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    async def transcribe(self, pcm: np.ndarray, *, hint: str | None = None) -> Transcript:
        started = time.perf_counter()
        text = await asyncio.to_thread(self._recognize, np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1))
        return _finish(text, pcm, started)


# ------------------------------------------------------------------------------ the gate


class GatedTranscriber:
    """Local tiny model decides who the audio is for; only addressed audio goes online."""

    gated = True

    def __init__(
        self,
        gate,
        online,
        is_addressed: Callable[[str], bool],
        *,
        full_local_factory: Callable[[], Any] | None = None,
        online_deadline: float = 4.0,
        cooloff: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gate = gate
        self.online = online
        self.is_addressed = is_addressed
        self.online_deadline = online_deadline
        self.cooloff = cooloff
        self.clock = clock
        self._full_factory = full_local_factory
        self._full: Any = None
        self._full_ready = False
        self._full_task: asyncio.Task[None] | None = None
        self._down_until = 0.0
        self._was_down = False
        self.online_calls = 0  # how many utterances were sent online (tests, diagnostics)

    @property
    def name(self) -> str:
        return f"{self.online.name} (wake word gated by {self.gate.name})"

    @property
    def online_down(self) -> bool:
        return self.clock() < self._down_until

    # -- outage handling
    def _mark_down(self, reason: object) -> None:
        self._down_until = self.clock() + self.cooloff
        if not self._was_down:
            log.warning(
                "Online speech recognition (%s) failed (%s); using the local recogniser for %.0f s",
                self.online.name, reason or "timeout", self.cooloff,
            )
        self._was_down = True
        self._start_full_local()

    def _start_full_local(self) -> None:
        """While online is out, an accurate local model is loaded in the background."""
        if self._full_factory is None or self._full_task is not None:
            return

        async def load() -> None:
            try:
                self._full = await asyncio.to_thread(self._full_factory)
                await self._full.warm_up()
                self._full_ready = True
                log.info("Loaded %s for use while online recognition is down", self._full.name)
            except Exception:
                log.exception("could not load the full local recogniser")

        self._full_task = asyncio.ensure_future(load())

    async def warm_up(self) -> None:
        await self.gate.warm_up()
        try:
            await asyncio.wait_for(self.online.warm_up(), self.online_deadline)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._mark_down(str(exc) or type(exc).__name__)
            if self._full_task is not None:
                await self._full_task  # start-up without a connection: be as accurate as before

    # -- transcription
    async def _online(self, pcm: np.ndarray, hint: str | None) -> Transcript | None:
        self.online_calls += 1
        try:
            result = await asyncio.wait_for(self.online.transcribe(pcm, hint=hint), self.online_deadline)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._mark_down(str(exc) or type(exc).__name__)
            return None
        if self._was_down:
            log.info("Online speech recognition is back")
            self._was_down = False
        return result

    async def transcribe(self, pcm: np.ndarray, *, hint: str | None = None, addressed: bool = False) -> Transcript:
        use_online = not self.online_down
        if addressed and use_online:
            online_task = asyncio.ensure_future(self._online(pcm, hint))  # local runs alongside as the fallback
            local = await self.gate.transcribe(pcm, hint=hint)
            online = await online_task
            if online is not None and online.text:
                return online
            return local
        local = await self.gate.transcribe(pcm, hint=hint)
        if not local.text:
            return local  # silence or noise: nothing is sent anywhere
        if not (addressed or self.is_addressed(local.text)):
            return local  # not for Jarvis: this audio never leaves the machine
        if not use_online:
            if self._full_ready:
                better = await self._full.transcribe(pcm, hint=hint)
                return better if better.text else local
            return local
        online = await self._online(pcm, hint)
        if online is None or not online.text:
            return local
        if not self.is_addressed(online.text):
            return local  # the gate heard the wake word; do not let a different spelling undo that
        return online
