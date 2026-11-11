"""Online speech synthesis with NVIDIA's hosted Magpie TTS (Riva over gRPC, existing NVIDIA key).

Verified for real: it answers with the free key, streams audio in about 0.25 s, but its
English voices are American only (no British voice), so it is the second choice after
Edge for the Jarvis character. The text of each sentence is sent to NVIDIA.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator, Callable

import numpy as np

from jarvis.voice.nvidia_riva import DEFAULT_TTS_MODEL, TTS_FUNCTIONS, function_id, riva_auth
from jarvis.voice.tts import clean_for_speech

DEFAULT_VOICE = "Magpie-Multilingual.EN-US.Ray.Calm"
NVIDIA_SAMPLE_RATE = 24000


class NvidiaSynthesizer:
    sample_rate = NVIDIA_SAMPLE_RATE

    def __init__(
        self,
        api_key: str,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_TTS_MODEL,
        *,
        service_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.name = f"nvidia {voice}"
        self.cache_key = f"nvidia:{voice}"
        self._factory = service_factory or self._default_service
        self._service: Any = None

    def _default_service(self):
        import riva.client

        fid = function_id(self.model, TTS_FUNCTIONS, DEFAULT_TTS_MODEL)
        return riva.client.SpeechSynthesisService(riva_auth(self.api_key, fid))

    def _language(self) -> str:
        parts = self.voice.split(".")
        if len(parts) > 2 and "-" in parts[1]:
            language, region = parts[1].split("-", 1)
            return f"{language.lower()}-{region.upper()}"
        return "en-US"

    async def warm_up(self) -> None:
        await self.synthesize("Ready.")

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        text = clean_for_speech(text)
        if not text:
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def work() -> None:
            try:
                if self._service is None:
                    self._service = self._factory()
                responses = self._service.synthesize_online(
                    text, self.voice, self._language(), sample_rate_hz=self.sample_rate,
                )
                try:
                    for response in responses:
                        if stop.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, bytes(response.audio))
                finally:
                    close = getattr(responses, "close", None)
                    if close is not None:
                        close()
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        future = loop.run_in_executor(None, work)
        got_audio = False
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                audio = np.frombuffer(item[: len(item) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
                if len(audio):
                    got_audio = True
                    yield audio
            if not got_audio:
                raise RuntimeError("the speech service returned no audio")
        finally:
            stop.set()  # the worker thread stops after the chunk it is waiting for
            del future

    async def synthesize(self, text: str) -> np.ndarray:
        parts = [part async for part in self.synthesize_stream(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
