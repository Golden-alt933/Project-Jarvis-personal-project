"""Vision model client: hosted (OpenAI-compatible) or local (Ollama), with fallback.

Ported from Friday's vision/vision_client.py (Ollama only) and extended with a hosted
path. Privacy rule: ``vision_provider=local`` never sends an image off the machine,
not even as a fallback. A hosted failure may fall back to local (that only reduces
exposure).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from jarvis.llm.providers import PROFILES
from jarvis.vision.capture import downscale, jpeg_base64

if TYPE_CHECKING:
    from PIL.Image import Image

    from jarvis.config import Settings

log = logging.getLogger(__name__)

# Verified 2026-11-11 by probing the NVIDIA key with a generated invoice image: read the
# text and the red circle correctly in 2.7 s. Other models that answered correctly
# (gemma-4-31b-it 13 s, nemotron-3-nano-omni 5.6 s, nemotron-3.5-bolt-e4b 1.3 s) are
# alternatives; llama-3.2-90b-vision timed out and the main brain model rejects images.
NVIDIA_VISION_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_VISION_MODELS = {"nvidia": NVIDIA_VISION_MODEL}

CLOUD_MAX_EDGE = 1568
LOCAL_MAX_EDGE = 896  # a 3B VLM on a 4 GB GPU: more pixels is more VRAM and time, not more detail
# Reasoning-capable vision models spend tokens thinking before the answer.
CLOUD_MAX_TOKENS = 1500
LOCAL_NUM_PREDICT = 400
CLOUD_TIMEOUT = 40.0
LOCAL_TIMEOUT = 70.0  # both together stay inside the tool's 120 s budget
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
SYSTEM_PROMPT = "You describe screenshots and images accurately and concisely. Say when text is unreadable rather than guessing."


class VisionError(RuntimeError):
    """A vision call failed; the message says what to do about it."""


class VisionBusy(VisionError):
    """The hosted model timed out or was busy: the same call is worth one more try."""


@dataclass(frozen=True)
class VisionAnswer:
    text: str
    model: str
    path: str  # "local" or the provider name, e.g. "nvidia"

    def label(self) -> str:
        return f"{self.model} ({self.path})"


class VisionClient:
    def __init__(
        self,
        *,
        provider: str,
        mode: str = "auto",
        cloud_base_url: str = "",
        cloud_api_key: str = "",
        cloud_model: str = "",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "",
        http: httpx.AsyncClient | None = None,
        allow_local: bool = True,
    ) -> None:
        self.provider = provider
        self.mode = mode
        self.allow_local = allow_local  # False: never load a local model as a fallback
        self.cloud_base_url = cloud_base_url.rstrip("/")
        self.cloud_api_key = cloud_api_key
        self.cloud_model = cloud_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self._http = http  # injected in tests; otherwise one short-lived client per call

    # -- selection -------------------------------------------------------------
    @property
    def cloud_available(self) -> bool:
        return bool(self.cloud_base_url and self.cloud_api_key and self.cloud_model)

    def order(self) -> list[str]:
        """Paths to try, in order. Local-only mode never lists the hosted path."""
        if self.mode == "local":
            return ["local"]
        if self.mode != "cloud" and not self.cloud_available:
            return ["local"]
        return ["cloud", "local"] if self.allow_local else ["cloud"]

    def primary_path(self) -> str:
        return self.order()[0]

    # -- public API ------------------------------------------------------------
    async def describe(self, image: "Image", question: str) -> str:
        return (await self.ask(image, question)).text

    async def ask(self, image: "Image", question: str) -> VisionAnswer:
        errors: list[str] = []
        for path in self.order():
            try:
                if path == "cloud":
                    try:
                        return await self._ask_cloud(image, question)
                    except VisionBusy as exc:
                        # The hosted model's first call after a pause can time out and the next
                        # answers in seconds: try again before loading a model onto this machine.
                        log.warning("vision cloud path was busy (%s); trying it once more", exc)
                        return await self._ask_cloud(image, question)
                return await self._ask_local(image, question)
            except VisionError as exc:
                log.warning("vision %s path failed: %s", path, exc)
                errors.append(f"{path}: {exc}")
        raise VisionError("; ".join(errors))

    # -- hosted ----------------------------------------------------------------
    def cloud_payload(self, image: "Image", question: str) -> dict:
        uri = "data:image/jpeg;base64," + jpeg_base64(downscale(image, CLOUD_MAX_EDGE))
        return {
            "model": self.cloud_model,
            "max_tokens": CLOUD_MAX_TOKENS,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                },
            ],
        }

    async def _ask_cloud(self, image: "Image", question: str) -> VisionAnswer:
        if not self.cloud_available:
            raise VisionError(
                f"no hosted vision model for provider '{self.provider}'; "
                "set JARVIS_VISION_MODEL and the provider's API key"
            )
        data = await self._post(
            f"{self.cloud_base_url}/chat/completions",
            self.cloud_payload(image, question),
            {"Authorization": f"Bearer {self.cloud_api_key}"},
            CLOUD_TIMEOUT,
            self.cloud_model,
        )
        try:
            text = (data["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise VisionError(f"{self.cloud_model} returned an unexpected response") from exc
        if not text:
            raise VisionError(f"{self.cloud_model} returned an empty answer")
        return VisionAnswer(text, self.cloud_model, self.provider)

    # -- local -----------------------------------------------------------------
    def local_payload(self, image: "Image", question: str) -> dict:
        return {
            "model": self.ollama_model,
            "stream": False,
            "keep_alive": "5m",
            "options": {"num_predict": LOCAL_NUM_PREDICT, "temperature": 0.2},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": question,
                    "images": [jpeg_base64(downscale(image, LOCAL_MAX_EDGE))],
                },
            ],
        }

    async def _ask_local(self, image: "Image", question: str) -> VisionAnswer:
        host = urlparse(self.ollama_base_url).hostname or ""
        if host not in LOOPBACK_HOSTS:
            raise VisionError(f"refusing to send an image to non-loopback host '{host}'")
        if not self.ollama_model:
            raise VisionError("no local vision model configured (OLLAMA_VISION_MODEL)")
        data = await self._post(
            f"{self.ollama_base_url}/api/chat",
            self.local_payload(image, question),
            {},
            LOCAL_TIMEOUT,
            self.ollama_model,
            hint=f"is Ollama running and `ollama pull {self.ollama_model}` done?",
        )
        text = ((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise VisionError(f"{self.ollama_model} returned an empty answer")
        return VisionAnswer(text, self.ollama_model, "local")

    # -- http ------------------------------------------------------------------
    async def _post(
        self, url: str, payload: dict, headers: dict, timeout: float, model: str, hint: str = ""
    ) -> dict:
        """POST JSON; every failure becomes a VisionError (never includes headers/keys)."""
        try:
            if self._http is not None:
                response = await self._http.post(url, json=payload, headers=headers, timeout=timeout)
            else:
                async with httpx.AsyncClient(trust_env=False) as http:
                    response = await http.post(url, json=payload, headers=headers, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise VisionBusy(f"{model} timed out after {timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise VisionError(f"{model} unreachable ({type(exc).__name__}){'; ' + hint if hint else ''}") from exc
        if response.status_code != 200:
            reason = {404: "not available", 429: "rate limited", 400: "rejected the request (no image support?)"}.get(
                response.status_code, f"HTTP {response.status_code}"
            )
            kind = VisionBusy if response.status_code in (429, 500, 502, 503, 504) else VisionError
            raise kind(f"{model} {reason} (HTTP {response.status_code}){'; ' + hint if hint and response.status_code == 404 else ''}")
        try:
            return response.json()
        except ValueError as exc:
            raise VisionError(f"{model} returned invalid JSON") from exc


def build_vision_client(settings: "Settings", http: httpx.AsyncClient | None = None) -> VisionClient:
    provider = settings.provider.lower()
    profile = PROFILES.get(provider)
    base_url = api_key = model = ""
    if profile is not None:
        base_url = settings.custom_base_url if provider == "custom" else profile.base_url
        api_key = getattr(settings, profile.key_setting_attr, "")
        model = settings.vision_model or DEFAULT_VISION_MODELS.get(provider, "")
    return VisionClient(
        provider=provider,
        mode=settings.vision_provider.lower(),
        cloud_base_url=base_url,
        cloud_api_key=api_key,
        cloud_model=model,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_vision_model,
        http=http,
        allow_local=bool(getattr(settings, "local_fallback", True)),
    )
