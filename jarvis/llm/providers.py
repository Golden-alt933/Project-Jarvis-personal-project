"""Provider profiles and client factories."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.config import Settings
from jarvis.llm.ollama import OllamaClient
from jarvis.llm.openai_compat import OpenAICompatClient
from jarvis.llm.types import LLMClient

# Default model ids. Only models verified to work with tool calling belong here.
# Verified by benchmark on 2026-11-10: the only free model on the NVIDIA key that passed
# every tool-use probe (12/12) and streams cleanly. nemotron-3-super-120b (Friday's
# default) returns 404 on this account.
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
# TODO: verified by benchmark on <date>
GROQ_DEFAULT_MODEL = ""
# TODO: verified by benchmark on <date>
OPENROUTER_DEFAULT_MODEL = ""
# TODO: verified by benchmark on <date>
GEMINI_DEFAULT_MODEL = ""

# TODO: confirm against the model cards; conservative until measured.
DEFAULT_CONTEXT_WINDOW = 128_000


class ConfigError(Exception):
    """Settings are missing or inconsistent; the message says what to set."""


@dataclass(frozen=True)
class ProviderProfile:
    base_url: str
    key_setting_attr: str
    default_model: str
    default_fast_model: str
    context_window: int


PROFILES: dict[str, ProviderProfile] = {
    "nvidia": ProviderProfile(
        "https://integrate.api.nvidia.com/v1", "nvidia_api_key",
        NVIDIA_DEFAULT_MODEL, "", DEFAULT_CONTEXT_WINDOW,
    ),
    "groq": ProviderProfile(
        "https://api.groq.com/openai/v1", "groq_api_key",
        GROQ_DEFAULT_MODEL, "", DEFAULT_CONTEXT_WINDOW,
    ),
    "openrouter": ProviderProfile(
        "https://openrouter.ai/api/v1", "openrouter_api_key",
        OPENROUTER_DEFAULT_MODEL, "", DEFAULT_CONTEXT_WINDOW,
    ),
    "gemini": ProviderProfile(
        "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini_api_key",
        GEMINI_DEFAULT_MODEL, "", DEFAULT_CONTEXT_WINDOW,
    ),
    # base_url is filled from settings.custom_base_url at build time.
    "custom": ProviderProfile("", "custom_api_key", "", "", DEFAULT_CONTEXT_WINDOW),
}

_KEY_ENV = {
    "nvidia_api_key": "NVIDIA_API_KEY",
    "groq_api_key": "GROQ_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
    "custom_api_key": "CUSTOM_API_KEY",
}


def _ollama(settings: Settings, model: str | None = None) -> OllamaClient:
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=model or settings.ollama_model,
        num_ctx=settings.ollama_num_ctx,
        request_timeout=settings.request_timeout,
    )


def _cloud(settings: Settings, model_override: str = "") -> OpenAICompatClient:
    provider = settings.provider.lower()
    profile = PROFILES.get(provider)
    if profile is None:
        raise ConfigError(
            f"Unknown JARVIS_PROVIDER '{settings.provider}'. "
            f"Choose one of: {', '.join(sorted([*PROFILES, 'ollama']))}."
        )
    base_url = profile.base_url
    if provider == "custom":
        base_url = settings.custom_base_url
        if not base_url:
            raise ConfigError("JARVIS_PROVIDER=custom needs CUSTOM_BASE_URL to be set.")
    api_key = getattr(settings, profile.key_setting_attr)
    if not api_key and provider != "custom":
        raise ConfigError(
            f"No API key for provider '{provider}': set {_KEY_ENV[profile.key_setting_attr]} in .env "
            "(or use JARVIS_PROVIDER=ollama to run fully offline)."
        )
    model = model_override or settings.model or profile.default_model
    if not model:
        raise ConfigError(
            f"Provider '{provider}' has no default model; set JARVIS_MODEL in .env."
        )
    return OpenAICompatClient(
        name=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        context_window=profile.context_window,
        request_timeout=settings.request_timeout,
    )


def build_clients(settings: Settings) -> tuple[LLMClient, LLMClient | None]:
    """Return (primary, local fallback or None)."""
    if settings.provider.lower() == "ollama":
        return _ollama(settings), None
    primary = _cloud(settings)
    fallback = _ollama(settings) if settings.local_fallback else None
    return primary, fallback


def build_fast_client(settings: Settings) -> LLMClient:
    """Client for cheap side jobs: the fast model if configured, else the primary model."""
    if settings.provider.lower() == "ollama":
        return _ollama(settings, settings.fast_model or None)
    profile = PROFILES.get(settings.provider.lower())
    fast = settings.fast_model or (profile.default_fast_model if profile else "")
    return _cloud(settings, fast)


def describe(client: LLMClient) -> str:
    return f"{client.name} · {client.model}"
