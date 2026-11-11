"""Settings. One object, loaded once, passed explicitly.

Lookup order, first match wins: environment variables, then ``<repo>/.env`` when running
from a checkout, then the per-user config file (``%APPDATA%\\Jarvis\\.env`` on Windows,
``${XDG_CONFIG_HOME:-~/.config}/jarvis/.env`` on Linux; ``jarvis --setup`` writes it).

Paths never depend on the current working directory: the data directory is the per-user
data folder (``%LOCALAPPDATA%\\Jarvis``, ``~/.local/share/jarvis``) unless JARVIS_DATA_DIR
overrides it, so memory survives launching from any folder.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from jarvis import platform as plat

REPO_ROOT = Path(__file__).resolve().parent.parent


def app_dir_name() -> str:
    return plat.current().app_dir_name


def is_checkout() -> bool:
    """True when running from a source checkout (as opposed to an installed wheel)."""
    return (REPO_ROOT / "pyproject.toml").is_file()


def repo_env_file() -> Path | None:
    """``<repo>/.env`` when running from a checkout, else None."""
    return REPO_ROOT / ".env" if is_checkout() else None


def user_config_file() -> Path:
    """The per-user config file. JARVIS_CONFIG_FILE overrides the location (used by tests)."""
    override = os.environ.get("JARVIS_CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    return Path(user_config_dir(app_dir_name(), appauthor=False, roaming=True)) / ".env"


def config_files() -> list[Path]:
    """Config files in priority order (highest first); only those that exist."""
    candidates = [repo_env_file(), user_config_file()]
    return [p for p in candidates if p is not None and p.is_file()]


def _env_file_chain() -> tuple[Path, ...]:
    """pydantic-settings applies later files over earlier ones: user file, then repo .env."""
    chain = [user_config_file()]
    repo = repo_env_file()
    if repo is not None:
        chain.append(repo)
    return tuple(chain)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file_chain(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Brain
    provider: str = Field("nvidia", alias="JARVIS_PROVIDER")
    model: str = Field("", alias="JARVIS_MODEL")
    fast_model: str = Field("", alias="JARVIS_FAST_MODEL")
    temperature: float = Field(0.3, alias="JARVIS_TEMPERATURE")
    max_output_tokens: int = Field(4096, alias="JARVIS_MAX_OUTPUT_TOKENS")
    request_timeout: float = Field(90.0, alias="JARVIS_REQUEST_TIMEOUT")
    # A streaming request that produces no bytes for this long is abandoned and retried
    # once (free-tier requests sometimes queue forever). 0 disables.
    first_byte_timeout: float = Field(25.0, alias="JARVIS_FIRST_BYTE_TIMEOUT")
    # Optional faster model for the voice/HUD frontends (same provider). Empty = same as text.
    voice_model: str = Field("", alias="JARVIS_VOICE_MODEL")

    nvidia_api_key: str = Field("", alias="NVIDIA_API_KEY")
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field("", alias="OPENROUTER_API_KEY")
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    custom_base_url: str = Field("", alias="CUSTOM_BASE_URL")
    custom_api_key: str = Field("", alias="CUSTOM_API_KEY")

    # Local
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("qwen3:4b-instruct", alias="OLLAMA_MODEL")
    ollama_num_ctx: int = Field(16384, alias="OLLAMA_NUM_CTX")
    local_fallback: bool = Field(True, alias="JARVIS_LOCAL_FALLBACK")
    # Start `ollama serve` (loopback only, once per process) when it is not running.
    ollama_autostart: bool = Field(True, alias="JARVIS_OLLAMA_AUTOSTART")

    # Agent
    user_name: str = Field("", alias="JARVIS_USER_NAME")
    max_steps: int = Field(25, alias="JARVIS_MAX_STEPS")
    turn_time_budget: float = Field(600.0, alias="JARVIS_TURN_TIME_BUDGET")
    # History is compacted (older turns summarised) above this many tokens (estimated).
    history_token_budget: int = Field(24000, alias="JARVIS_HISTORY_TOKEN_BUDGET")
    tool_result_max_chars: int = Field(12000, alias="JARVIS_TOOL_RESULT_MAX_CHARS")
    # Tool groups offered to a small local model (a 4B model degrades with a huge menu).
    local_tool_groups: list[str] = Field(
        default_factory=lambda: ["core", "files", "web", "memory"],
        alias="JARVIS_LOCAL_TOOL_GROUPS",
    )

    # Voice
    wake_word: str = Field("jarvis", alias="JARVIS_WAKE_WORD")
    # False = every utterance is addressed to Jarvis (no wake word needed).
    require_wake_word: bool = Field(True, alias="JARVIS_REQUIRE_WAKE_WORD")
    # After Jarvis replies, the user can answer without the wake word for this long.
    followup_seconds: float = Field(10.0, alias="JARVIS_FOLLOWUP_SECONDS")
    end_silence_ms: int = Field(650, alias="JARVIS_END_SILENCE_MS")
    # vad = speaking over Jarvis interrupts it; wake = only "<wake word> ..." interrupts
    # (use on loud speakers without headphones); off = never.
    barge_in: str = Field("vad", alias="JARVIS_BARGE_IN")
    stt_model: str = Field("auto", alias="JARVIS_STT_MODEL")  # auto | tiny.en | base.en | small.en | ...
    stt_device: str = Field("auto", alias="JARVIS_STT_DEVICE")  # auto | cuda | cpu
    tts_voice: str = Field("en_GB-alan-medium", alias="JARVIS_TTS_VOICE")
    tts_speed: float = Field(1.0, alias="JARVIS_TTS_SPEED")
    input_device: str = Field("", alias="JARVIS_INPUT_DEVICE")  # empty = system default
    output_device: str = Field("", alias="JARVIS_OUTPUT_DEVICE")

    # Online speech (free services; the local engines stay as the offline fallback)
    # auto = Edge voice, then NVIDIA if a key is set, then Piper (local mode: Piper only)
    tts_provider: str = Field("auto", alias="JARVIS_TTS_PROVIDER")  # auto | edge | nvidia | piper
    tts_online_voice: str = Field("", alias="JARVIS_TTS_ONLINE_VOICE")  # empty = the provider's default
    tts_rate: str = Field("-5%", alias="JARVIS_TTS_RATE")  # Edge speaking rate, e.g. -10% or +5%
    tts_pitch: str = Field("+0Hz", alias="JARVIS_TTS_PITCH")  # Edge pitch, e.g. -3Hz
    # auto = hosted recogniser if a key is set (idle audio is still gated locally), else local
    stt_provider: str = Field("auto", alias="JARVIS_STT_PROVIDER")  # auto | local | nvidia | openai
    stt_base_url: str = Field("", alias="JARVIS_STT_BASE_URL")  # openai: e.g. https://api.groq.com/openai/v1
    stt_api_key: str = Field("", alias="JARVIS_STT_API_KEY")
    stt_online_model: str = Field("", alias="JARVIS_STT_ONLINE_MODEL")  # empty = the provider's default

    # Screen vision
    # auto = hosted vision model if one works on the configured key, else local Ollama;
    # local = images never leave this machine.
    vision_provider: str = Field("auto", alias="JARVIS_VISION_PROVIDER")  # auto | cloud | local
    vision_model: str = Field("", alias="JARVIS_VISION_MODEL")  # empty = provider default
    ollama_vision_model: str = Field("qwen2.5vl:3b", alias="OLLAMA_VISION_MODEL")

    # Chrome extension bridge (loopback only)
    browser_bridge_port: int = Field(8765, alias="JARVIS_BROWSER_PORT")

    data_dir_override: str = Field("", alias="JARVIS_DATA_DIR")

    @property
    def data_dir(self) -> Path:
        path = (
            Path(self.data_dir_override).expanduser()
            if self.data_dir_override
            else Path(user_data_dir(app_dir_name(), appauthor=False))
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jarvis.db"

    @property
    def voices_dir(self) -> Path:
        path = self.data_dir / "voices"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        path = self.data_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
