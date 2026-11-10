"""Settings. One object, loaded once, passed explicitly.

Paths never depend on the current working directory: the data directory is
``%LOCALAPPDATA%\\Jarvis`` unless JARVIS_DATA_DIR overrides it, so memory
survives launching from any folder.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from platformdirs import user_data_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
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

    data_dir_override: str = Field("", alias="JARVIS_DATA_DIR")

    @property
    def data_dir(self) -> Path:
        path = (
            Path(self.data_dir_override).expanduser()
            if self.data_dir_override
            else Path(user_data_dir("Jarvis", appauthor=False))
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jarvis.db"

    @property
    def log_dir(self) -> Path:
        path = self.data_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
