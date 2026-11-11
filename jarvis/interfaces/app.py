"""Wiring: settings -> store, registry, clients, agent. Frontends call build_app()."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

from jarvis.agent.loop import Agent
from jarvis.config import Settings
from jarvis.llm.providers import build_clients, build_fast_client
from jarvis.llm.types import LLMClient
from jarvis.memory.store import MemoryStore
from jarvis.tools.registry import ToolRegistry

_LOG_HANDLER_NAME = "jarvis-file"


@dataclass
class App:
    settings: Settings
    store: MemoryStore
    registry: ToolRegistry
    primary: LLMClient
    fallback: LLMClient | None
    fast: LLMClient
    agent: Agent
    # Latest conversation that existed before this run started (see resume()).
    previous_conversation: str | None = None

    def resume(self) -> bool:
        """Continue the most recent conversation if it is recent enough.

        Agent.__init__ already starts a fresh conversation, which would always be
        the "latest" one, so the candidate was captured before the agent was built.
        """
        previous = self.previous_conversation
        if previous is None:
            return False
        fresh = self.agent.conversation_id
        self.agent.conversation_id = previous
        self.agent.context.repair_stored(previous)
        try:  # the empty conversation created by Agent.__init__ is now unused
            self.store.execute("DELETE FROM conversations WHERE id = ?", (fresh,))
        except Exception:
            logging.getLogger(__name__).debug("could not drop unused conversation", exc_info=True)
        return True

    def use_voice_model(self) -> str | None:
        """Switch the agent to the faster model meant for spoken turns, if there is one.
        Returns its description, or None when the main model stays in use."""
        from jarvis.llm.providers import build_voice_client, describe

        if self.agent.primary.is_local:
            return None
        client = build_voice_client(self.settings)
        if client is None:
            return None
        self.agent.primary = client
        return describe(client)

    def close(self) -> None:
        self.store.close()


def configure_logging(settings: Settings) -> None:
    """Log to a rotating file only. Never to the console; never request bodies or keys."""
    root = logging.getLogger()
    if any(h.get_name() == _LOG_HANDLER_NAME for h in root.handlers):
        return
    handler = RotatingFileHandler(
        settings.log_dir / "jarvis.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.set_name(_LOG_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_app(settings: Settings | None = None, *, force_local: bool = False) -> App:
    if settings is None:
        from jarvis.config import get_settings

        settings = get_settings()
    if force_local:
        settings = settings.model_copy(update={"provider": "ollama"})
    configure_logging(settings)

    store = MemoryStore(settings.db_path)
    try:
        registry = ToolRegistry(settings.tool_result_max_chars).load_builtin()
        primary, fallback = build_clients(settings)
        fast = build_fast_client(settings)
        previous = store.latest_conversation(12)
        agent = Agent(primary, registry, store, settings, fallback=fallback, fast=fast)
    except BaseException:
        store.close()
        raise
    return App(settings, store, registry, primary, fallback, fast, agent, previous)
