"""CLI tests: no network, no real terminal."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from rich.console import Console

from jarvis.agent.events import Notice, TextChunk, ToolFinished, ToolStarted, TurnDone
from jarvis.config import Settings
from jarvis.interfaces import cli
from jarvis.interfaces.app import build_app
from jarvis.llm.providers import ConfigError
from jarvis.llm.types import Message, ToolCall, Usage
from jarvis.tools.base import ToolResult


def local_settings(tmp_path: Path, **extra: Any) -> Settings:
    return Settings(provider="ollama", data_dir_override=str(tmp_path), **extra)


def recording_console() -> Console:
    return Console(record=True, width=100, color_system=None)


# -- build_app ---------------------------------------------------------------------
def test_build_app_with_ollama(tmp_path: Path) -> None:
    app = build_app(local_settings(tmp_path))
    try:
        assert app.primary.is_local
        assert app.fallback is None
        assert len(app.registry) > 0
        assert (tmp_path / "jarvis.db").exists()
        assert (tmp_path / "logs" / "jarvis.log").exists()
        assert app.agent.conversation_id
    finally:
        app.close()


def test_force_local_overrides_cloud_provider(tmp_path: Path) -> None:
    settings = Settings(provider="nvidia", nvidia_api_key="", data_dir_override=str(tmp_path))
    app = build_app(settings, force_local=True)
    try:
        assert app.primary.is_local
    finally:
        app.close()


def test_resume_continues_latest_conversation(tmp_path: Path) -> None:
    first = build_app(local_settings(tmp_path))
    cid = first.agent.conversation_id
    first.store.append_message(cid, Message("user", "hello"))
    first.close()

    second = build_app(local_settings(tmp_path))
    try:
        assert second.resume() is True
        assert second.agent.conversation_id == cid
    finally:
        second.close()


def test_resume_with_nothing_to_resume(tmp_path: Path) -> None:
    app = build_app(local_settings(tmp_path))
    try:
        assert app.resume() is False
    finally:
        app.close()


# -- renderer ----------------------------------------------------------------------
def test_renderer_scripted_turn() -> None:
    console = recording_console()
    renderer = cli.TurnRenderer(console)
    call = ToolCall("1", "get_time", {"timezone": "UTC"})
    events = [
        TextChunk("Let me "),
        TextChunk("check."),
        ToolStarted(call),
        ToolFinished(call, ToolResult.success("12:00"), 0.2),
        Notice("cloud unreachable"),
        TextChunk("It is **noon**."),
        TurnDone("It is noon.", 2, 1, 1.5, "qwen", Usage(100, 20)),
    ]
    for event in events:
        renderer.handle(event)
    out = console.export_text()
    assert "Let me check." in out
    assert "get_time" in out and "timezone=UTC" in out
    assert "✓ get_time 0.2s" in out
    assert "cloud unreachable" in out
    assert "It is noon." in out  # markdown bold markers are rendered away
    assert "1.5s · 2 steps · 1 tool call · qwen · 100 in / 20 out tokens" in out
    assert out.index("get_time") < out.index("It is noon.")


def test_renderer_failed_tool_and_verbose_preview() -> None:
    console = recording_console()
    renderer = cli.TurnRenderer(console, verbose=True)
    call = ToolCall("1", "read_file", {"path": "C:/x"})
    renderer.handle(ToolStarted(call))
    renderer.handle(ToolFinished(call, ToolResult.fail("no such file", "check path"), 0.0))
    out = console.export_text()
    assert "✗ read_file" in out
    assert "no such file" in out


def test_renderer_reports_why_a_turn_stopped() -> None:
    console = recording_console()
    renderer = cli.TurnRenderer(console)
    renderer.handle(TurnDone("Sorry, error.", 1, 0, 0.1, "m", stopped="error"))
    out = console.export_text()
    assert "Sorry, error." in out  # nothing streamed, so the text is shown at the end
    assert "something went wrong" in out

    console = recording_console()
    renderer = cli.TurnRenderer(console)
    renderer.handle(TurnDone("partial", 25, 24, 9.0, "m", stopped="max_steps"))
    assert "step limit" in console.export_text()


def test_summarize_args_is_one_line_and_truncated() -> None:
    line = cli.summarize_args({"text": "a\nb\n" * 100, "n": 3}, limit=60)
    assert "\n" not in line and len(line) <= 60 and line.endswith("…")


# -- confirmation ------------------------------------------------------------------
async def test_confirmer_default_is_no() -> None:
    console = recording_console()
    answers = iter(["", "n", "y", "YES"])

    async def ask(_: str) -> str:
        return next(answers)

    confirm = cli.Confirmer(console, cli.TurnRenderer(console), ask)
    results = [await confirm("delete C:/x") for _ in range(4)]
    assert results == [False, False, True, True]
    assert "delete C:/x" in console.export_text()


async def test_confirmer_denies_on_ctrl_c() -> None:
    console = recording_console()

    async def ask(_: str) -> str:
        raise KeyboardInterrupt

    assert await cli.Confirmer(console, cli.TurnRenderer(console), ask)("x") is False


# -- slash commands ----------------------------------------------------------------
def test_parse_command() -> None:
    assert cli.parse_command("/help") == ("help", "")
    assert cli.parse_command("  /Forget 12 ") == ("forget", "12")
    assert cli.parse_command("hello /help") is None
    assert cli.parse_command("//not a command") is None


def test_slash_commands(tmp_path: Path) -> None:
    app = build_app(local_settings(tmp_path))
    try:
        console = recording_console()
        commands = cli.SlashCommands(app, console)

        assert commands.dispatch("/help") is False
        assert "/forget" in console.export_text()

        commands.dispatch("/tools")
        assert "get_time" in console.export_text() or "group" in console.export_text()

        commands.dispatch("/model")
        assert "ollama" in console.export_text()

        before = app.agent.conversation_id
        commands.dispatch("/new")
        assert app.agent.conversation_id != before

        fact = app.store.remember("Likes tea")
        commands.dispatch("/memory")
        assert "Likes tea" in console.export_text()
        commands.dispatch(f"/forget {fact.id}")
        assert app.store.list_facts() == []
        commands.dispatch("/forget abc")
        assert "Usage: /forget" in console.export_text()

        commands.dispatch("/nope")
        assert "Unknown command /nope" in console.export_text()

        assert commands.dispatch("/exit") is True
    finally:
        app.close()


def test_local_and_cloud_switch(tmp_path: Path) -> None:
    settings = Settings(provider="nvidia", nvidia_api_key="k", model="m", data_dir_override=str(tmp_path))
    app = build_app(settings)
    try:
        console = recording_console()
        commands = cli.SlashCommands(app, console)
        commands.dispatch("/local")
        assert app.agent._force_local is True
        assert "qwen3" in console.export_text()
        commands.dispatch("/cloud")
        assert app.agent._force_local is False
    finally:
        app.close()


# -- turn driving and --once -------------------------------------------------------
class FakeAgent:
    def __init__(self, script: Any = None) -> None:
        self.cancelled = False
        self.confirm_answer: bool | None = None
        self.script = script

    def cancel(self) -> None:
        self.cancelled = True

    async def run_turn(self, text: str, *, confirm: Any = None, notify: Any = None, **_: Any) -> AsyncIterator[Any]:
        if self.script is not None:
            async for event in self.script(self, text, confirm):
                yield event
            return
        yield TextChunk(f"echo: {text}")
        yield TurnDone(f"echo: {text}", 1, 0, 0.1, "fake")


def fake_app(agent: FakeAgent) -> SimpleNamespace:
    closed: list[bool] = []
    return SimpleNamespace(
        agent=agent,
        close=lambda: closed.append(True),
        resume=lambda: False,
        settings=SimpleNamespace(),
        closed=closed,
    )


def patch_main(monkeypatch: pytest.MonkeyPatch, app: Any) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir_override="."))
    monkeypatch.setattr(cli, "build_app", lambda settings, force_local=False: app)


def test_once_prints_reply_and_exits_zero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    app = fake_app(FakeAgent())
    patch_main(monkeypatch, app)
    assert cli.main(["--once", "hi there"]) == 0
    assert "echo: hi there" in capsys.readouterr().out
    assert app.closed == [True]


@pytest.mark.parametrize(("flags", "expected"), [([], False), (["--yes"], True)])
def test_once_confirmations(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flags: list[str], expected: bool
) -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        agent.confirm_answer = await confirm("delete C:/x")
        yield TurnDone("done", 1, 1, 0.1, "fake")

    agent = FakeAgent(script)
    patch_main(monkeypatch, fake_app(agent))
    assert cli.main(["--once", "go", *flags]) == 0
    assert agent.confirm_answer is expected
    assert "delete C:/x" in capsys.readouterr().out


def test_once_error_turn_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        yield TurnDone("failed", 0, 0, 0.1, "fake", stopped="error")

    patch_main(monkeypatch, fake_app(FakeAgent(script)))
    assert cli.main(["--once", "go"]) == 1


async def test_cancel_during_turn_returns_cleanly() -> None:
    async def script(agent: FakeAgent, text: str, confirm: Any) -> AsyncIterator[Any]:
        yield TextChunk("working")
        raise asyncio.CancelledError

    agent = FakeAgent(script)
    app = fake_app(agent)
    console = recording_console()
    renderer = cli.TurnRenderer(console)
    confirm = cli.Confirmer(console, renderer, lambda _: asyncio.sleep(0, "n"))
    assert await cli.run_one_turn(app, "x", console, renderer, confirm) is None  # type: ignore[arg-type]
    assert agent.cancelled is True
    assert "Cancelled." in console.export_text()


# -- configuration errors ----------------------------------------------------------
def test_missing_api_key_exits_2_with_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(provider="nvidia", nvidia_api_key="", data_dir_override=str(tmp_path))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["--once", "hi"]) == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "NVIDIA_API_KEY" in text
    assert ".env" in text
    assert "Traceback" not in text


def test_config_error_from_build_app_exits_2(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def boom(settings: Any, force_local: bool = False) -> Any:
        raise ConfigError("Provider 'groq' has no default model; set JARVIS_MODEL in .env.")

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir_override="."))
    monkeypatch.setattr(cli, "build_app", boom)
    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert "JARVIS_MODEL" in captured.out + captured.err
