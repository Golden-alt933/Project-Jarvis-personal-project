"""How the autonomy features are wired in: the websites prompt section, `jarvis --daemon`,
`jarvis --brief`, the frontends' hook-up, the doctor lines and the desktop notification.
Fakes everywhere: no network, no browser, no real OS notification."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from rich.console import Console

from jarvis import doctor
from jarvis.agent.loop import Agent
from jarvis.agent.prompt import _WEBSITES, build_system_prompt
from jarvis.config import Settings
from jarvis.interfaces import cli
from jarvis.llm.types import LLMResponse, ResponseDone, TextDelta
from jarvis.memory.store import MemoryStore
from jarvis.proactive import notify as notify_mod
from jarvis.proactive import runtime, taskstore as ts
from jarvis.proactive.schedule_math import first_run, parse_repeat, utc_iso
from jarvis.proactive.tasks import RunResult
from jarvis.tools.base import ToolResult, tool
from jarvis.tools.registry import ToolRegistry


# -- the "working on websites" prompt section ------------------------------------------------------------------
def prompt(tmp_path: Path, groups, local: bool = False) -> str:
    return build_system_prompt(
        settings=Settings(data_dir_override=str(tmp_path)), facts=[], summary=None, output_mode="text",
        is_local=local, now=datetime(2026, 11, 10, 14, 0), tool_groups=groups,
    )


def test_websites_section_only_with_browser_tools(tmp_path: Path):
    assert "Working on websites" in prompt(tmp_path, {"core", "browser"})
    assert "Working on websites" not in prompt(tmp_path, {"core", "files"})
    assert "Working on websites" not in prompt(tmp_path, None)
    assert "Working on websites" not in prompt(tmp_path, [])


def test_websites_section_is_a_sensible_length_and_covers_the_rules():
    words = len(_WEBSITES.split())
    assert 120 <= words <= 175, words
    for needle in ("browser_status", "not connected", "Inspect", "again after every page change", "comparison",
                   "ask for anything missing", "payment", "stop", "irreversible", "read back", "truthfully"):
        assert needle.lower() in _WEBSITES.lower(), needle


class Script:
    def __init__(self, *replies: str, local: bool = False) -> None:
        self.replies = list(replies)
        self.name, self.model, self.context_window, self.is_local = "script", "m", 100000, local
        self.seen: list[Any] = []

    async def stream(self, messages, tools=None, *, max_tokens=None, temperature=None) -> AsyncIterator:
        self.seen.append((list(messages), tools))
        reply = self.replies.pop(0) if self.replies else "ok"
        yield TextDelta(reply)
        yield ResponseDone(LLMResponse(reply, []))

    async def complete(self, messages, tools=None, *, max_tokens=None, temperature=None) -> LLMResponse:
        async for ev in self.stream(messages, tools):
            if isinstance(ev, ResponseDone):
                return ev.response
        raise AssertionError


def registry_with(*groups: str) -> ToolRegistry:
    registry = ToolRegistry()
    for group in groups:
        def make(g: str):
            @tool(name=f"tool_{g}", group=g)
            def fn() -> ToolResult:
                """Do something."""
                return ToolResult.success("ok")

            return fn

        registry.register(make(group).__tool_spec__)
    return registry


async def run_one(tmp_path: Path, registry: ToolRegistry, llm: Script, **settings) -> str:
    store = MemoryStore(tmp_path / "j.db")
    try:
        agent = Agent(llm, registry, store, Settings(data_dir_override=str(tmp_path), **settings))
        async for _ in agent.run_turn("hi"):
            pass
    finally:
        store.close()
    return llm.seen[0][0][0].content


async def test_the_agent_adds_the_section_only_when_browser_tools_are_offered(tmp_path: Path):
    with_browser = await run_one(tmp_path, registry_with("core", "browser"), Script("hello"))
    assert "Working on websites" in with_browser
    without = await run_one(tmp_path / "b", registry_with("core", "files"), Script("hello"))
    assert "Working on websites" not in without


async def test_a_small_local_model_without_browser_tools_gets_no_website_guidance(tmp_path: Path):
    (tmp_path / "a").mkdir()
    text = await run_one(tmp_path / "a", registry_with("core", "browser"), Script("hi", local=True))
    assert "Working on websites" not in text  # the local menu is core/files/web/memory
    (tmp_path / "b").mkdir()
    text = await run_one(tmp_path / "b", registry_with("core", "browser"), Script("hi", local=True),
                         local_tool_groups=["core", "browser"])
    assert "Working on websites" in text


# -- a fake app --------------------------------------------------------------------------------------------------
@pytest.fixture
def app(tmp_path: Path):
    store = MemoryStore(tmp_path / "jarvis.db")
    settings = Settings(data_dir_override=str(tmp_path), task_max_steps=4)
    llm = Script("Good morning. It is Tuesday.")
    agent = Agent(llm, ToolRegistry(), store, settings)
    yield SimpleNamespace(settings=settings, store=store, agent=agent, llm=llm, closed=[], resume=lambda: False,
                          close=lambda: None)
    store.close()


# -- the daemon ------------------------------------------------------------------------------------------------------
class FakeRunner:
    last: "FakeRunner | None" = None

    def __init__(self, source, settings, *, confirm=None, output_mode="text") -> None:
        self.confirm, self.output_mode = confirm, output_mode
        self.calls: list[int] = []
        FakeRunner.last = self

    async def __call__(self, task, scheduled_for):
        self.calls.append(task["id"])
        return RunResult("ok", f"{task['title']} done", "conv", 1)


async def test_daemon_runs_due_tasks_notifies_stores_and_denies_confirmations(app, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "AgentTaskRunner", FakeRunner)
    now = datetime.now(timezone.utc).astimezone()
    rule = parse_repeat("once")
    anchor = (now - timedelta(minutes=1)).replace(tzinfo=None)
    tid = ts.add_task(app.store, title="Backup check", instruction="check the backup", repeat_text="once", anchor=anchor,
                      next_run=utc_iso(first_run(rule, anchor)), created=utc_iso(now))
    notes: list[tuple[str, str]] = []
    job = asyncio.ensure_future(runtime.run_daemon(app, None, notifier=lambda t, x: notes.append((t, x)) or True))
    for _ in range(300):
        if notes:
            break
        await asyncio.sleep(0.01)
    job.cancel()
    assert await job == 0
    assert FakeRunner.last.calls == [tid]
    assert FakeRunner.last.confirm is None  # nobody is present: every confirmation is declined
    assert notes == [("Jarvis", "Scheduled task Backup check: Backup check done")]
    out = capsys.readouterr().out
    assert "daemon running" in out and "Backup check done" in out
    assert ts.recent_runs(app.store, tid)[0]["summary"] == "Backup check done"  # stored for later


async def test_daemon_survives_a_failing_notifier(app, monkeypatch):
    monkeypatch.setattr(runtime, "AgentTaskRunner", FakeRunner)
    now = datetime.now(timezone.utc).astimezone()
    anchor = (now - timedelta(minutes=1)).replace(tzinfo=None)
    tid = ts.add_task(app.store, title="T", instruction="x", repeat_text="once", anchor=anchor,
                      next_run=utc_iso(first_run(parse_repeat("once"), anchor)), created=utc_iso(now))

    def explode(title, text):
        raise OSError("no notification daemon")

    job = asyncio.ensure_future(runtime.run_daemon(app, None, notifier=explode))
    for _ in range(300):
        if FakeRunner.last and FakeRunner.last.calls:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    job.cancel()
    assert await job == 0
    assert ts.recent_runs(app.store, tid)[0]["status"] == "ok"


def test_background_wiring_for_the_windows(app, monkeypatch):
    monkeypatch.setattr(runtime, "AgentTaskRunner", FakeRunner)
    said: list[str] = []

    async def say(text: str) -> None:
        said.append(text)

    bg = runtime.build_background(app, say, lambda: True, spoken=True, output_mode="voice")
    assert bg.tasks.runner.output_mode == "voice" and bg.tasks.runner.confirm is None
    assert [w.name for w in bg.proactive.watchers] == ["battery", "disk", "long-turn"]
    assert bg.proactive.deliver_busy is None  # no mid-turn channel unless the frontend gives one

    async def go():
        await bg.tasks.deliver("word " * 400)

    asyncio.run(go())
    assert len(said[0]) < 700 and "ask me for the full result" in said[0]  # spoken results are shortened


def test_background_is_fail_soft_for_a_frontend_with_an_odd_app():
    async def go():
        bg = runtime.start_background(SimpleNamespace(settings=SimpleNamespace(), store=None, agent=None),
                                      lambda t: asyncio.sleep(0), lambda: True)
        await bg.stop()
        return bg

    bg = asyncio.run(go())
    assert bg.tasks is None and bg.proactive is None


async def test_proactive_off_starts_no_watcher_task(app, monkeypatch):
    monkeypatch.setattr(runtime, "AgentTaskRunner", FakeRunner)
    app.settings = app.settings.model_copy(update={"proactive": "off"})
    bg = runtime.start_background(app, lambda t: asyncio.sleep(0), lambda: True)
    assert bg.proactive.watchers == [] and bg.proactive._task is None
    await bg.stop()


# -- the briefing ------------------------------------------------------------------------------------------------------------
async def test_brief_prints_one_turn_in_its_own_conversation(app):
    console = Console(record=True, width=120, color_system=None)
    before = app.agent.conversation_id
    assert await runtime.run_brief(app, console) == 0
    assert "Good morning. It is Tuesday." in console.export_text()
    sent = app.llm.seen[0][0]
    user = next(m for m in sent if m.role == "user").content
    for word in ("weather", "list_reminders", "list_scheduled_tasks", "sign-in", "home location"):
        assert word in user
    assert "Voice" not in user and app.agent.conversation_id == before  # the user's conversation is untouched
    assert "Your reply will be spoken aloud" in sent[0].content  # spoken-style output


async def test_brief_reports_failure_with_a_nonzero_exit(app, capsys):
    app.llm.replies = []

    class Boom(Script):
        async def stream(self, *a, **k):
            raise RuntimeError("down")
            yield  # pragma: no cover

    app.agent.primary = Boom()
    app.agent.fallback = None
    assert await runtime.run_brief(app, None) == 1
    assert capsys.readouterr().out.strip()


# -- the command line -----------------------------------------------------------------------------------------------------------
def patch_main(monkeypatch, app) -> None:
    async def no_bridge(settings):
        return None

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir_override="."))
    monkeypatch.setattr(cli, "build_app", lambda settings, force_local=False: app)
    monkeypatch.setattr(cli, "_start_browser_bridge", no_bridge)


def test_flags_exist_and_are_documented():
    parser = cli.build_parser()
    args = parser.parse_args(["--daemon"])
    assert args.daemon and not args.brief
    assert parser.parse_args(["--brief"]).brief
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    for needle in ("`--daemon`", "`--brief`", "## Scheduled tasks and the daemon", "## Proactive alerts",
                   "## Errands (email, calendar, maps, messages)", "schtasks /Create", "systemctl --user enable",
                   "ExecStart=%h/.local/bin/jarvis --daemon"):
        assert needle in readme, needle


@pytest.mark.parametrize("argv", [["--daemon", "--brief"], ["--daemon", "--once", "hi"], ["--daemon", "--voice"],
                                  ["--brief", "--hud"], ["--brief", "--once", "hi"]])
def test_incompatible_modes_are_refused(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_daemon_flag_runs_the_daemon_and_closes_the_app(app, monkeypatch):
    patch_main(monkeypatch, app)
    closed: list[bool] = []
    app.close = lambda: closed.append(True)
    seen: list[Any] = []

    async def fake_daemon(a, console):
        seen.append(a)
        return 0

    monkeypatch.setattr(runtime, "run_daemon", fake_daemon)
    assert cli.main(["--daemon"]) == 0
    assert seen == [app] and closed == [True]


def test_brief_flag_runs_the_briefing(app, monkeypatch, capsys):
    patch_main(monkeypatch, app)
    assert cli.main(["--brief"]) == 0
    assert "Good morning. It is Tuesday." in capsys.readouterr().out


# -- the doctor ---------------------------------------------------------------------------------------------------------------------
def doctor_lines(tmp_path: Path, **kw) -> dict[str, doctor.Check]:
    settings = Settings(_env_file=None, data_dir_override=str(tmp_path), **kw)
    return {c.name: c for c in doctor.check_autonomy(settings)}


def test_doctor_shows_no_tasks_and_alerts_on(tmp_path: Path):
    checks = doctor_lines(tmp_path)
    assert checks["Scheduled tasks"].status == doctor.PASS and "no tasks stored" in checks["Scheduled tasks"].detail
    assert checks["Proactive alerts"].detail == "on: battery, disk, long-turn"
    assert not (tmp_path / "jarvis.db").exists()  # looking never creates the database


def test_doctor_counts_tasks_and_shows_the_next_run(tmp_path: Path):
    store = MemoryStore(tmp_path / "jarvis.db")
    now = datetime(2026, 11, 10, 14, 0, tzinfo=timezone.utc)
    for title, nxt in (("a", "2026-11-12T03:30:00+00:00"), ("b", "2026-11-11T03:30:00+00:00")):
        ts.add_task(store, title=title, instruction="x", repeat_text="daily", anchor=datetime(2026, 11, 11, 9), next_run=nxt, created=utc_iso(now))
    ts.cancel_task(store, 1)
    store.close()
    detail = doctor_lines(tmp_path)["Scheduled tasks"].detail
    assert detail.startswith("1 stored, next run ") and "Nov 2026" in detail


def test_doctor_reports_proactive_off(tmp_path: Path):
    line = doctor_lines(tmp_path, proactive="off")["Proactive alerts"]
    assert line.status == doctor.SKIP and "off" in line.detail
    assert "disk" not in doctor_lines(tmp_path, watch_disk=False)["Proactive alerts"].detail


def test_doctor_does_not_break_on_a_corrupt_database(tmp_path: Path):
    (tmp_path / "jarvis.db").write_bytes(b"this is not sqlite" * 50)
    assert doctor_lines(tmp_path)["Scheduled tasks"].status == doctor.WARN


def test_the_checks_list_includes_the_new_lines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(doctor, "probe_ollama", lambda s: (False, []))
    names = [c.name for c in doctor.run_checks(Settings(_env_file=None, data_dir_override=str(tmp_path)), offline=True)]
    assert "Scheduled tasks" in names and "Proactive alerts" in names


# -- the notification ------------------------------------------------------------------------------------------------------------------
def test_windows_toast_command_keeps_user_text_out_of_the_script(monkeypatch):
    monkeypatch.setattr(notify_mod.shutil, "which", lambda name: r"C:\Windows\powershell.exe" if name == "powershell" else None)
    evil = "'; Remove-Item -Recurse C:\\ #"
    argv, env = notify_mod.command_for("Jarvis", evil, "win32")
    assert argv[0].endswith("powershell.exe") and "-NoProfile" in argv
    assert evil not in " ".join(argv)  # travels in the environment, escaped by PowerShell itself
    assert env["JARVIS_NOTIFY_TEXT"] == evil and env["JARVIS_NOTIFY_TITLE"] == "Jarvis"
    assert "SecurityElement]::Escape" in argv[-1]


def test_linux_notification_command(monkeypatch):
    monkeypatch.setattr(notify_mod.shutil, "which", lambda name: "/usr/bin/notify-send")
    argv, env = notify_mod.command_for("Jarvis", "-rf everything", "linux")
    assert argv == ["/usr/bin/notify-send", "--app-name=Jarvis", "--", "Jarvis", "-rf everything"] and env == {}


def test_notification_text_is_clipped_and_flattened(monkeypatch):
    monkeypatch.setattr(notify_mod.shutil, "which", lambda name: "/usr/bin/notify-send")
    argv, _ = notify_mod.command_for("T", "line one\nline two " + "x" * 500, "linux")
    assert "\n" not in argv[-1] and len(argv[-1]) <= notify_mod.MAX_TEXT


def test_no_way_to_notify_is_not_an_error(monkeypatch):
    monkeypatch.setattr(notify_mod.shutil, "which", lambda name: None)
    assert notify_mod.command_for("a", "b", "linux") is None
    assert notify_mod.command_for("a", "b", "win32") is None
    assert notify_mod.command_for("a", "b", "darwin") is None
    monkeypatch.setattr(notify_mod.sys, "platform", "linux")
    assert notify_mod.notify("a", "b") is False


def test_notify_runs_the_command_and_fails_quietly(monkeypatch):
    monkeypatch.setattr(notify_mod.shutil, "which", lambda name: "/usr/bin/notify-send")
    monkeypatch.setattr(notify_mod.sys, "platform", "linux")
    calls: list[list[str]] = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(notify_mod.subprocess, "run", fake_run)
    assert notify_mod.notify("Jarvis", "hello") is True and calls[0][-1] == "hello"
    monkeypatch.setattr(notify_mod.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))
    assert notify_mod.notify("Jarvis", "hello") is False

    def broken(*a, **k):
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(notify_mod.subprocess, "run", broken)
    assert notify_mod.notify("Jarvis", "hello") is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows toast")
def test_the_toast_script_is_valid_powershell_syntax():
    import shutil

    shell = shutil.which("powershell")
    if shell is None:
        pytest.skip("no powershell")
    check = "$null = [scriptblock]::Create($env:SCRIPT_UNDER_TEST); 'ok'"
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", check],
                            env={**__import__("os").environ, "SCRIPT_UNDER_TEST": notify_mod._TOAST},
                            capture_output=True, text=True, timeout=60)
    assert result.stdout.strip() == "ok", result.stderr
