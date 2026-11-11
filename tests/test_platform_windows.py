"""Windows backend logic (moved out of the tool modules). The system calls are mocked, so
these run on every OS; they check the code paths, not a real Windows machine."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jarvis.platform import windows as win
from jarvis.platform.base import AppEntry
from jarvis.platform.windows import WindowsPlatform


@pytest.fixture
def backend():
    return WindowsPlatform()


def test_scan_shortcuts_filters_junk(tmp_path):
    (tmp_path / "Sub").mkdir()
    for name in ("Spotify.lnk", "Sub/Uninstall Foo.lnk", "Sub/Foo Readme.lnk", "Sub/Foo.lnk"):
        (tmp_path / name).write_text("")
    names = sorted(e.name for e in win.scan_shortcuts([tmp_path]))
    assert names == ["Foo", "Spotify"]


def test_scan_shortcuts_marks_browser_site_shortcuts(tmp_path):
    apps = tmp_path / "Chrome Apps"
    apps.mkdir()
    (apps / "YouTube.lnk").write_text("")
    (tmp_path / "Real.lnk").write_bytes(b"xx chrome_proxy.exe xx")
    (tmp_path / "Plain.lnk").write_text("")
    kinds = {e.name: e.kind for e in win.scan_shortcuts([tmp_path])}
    assert kinds == {"YouTube": "webapp", "Real": "webapp", "Plain": "lnk"}


def test_launch_dispatch(backend, monkeypatch):
    calls = {}
    monkeypatch.setattr(win.os, "startfile", lambda p: calls.setdefault("start", p), raising=False)
    monkeypatch.setattr(win.subprocess, "Popen", lambda args, **k: calls.setdefault("popen", args))
    backend.launch_app(AppEntry("A", "lnk", "x.lnk"))
    backend.launch_app(AppEntry("S", "uwp", "Pkg!App"))
    assert calls["start"] == "x.lnk" and calls["popen"][1] == "shell:AppsFolder\\Pkg!App"


def test_open_path_uses_startfile(backend, monkeypatch):
    opened = []
    monkeypatch.setattr(win.os, "startfile", opened.append, raising=False)
    backend.open_path("C:/a.txt")
    assert opened == ["C:/a.txt"]


def test_common_exes_are_stock_windows_tools():
    names = {e.name for e in win.common_exes()}
    assert {"Notepad", "Calculator", "Command Prompt"} <= names


def test_uwp_scan_parses_get_startapps(monkeypatch):
    out = '[{"Name":"Spotify","AppID":"Spotify.App"},{"Name":"Calc","AppID":"Chrome._crx_abc"},{"Name":"Uninstall X","AppID":"u"}]'
    monkeypatch.setattr(win, "_powershell", lambda script, timeout=20.0: subprocess.CompletedProcess([], 0, out, ""))
    entries = {e.name: e.kind for e in win.scan_uwp()}
    assert entries == {"Spotify": "uwp", "Calc": "webapp"}
    monkeypatch.setattr(win, "_powershell", lambda script, timeout=20.0: subprocess.CompletedProcess([], 1, "", "x"))
    assert win.scan_uwp() == []


def _ps(stdout, code=0, scripts=None):
    def run(script, timeout=20.0):
        if scripts is not None:
            scripts.append(script)
        return subprocess.CompletedProcess([], code, stdout, "")

    return run


def test_volume_goes_through_the_powershell_snippet(backend, monkeypatch):
    scripts = []
    monkeypatch.setattr(win, "_powershell", _ps("40,0\n", scripts=scripts))
    assert backend.set_volume(40) == (40, False) and "SetLevel(40)" in scripts[0]
    assert backend.set_mute(True) == (40, False) and "SetMute($true)" in scripts[1]
    monkeypatch.setattr(win, "_powershell", _ps("25,1"))
    assert backend.get_volume() == (25, True)


def test_volume_failure_raises_oserror(backend, monkeypatch):
    monkeypatch.setattr(win, "_powershell", _ps("", code=1))
    with pytest.raises(OSError):
        backend.get_volume()
    monkeypatch.setattr(win, "_powershell", _ps("garbage"))
    with pytest.raises(OSError):
        backend.get_volume()


def test_shell_and_flags(backend):
    argv = backend.shell_argv("Get-Date")
    assert argv[:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    assert argv[-1].endswith("Get-Date") and "UTF8" in argv[-1]
    assert backend.shell_kind == "powershell" and backend.reserved_device_names
    assert backend.trash_name == "Recycle Bin"
    assert backend.never_close_titles() == {"program manager"}


def test_protected_paths_follow_environment(backend, tmp_path, monkeypatch):
    import os

    windir = tmp_path / "Win"
    (windir / "System32").mkdir(parents=True)
    monkeypatch.setenv("SystemRoot", str(windir))
    monkeypatch.setenv("windir", str(windir))
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData"):
        monkeypatch.delenv(var, raising=False)
    norm = lambda p: os.path.normcase(os.path.realpath(str(p)))  # noqa: E731
    assert "system folder" in backend.protected_path_reason(norm(windir / "System32"))
    assert backend.protected_path_reason(norm(windir)) is not None
    assert backend.protected_path_reason(norm(tmp_path / "Documents")) is None


def test_ollama_path_uses_localappdata(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert backend.ollama_extra_paths() == [tmp_path / "Programs" / "Ollama" / "ollama.exe"]
    monkeypatch.delenv("LOCALAPPDATA")
    assert backend.ollama_extra_paths() == []


def test_find_ollama_exe_prefers_path_then_platform_locations(monkeypatch, tmp_path):
    from jarvis import platform as plat
    from jarvis.llm import ollama as ollama_mod

    plat.set_backend(WindowsPlatform())
    monkeypatch.setattr(ollama_mod.shutil, "which", lambda name: "C:/bin/ollama.exe")
    assert ollama_mod.find_ollama_exe() == "C:/bin/ollama.exe"
    monkeypatch.setattr(ollama_mod.shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert ollama_mod.find_ollama_exe() is None
    exe = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert ollama_mod.find_ollama_exe() == str(exe)


def test_find_ollama_exe_on_linux_locations(monkeypatch, tmp_path):
    from jarvis import platform as plat
    from jarvis.llm import ollama as ollama_mod
    from jarvis.platform.linux import LinuxPlatform

    plat.set_backend(LinuxPlatform())
    monkeypatch.setattr(ollama_mod.shutil, "which", lambda name: None)
    fake = tmp_path / "ollama"
    fake.write_text("")
    monkeypatch.setattr(LinuxPlatform, "ollama_extra_paths", lambda self: [fake])
    assert ollama_mod.find_ollama_exe() == str(fake)


def test_detached_launch_flags(monkeypatch):
    from jarvis import platform as plat
    from jarvis.llm import ollama as ollama_mod
    from jarvis.platform.linux import LinuxPlatform

    seen = {}
    monkeypatch.setattr(ollama_mod.subprocess, "Popen", lambda argv, **kw: seen.update(argv=argv, **kw))
    plat.set_backend(LinuxPlatform())
    ollama_mod.launch_ollama_serve("/usr/bin/ollama")
    assert seen["argv"] == ["/usr/bin/ollama", "serve"] and seen["start_new_session"] is True
    assert "creationflags" not in seen


def test_unsupported_platform_imports_and_fails_clearly():
    from jarvis import platform as plat
    from jarvis.platform import PlatformUnsupported

    other = plat.create("darwin")
    with pytest.raises(PlatformUnsupported, match="not supported"):
        other.get_volume()
    with pytest.raises(PlatformUnsupported):
        other.windows()
    with pytest.raises(PlatformUnsupported):
        other.lock_screen()
    assert other.known_folder("Desktop") == Path.home() / "Desktop"
    assert other.clipboard_problem() is None
    assert other.shell_argv("ls")[0] == "bash"


def test_create_picks_backend_by_platform_name():
    from jarvis import platform as plat

    assert plat.create("win32").name == "windows"
    assert plat.create("linux").name == "linux"
    assert plat.create("linux2").name == "linux"
    assert plat.create("darwin").name == "other"
    assert isinstance(plat.create("win32"), plat.Platform)
    assert isinstance(plat.create("linux"), plat.Platform)
