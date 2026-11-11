"""Linux desktop-input backend (xdotool mocked), camera helpers per OS, and the doctor lines.
Nothing here touches a real display, input device or camera."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jarvis import doctor
from jarvis import platform as plat
from jarvis.config import Settings
from jarvis.platform import linux as lx
from jarvis.platform import windows as wn
from jarvis.platform.base import BasePlatform, PlatformUnsupported
from jarvis.platform.keys import MAIN_KEYS, MODIFIERS
from jarvis.platform.linux import LinuxPlatform
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import desktop as tools


# ---------------------------------------------------------------- mocked helpers
class Cmds:
    def __init__(self):
        self.present: set[str] = set()
        self.calls: list[list[str]] = []
        self.out: dict[tuple[str, str], str] = {}  # (program, first arg) -> stdout
        self.fail: set[str] = set()

    def which(self, name):
        return f"/usr/bin/{name}" if name in self.present else None

    def run(self, argv, timeout=10.0):
        self.calls.append(list(argv))
        if argv[0] in self.fail or (argv[0], argv[1] if len(argv) > 1 else "") in self.fail:
            return subprocess.CompletedProcess(argv, 1, "", "boom")
        return subprocess.CompletedProcess(argv, 0, self.out.get((argv[0], argv[1] if len(argv) > 1 else ""), ""), "")

    def xdo(self):
        return [c[1:] for c in self.calls if c[0] == "xdotool"]


@pytest.fixture
def cmds(monkeypatch):
    c = Cmds()
    monkeypatch.setattr(lx.shutil, "which", c.which)
    monkeypatch.setattr(lx, "run_quiet", c.run)
    monkeypatch.setattr(lx, "read_os_release", lambda path="": {"ID": "ubuntu", "ID_LIKE": "debian"})
    return c


@pytest.fixture
def x11(monkeypatch, cmds):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    cmds.present.add("xdotool")


@pytest.fixture
def linux():
    return LinuxPlatform()


# ---------------------------------------------------------------- Linux: availability
def test_wayland_is_refused_clearly_and_names_ydotool(monkeypatch, cmds, linux):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    cmds.present.add("xdotool")
    problem = linux.desktop_problem()
    assert "Wayland" in problem and "ydotool" in problem and "X11" in problem
    for call in (lambda: linux.input_click(1, 1), lambda: linux.input_text("x"), lambda: linux.input_keys(["enter"]),
                 lambda: linux.input_scroll(1, 1, "down", 1), lambda: linux.active_window(), lambda: linux.focus_window(5)):
        with pytest.raises(PlatformUnsupported, match="Wayland"):
            call()
    assert cmds.xdo() == [], "nothing was sent"


def test_missing_xdotool_names_the_package(monkeypatch, cmds, linux):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    problem = linux.desktop_problem()
    assert "xdotool" in problem and "sudo apt install xdotool" in problem
    with pytest.raises(PlatformUnsupported, match="xdotool"):
        linux.input_text("hi")


def test_headless_session(monkeypatch, cmds, linux):
    for var in ("XDG_SESSION_TYPE", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)
    assert "graphical session" in linux.desktop_problem()


def test_x11_with_xdotool_is_ready_and_has_no_element_listing(x11, linux):
    assert linux.desktop_problem() is None
    assert linux.can_inspect_elements() is False
    with pytest.raises(PlatformUnsupported, match="not implemented on Linux"):
        linux.ui_snapshot(1)


# ---------------------------------------------------------------- Linux: input
def test_click_and_double_click(x11, cmds, linux):
    linux.input_click(100, 200)
    linux.input_click(5, 6, double=True)
    assert cmds.xdo() == [
        ["mousemove", "100", "200"], ["click", "1"],
        ["mousemove", "5", "6"], ["click", "--repeat", "2", "--delay", "80", "1"],
    ]


def test_typing_passes_the_text_after_double_dash(x11, cmds, linux):
    linux.input_text("-rf héllo\nworld")
    (call,) = cmds.xdo()
    assert call[0] == "type" and call[-2:] == ["--", "-rf héllo\nworld"]


def test_key_chords_map_to_xdotool_names(x11, cmds, linux):
    for keys in (["ctrl", "s"], ["alt", "tab"], ["win", "d"], ["enter"], ["pagedown"], ["f5"], ["ctrl", "plus"], ["escape"]):
        linux.input_keys(keys)
    sent = [c[-1] for c in cmds.xdo()]
    assert sent == ["ctrl+s", "alt+Tab", "super+d", "Return", "Next", "F5", "ctrl+plus", "Escape"]


def test_every_parseable_key_has_an_xdotool_name(x11, cmds, linux):
    for name in (*MODIFIERS, *MAIN_KEYS):
        linux.input_keys([name])  # raises if there is no mapping
    assert len(cmds.xdo()) == len(MODIFIERS) + len(MAIN_KEYS)
    with pytest.raises(PlatformUnsupported):
        linux.input_keys(["nonsense"])


def test_scroll_uses_wheel_buttons(x11, cmds, linux):
    linux.input_scroll(10, 20, "down", 3)
    linux.input_scroll(10, 20, "left", 1)
    assert cmds.xdo() == [
        ["mousemove", "10", "20"], ["click", "--repeat", "3", "--delay", "20", "5"],
        ["mousemove", "10", "20"], ["click", "--repeat", "1", "--delay", "20", "6"],
    ]


def test_a_failing_xdotool_becomes_an_oserror(x11, cmds, linux):
    cmds.fail.add("xdotool")
    with pytest.raises(OSError, match="xdotool failed"):
        linux.input_text("x")


# ---------------------------------------------------------------- Linux: windows
def test_active_window_info(x11, cmds, linux, monkeypatch):
    cmds.out[("xdotool", "getactivewindow")] = "12345\n"
    cmds.out[("xdotool", "getwindowname")] = "Terminal - bash\n"
    cmds.out[("xdotool", "getwindowpid")] = "999\n"
    cmds.out[("xdotool", "getwindowclassname")] = "gnome-terminal-server\n"

    class FakeProc:
        def __init__(self, pid):
            pass

        def name(self):
            return "gnome-terminal-"

        def uids(self):
            return type("U", (), {"real": 1000})()

    import psutil

    monkeypatch.setattr(psutil, "Process", FakeProc)
    info = linux.active_window()
    assert (info.title, info.process, info.class_name, info.pid, info.handle, info.elevated) == (
        "Terminal - bash", "gnome-terminal-", "gnome-terminal-server", 999, 12345, False,
    )
    from jarvis.desktop import policy

    assert policy.window_kind(info) == "terminal"


def test_a_root_owned_window_counts_as_elevated_for_a_normal_user(x11, cmds, linux, monkeypatch):
    cmds.out[("xdotool", "getwindowname")] = "Synaptic\n"
    cmds.out[("xdotool", "getwindowpid")] = "50\n"

    class RootProc:
        def __init__(self, pid):
            pass

        def name(self):
            return "synaptic"

        def uids(self):
            return type("U", (), {"real": 0})()

    import psutil

    monkeypatch.setattr(psutil, "Process", RootProc)
    monkeypatch.setattr(LinuxPlatform, "self_elevated", lambda self: False)
    assert linux.window_info(7).elevated is True


def test_focus_window_activates_and_verifies(x11, cmds, linux):
    cmds.out[("xdotool", "getactivewindow")] = "42\n"
    cmds.out[("xdotool", "getwindowname")] = "Files\n"
    cmds.out[("xdotool", "getwindowpid")] = "0\n"
    assert linux.focus_window(42) is True
    assert ["windowactivate", "--sync", "42"] in cmds.xdo()
    assert linux.focus_window(43) is False  # activation ran but another window is in front


def test_tools_on_linux_say_listing_is_not_implemented(x11, cmds, linux):
    plat.set_backend(linux)
    import asyncio

    result = asyncio.run(tools.desktop_inspect())
    assert not result.ok and "not implemented" in result.content and "desktop_click" in result.content


# ---------------------------------------------------------------- the base class
def test_unsupported_platform_fails_clearly():
    base = BasePlatform()
    assert "not supported" in base.desktop_problem()
    with pytest.raises(PlatformUnsupported):
        base.input_text("x")
    assert base.camera_devices() is None and base.camera_access_problem() is None and base.camera_backends() == ["any"]


def test_linux_camera_helpers(monkeypatch, tmp_path, linux):
    monkeypatch.setattr(lx, "Path", lambda p: tmp_path if p == "/dev" else Path(p))
    assert linux.camera_backends() == ["v4l2"]
    assert linux.camera_devices() == []
    video = tmp_path / "video0"
    video.write_text("")
    assert linux.camera_devices() == [str(video)]
    video.chmod(0o000)
    import os

    if os.name == "posix" and os.geteuid() != 0:  # root can open anything
        assert "video' group" in linux.camera_access_problem()


# ---------------------------------------------------------------- Windows pieces that run anywhere
def test_windows_camera_backends_prefer_directshow():
    assert wn.WindowsPlatform().camera_backends() == ["dshow", "msmf"]


def test_windows_camera_devices_parse_powershell_json(monkeypatch):
    def fake_ps(script, timeout=0):
        return subprocess.CompletedProcess([], 0, '["Integrated Camera","OBS Virtual Camera"]\n', "")

    monkeypatch.setattr(wn, "_powershell", fake_ps)
    assert wn.WindowsPlatform().camera_devices() == ["Integrated Camera", "OBS Virtual Camera"]
    monkeypatch.setattr(wn, "_powershell", lambda s, timeout=0: subprocess.CompletedProcess([], 0, '"Webcam"', ""))
    assert wn.WindowsPlatform().camera_devices() == ["Webcam"]
    monkeypatch.setattr(wn, "_powershell", lambda s, timeout=0: subprocess.CompletedProcess([], 0, "", ""))
    assert wn.WindowsPlatform().camera_devices() == []
    monkeypatch.setattr(wn, "_powershell", lambda s, timeout=0: subprocess.CompletedProcess([], 1, "", "x"))
    assert wn.WindowsPlatform().camera_devices() is None

    def boom(script, timeout=0):
        raise subprocess.TimeoutExpired("powershell", 1)

    monkeypatch.setattr(wn, "_powershell", boom)
    assert wn.WindowsPlatform().camera_devices() is None


# ---------------------------------------------------------------- doctor lines
def settings(tmp_path, **kw) -> Settings:
    base = dict(provider="nvidia", nvidia_api_key="k", data_dir_override=str(tmp_path))
    base.update(kw)
    return Settings(_env_file=None, **base)


class DoctorPlatform(BasePlatform):
    name = "linux"

    def __init__(self, problem=None, inspect=False, access=None):
        self._problem, self._inspect, self._access = problem, inspect, access

    def desktop_problem(self):
        return self._problem

    def can_inspect_elements(self):
        return self._inspect

    def camera_access_problem(self):
        return self._access


@pytest.fixture
def cv2_present(monkeypatch):
    monkeypatch.setattr(doctor, "module_present", lambda name: True)


def test_doctor_camera_lines(monkeypatch, tmp_path, cv2_present):
    plat.set_backend(DoctorPlatform())
    monkeypatch.setattr(doctor, "probe_cameras", lambda: ["Integrated Camera"])
    ok = doctor.check_camera(settings(tmp_path))
    assert ok.name == "Camera" and ok.status == doctor.PASS and "1 found (Integrated Camera)" in ok.detail and "JARVIS_CAMERA=ask" in ok.detail
    monkeypatch.setattr(doctor, "probe_cameras", lambda: [])
    none = doctor.check_camera(settings(tmp_path, camera="on"))
    assert none.status == doctor.WARN and "no camera found" in none.detail and "JARVIS_CAMERA=on" in none.detail
    monkeypatch.setattr(doctor, "probe_cameras", lambda: None)
    unknown = doctor.check_camera(settings(tmp_path))
    assert unknown.status == doctor.PASS and "does not list cameras" in unknown.detail
    many = ["A", "B", "C", "D", "E"]
    monkeypatch.setattr(doctor, "probe_cameras", lambda: many)
    assert "5 found (A, B, C and 2 more)" in doctor.check_camera(settings(tmp_path)).detail


def test_doctor_camera_off_bad_mode_privacy_and_missing_opencv(monkeypatch, tmp_path, cv2_present):
    plat.set_backend(DoctorPlatform())
    assert doctor.check_camera(settings(tmp_path, camera="off")).status == doctor.SKIP
    bad = doctor.check_camera(settings(tmp_path, camera="maybe"))
    assert bad.status == doctor.WARN and "maybe" in bad.detail and "ask, on or off" in bad.fix
    plat.set_backend(DoctorPlatform(access="'Let desktop apps access your camera' is off"))
    blocked = doctor.check_camera(settings(tmp_path))
    assert blocked.status == doctor.WARN and "desktop apps" in blocked.detail
    plat.set_backend(DoctorPlatform())
    monkeypatch.setattr(doctor, "module_present", lambda name: name != "cv2")
    missing = doctor.check_camera(settings(tmp_path))
    assert missing.status == doctor.WARN and "OpenCV" in missing.detail and "screen" in missing.fix


def test_doctor_desktop_lines(monkeypatch):
    plat.set_backend(DoctorPlatform(inspect=True))
    win = doctor.check_desktop()
    assert win.name == "Desktop control" and win.status == doctor.PASS and "UI Automation" in win.detail
    plat.set_backend(DoctorPlatform(inspect=False))
    lin = doctor.check_desktop()
    assert lin.status == doctor.PASS and "xdotool present" in lin.detail and "not implemented on Linux" in lin.detail
    monkeypatch.setattr(lx, "read_os_release", lambda path="": {"ID": "ubuntu", "ID_LIKE": "debian"})
    plat.set_backend(DoctorPlatform(problem="desktop control needs xdotool: install xdotool"))
    missing = doctor.check_desktop()
    assert missing.status == doctor.WARN and "xdotool" in missing.detail and missing.fix == "sudo apt install xdotool"
    wayland = DoctorPlatform(problem="desktop input is not supported in a Wayland session")
    wayland.name = "windows"
    plat.set_backend(wayland)
    assert "desktop extra" in doctor.check_desktop().fix


def test_doctor_runs_both_checks_in_the_full_list(monkeypatch, tmp_path, fake_platform):
    monkeypatch.setattr(doctor, "probe_hosted", lambda s: (doctor.PASS, "ok"))
    monkeypatch.setattr(doctor, "find_ollama", lambda: None)
    monkeypatch.setattr(doctor, "probe_ollama", lambda s: (False, []))
    monkeypatch.setattr(doctor, "probe_cameras", lambda: [])
    names = [c.name for c in doctor.run_checks(settings(tmp_path), offline=True)]
    assert "Camera" in names and "Desktop control" in names


def test_the_desktop_extra_is_only_checked_where_it_applies(monkeypatch, tmp_path, fake_platform):
    monkeypatch.setattr(doctor, "EXTRAS", {"office": ("openpyxl",), "desktop": ()})
    checks = doctor.check_extras()
    assert [c.name for c in checks] == ["Extra: office"]
