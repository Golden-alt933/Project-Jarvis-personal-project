"""Linux backend. Everything is driven by fake .desktop files and mocked subprocess/which, so
these tests run on any OS and prove the parsing, ordering and failure messages. They say
nothing about how a real desktop behaves."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.platform import PlatformUnsupported
from jarvis.platform import linux as lx
from jarvis.platform.base import AppEntry
from jarvis.platform.linux import LinuxPlatform


# ----------------------------------------------------------------- fixtures
class Cmds:
    """Fake helper programs: which() finds exactly the registered names and run_quiet
    answers from a table keyed by the command name."""

    def __init__(self) -> None:
        self.present: set[str] = set()
        self.handlers: dict[str, object] = {}
        self.calls: list[list[str]] = []
        self.popens: list[list[str]] = []
        self.popen_codes: dict[str, int | None] = {}

    def add(self, name: str, handler=None) -> None:
        self.present.add(name)
        if handler is not None:
            self.handlers[name] = handler

    def which(self, name: str):
        return f"/usr/bin/{name}" if name in self.present else None

    def run(self, argv, timeout=10.0):
        self.calls.append(list(argv))
        handler = self.handlers.get(argv[0])
        if handler is None:
            raise FileNotFoundError(argv[0])
        code, out, err = handler(argv) if callable(handler) else handler
        return subprocess.CompletedProcess(argv, code, out, err)


@pytest.fixture
def cmds(monkeypatch):
    c = Cmds()
    monkeypatch.setattr(lx.shutil, "which", c.which)
    monkeypatch.setattr(lx, "run_quiet", c.run)
    monkeypatch.setattr(lx.ctypes.util, "find_library", lambda name: None)  # real one spawns a subprocess

    class FakePopen:
        def __init__(self, argv, **kwargs):
            c.popens.append(list(argv))
            self.argv = argv
            self.kwargs = kwargs

        def wait(self, timeout=None):
            code = c.popen_codes.get(self.argv[0], None)
            if code is None:
                raise subprocess.TimeoutExpired(self.argv, timeout)
            return code

    monkeypatch.setattr(lx.subprocess, "Popen", FakePopen)
    return c


@pytest.fixture
def x11(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


@pytest.fixture
def wayland(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")  # XWayland


@pytest.fixture
def headless(monkeypatch):
    for var in ("XDG_SESSION_TYPE", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def apt(monkeypatch):
    monkeypatch.setattr(lx, "read_os_release", lambda path="": {"ID": "ubuntu", "ID_LIKE": "debian"})


@pytest.fixture
def linux():
    return LinuxPlatform()


# ----------------------------------------------------------------- session / distro
def test_session_type_matrix(monkeypatch):
    def check(env: dict, expected: str) -> None:
        for var in ("XDG_SESSION_TYPE", "DISPLAY", "WAYLAND_DISPLAY"):
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert lx.session_type() == expected, env

    check({}, "headless")
    check({"DISPLAY": ":0"}, "x11")
    check({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"}, "x11")
    check({"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, "wayland")
    check({"WAYLAND_DISPLAY": "wayland-0"}, "wayland")
    check({"XDG_SESSION_TYPE": "tty"}, "headless")
    check({"XDG_SESSION_TYPE": "wayland"}, "headless")  # declared, but no display at all


@pytest.mark.parametrize(
    "release, expected",
    [
        ({"ID": "ubuntu", "ID_LIKE": "debian"}, "apt"),
        ({"ID": "debian"}, "apt"),
        ({"ID": "linuxmint", "ID_LIKE": "ubuntu debian"}, "apt"),
        ({"ID": "fedora"}, "dnf"),
        ({"ID": "rocky", "ID_LIKE": "rhel centos fedora"}, "dnf"),
        ({"ID": "arch"}, "pacman"),
        ({"ID": "endeavouros", "ID_LIKE": "arch"}, "pacman"),
        ({"ID": "opensuse-tumbleweed", "ID_LIKE": "opensuse suse"}, "zypper"),
        ({"ID": "nixos"}, None),
        ({}, None),
    ],
)
def test_package_manager(release, expected):
    assert lx.package_manager(release) == expected


def test_read_os_release(tmp_path):
    f = tmp_path / "os-release"
    f.write_text('NAME="Ubuntu"\nID=ubuntu\n# comment\nID_LIKE="debian"\nPRETTY_NAME="Ubuntu 24.04 LTS"\n')
    info = lx.read_os_release(str(f))
    assert info["ID"] == "ubuntu" and info["PRETTY_NAME"] == "Ubuntu 24.04 LTS"
    assert lx.read_os_release(str(tmp_path / "missing")) == {}


def test_install_command_per_distro():
    assert lx.install_command("wmctrl", "apt") == "sudo apt install wmctrl"
    assert lx.install_command("wpctl", "dnf") == "sudo dnf install wireplumber"
    assert lx.install_command("pactl", "pacman") == "sudo pacman -S libpulse"
    assert lx.install_command("portaudio", "apt") == "sudo apt install libportaudio2"
    assert lx.install_command("nothing", "apt") == ""


def test_need_names_the_package(apt):
    err = lx.need("wmctrl", "listing windows needs wmctrl")
    assert isinstance(err, PlatformUnsupported)
    assert "install wmctrl" in str(err) and "sudo apt install wmctrl" in str(err)


# ----------------------------------------------------------------- .desktop files
FIREFOX = """\
[Desktop Entry]
Version=1.0
Type=Application
Name=Firefox Web Browser
Name[de]=Firefox Webbrowser
Name[en_GB]=Firefox Browser
Exec=firefox %u
Icon=firefox
Categories=Network;WebBrowser;

[Desktop Action new-window]
Name=New Window
Exec=firefox --new-window %u
"""


def write_desktop(directory: Path, name: str, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text, encoding="utf-8")


def entry(name: str, exec_line: str = "app %U", extra: str = "") -> str:
    return f"[Desktop Entry]\nType=Application\nName={name}\nExec={exec_line}\n{extra}"


def test_parse_desktop_entry_localised_names():
    assert lx.parse_desktop_entry(FIREFOX)["Name"] == "Firefox Web Browser"
    assert lx.parse_desktop_entry(FIREFOX, ["de"])["Name"] == "Firefox Webbrowser"
    assert lx.parse_desktop_entry(FIREFOX, ["en_GB", "en"])["Name"] == "Firefox Browser"
    assert lx.parse_desktop_entry(FIREFOX, ["fr"])["Name"] == "Firefox Web Browser"
    # keys of later groups (desktop actions) never leak into the main entry
    assert lx.parse_desktop_entry(FIREFOX)["Exec"] == "firefox %u"
    assert lx.parse_desktop_entry("not a desktop file") is None


@pytest.mark.parametrize(
    "lang, keys",
    [("en_GB.UTF-8", ["en_GB", "en"]), ("de_DE.UTF-8@euro", ["de_DE", "de"]), ("fr", ["fr"]), ("C", []), ("", [])],
)
def test_locale_keys(lang, keys):
    assert lx.locale_keys({"LANG": lang}) == keys
    assert lx.locale_keys({"LC_ALL": lang, "LANG": "xx_YY"}) == (keys if lang else ["xx_YY", "xx"])


@pytest.mark.parametrize(
    "exec_line, cleaned",
    [
        ("firefox %u", "firefox"),
        ("firefox %U", "firefox"),
        ("gimp-2.10 %F", "gimp-2.10"),
        ('"/opt/My App/run" --flag %f', "'/opt/My App/run' --flag"),
        ("app --icon=%i --name=%c %k", "app --icon= --name="),
        ("env GTK_THEME=x app %u", "env GTK_THEME=x app"),
        ("sh -c 'echo 100%% done'", "sh -c 'echo 100% done'"),
        ("app --file=%f", "app --file="),
        ("", ""),
        ("app 'unterminated", ""),
    ],
)
def test_clean_exec(exec_line, cleaned):
    assert lx.clean_exec(exec_line) == cleaned


def test_scan_desktop_files(tmp_path):
    user, system = tmp_path / "user", tmp_path / "system"
    write_desktop(system, "firefox.desktop", FIREFOX)
    write_desktop(system, "hidden-app.desktop", entry("Hidden App", extra="NoDisplay=true\n"))
    write_desktop(system, "gone.desktop", entry("Gone App", extra="Hidden=true\n"))
    write_desktop(system, "htop.desktop", entry("htop", "htop", "Terminal=true\n"))
    write_desktop(system, "link.desktop", "[Desktop Entry]\nType=Link\nName=A Link\nURL=http://x\n")
    write_desktop(system, "broken.desktop", entry("Broken", extra="TryExec=/definitely/not/here\n"))
    write_desktop(system, "uninstall.desktop", entry("Uninstall Thing"))
    write_desktop(system, "noexec.desktop", "[Desktop Entry]\nType=Application\nName=No Exec\n")
    write_desktop(system, "org.gnome.Calculator.desktop", entry("Calculator", "gnome-calculator"))
    write_desktop(system / "kde", "dolphin.desktop", entry("Dolphin", "dolphin %u"))
    # the user directory comes first: its Firefox wins, and its Hidden entry removes Gone App
    write_desktop(user, "firefox.desktop", entry("My Firefox", "firefox-custom %u"))
    write_desktop(user, "gone.desktop", entry("Gone App", extra="Hidden=true\n"))
    entries = {e.name: e for e in lx.scan_desktop_files([user, system, tmp_path / "does-not-exist"], ["en_GB", "en"])}
    assert sorted(entries) == ["Calculator", "Dolphin", "My Firefox"]
    assert entries["My Firefox"].kind == "desktop" and entries["My Firefox"].target == "firefox.desktop"
    assert entries["My Firefox"].extra == "firefox-custom"
    assert entries["Dolphin"].target == "kde-dolphin.desktop"
    assert entries["Calculator"].target == "org.gnome.Calculator.desktop"


def test_scan_uses_localised_name(tmp_path):
    write_desktop(tmp_path, "firefox.desktop", FIREFOX)
    names = [e.name for e in lx.scan_desktop_files([tmp_path], ["de"])]
    assert names == ["Firefox Webbrowser"]


def test_try_exec_that_exists_is_kept(tmp_path, cmds):
    cmds.add("present-tool")
    write_desktop(tmp_path, "a.desktop", entry("Present", extra="TryExec=present-tool\n"))
    write_desktop(tmp_path, "b.desktop", entry("Absent", extra="TryExec=absent-tool\n"))
    assert [e.name for e in lx.scan_desktop_files([tmp_path])] == ["Present"]


def test_application_dirs_order_and_flatpak_snap(tmp_path):
    env = {"XDG_DATA_HOME": str(tmp_path / "dh"), "XDG_DATA_DIRS": "/opt/a:/opt/b"}
    dirs = lx.application_dirs(env, tmp_path)
    assert dirs[0] == tmp_path / "dh" / "applications"
    assert dirs.index(Path("/opt/a/applications")) < dirs.index(Path("/opt/b/applications"))
    for expected in (
        "/usr/share/applications", "/usr/local/share/applications",
        "/var/lib/flatpak/exports/share/applications", "/var/lib/snapd/desktop/applications",
    ):
        assert Path(expected) in dirs
    assert tmp_path / ".local" / "share" / "flatpak" / "exports" / "share" / "applications" in dirs
    assert len(dirs) == len(set(dirs))
    defaults = lx.application_dirs({}, tmp_path)
    assert defaults[0] == tmp_path / ".local" / "share" / "applications"


def test_launch_with_gtk_launch(linux, cmds):
    cmds.add("gtk-launch")
    linux.launch_app(AppEntry("Firefox", "desktop", "firefox.desktop", "firefox"))
    assert cmds.popens == [["gtk-launch", "firefox.desktop"]]


def test_launch_falls_back_to_the_exec_line(linux, cmds):
    linux.launch_app(AppEntry("Firefox", "desktop", "firefox.desktop", "firefox --new-window"))
    assert cmds.popens == [["firefox", "--new-window"]]
    cmds.popens.clear()
    cmds.add("gtk-launch")
    cmds.popen_codes["gtk-launch"] = 1  # gtk-launch exists but does not know the id
    linux.launch_app(AppEntry("Firefox", "desktop", "firefox.desktop", "firefox"))
    assert cmds.popens == [["gtk-launch", "firefox.desktop"], ["firefox"]]


def test_launch_without_a_command_fails(linux, cmds):
    with pytest.raises(OSError, match="no command"):
        linux.launch_app(AppEntry("Ghost", "desktop", "ghost.desktop", ""))


def test_scan_apps_uses_xdg_dirs(linux, tmp_path, monkeypatch):
    data = tmp_path / "share"
    write_desktop(data / "applications", "a.desktop", entry("Alpha", "alpha"))
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path / "none"))
    assert "Alpha" in [e.name for e in linux.scan_apps()]


# ----------------------------------------------------------------- wmctrl / windows
WMCTRL = """\
0x01c00003 -1 1021   myhost Desktop
0x03a00004  0 2231   myhost Untitled Document 1 - gedit
0x04200006  1 3302   myhost Inbox - user@example.com - Mozilla Thunderbird
0x04400007  0 0      myhost Window of an app without _NET_WM_PID
0x04600008  0 4410   myhost
"""


def test_parse_wmctrl():
    rows = lx.parse_wmctrl(WMCTRL)
    assert rows[0] == (0x01C00003, -1, 1021, "Desktop")
    assert rows[1] == (0x03A00004, 0, 2231, "Untitled Document 1 - gedit")
    assert rows[2][3] == "Inbox - user@example.com - Mozilla Thunderbird"
    assert rows[3][2] == 0
    assert lx.parse_wmctrl("garbage\n\n") == []


def test_windows_lists_through_wmctrl(linux, cmds, x11, monkeypatch):
    cmds.add("wmctrl", lambda argv: (0, WMCTRL, ""))
    import psutil

    class Proc:
        def __init__(self, pid):
            if pid == 3302:
                raise psutil.NoSuchProcess(pid)
            self.pid = pid

        def name(self):
            return {1021: "gnome-shell", 2231: "gedit"}[self.pid]

    monkeypatch.setattr(psutil, "Process", Proc)
    rows = linux.windows()
    assert ["wmctrl", "-lp"] in cmds.calls
    assert ("gedit", "Untitled Document 1 - gedit", 2231, 0x03A00004) in rows
    assert ("?", "Inbox - user@example.com - Mozilla Thunderbird", 3302, 0x04200006) in rows
    assert all(title for _, title, _, _ in rows)  # the untitled window is dropped


def test_windows_not_supported_on_wayland(linux, cmds, wayland):
    cmds.add("wmctrl", lambda argv: (0, WMCTRL, ""))
    with pytest.raises(PlatformUnsupported, match="Wayland"):
        linux.windows()
    with pytest.raises(PlatformUnsupported, match="Wayland"):
        linux.close_window(1)
    assert cmds.calls == []  # never guessed


def test_windows_needs_a_display(linux, cmds, headless):
    cmds.add("wmctrl")
    with pytest.raises(PlatformUnsupported, match="graphical session"):
        linux.windows()


def test_windows_missing_wmctrl_names_the_package(linux, cmds, x11, apt):
    with pytest.raises(PlatformUnsupported, match="install wmctrl"):
        linux.windows()


def test_close_window_uses_wmctrl_close_by_id(linux, cmds, x11):
    cmds.add("wmctrl", lambda argv: (0, "", ""))
    linux.close_window(0x03A00004)
    assert cmds.calls == [["wmctrl", "-ic", "0x3a00004"]]
    cmds.handlers["wmctrl"] = lambda argv: (1, "", "Cannot find window")
    with pytest.raises(OSError, match="Cannot find window"):
        linux.close_window(5)


def test_close_never_kills_processes(linux, cmds, x11):
    cmds.add("wmctrl", lambda argv: (0, "", ""))
    linux.close_window(1)
    assert all(c[0] == "wmctrl" for c in cmds.calls)


def test_foreground_window_title(linux, cmds, x11):
    assert linux.foreground_window_title() == ""  # no xdotool
    cmds.add("xdotool", lambda argv: (0, "Terminal - bash\n", ""))
    assert linux.foreground_window_title() == "Terminal - bash"


def test_foreground_title_empty_on_wayland(linux, cmds, wayland):
    cmds.add("xdotool", lambda argv: (0, "x", ""))
    assert linux.foreground_window_title() == ""


def test_cursor_position(linux, cmds, x11):
    assert linux.cursor_position() is None
    cmds.add("xdotool", lambda argv: (0, "X=1280\nY=-40\nSCREEN=0\nWINDOW=123\n", ""))
    assert linux.cursor_position() == (1280, -40)
    cmds.handlers["xdotool"] = lambda argv: (0, "nonsense", "")
    assert linux.cursor_position() is None


# ----------------------------------------------------------------- volume
WPCTL_OK = "Volume: 0.40\n"
WPCTL_MUTED = "Volume: 0.65 [MUTED]\n"
PACTL_VOL = (
    "Volume: front-left: 26214 /  40% / -23.88 dB,   front-right: 26214 /  40% / -23.88 dB\n"
    "        balance 0.00\n"
)
AMIXER_MONO = (
    "Simple mixer control 'Master',0\n"
    "  Capabilities: pvolume pvolume-joined pswitch pswitch-joined\n"
    "  Playback channels: Mono\n"
    "  Limits: Playback 0 - 87\n"
    "  Mono: Playback 52 [60%] [-20.00dB] [on]\n"
)
AMIXER_STEREO_OFF = (
    "Simple mixer control 'Master',0\n"
    "  Front Left: Playback 40 [46%] [off]\n"
    "  Front Right: Playback 40 [46%] [off]\n"
)


def test_parse_wpctl():
    assert lx.parse_wpctl(WPCTL_OK) == (40, False)
    assert lx.parse_wpctl(WPCTL_MUTED) == (65, True)
    assert lx.parse_wpctl("Volume: 1.00") == (100, False)
    with pytest.raises(OSError):
        lx.parse_wpctl("nonsense")


def test_parse_pactl():
    assert lx.parse_pactl_volume(PACTL_VOL) == 40
    assert lx.parse_pactl_mute("Mute: yes\n") is True
    assert lx.parse_pactl_mute("Mute: no\n") is False
    with pytest.raises(OSError):
        lx.parse_pactl_volume("")
    with pytest.raises(OSError):
        lx.parse_pactl_mute("")


def test_parse_amixer():
    assert lx.parse_amixer(AMIXER_MONO) == (60, False)
    assert lx.parse_amixer(AMIXER_STEREO_OFF) == (46, True)
    with pytest.raises(OSError):
        lx.parse_amixer("Simple mixer control")


def test_volume_with_wpctl(linux, cmds):
    state = {"vol": "0.40", "muted": False}

    def wpctl(argv):
        if argv[1] == "get-volume":
            return 0, f"Volume: {state['vol']}{' [MUTED]' if state['muted'] else ''}\n", ""
        if argv[1] == "set-volume":
            state["vol"] = f"{int(argv[3].rstrip('%')) / 100:.2f}"
        if argv[1] == "set-mute":
            state["muted"] = argv[3] == "1"
        return 0, "", ""

    cmds.add("wpctl", wpctl)
    assert linux.get_volume() == (40, False)
    assert linux.set_volume(75) == (75, False)
    assert ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "75%"] in cmds.calls
    assert linux.set_mute(True) == (75, True)
    assert ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1"] in cmds.calls


def test_volume_with_pactl(linux, cmds):
    state = {"pct": 40, "mute": "no"}

    def pactl(argv):
        if argv[1] == "get-sink-volume":
            return 0, PACTL_VOL.replace("40%", f"{state['pct']}%"), ""
        if argv[1] == "get-sink-mute":
            return 0, f"Mute: {state['mute']}\n", ""
        if argv[1] == "set-sink-volume":
            state["pct"] = int(argv[3].rstrip("%"))
        if argv[1] == "set-sink-mute":
            state["mute"] = "yes" if argv[3] == "1" else "no"
        return 0, "", ""

    cmds.add("pactl", pactl)
    assert linux.get_volume() == (40, False)
    assert linux.set_volume(20) == (20, False)
    assert ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "20%"] in cmds.calls
    assert linux.set_mute(True) == (20, True)


def test_volume_with_amixer(linux, cmds):
    state = {"pct": 60, "state": "on"}

    def amixer(argv):
        if argv[1] == "sget":
            return 0, AMIXER_MONO.replace("[60%]", f"[{state['pct']}%]").replace("[on]", f"[{state['state']}]"), ""
        if argv[2] == "sset" and argv[4].endswith("%"):
            state["pct"] = int(argv[4].rstrip("%"))
        elif argv[4] in ("mute", "unmute"):
            state["state"] = "off" if argv[4] == "mute" else "on"
        return 0, "", ""

    cmds.add("amixer", amixer)
    assert linux.get_volume() == (60, False)
    assert linux.set_volume(30) == (30, False)
    assert ["amixer", "-q", "sset", "Master", "30%"] in cmds.calls
    assert linux.set_mute(True) == (30, True)


def test_volume_prefers_wpctl_then_pactl_then_amixer(linux, cmds):
    def answer(argv):
        if argv[0] == "wpctl":
            return 0, WPCTL_OK, ""
        if argv[0] == "pactl":
            return 0, "Mute: no\n" if argv[1] == "get-sink-mute" else PACTL_VOL, ""
        return 0, AMIXER_MONO, ""

    for name in ("wpctl", "pactl", "amixer"):
        cmds.add(name, answer)
    assert linux.get_volume() == (40, False)
    assert cmds.calls[0][0] == "wpctl"
    cmds.calls.clear()
    cmds.handlers["wpctl"] = lambda argv: (1, "", "PipeWire not running")
    assert linux.get_volume() == (40, False)  # pactl answered after wpctl failed
    assert [c[0] for c in cmds.calls][:2] == ["wpctl", "pactl"]
    cmds.handlers["pactl"] = lambda argv: (1, "", "Connection refused")
    assert linux.get_volume() == (60, False)  # amixer is the last resort


def test_volume_all_backends_failing_reports_each(linux, cmds):
    cmds.add("wpctl", lambda argv: (1, "", "no pipewire"))
    cmds.add("amixer", lambda argv: (1, "", "no card"))
    with pytest.raises(OSError) as exc:
        linux.get_volume()
    assert "no pipewire" in str(exc.value) and "no card" in str(exc.value)


def test_volume_without_any_tool_names_packages(linux, cmds, apt):
    with pytest.raises(PlatformUnsupported) as exc:
        linux.set_volume(10)
    text = str(exc.value)
    assert "wpctl" in text and "pactl" in text and "amixer" in text and "install wireplumber" in text


# ----------------------------------------------------------------- media / lock / clipboard
@pytest.mark.parametrize(
    "action, arg", [("play_pause", "play-pause"), ("next", "next"), ("previous", "previous"), ("stop", "stop")]
)
def test_media_key_maps_to_playerctl(linux, cmds, action, arg):
    cmds.add("playerctl", lambda argv: (0, "", ""))
    linux.media_key(action)
    assert cmds.calls == [["playerctl", arg]]


def test_media_key_errors(linux, cmds, apt):
    with pytest.raises(PlatformUnsupported, match="install playerctl"):
        linux.media_key("next")
    cmds.add("playerctl", lambda argv: (1, "", "No players found"))
    with pytest.raises(OSError, match="No players found"):
        linux.media_key("next")


def test_lock_prefers_loginctl_then_screensaver(linux, cmds):
    cmds.add("loginctl", lambda argv: (0, "", ""))
    cmds.add("xdg-screensaver", lambda argv: (0, "", ""))
    linux.lock_screen()
    assert cmds.calls == [["loginctl", "lock-session"]]
    cmds.calls.clear()
    cmds.handlers["loginctl"] = lambda argv: (1, "", "Could not get session")
    linux.lock_screen()
    assert cmds.calls == [["loginctl", "lock-session"], ["xdg-screensaver", "lock"]]


def test_lock_failures(linux, cmds):
    with pytest.raises(PlatformUnsupported, match="loginctl"):
        linux.lock_screen()
    cmds.add("loginctl", lambda argv: (1, "", "denied"))
    with pytest.raises(OSError, match="denied"):
        linux.lock_screen()


def test_clipboard_problem_matrix(linux, cmds, monkeypatch, apt):
    for var in ("XDG_SESSION_TYPE", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)
    assert "graphical session" in linux.clipboard_problem()
    monkeypatch.setenv("DISPLAY", ":0")
    assert "install xclip" in linux.clipboard_problem()
    cmds.add("xsel")
    assert linux.clipboard_problem() is None
    cmds.present.clear()
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY")
    assert "install wl-clipboard" in linux.clipboard_problem()
    cmds.add("wl-copy")
    cmds.add("wl-paste")
    assert linux.clipboard_problem() is None
    cmds.present.clear()
    monkeypatch.setenv("DISPLAY", ":0")
    cmds.add("xclip")  # XWayland clipboard
    assert linux.clipboard_problem() is None


# ----------------------------------------------------------------- opening things
def test_open_path_uses_xdg_open(linux, cmds):
    cmds.add("xdg-open")
    linux.open_path("/home/u/report.pdf")
    assert cmds.popens == [["xdg-open", "/home/u/report.pdf"]]


def test_open_url_returns_true_and_uses_xdg_open(linux, cmds):
    cmds.add("xdg-open")
    assert linux.open_url("https://example.com") is True
    assert cmds.popens == [["xdg-open", "https://example.com"]]


def test_open_falls_back_to_gio(linux, cmds):
    cmds.add("gio")
    linux.open_path("/x")
    assert cmds.popens == [["gio", "open", "/x"]]


def test_open_failure_and_missing_helper(linux, cmds, apt):
    with pytest.raises(PlatformUnsupported, match="install xdg-utils"):
        linux.open_path("/x")
    cmds.add("xdg-open")
    cmds.popen_codes["xdg-open"] = 3
    with pytest.raises(OSError, match="exit code 3"):
        linux.open_path("/x")
    cmds.popen_codes["xdg-open"] = 0
    linux.open_path("/x")  # exiting 0 quickly is success


# ----------------------------------------------------------------- known folders
def test_known_folder_uses_xdg_user_dir(linux, cmds, tmp_path):
    answers = {"DESKTOP": str(tmp_path / "Schreibtisch"), "DOWNLOAD": str(tmp_path / "Herunterladen")}
    cmds.add("xdg-user-dir", lambda argv: (0, answers.get(argv[1], str(Path.home())) + "\n", ""))
    assert linux.known_folder("Desktop") == tmp_path / "Schreibtisch"
    assert linux.known_folder("Downloads") == tmp_path / "Herunterladen"
    assert ["xdg-user-dir", "DOWNLOAD"] in cmds.calls
    # unconfigured folders make xdg-user-dir answer $HOME: use ~/Name instead
    assert linux.known_folder("Music") == Path.home() / "Music"


def test_known_folder_fallbacks(linux, cmds):
    assert linux.known_folder("Documents") == Path.home() / "Documents"  # no xdg-user-dir
    cmds.add("xdg-user-dir", lambda argv: (0, "", ""))
    assert linux.known_folder("Documents") == Path.home() / "Documents"
    assert linux.known_folder("Mystery") == Path.home() / "Mystery"
    cmds.handlers["xdg-user-dir"] = lambda argv: (_ for _ in ()).throw(OSError("boom"))
    assert linux.known_folder("Pictures") == Path.home() / "Pictures"


# ----------------------------------------------------------------- protected paths, ollama, misc
@pytest.mark.parametrize(
    "path, protected",
    [
        ("/usr", True), ("/usr/lib/python3", True), ("/etc/passwd", True), ("/bin/ls", True),
        ("/var/lib/dpkg", True), ("/boot", True), ("/root", True), ("/opt/app", True),
        ("/var/tmp/scratch", False), ("/run/user/1000/x", False), ("/run/media/me/disk", False),
        ("/home/me/Documents", False), ("/tmp/work", False), ("/mnt/data", False), ("/media/usb", False),
        ("/usrx/other", False), ("/etcetera", False),
    ],
)
def test_protected_path_reason(linux, path, protected):
    assert (linux.protected_path_reason(path) is not None) is protected


def test_ollama_paths(linux):
    assert Path("/usr/local/bin/ollama") in linux.ollama_extra_paths()


def test_executable_suffixes_and_open_risk(fake_platform, monkeypatch):
    from jarvis.tools.base import Risk
    from jarvis.tools.builtin import files

    monkeypatch.setattr(fake_platform, "executable_suffixes", LinuxPlatform.executable_suffixes, raising=False)
    assert files._open_risk({"path": "/home/u/run.sh"}) is Risk.CONFIRM
    assert files._open_risk({"path": "/home/u/app.AppImage"}) is Risk.CONFIRM
    assert files._open_risk({"path": "/home/u/launcher.desktop"}) is Risk.CONFIRM
    assert files._open_risk({"path": "/home/u/report.pdf"}) is Risk.SAFE


def test_detached_and_subprocess_kwargs(linux):
    assert linux.detached_popen_kwargs() == {"start_new_session": True}
    assert linux.subprocess_kwargs() == {"start_new_session": True}
    assert linux.reserved_device_names is False and linux.shell_kind == "bash"


def test_describe_names_distro_and_session(linux, x11, apt, monkeypatch):
    monkeypatch.setattr(lx, "read_os_release", lambda path="": {"PRETTY_NAME": "Ubuntu 24.04 LTS", "ID": "ubuntu"})
    assert linux.describe() == "Ubuntu 24.04 LTS (X11)"


# ----------------------------------------------------------------- screen
def test_screen_grab_x11_defers_to_mss(linux, x11):
    assert linux.screen_grab() is None


def test_screen_grab_headless_fails_clearly(linux, headless):
    with pytest.raises(PlatformUnsupported, match="graphical session"):
        linux.screen_grab()


def png_bytes() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_screen_grab_wayland_uses_grim(linux, wayland, monkeypatch):
    shot = png_bytes()
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout=shot, stderr=b"")

    monkeypatch.setattr(lx.shutil, "which", lambda n: "/usr/bin/grim" if n == "grim" else None)
    monkeypatch.setattr(lx.subprocess, "run", run)
    image = linux.screen_grab()
    assert image.size == (8, 6) and seen == [["grim", "-"]]


def test_screen_grab_wayland_falls_back_to_gnome_screenshot(linux, wayland, monkeypatch):
    shot = png_bytes()

    def run(argv, **kwargs):
        if argv[0] == "grim":
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"compositor does not support it")
        Path(argv[2]).write_bytes(shot)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(lx.shutil, "which", lambda n: f"/usr/bin/{n}" if n in ("grim", "gnome-screenshot") else None)
    monkeypatch.setattr(lx.subprocess, "run", run)
    assert linux.screen_grab().size == (8, 6)


def test_screen_grab_wayland_without_tools(linux, wayland, cmds, apt):
    with pytest.raises(PlatformUnsupported, match="install grim"):
        linux.screen_grab()


def test_capture_screen_reports_unsupported_as_capture_error(monkeypatch, fake_platform):
    from jarvis.vision import capture

    def refuse():
        raise PlatformUnsupported("screen capture needs a graphical session; DISPLAY is not set")

    monkeypatch.setattr(fake_platform, "screen_grab", refuse)
    pytest.importorskip("mss")
    pytest.importorskip("PIL")
    with pytest.raises(capture.CaptureError, match="graphical session"):
        capture.capture_screen()


# ----------------------------------------------------------------- doctor helper table
def test_helper_report_lists_what_is_missing(linux, cmds, apt):
    rows = {name: (ok, fix) for name, ok, fix in linux.helper_report()}
    assert {"xdg-open", "wmctrl", "wpctl/pactl/amixer", "playerctl", "xclip or wl-clipboard", "libportaudio"} <= set(rows)
    assert rows["wmctrl"] == (False, "sudo apt install wmctrl")
    cmds.add("wmctrl")
    cmds.add("pactl")
    rows = {name: ok for name, ok, _ in linux.helper_report()}
    assert rows["wmctrl"] is True and rows["wpctl/pactl/amixer"] is True and rows["playerctl"] is False
