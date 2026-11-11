"""Linux backend: what an ordinary desktop provides, degrading with precise messages.

Everything runs through small helper programs (xdg-open, wmctrl, wpctl, playerctl, ...).
When one is missing the failure text names the package to install. The parsers are plain
functions so they are unit-tested against real command output on any OS.

Status: implemented, tested with mocked subprocess output and in CI, not yet tried on a
real Linux desktop.
"""

from __future__ import annotations

import ctypes.util
import io
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.platform.base import (
    JUNK_NAME,
    AppEntry,
    BasePlatform,
    PlatformUnsupported,
    UiSnapshot,
    WindowInfo,
    WindowRow,
    run_quiet,
)

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ session and distro
def session_type() -> str:
    """'wayland', 'x11' or 'headless' (no graphical session variables at all)."""
    declared = os.environ.get("XDG_SESSION_TYPE", "").lower()
    wayland, x11 = bool(os.environ.get("WAYLAND_DISPLAY")), bool(os.environ.get("DISPLAY"))
    if declared == "wayland" and (wayland or x11):
        return "wayland"
    if declared == "x11" and x11:
        return "x11"
    if wayland:
        return "wayland"
    return "x11" if x11 else "headless"


def read_os_release(path: str = "/etc/os-release") -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return data
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key and not key.startswith("#"):
            data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def package_manager(os_release: dict[str, str] | None = None) -> str | None:
    """'apt', 'dnf', 'pacman' or 'zypper' from the distribution ids, else None."""
    info = os_release if os_release is not None else read_os_release()
    ids = [info.get("ID", "").lower(), *info.get("ID_LIKE", "").lower().split()]
    for ident in ids:
        if ident in ("debian", "ubuntu", "linuxmint", "pop", "raspbian", "kali", "elementary"):
            return "apt"
        if ident in ("fedora", "rhel", "centos", "rocky", "almalinux", "nobara"):
            return "dnf"
        if ident in ("arch", "manjaro", "endeavouros", "cachyos"):
            return "pacman"
        if ident in ("opensuse", "suse", "sles", "opensuse-leap", "opensuse-tumbleweed"):
            return "zypper"
    return None


@dataclass(frozen=True)
class Helper:
    purpose: str
    apt: str
    dnf: str
    pacman: str
    zypper: str = ""


HELPERS: dict[str, Helper] = {
    "xdg-open": Helper("open files and links", "xdg-utils", "xdg-utils", "xdg-utils", "xdg-utils"),
    "wmctrl": Helper("list and close windows (X11)", "wmctrl", "wmctrl", "wmctrl", "wmctrl"),
    "xdotool": Helper("foreground window, cursor and desktop input (X11)", "xdotool", "xdotool", "xdotool", "xdotool"),
    "wpctl": Helper("volume (PipeWire)", "wireplumber", "wireplumber", "wireplumber", "wireplumber"),
    "pactl": Helper("volume (PulseAudio)", "pulseaudio-utils", "pulseaudio-utils", "libpulse", "pulseaudio-utils"),
    "amixer": Helper("volume (ALSA)", "alsa-utils", "alsa-utils", "alsa-utils", "alsa-utils"),
    "playerctl": Helper("media keys", "playerctl", "playerctl", "playerctl", "playerctl"),
    "xclip": Helper("clipboard (X11)", "xclip", "xclip", "xclip", "xclip"),
    "wl-clipboard": Helper("clipboard (Wayland)", "wl-clipboard", "wl-clipboard", "wl-clipboard", "wl-clipboard"),
    "portaudio": Helper("microphone and speaker", "libportaudio2", "portaudio", "portaudio", "libportaudio2"),
    "grim": Helper("screenshots (Wayland)", "grim", "grim", "grim", "grim"),
    "gtk-launch": Helper("launching apps", "libgtk-3-bin", "gtk3", "gtk3", "gtk3-tools"),
    "soffice": Helper("recalculating spreadsheets", "libreoffice-calc", "libreoffice-calc", "libreoffice-fresh", "libreoffice-calc"),
}

_MANAGER_PREFIX = {
    "apt": "sudo apt install",
    "dnf": "sudo dnf install",
    "pacman": "sudo pacman -S",
    "zypper": "sudo zypper install",
}


def install_command(helper: str, manager: str | None = None) -> str:
    """'sudo apt install wmctrl' for this distribution, or '' when it is not known."""
    spec = HELPERS.get(helper)
    manager = manager or package_manager()
    if spec is None or manager is None:
        return ""
    package = getattr(spec, manager, "")
    return f"{_MANAGER_PREFIX[manager]} {package}" if package else ""


def need(helper: str, why: str) -> PlatformUnsupported:
    """The failure for a missing helper: names the package and, if known, the command."""
    spec = HELPERS.get(helper)
    package = None
    if spec is not None:
        manager = package_manager()
        package = getattr(spec, manager, "") if manager else spec.apt
    cmd = install_command(helper)
    extra = f" ({cmd})" if cmd else ""
    return PlatformUnsupported(f"{why}: install {package or helper}{extra}")


# ------------------------------------------------------------------ .desktop files
_FIELD_CODE = re.compile(r"%[a-zA-Z]")


def clean_exec(exec_line: str) -> str:
    """Drop the freedesktop field codes (%f %U %i ...) from an Exec line; ``%%`` becomes ``%``."""
    try:
        tokens = shlex.split(exec_line)
    except ValueError:
        return ""
    out: list[str] = []
    for token in tokens:
        if _FIELD_CODE.fullmatch(token):
            continue
        token = token.replace("%%", "\0")
        token = _FIELD_CODE.sub("", token).replace("\0", "%")
        out.append(token)
    return shlex.join(out)


def locale_keys(env: dict[str, str] | None = None) -> list[str]:
    """Localised-key suffixes to try, best first: ['en_GB', 'en'] for LANG=en_GB.UTF-8."""
    env = os.environ if env is None else env
    raw = env.get("LC_ALL") or env.get("LC_MESSAGES") or env.get("LANG") or ""
    raw = raw.split(".")[0].split("@")[0]
    if not raw or raw in ("C", "POSIX"):
        return []
    keys = [raw]
    if "_" in raw:
        keys.append(raw.split("_")[0])
    return keys


def parse_desktop_entry(text: str, locales: list[str] | None = None) -> dict[str, str] | None:
    """Fields of the ``[Desktop Entry]`` group, with ``Name`` replaced by its best
    localised value. None when there is no such group."""
    locales = locales or []
    fields: dict[str, str] = {}
    localised: dict[str, str] = {}
    in_group = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            if in_group:
                break
            in_group = line == "[Desktop Entry]"
            continue
        if not in_group:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        match = re.fullmatch(r"Name\[([^\]]+)\]", key)
        if match:
            localised[match.group(1)] = value
        else:
            fields[key] = value
    if not in_group and not fields:
        return None
    for loc in locales:
        if loc in localised:
            fields["Name"] = localised[loc]
            break
    return fields


def application_dirs(env: dict[str, str] | None = None, home: Path | None = None) -> list[Path]:
    """XDG application directories in priority order, plus flatpak and snap exports."""
    env = os.environ if env is None else env
    home = home or Path.home()
    data_home = Path(env.get("XDG_DATA_HOME") or home / ".local" / "share")
    data_dirs = [Path(p) for p in (env.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":") if p]
    candidates = [
        data_home / "applications",
        home / ".local" / "share" / "flatpak" / "exports" / "share" / "applications",
        *(d / "applications" for d in data_dirs),
        Path("/usr/local/share/applications"),
        Path("/usr/share/applications"),
        Path("/var/lib/flatpak/exports/share/applications"),
        Path("/var/lib/snapd/desktop/applications"),
    ]
    seen: set[Path] = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


def scan_desktop_files(dirs: list[Path], locales: list[str] | None = None) -> list[AppEntry]:
    """Launchable applications from .desktop files. The first directory that has a given
    desktop id wins; NoDisplay/Hidden entries are left out (and hide lower-priority copies)."""
    seen_ids: set[str] = set()
    entries: list[AppEntry] = []
    for base in dirs:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.desktop")):
            desktop_id = "-".join(path.relative_to(base).parts)
            if desktop_id in seen_ids:
                continue
            seen_ids.add(desktop_id)
            try:
                fields = parse_desktop_entry(path.read_text(encoding="utf-8", errors="replace"), locales)
            except OSError:
                continue
            if not fields:
                continue
            if fields.get("Type", "Application") != "Application":
                continue
            if _truthy(fields.get("NoDisplay")) or _truthy(fields.get("Hidden")) or _truthy(fields.get("Terminal")):
                continue
            name, exec_line = fields.get("Name", "").strip(), fields.get("Exec", "")
            if not name or not exec_line or JUNK_NAME.search(name):
                continue
            try_exec = fields.get("TryExec")
            if try_exec and not (shutil.which(try_exec) or Path(try_exec).is_file()):
                continue
            entries.append(AppEntry(name, "desktop", desktop_id, clean_exec(exec_line)))
    return entries


# ------------------------------------------------------------------ wmctrl
_WMCTRL_LINE = re.compile(r"^(0x[0-9a-fA-F]+)\s+(-?\d+)\s+(\d+)\s+(\S+)\s+(.*)$")


def parse_wmctrl(output: str) -> list[tuple[int, int, int, str]]:
    """(window id, desktop, pid, title) per line of ``wmctrl -lp``."""
    rows = []
    for line in output.splitlines():
        match = _WMCTRL_LINE.match(line.rstrip())
        if match:
            rows.append((int(match.group(1), 16), int(match.group(2)), int(match.group(3)), match.group(5).strip()))
    return rows


# ------------------------------------------------------------------ volume parsers
def parse_wpctl(output: str) -> tuple[int, bool]:
    """'Volume: 0.40' or 'Volume: 0.40 [MUTED]' -> (40, muted)."""
    match = re.search(r"Volume:\s*([0-9]*\.?[0-9]+)", output)
    if not match:
        raise OSError(f"unexpected wpctl output: {output.strip()[:80]!r}")
    return round(float(match.group(1)) * 100), "[MUTED]" in output


def parse_pactl_volume(output: str) -> int:
    """First percentage in 'Volume: front-left: 26214 /  40% / -23.88 dB, ...'."""
    match = re.search(r"(\d+)%", output)
    if not match:
        raise OSError(f"unexpected pactl output: {output.strip()[:80]!r}")
    return int(match.group(1))


def parse_pactl_mute(output: str) -> bool:
    match = re.search(r"Mute:\s*(yes|no)", output, re.IGNORECASE)
    if not match:
        raise OSError(f"unexpected pactl output: {output.strip()[:80]!r}")
    return match.group(1).lower() == "yes"


def parse_amixer(output: str) -> tuple[int, bool]:
    """'Mono: Playback 52 [60%] [on]' or 'Front Left: Playback 52 [60%] [-20.00dB] [off]'."""
    pct = re.search(r"\[(\d+)%\]", output)
    if not pct:
        raise OSError(f"unexpected amixer output: {output.strip()[:80]!r}")
    state = re.search(r"\[(on|off)\]", output)
    return int(pct.group(1)), bool(state and state.group(1) == "off")


def _run_ok(argv: list[str], timeout: float = 10.0) -> str:
    """Run a helper; its stdout on success, OSError carrying its error text otherwise."""
    done = run_quiet(argv, timeout=timeout)
    if done.returncode != 0:
        raise OSError(f"{argv[0]} failed: {(done.stderr or done.stdout).strip()[:150] or done.returncode}")
    return done.stdout


class _Wpctl:
    name, helper = "wpctl", "wpctl"
    sink = "@DEFAULT_AUDIO_SINK@"

    def get(self) -> tuple[int, bool]:
        return parse_wpctl(_run_ok(["wpctl", "get-volume", self.sink]))

    def set_level(self, pct: int) -> None:
        _run_ok(["wpctl", "set-volume", self.sink, f"{pct}%"])

    def set_mute(self, muted: bool) -> None:
        _run_ok(["wpctl", "set-mute", self.sink, "1" if muted else "0"])


class _Pactl:
    name, helper = "pactl", "pactl"
    sink = "@DEFAULT_SINK@"

    def get(self) -> tuple[int, bool]:
        pct = parse_pactl_volume(_run_ok(["pactl", "get-sink-volume", self.sink]))
        return pct, parse_pactl_mute(_run_ok(["pactl", "get-sink-mute", self.sink]))

    def set_level(self, pct: int) -> None:
        _run_ok(["pactl", "set-sink-volume", self.sink, f"{pct}%"])

    def set_mute(self, muted: bool) -> None:
        _run_ok(["pactl", "set-sink-mute", self.sink, "1" if muted else "0"])


class _Amixer:
    name, helper = "amixer", "amixer"

    def get(self) -> tuple[int, bool]:
        return parse_amixer(_run_ok(["amixer", "sget", "Master"]))

    def set_level(self, pct: int) -> None:
        _run_ok(["amixer", "-q", "sset", "Master", f"{pct}%"])

    def set_mute(self, muted: bool) -> None:
        _run_ok(["amixer", "-q", "sset", "Master", "mute" if muted else "unmute"])



# canonical key names (jarvis.platform.keys) -> xdotool key names
_XDO_KEYS = {
    "ctrl": "ctrl", "alt": "alt", "shift": "shift", "win": "super",
    "enter": "Return", "tab": "Tab", "space": "space", "escape": "Escape",
    "backspace": "BackSpace", "delete": "Delete", "insert": "Insert", "home": "Home",
    "end": "End", "pageup": "Prior", "pagedown": "Next", "up": "Up", "down": "Down",
    "left": "Left", "right": "Right", "capslock": "Caps_Lock", "printscreen": "Print",
    "plus": "plus", "minus": "minus", "equals": "equal", "comma": "comma", "period": "period",
    "slash": "slash", "semicolon": "semicolon", "quote": "apostrophe", "backquote": "grave",
    "bracketleft": "bracketleft", "bracketright": "bracketright", "backslash": "backslash",
}

_MEDIA_PLAYERCTL = {"play_pause": "play-pause", "next": "next", "previous": "previous", "stop": "stop"}


# ------------------------------------------------------------------ the backend
class LinuxPlatform(BasePlatform):
    name = "linux"
    app_dir_name = "jarvis"
    shell_kind = "bash"
    shell_name = "bash"
    reserved_device_names = False
    executable_suffixes = frozenset({".sh", ".bash", ".desktop", ".appimage", ".run", ".bin"})

    def describe(self) -> str:
        info = read_os_release()
        distro = info.get("PRETTY_NAME") or "Linux"
        session = session_type()
        label = {"x11": "X11", "wayland": "Wayland", "headless": "no graphical session"}[session]
        return f"{distro} ({label})"

    # -- opening things ---------------------------------------------------
    def _spawn_detached(self, argv: list[str], settle: float = 1.5) -> int | None:
        """Start a program in its own session. Returns its exit code if it ended within
        ``settle`` seconds, None if it is still running (normal for a launched app)."""
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )
        try:
            return proc.wait(timeout=settle)
        except subprocess.TimeoutExpired:
            return None

    def _open_with_handler(self, target: str) -> None:
        for argv in (["xdg-open", target], ["gio", "open", target]):
            if shutil.which(argv[0]) is None:
                continue
            code = self._spawn_detached(argv)
            if code not in (None, 0):
                raise OSError(f"{argv[0]} could not open it (exit code {code})")
            return
        raise need("xdg-open", "opening files and links needs xdg-open (or gio)")

    def open_path(self, path: str) -> None:
        self._open_with_handler(path)

    def open_url(self, url: str) -> bool:
        self._open_with_handler(url)
        return True

    def known_folder(self, name: str) -> Path:
        fallback = Path.home() / name
        key = {"Downloads": "DOWNLOAD"}.get(name, name.upper())
        if name not in ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"):
            return fallback
        if shutil.which("xdg-user-dir") is None:
            return fallback
        try:
            out = run_quiet(["xdg-user-dir", key], timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return fallback
        # xdg-user-dir answers with $HOME itself when the folder is not configured.
        if not out or Path(out) == Path.home():
            return fallback
        return Path(out)

    # -- apps and windows -------------------------------------------------
    def scan_apps(self) -> list[AppEntry]:
        return scan_desktop_files(application_dirs(), locale_keys())

    def launch_app(self, entry: AppEntry) -> None:
        if entry.kind == "desktop" and shutil.which("gtk-launch"):
            code = self._spawn_detached(["gtk-launch", entry.target])
            if code in (None, 0):
                return
            log.info("gtk-launch %s failed with %s; running the Exec line", entry.target, code)
        command = entry.extra or (entry.target if entry.kind == "exe" else "")
        if not command:
            raise OSError(f"{entry.name} has no command to run")
        argv = shlex.split(command)
        if not argv:
            raise OSError(f"{entry.name} has an empty command")
        subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )

    def _require_wmctrl(self, what: str) -> None:
        session = session_type()
        if session == "wayland":
            raise PlatformUnsupported(
                f"{what} is not supported in a Wayland session (it needs X11 and wmctrl); "
                "log in with an X11 session to use it"
            )
        if session == "headless":
            raise PlatformUnsupported(f"{what} needs a graphical session; DISPLAY is not set")
        if shutil.which("wmctrl") is None:
            raise need("wmctrl", f"{what} needs wmctrl")

    def windows(self) -> list[WindowRow]:
        self._require_wmctrl("listing windows")
        import psutil

        rows: list[WindowRow] = []
        for wid, _desktop, pid, title in parse_wmctrl(_run_ok(["wmctrl", "-lp"])):
            if not title:
                continue
            try:
                proc = psutil.Process(pid).name() if pid else "?"
            except psutil.Error:
                proc = "?"
            rows.append((proc, title, pid, wid))
        return rows

    def close_window(self, handle: int) -> None:
        self._require_wmctrl("closing windows")
        _run_ok(["wmctrl", "-ic", f"0x{handle:x}"])

    def never_close_titles(self) -> set[str]:
        return {"desktop", "xfdesktop", "plasma"}

    def foreground_window_title(self) -> str:
        if session_type() != "x11" or shutil.which("xdotool") is None:
            return ""
        try:
            return run_quiet(["xdotool", "getactivewindow", "getwindowname"], timeout=3).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    # -- audio, media, session -------------------------------------------
    def _volume_backends(self) -> list[Any]:
        found = [b for b in (_Wpctl(), _Pactl(), _Amixer()) if shutil.which(b.helper)]
        if not found:
            raise PlatformUnsupported(
                "volume control needs wpctl (PipeWire), pactl (PulseAudio) or amixer (ALSA): "
                f"install {HELPERS['wpctl'].apt if package_manager() == 'apt' else 'wireplumber'}"
                + (f" ({install_command('wpctl')})" if install_command("wpctl") else "")
            )
        return found

    def _with_backend(self, op: str, *args: Any) -> tuple[int, bool]:
        errors: list[str] = []
        for backend in self._volume_backends():
            try:
                if op == "set_level":
                    backend.set_level(*args)
                elif op == "set_mute":
                    backend.set_mute(*args)
                return backend.get()
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{backend.name}: {exc}")
        raise OSError("; ".join(errors))

    def get_volume(self) -> tuple[int, bool]:
        return self._with_backend("get")

    def set_volume(self, percent: int) -> tuple[int, bool]:
        return self._with_backend("set_level", int(percent))

    def set_mute(self, muted: bool) -> tuple[int, bool]:
        return self._with_backend("set_mute", bool(muted))

    def media_key(self, action: str) -> None:
        if shutil.which("playerctl") is None:
            raise need("playerctl", "media keys need playerctl")
        _run_ok(["playerctl", _MEDIA_PLAYERCTL[action]])

    def lock_screen(self) -> None:
        errors = []
        for argv in (["loginctl", "lock-session"], ["xdg-screensaver", "lock"]):
            if shutil.which(argv[0]) is None:
                continue
            try:
                _run_ok(argv)
                return
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(str(exc))
        if errors:
            raise OSError("; ".join(errors))
        raise PlatformUnsupported("locking the screen needs loginctl (systemd) or xdg-screensaver: install xdg-utils")

    def clipboard_problem(self) -> str | None:
        session = session_type()
        if session == "headless":
            return "there is no clipboard without a graphical session (DISPLAY and WAYLAND_DISPLAY are not set)"
        has_x = shutil.which("xclip") or shutil.which("xsel")
        has_wl = shutil.which("wl-copy") and shutil.which("wl-paste")
        if session == "wayland":
            if has_wl or (has_x and os.environ.get("DISPLAY")):
                return None
            return str(need("wl-clipboard", "the clipboard needs wl-clipboard (Wayland)"))
        if has_x or has_wl:
            return None
        return str(need("xclip", "the clipboard needs xclip or xsel (X11)"))

    # -- files and processes ---------------------------------------------
    _SYSTEM_DIRS = (
        "/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/opt", "/proc",
        "/root", "/run", "/sbin", "/snap", "/srv", "/sys", "/usr", "/var",
    )
    _SYSTEM_EXCEPTIONS = ("/var/tmp", "/run/user", "/run/media")

    def protected_path_reason(self, real_path: str) -> str | None:
        for exempt in self._SYSTEM_EXCEPTIONS:
            if real_path.startswith(exempt + "/"):
                return None
        for base in self._SYSTEM_DIRS:
            if real_path == base or real_path.startswith(base + "/"):
                return f"it is inside the system folder {base}"
        return None

    def ollama_extra_paths(self) -> list[Path]:
        return [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama"), Path.home() / ".local" / "bin" / "ollama"]

    # -- screen -----------------------------------------------------------
    def screen_grab(self) -> "Image | None":
        session = session_type()
        if session == "headless":
            raise PlatformUnsupported("screen capture needs a graphical session; DISPLAY is not set")
        if session != "wayland":
            return None  # X11: mss reads the screen directly
        from PIL import Image

        if shutil.which("grim"):
            done = subprocess.run(["grim", "-"], capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
            if done.returncode == 0 and done.stdout:
                return Image.open(io.BytesIO(done.stdout)).convert("RGB")
            log.info("grim failed: %s", done.stderr.decode("utf-8", "replace")[:150])
        if shutil.which("gnome-screenshot"):
            fd, name = tempfile.mkstemp(prefix="jarvis-shot-", suffix=".png")
            os.close(fd)
            try:
                done = subprocess.run(
                    ["gnome-screenshot", "-f", name], capture_output=True, timeout=20, stdin=subprocess.DEVNULL
                )
                if done.returncode == 0 and Path(name).stat().st_size > 0:
                    with Image.open(name) as shot:
                        return shot.convert("RGB")
            finally:
                try:
                    os.unlink(name)
                except OSError:
                    pass
        raise PlatformUnsupported(
            "screen capture on Wayland needs grim or gnome-screenshot: "
            f"install grim{(' (' + install_command('grim') + ')') if install_command('grim') else ''}"
        )

    def cursor_position(self) -> tuple[int, int] | None:
        if session_type() != "x11" or shutil.which("xdotool") is None:
            return None
        try:
            out = run_quiet(["xdotool", "getmouselocation", "--shell"], timeout=3).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        x, y = re.search(r"^X=(-?\d+)", out, re.M), re.search(r"^Y=(-?\d+)", out, re.M)
        return (int(x.group(1)), int(y.group(1))) if x and y else None

    # -- camera -----------------------------------------------------------
    def camera_backends(self) -> list[str]:
        return ["v4l2"]

    def camera_access_problem(self) -> str | None:
        devices = sorted(Path("/dev").glob("video[0-9]*"))
        if devices and not any(os.access(d, os.R_OK | os.W_OK) for d in devices):
            return (
                f"you have no permission to use {devices[0]}. Add your user to the 'video' group "
                "(sudo usermod -aG video $USER), then log out and in again"
            )
        return None

    def camera_devices(self) -> list[str] | None:
        devices = sorted(str(d) for d in Path("/dev").glob("video[0-9]*"))
        return devices

    # -- desktop control (input through xdotool on X11; no element listing yet) ------
    def desktop_problem(self) -> str | None:
        session = session_type()
        if session == "wayland":
            return (
                "desktop input is not supported in a Wayland session (xdotool only works on X11); "
                "log in with an X11 session. ydotool could be supported in a future version"
            )
        if session == "headless":
            return "desktop control needs a graphical session; DISPLAY is not set"
        if shutil.which("xdotool") is None:
            return str(need("xdotool", "desktop control needs xdotool"))
        return None

    def _require_input(self, what: str) -> None:
        problem = self.desktop_problem()
        if problem:
            raise PlatformUnsupported(f"{what}: {problem}")

    def self_elevated(self) -> bool:
        return hasattr(os, "geteuid") and os.geteuid() == 0

    def _xdo(self, *args: str, timeout: float = 10.0) -> str:
        return _run_ok(["xdotool", *args], timeout=timeout)

    def window_info(self, handle: int) -> WindowInfo | None:
        if self.desktop_problem():
            return None
        try:
            title = self._xdo("getwindowname", str(handle)).strip()
            pid = int(self._xdo("getwindowpid", str(handle)).strip() or 0)
        except (OSError, ValueError):
            return None
        try:
            class_name = self._xdo("getwindowclassname", str(handle)).strip()
        except OSError:
            class_name = ""
        proc, elevated = "?", False
        if pid:
            import psutil

            try:
                process = psutil.Process(pid)
                proc = process.name()
                elevated = process.uids().real == 0 and not self.self_elevated()
            except psutil.Error:
                pass
        return WindowInfo(title=title, process=proc, class_name=class_name, pid=pid, handle=handle, elevated=elevated)

    def active_window(self) -> WindowInfo | None:
        self._require_input("identifying the foreground window")
        try:
            handle = int(self._xdo("getactivewindow").strip())
        except (OSError, ValueError):
            return None
        return self.window_info(handle)

    def focus_window(self, handle: int) -> bool:
        self._require_input("focusing a window")
        try:
            self._xdo("windowactivate", "--sync", str(handle), timeout=5)
        except OSError:
            if shutil.which("wmctrl") is None:
                raise
            _run_ok(["wmctrl", "-ia", f"0x{handle:x}"])
        current = self.active_window()
        return current is not None and current.handle == handle

    def ui_snapshot(self, handle: int) -> UiSnapshot:
        raise PlatformUnsupported(
            "listing the elements of a window is not implemented on Linux yet; use read_screen_text "
            "to see the text on screen and desktop_click with text= to click it (OCR)"
        )

    def input_click(self, x: int, y: int, double: bool = False) -> None:
        self._require_input("clicking")
        self._xdo("mousemove", str(int(x)), str(int(y)))
        if double:
            self._xdo("click", "--repeat", "2", "--delay", "80", "1")
        else:
            self._xdo("click", "1")

    def input_text(self, text: str) -> None:
        self._require_input("typing")
        self._xdo("type", "--delay", "12", "--clearmodifiers", "--", text, timeout=max(10.0, len(text) * 0.05))

    def input_keys(self, keys: list[str]) -> None:
        self._require_input("pressing keys")
        names = []
        for key in keys:
            if key not in _XDO_KEYS and not re.fullmatch(r"[a-z0-9]|f\d{1,2}", key):
                raise PlatformUnsupported(f"no xdotool key name for '{key}'")
            names.append(_XDO_KEYS.get(key, key.upper() if re.fullmatch(r"f\d{1,2}", key) else key))
        self._xdo("key", "--clearmodifiers", "+".join(names))

    def input_scroll(self, x: int, y: int, direction: str, amount: int) -> None:
        self._require_input("scrolling")
        button = {"up": "4", "down": "5", "left": "6", "right": "7"}[direction]
        self._xdo("mousemove", str(int(x)), str(int(y)))
        self._xdo("click", "--repeat", str(max(1, int(amount))), "--delay", "20", button)

    # -- voice ------------------------------------------------------------
    def prepare_gpu_libs(self) -> None:
        """Load the pip-installed cuBLAS/cuDNN (nvidia-*-cu12) so ctranslate2 can find them.
        Untested on a machine with a GPU."""
        found: list[Path] = []
        for base in sys.path:
            root = Path(base) / "nvidia"
            if root.is_dir():
                found.extend(sorted(root.glob("*/lib/lib*.so*")))
        for _ in range(2):  # a second pass resolves inter-library dependencies
            for lib in found:
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)  # type: ignore[attr-defined]
                except OSError:
                    pass

    # -- doctor -----------------------------------------------------------
    def helper_report(self) -> list[tuple[str, bool, str]]:
        def fix(helper: str) -> str:
            cmd = install_command(helper)
            return cmd or f"install the package that provides {helper}"

        rows: list[tuple[str, bool, str]] = []
        for helper in ("xdg-open", "wmctrl"):
            rows.append((helper, shutil.which(helper) is not None, fix(helper)))
        has_volume = any(shutil.which(h) for h in ("wpctl", "pactl", "amixer"))
        rows.append(("wpctl/pactl/amixer", has_volume, fix("wpctl")))
        rows.append(("playerctl", shutil.which("playerctl") is not None, fix("playerctl")))
        has_clip = bool(shutil.which("xclip") or shutil.which("xsel") or shutil.which("wl-copy"))
        rows.append(("xclip or wl-clipboard", has_clip, fix("xclip") + " (or wl-clipboard on Wayland)"))
        rows.append(("libportaudio", ctypes.util.find_library("portaudio") is not None, fix("portaudio")))
        return rows
