"""System tools: time, machine status, apps, volume, media keys, clipboard, lock.

Only the standard library, psutil and pyperclip are used. Windows-specific
features go through ctypes, winreg-free file scans, or PowerShell, so this
module always imports (on any OS) and individual tools fail cleanly instead.

Volume note: master volume is driven by a small C# COM interop snippet compiled
by PowerShell (``Add-Type``) against the Core Audio ``IAudioEndpointVolume``
interface. It needs no extra Python packages, reads and sets an exact percent,
and takes roughly 1-2 seconds per call because of the compile step.
"""

from __future__ import annotations

import base64
import ctypes
import difflib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import psutil

from jarvis.tools.base import Risk, ToolResult, tool

log = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _powershell(script: str, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    """Run a PowerShell script (passed encoded, so no quoting problems)."""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_NO_WINDOW,
    )


def _gb(n: float) -> str:
    return f"{n / 1024**3:.1f} GB"


# --------------------------------------------------------------------------
# time and status
# --------------------------------------------------------------------------
@tool(group="core")
def get_time() -> ToolResult:
    """Get the current local date, time, weekday and timezone.

    Use whenever the answer depends on "now" (today's date, how long until
    something, scheduling). Returns a single line.
    """
    now = datetime.now().astimezone()
    text = (
        f"{now:%A, %d %B %Y}, {now:%H:%M:%S} "
        f"(timezone {now.tzname()}, UTC{now:%z})"
    )
    return ToolResult.success(text, data={"iso": now.isoformat(timespec="seconds")})


def _gpu_line() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        return None
    name, util, used, total, temp = parts
    return f"GPU: {name}, {util}% load, {used}/{total} MiB VRAM, {temp} C"


def _top_processes(limit: int = 5) -> list[tuple[str, int, float]]:
    """(name, process count, total MB) grouped by name, biggest memory first."""
    grouped: dict[str, list[float]] = {}
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            info = proc.info
            mem = info["memory_info"].rss if info["memory_info"] else 0
            grouped.setdefault(info["name"] or "?", []).append(mem)
        except (psutil.Error, AttributeError):
            continue
    rows = [(n, len(v), sum(v) / 1024**2) for n, v in grouped.items()]
    return sorted(rows, key=lambda r: r[2], reverse=True)[:limit]


def _network_connected() -> bool:
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1.5).close()
        return True
    except OSError:
        return False


@tool(group="core", timeout=30)
def system_status() -> ToolResult:
    """Report the machine's health: CPU, RAM, battery, disks, top memory-hungry
    processes, uptime, network connectivity and NVIDIA GPU load (if present).

    Use for "how is my PC doing", "is my battery low", "how much disk space is
    left". Read-only.
    """
    lines: list[str] = []
    lines.append(f"CPU: {psutil.cpu_percent(interval=0.5):.0f}% ({psutil.cpu_count()} logical cores)")
    mem = psutil.virtual_memory()
    lines.append(f"RAM: {_gb(mem.used)} used of {_gb(mem.total)} ({mem.percent:.0f}%)")

    try:
        battery = psutil.sensors_battery()
    except (AttributeError, OSError):
        battery = None
    if battery is None:
        lines.append("Battery: none detected (desktop or unavailable)")
    else:
        state = "charging" if battery.power_plugged else "on battery"
        lines.append(f"Battery: {battery.percent:.0f}% ({state})")

    disks = []
    for part in psutil.disk_partitions(all=False):
        if "cdrom" in part.opts.lower() or not part.fstype:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        disks.append(f"{part.mountpoint} {_gb(usage.free)} free of {_gb(usage.total)}")
    lines.append("Disks: " + ("; ".join(disks) if disks else "unavailable"))

    top = _top_processes()
    lines.append(
        "Top memory: "
        + ", ".join(f"{n} x{c} {mb:.0f} MB" if c > 1 else f"{n} {mb:.0f} MB" for n, c, mb in top)
    )

    up = int(time.time() - psutil.boot_time())
    days, rem = divmod(up, 86400)
    lines.append(f"Uptime: {days}d {rem // 3600}h {rem % 3600 // 60}m")
    lines.append(f"Network: {'connected' if _network_connected() else 'no internet connection'}")
    gpu = _gpu_line()
    if gpu:
        lines.append(gpu)
    return ToolResult.success("\n".join(lines))


# --------------------------------------------------------------------------
# apps: catalog, fuzzy resolve, launch
# --------------------------------------------------------------------------
class AppEntry:
    """One launchable thing. kind: 'lnk' (shortcut path), 'uwp' (AppID), 'exe' (command)."""

    __slots__ = ("name", "kind", "target")

    def __init__(self, name: str, kind: str, target: str) -> None:
        self.name, self.kind, self.target = name, kind, target

    def __repr__(self) -> str:  # for debugging and test failures
        return f"AppEntry({self.name!r}, {self.kind!r})"


# Shortcut names that are never what someone means by "open X".
_JUNK = re.compile(
    r"\b(uninstall|readme|release notes|license|documentation|help|manual|website|"
    r"changelog|repair|updater?|configure|setup)\b",
    re.IGNORECASE,
)

# Executables that exist on a stock Windows install, reachable by name.
_COMMON_EXES: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "command prompt": "cmd.exe",
    "task manager": "taskmgr.exe",
    "file explorer": "explorer.exe",
    "snipping tool": "snippingtool.exe",
    "control panel": "control.exe",
    "registry editor": "regedit.exe",
    "wordpad": "write.exe",
}

_catalog_cache: tuple[float, list[AppEntry]] | None = None
_CATALOG_TTL = 600.0


def _start_menu_dirs() -> list[Path]:
    roots = [os.environ.get("PROGRAMDATA"), os.environ.get("APPDATA")]
    return [
        Path(r) / "Microsoft" / "Windows" / "Start Menu" / "Programs" for r in roots if r
    ]


def _scan_shortcuts(dirs: list[Path]) -> list[AppEntry]:
    found: dict[str, AppEntry] = {}
    for base in dirs:
        if not base.is_dir():
            continue
        for path in base.rglob("*.lnk"):
            name = path.stem
            if _JUNK.search(name):
                continue
            found.setdefault(name.lower(), AppEntry(name, "lnk", str(path)))
    return list(found.values())


def _scan_uwp() -> list[AppEntry]:
    if not _IS_WINDOWS:
        return []
    try:
        out = _powershell("Get-StartApps | ConvertTo-Json -Compress", timeout=25)
        data = json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        log.warning("Get-StartApps failed", exc_info=True)
        return []
    if isinstance(data, dict):
        data = [data]
    entries = []
    for item in data:
        name, app_id = item.get("Name"), item.get("AppID")
        if name and app_id and not _JUNK.search(name):
            entries.append(AppEntry(name, "uwp", app_id))
    return entries


def _common_exes() -> list[AppEntry]:
    return [AppEntry(n.title(), "exe", exe) for n, exe in _COMMON_EXES.items()]


def _catalog(refresh: bool = False) -> list[AppEntry]:
    global _catalog_cache
    now = time.monotonic()
    if not refresh and _catalog_cache and now - _catalog_cache[0] < _CATALOG_TTL:
        return _catalog_cache[1]
    # Shortcuts first so a real install wins over a same-named UWP or exe entry.
    merged: dict[str, AppEntry] = {}
    for entry in _scan_shortcuts(_start_menu_dirs()) + _scan_uwp() + _common_exes():
        merged.setdefault(entry.name.lower(), entry)
    _catalog_cache = (now, sorted(merged.values(), key=lambda e: e.name.lower()))
    return _catalog_cache[1]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _score(wanted: str, candidate: str) -> float:
    """0..1 similarity between a spoken name and an app name (both normalised)."""
    if not wanted or not candidate:
        return 0.0
    if wanted == candidate:
        return 1.0
    w_tokens, c_tokens = set(wanted.split()), set(candidate.split())
    # Whole-word matches outrank character prefixes: "word" must find
    # "Microsoft Word", not "WordPad".
    if c_tokens <= w_tokens:  # "google chrome browser" -> "google chrome"
        return 0.92
    if w_tokens <= c_tokens:  # "code" -> "visual studio code"
        return 0.86
    if candidate.startswith(wanted) or wanted.startswith(candidate):
        return 0.8
    if wanted in candidate:
        return 0.74
    return difflib.SequenceMatcher(None, wanted, candidate).ratio() * 0.9


def resolve_app(name: str, entries: list[AppEntry]) -> tuple[AppEntry | None, list[AppEntry]]:
    """Best match (score >= 0.72) and the closest few candidates, best first."""
    wanted = _norm(name)
    scored = sorted(
        ((_score(wanted, _norm(e.name)), e) for e in entries),
        key=lambda s: (-s[0], len(s[1].name)),
    )
    scored = [s for s in scored if s[0] > 0.4]
    best = scored[0][1] if scored and scored[0][0] >= 0.72 else None
    return best, [e for _, e in scored[:5]]


def _launch(entry: AppEntry) -> None:
    if entry.kind == "uwp":
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{entry.target}"], creationflags=_NO_WINDOW)
    else:
        os.startfile(entry.target)  # type: ignore[attr-defined]  # Windows only


@tool(group="system", timeout=40)
def list_installed_apps(query: str | None = None) -> ToolResult:
    """List applications that can be opened by name (Start Menu apps, Store apps
    and common Windows tools).

    Use to discover an exact app name, or to check whether something is
    installed. Returns up to 40 names, best matches first when a query is given.

    Args:
        query: Optional text to filter/rank by, for example "chrome" or "office".
    """
    entries = _catalog()
    if query and query.strip():
        wanted = _norm(query)
        ranked = sorted(
            ((_score(wanted, _norm(e.name)), e) for e in entries), key=lambda s: -s[0]
        )
        entries = [e for s, e in ranked if s >= 0.72]
    if not entries:
        return ToolResult.success("No matching installed apps found.", data=[])
    names = [e.name for e in entries[:40]]
    more = f" (+{len(entries) - 40} more)" if len(entries) > 40 else ""
    return ToolResult.success("\n".join(names) + more, data=names)


@tool(group="system", timeout=40)
def open_app(name: str) -> ToolResult:
    """Launch an installed application by its everyday name, such as "Chrome",
    "Spotify", "Word", "Calculator".

    Use to start desktop or Store apps. Do not use it for websites (use
    open_url) or files. Returns which app was started, or the closest candidates
    when the name did not match anything.

    Args:
        name: The app's name as a person would say it.
    """
    if not _IS_WINDOWS:
        return ToolResult.fail("opening apps is only supported on Windows")
    best, candidates = resolve_app(name, _catalog())
    if best is None:
        # The cache may be stale (app just installed): rescan once before giving up.
        best, candidates = resolve_app(name, _catalog(refresh=True))
    if best is None:
        hint = (
            "Closest installed apps: " + ", ".join(e.name for e in candidates)
            if candidates
            else "Use list_installed_apps to see what is available."
        )
        return ToolResult.fail(f"no installed app matches '{name}'", hint)
    try:
        _launch(best)
    except OSError as exc:
        return ToolResult.fail(f"could not launch {best.name}: {exc}")
    return ToolResult.success(f"Launched {best.name}.", data={"app": best.name, "kind": best.kind})


# -- running apps ---------------------------------------------------------
def _visible_windows() -> list[tuple[str, str, int]]:
    """(process name, window title, pid) for each top-level visible window."""
    if not _IS_WINDOWS:
        return []
    user32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi
    results: list[tuple[str, str, int]] = []
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd: int, _: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        cloaked = ctypes.c_int(0)  # DWMWA_CLOAKED: hidden UWP shells and other virtual-desktop windows
        dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        length = user32.GetWindowTextLengthW(hwnd)
        if cloaked.value or length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            proc = psutil.Process(pid.value).name()
        except psutil.Error:
            return True
        results.append((proc, buf.value, pid.value))
        return True

    user32.EnumWindows(enum_proc(callback), 0)
    return results


@tool(group="system")
def list_running_apps() -> ToolResult:
    """List applications that currently have a visible window, with the window
    title of each.

    Use to see what the user has open, or to find the exact name before
    close_app. Returns one line per process.
    """
    if not _IS_WINDOWS:
        return ToolResult.fail("listing windows is only supported on Windows")
    by_proc: dict[str, list[str]] = {}
    for proc, title, _pid in _visible_windows():
        by_proc.setdefault(proc, []).append(title)
    if not by_proc:
        return ToolResult.success("No visible application windows.", data={})
    lines = [f"{proc}: {' | '.join(t[:60] for t in titles[:3])}" for proc, titles in sorted(by_proc.items())]
    return ToolResult.success("\n".join(lines), data=by_proc)


def _match_processes(name: str, windows: list[tuple[str, str, int]]) -> list[str]:
    """Process names (with visible windows) that correspond to a spoken app name."""
    wanted = _norm(name)
    matched: dict[str, float] = {}
    for proc, title, _ in windows:
        stem = _norm(Path(proc).stem)
        score = max(_score(wanted, stem), 0.85 if wanted and wanted in _norm(title) else 0.0)
        if score >= 0.72:
            matched[proc] = max(matched.get(proc, 0.0), score)
    return sorted(matched, key=lambda p: -matched[p])


def _still_running(proc_names: list[str]) -> bool:
    wanted = {p.lower() for p in proc_names}
    return any((p.info["name"] or "").lower() in wanted for p in psutil.process_iter(["name"]))


@tool(
    group="system",
    risk=Risk.CONFIRM,
    timeout=30,
    describe=lambda a: f"Close the application '{a.get('name')}' (unsaved work may be lost)",
)
def close_app(name: str) -> ToolResult:
    """Ask a running application to close, the way clicking its X button would.

    Use when the user asks to quit or close an app. It is graceful (no force
    kill), so an app with unsaved work may stay open and prompt the user; the
    result says whether it is still running.

    Args:
        name: App or process name, for example "notepad" or "Chrome".
    """
    if not _IS_WINDOWS:
        return ToolResult.fail("closing apps is only supported on Windows")
    windows = _visible_windows()
    procs = _match_processes(name, windows)
    if not procs:
        open_now = sorted({p for p, _, _ in windows})
        return ToolResult.fail(
            f"no running app with a window matches '{name}'",
            "Apps with windows: " + (", ".join(open_now) or "none"),
        )
    for proc in procs:
        subprocess.run(["taskkill", "/IM", proc], capture_output=True, timeout=10, creationflags=_NO_WINDOW)
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and _still_running(procs):
        time.sleep(0.4)
    if _still_running(procs):
        return ToolResult.success(
            f"Asked {', '.join(procs)} to close, but it is still running "
            "(probably waiting on the user, for example an unsaved-changes prompt).",
            data={"closed": False},
        )
    return ToolResult.success(f"Closed {', '.join(procs)}.", data={"closed": True})


# --------------------------------------------------------------------------
# volume and media keys
# --------------------------------------------------------------------------
_AUDIO_CS = r"""
using System;
using System.Runtime.InteropServices;
[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {
  int a(); int b(); int c(); int d();
  int SetMasterVolumeLevelScalar(float level, Guid ctx);
  int e();
  int GetMasterVolumeLevelScalar(out float level);
  int f(); int g(); int h(); int i();
  int SetMute([MarshalAs(UnmanagedType.Bool)] bool mute, Guid ctx);
  int GetMute(out bool mute);
}
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice { int Activate(ref Guid id, int clsCtx, IntPtr p, out IAudioEndpointVolume v); }
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator { int f(); int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice d); }
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumeratorComObject { }
public class JarvisAudio {
  static IAudioEndpointVolume Ep() {
    var en = (IMMDeviceEnumerator)(new MMDeviceEnumeratorComObject());
    IMMDevice dev; en.GetDefaultAudioEndpoint(0, 1, out dev);
    Guid iid = typeof(IAudioEndpointVolume).GUID; IAudioEndpointVolume v;
    dev.Activate(ref iid, 23, IntPtr.Zero, out v); return v;
  }
  public static string Get() { float l; bool m; var v = Ep(); v.GetMasterVolumeLevelScalar(out l); v.GetMute(out m);
    return ((int)Math.Round(l * 100)).ToString() + "," + (m ? "1" : "0"); }
  public static void SetLevel(int pct) { Ep().SetMasterVolumeLevelScalar(pct / 100f, Guid.Empty); }
  public static void SetMute(bool m) { Ep().SetMute(m, Guid.Empty); }
}
"""


def _audio(action: str) -> tuple[int, bool]:
    """Run an audio action ('Get()', 'SetLevel(40)', 'SetMute($true)'); return (percent, muted)."""
    if not _IS_WINDOWS:
        raise OSError("volume control is only supported on Windows")
    script = (
        "$ProgressPreference = 'SilentlyContinue'\n"  # else stderr fills with CLIXML noise
        f"$src = @'\n{_AUDIO_CS}\n'@\n"
        "Add-Type -TypeDefinition $src\n"
        f"[JarvisAudio]::{action}\n"
        "[JarvisAudio]::Get()"
    )
    out = _powershell(script, timeout=25)
    last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if out.returncode != 0 or not re.fullmatch(r"\d+,[01]", last):
        raise OSError((out.stderr or out.stdout or "no audio endpoint").strip()[:200])
    pct, muted = last.split(",")
    return int(pct), muted == "1"


def _volume_text(pct: int, muted: bool) -> str:
    return f"Volume {pct}%{' (muted)' if muted else ''}."


@tool(group="system", timeout=40)
def get_volume() -> ToolResult:
    """Read the master speaker volume (0-100) and whether it is muted.

    Use before adjusting relatively ("a bit louder") or to answer "how loud is
    it". Read-only.
    """
    try:
        pct, muted = _audio("Get()")
    except (OSError, subprocess.SubprocessError) as exc:
        return ToolResult.fail(f"could not read volume: {exc}")
    return ToolResult.success(_volume_text(pct, muted), data={"percent": pct, "muted": muted})


@tool(group="system", timeout=40)
def set_volume(percent: int) -> ToolResult:
    """Set the master speaker volume to an exact percentage.

    Use for "set volume to 30" or relative requests after reading get_volume.
    Returns the volume after the change.

    Args:
        percent: Target volume from 0 to 100.
    """
    if not 0 <= percent <= 100:
        return ToolResult.fail(f"percent must be 0-100, got {percent}")
    try:
        pct, muted = _audio(f"SetLevel({int(percent)})")
    except (OSError, subprocess.SubprocessError) as exc:
        return ToolResult.fail(f"could not set volume: {exc}")
    return ToolResult.success(_volume_text(pct, muted), data={"percent": pct, "muted": muted})


@tool(group="system", timeout=40)
def set_mute(muted: bool) -> ToolResult:
    """Mute or unmute the master speaker output without changing the level.

    Args:
        muted: True to mute, False to unmute.
    """
    try:
        pct, now_muted = _audio(f"SetMute(${'true' if muted else 'false'})")
    except (OSError, subprocess.SubprocessError) as exc:
        return ToolResult.fail(f"could not change mute: {exc}")
    return ToolResult.success(_volume_text(pct, now_muted), data={"percent": pct, "muted": now_muted})


_MEDIA_VK = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}


def _press_key(vk: int) -> None:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP


@tool(group="system")
def media_key(action: Literal["play_pause", "next", "previous", "stop"]) -> ToolResult:
    """Send a media key to whatever player is active (Spotify, YouTube in a
    browser, VLC, ...): play/pause, next track, previous track or stop.

    The key goes to the system, not a specific app, and Windows gives no
    feedback on whether a player reacted.

    Args:
        action: One of play_pause, next, previous, stop.
    """
    if not _IS_WINDOWS:
        return ToolResult.fail("media keys are only supported on Windows")
    try:
        _press_key(_MEDIA_VK[action])
    except OSError as exc:
        return ToolResult.fail(f"could not send media key: {exc}")
    return ToolResult.success(f"Sent media key: {action}.")


# --------------------------------------------------------------------------
# clipboard, lock, url
# --------------------------------------------------------------------------
@tool(group="system")
def get_clipboard() -> ToolResult:
    """Read the text currently on the clipboard.

    Use when the user says "what I copied", "this", or asks to work with
    copied text. Returns the text (empty if the clipboard holds no text).
    """
    import pyperclip

    try:
        text = pyperclip.paste()
    except pyperclip.PyperclipException as exc:
        return ToolResult.fail(f"clipboard unavailable: {exc}")
    if not text:
        return ToolResult.success("The clipboard is empty or holds non-text content.", data="")
    return ToolResult.success(text, data=text)


@tool(group="system")
def set_clipboard(text: str) -> ToolResult:
    """Put text on the clipboard so the user can paste it.

    Args:
        text: The exact text to copy.
    """
    import pyperclip

    try:
        pyperclip.copy(text)
    except pyperclip.PyperclipException as exc:
        return ToolResult.fail(f"clipboard unavailable: {exc}")
    return ToolResult.success(f"Copied {len(text)} characters to the clipboard.")


def _lock_workstation() -> None:
    if not ctypes.windll.user32.LockWorkStation():  # type: ignore[attr-defined]
        raise OSError("LockWorkStation failed")


@tool(group="system", risk=Risk.CONFIRM, describe=lambda a: "Lock the screen")
def lock_screen() -> ToolResult:
    """Lock the Windows session (the user must sign in again to continue).

    Use only when the user asks to lock the computer.
    """
    if not _IS_WINDOWS:
        return ToolResult.fail("locking is only supported on Windows")
    try:
        _lock_workstation()
    except OSError as exc:
        return ToolResult.fail(f"could not lock the screen: {exc}")
    return ToolResult.success("Screen locked.")


def _open_in_browser(url: str) -> bool:
    return webbrowser.open(url)


@tool(group="system")
def open_url(url: str) -> ToolResult:
    """Open a web page in the user's default browser so they can see it.

    Use when the user wants to view a site themselves. To read a page's content
    yourself, use fetch_url instead.

    Args:
        url: Full http:// or https:// address.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ToolResult.fail(
            f"only http and https URLs can be opened, got '{url[:80]}'",
            "Include the scheme, for example https://example.com",
        )
    if not _open_in_browser(url.strip()):
        return ToolResult.fail("no default browser accepted the URL")
    return ToolResult.success(f"Opened {url.strip()} in the default browser.")
