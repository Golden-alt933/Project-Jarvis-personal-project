"""System tools: time, machine status, apps, volume, media keys, clipboard, lock.

Everything that differs between operating systems (launching apps, windows, volume,
media keys, locking, the clipboard check) lives in ``jarvis.platform``; this module only
decides what to ask and how to word the answer, so it imports and behaves the same on
every OS. When the platform cannot do something the backend raises PlatformUnsupported
with a message that names what to install, and the tool reports it as its failure.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import psutil

from jarvis import platform as plat
from jarvis.platform.base import JUNK_NAME, AppEntry
from jarvis.tools.base import Risk, ToolResult, tool

log = logging.getLogger(__name__)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_JUNK = JUNK_NAME


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
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
_catalog_cache: tuple[float, list[AppEntry]] | None = None
_CATALOG_TTL = 600.0


def _catalog(refresh: bool = False) -> list[AppEntry]:
    global _catalog_cache
    now = time.monotonic()
    if not refresh and _catalog_cache and now - _catalog_cache[0] < _CATALOG_TTL:
        return _catalog_cache[1]
    # Real installs first so they win over a same-named Store or stock entry. Browser
    # site shortcuts go last and keep a distinct name: a Chrome shortcut called
    # "Calculator" must never be what "open calculator" launches.
    try:
        found = [e for e in plat.current().scan_apps() if not _JUNK.search(e.name)]
    except OSError:
        log.warning("app scan failed", exc_info=True)
        found = []
    merged: dict[str, AppEntry] = {}
    for entry in (e for e in found if e.kind != "webapp"):
        merged.setdefault(entry.name.lower(), entry)
    for entry in (e for e in found if e.kind == "webapp"):
        label = entry.name if entry.name.lower() not in merged else f"{entry.name} (browser app)"
        merged.setdefault(label.lower(), AppEntry(label, "webapp", entry.target))
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
    plat.current().launch_app(entry)


_LAUNCH_WAIT = 5.0


def _window_keys() -> list[tuple[str, str]]:
    try:
        return [(w[0], w[1]) for w in _visible_windows()]
    except Exception:  # window enumeration is best effort
        return []


def _wait_for_new_window(before: set[tuple[str, str]]) -> str | None:
    """Poll for a window that was not there before the launch; say what actually opened."""
    deadline = time.monotonic() + _LAUNCH_WAIT
    while time.monotonic() < deadline:
        time.sleep(0.3)
        fresh = [k for k in _window_keys() if k not in before]
        if fresh:
            proc, title = fresh[0]
            return f"{title} ({proc})"
    return None


@tool(group="system", timeout=40)
def list_installed_apps(query: str | None = None) -> ToolResult:
    """List applications that can be opened by name (Start Menu and Store apps on
    Windows, desktop-menu applications on Linux).

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
    before = set(_window_keys())
    try:
        _launch(best)
    except OSError as exc:
        return ToolResult.fail(f"could not launch {best.name}: {exc}")
    opened = _wait_for_new_window(before)
    what = "a browser shortcut" if best.kind == "webapp" else "an app"
    if opened:
        text = f"Launched {best.name} ({what}). New window: {opened}."
    else:
        text = (
            f"Asked the system to launch {best.name} ({what}), but no new window appeared within "
            f"{_LAUNCH_WAIT:.0f} s. It may still be starting, or it reused an existing window."
        )
    return ToolResult.success(text, data={"app": best.name, "kind": best.kind, "window": opened})


# -- running apps ---------------------------------------------------------
def _visible_windows() -> list[tuple[str, str, int]]:
    """(process name, window title, pid) for each top-level visible window."""
    return [(proc, title, pid) for proc, title, pid, _ in _windows()]


def _windows() -> list[tuple[str, str, int, int]]:
    """(process name, window title, pid, window handle) for each top-level visible window.
    Raises OSError (PlatformUnsupported) where windows cannot be listed."""
    return list(plat.current().windows())


@tool(group="system")
def list_running_apps() -> ToolResult:
    """List applications that currently have a visible window, with the window
    title of each.

    Use to see what the user has open, or to find the exact name before
    close_app. Returns one line per process.
    """
    try:
        visible = _visible_windows()
    except OSError as exc:
        return ToolResult.fail(f"cannot list windows: {exc}")
    by_proc: dict[str, list[str]] = {}
    for proc, title, _pid in visible:
        by_proc.setdefault(proc, []).append(title)
    if not by_proc:
        return ToolResult.success("No visible application windows.", data={})
    lines = [f"{proc}: {' | '.join(t[:60] for t in titles[:3])}" for proc, titles in sorted(by_proc.items())]
    return ToolResult.success("\n".join(lines), data=by_proc)


# Processes that host windows belonging to many different apps. Matching these by
# process name, or closing the process, would hit every app they host.
_SHARED_HOSTS = {"applicationframehost.exe", "explorer.exe", "systemsettings.exe"}
_TITLE_SEPARATOR = re.compile(r"\s+[-–—|]\s+")


def _title_app_name(title: str) -> str:
    """The app part of a window title: "Untitled - Notepad" -> "notepad"."""
    return _norm(_TITLE_SEPARATOR.split(title)[-1])


def _match_windows(
    name: str, windows: list[tuple[str, str, int, int]]
) -> list[tuple[str, str, int, int]]:
    """Windows that belong to the app a person means by `name`.

    A window matches when its process is that app ("chrome" -> chrome.exe, all of its
    windows), or when the app part of its title is that app ("calculator" -> the
    Calculator window inside the shared Store-app host). A word merely appearing in a
    title, such as a browser tab about calculators, is not a match.
    """
    wanted = _norm(name)
    never = plat.current().never_close_titles()
    matched = []
    for window in windows:
        proc, title = window[0], window[1]
        if _norm(title) in never:
            continue
        by_process = (
            proc.lower() not in _SHARED_HOSTS and _score(wanted, _norm(Path(proc).stem)) >= 0.72
        )
        by_title = _score(wanted, _title_app_name(title)) >= 0.86
        if by_process or by_title:
            matched.append(window)
    return matched


def _post_close(hwnd: int) -> None:
    plat.current().close_window(hwnd)


def _open_handles() -> set[int]:
    return {w[3] for w in _windows()}


@tool(
    group="system",
    risk=Risk.CONFIRM,
    timeout=30,
    describe=lambda a: f"Close the application '{a.get('name')}' (unsaved work may be lost)",
)
def close_app(name: str) -> ToolResult:
    """Ask a running application to close, the way clicking its X button would.

    Use when the user asks to quit or close an app. Only that app's own windows are
    closed. It is graceful (no force kill), so an app with unsaved work may stay open
    and prompt the user; the result says which windows closed and which are still open.

    Args:
        name: App name as a person would say it, for example "notepad" or "Chrome".
    """
    try:
        windows = _windows()
    except OSError as exc:
        return ToolResult.fail(f"cannot close apps: {exc}")
    targets = _match_windows(name, windows)
    if not targets:
        open_now = sorted({f"{t} ({p})" for p, t, _, _ in windows})[:12]
        return ToolResult.fail(
            f"no open window belongs to an app called '{name}'",
            "Open windows: " + ("; ".join(open_now) or "none"),
        )
    try:
        for _, _, _, hwnd in targets:
            _post_close(hwnd)
    except OSError as exc:
        return ToolResult.fail(f"could not close {name}: {exc}")
    deadline = time.monotonic() + 4
    remaining = targets
    while time.monotonic() < deadline:
        alive = _open_handles()
        remaining = [w for w in targets if w[3] in alive]
        if not remaining:
            break
        time.sleep(0.4)
    closed = [w[1] for w in targets if w not in remaining]
    if remaining:
        still = "; ".join(w[1] for w in remaining)
        done = f"Closed: {'; '.join(closed)}. " if closed else ""
        return ToolResult.success(
            f"{done}Still open: {still} (probably waiting on the user, for example an "
            "unsaved-changes prompt).",
            data={"closed": False, "closed_windows": closed},
        )
    return ToolResult.success(
        f"Closed {'; '.join(closed)}.", data={"closed": True, "closed_windows": closed}
    )


# --------------------------------------------------------------------------
# volume and media keys
# --------------------------------------------------------------------------
def _audio(action: str, *args: object) -> tuple[int, bool]:
    """Volume through the platform backend: action is get, set or mute. Returns (percent, muted)."""
    backend = plat.current()
    if action == "get":
        return backend.get_volume()
    if action == "set":
        return backend.set_volume(int(args[0]))  # type: ignore[arg-type]
    return backend.set_mute(bool(args[0]))


def _volume_text(pct: int, muted: bool) -> str:
    return f"Volume {pct}%{' (muted)' if muted else ''}."


@tool(group="system", timeout=40)
def get_volume() -> ToolResult:
    """Read the master speaker volume (0-100) and whether it is muted.

    Use before adjusting relatively ("a bit louder") or to answer "how loud is
    it". Read-only.
    """
    try:
        pct, muted = _audio("get")
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
        pct, muted = _audio("set", int(percent))
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
        pct, now_muted = _audio("mute", muted)
    except (OSError, subprocess.SubprocessError) as exc:
        return ToolResult.fail(f"could not change mute: {exc}")
    return ToolResult.success(_volume_text(pct, now_muted), data={"percent": pct, "muted": now_muted})


@tool(group="system")
def media_key(action: Literal["play_pause", "next", "previous", "stop"]) -> ToolResult:
    """Send a media key to whatever player is active (Spotify, YouTube in a
    browser, VLC, ...): play/pause, next track, previous track or stop.

    The key goes to the system (on Linux, to the active MPRIS player), and there
    is usually no feedback on whether a player reacted.

    Args:
        action: One of play_pause, next, previous, stop.
    """
    try:
        plat.current().media_key(action)
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

    if problem := plat.current().clipboard_problem():
        return ToolResult.fail(f"clipboard unavailable: {problem}")
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

    if problem := plat.current().clipboard_problem():
        return ToolResult.fail(f"clipboard unavailable: {problem}")
    try:
        pyperclip.copy(text)
    except pyperclip.PyperclipException as exc:
        return ToolResult.fail(f"clipboard unavailable: {exc}")
    return ToolResult.success(f"Copied {len(text)} characters to the clipboard.")


@tool(group="system", risk=Risk.CONFIRM, describe=lambda a: "Lock the screen")
def lock_screen() -> ToolResult:
    """Lock the screen (the user must sign in again to continue).

    Use only when the user asks to lock the computer.
    """
    try:
        plat.current().lock_screen()
    except OSError as exc:
        return ToolResult.fail(f"could not lock the screen: {exc}")
    return ToolResult.success("Screen locked.")


def _open_in_browser(url: str) -> bool:
    return plat.current().open_url(url)


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
    try:
        opened = _open_in_browser(url.strip())
    except OSError as exc:
        return ToolResult.fail(f"cannot open the URL: {exc}")
    if not opened:
        return ToolResult.fail("no default browser accepted the URL")
    return ToolResult.success(f"Opened {url.strip()} in the default browser.")
