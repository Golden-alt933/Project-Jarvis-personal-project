"""The platform contract: everything an OS does differently, behind one interface.

Tool modules never look at ``sys.platform``; they ask ``jarvis.platform.current()`` for a
backend and call these methods. Operations a backend cannot do raise
``PlatformUnsupported`` (an ``OSError``) whose message says what is missing and, where
there is one, which package provides it. Callers turn that text into a tool failure.
"""

from __future__ import annotations

import os
import platform as _stdlib_platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL.Image import Image

# Shortcut or menu entries that are never what someone means by "open X".
JUNK_NAME = re.compile(
    r"\b(uninstall|readme|release notes|license|documentation|help|manual|website|"
    r"changelog|repair|updater?|configure|setup)\b",
    re.IGNORECASE,
)


class PlatformUnsupported(OSError):
    """The operation is not available here (missing helper, wrong session type, other OS)."""


class AppEntry:
    """One launchable thing.

    kind: 'lnk' (Windows shortcut path), 'uwp' (Windows AppID), 'exe' (command),
    'webapp' (a shortcut that opens a site in a browser window), 'desktop' (a Linux
    .desktop entry: ``target`` is the desktop id, ``extra`` the cleaned Exec line).
    """

    __slots__ = ("name", "kind", "target", "extra")

    def __init__(self, name: str, kind: str, target: str, extra: str = "") -> None:
        self.name, self.kind, self.target, self.extra = name, kind, target, extra

    def __repr__(self) -> str:  # for debugging and test failures
        return f"AppEntry({self.name!r}, {self.kind!r})"


# (process name, window title, pid, window handle)
WindowRow = tuple[str, str, int, int]


@dataclass(frozen=True)
class WindowInfo:
    """A top-level window as the desktop-control tools need to know it."""

    title: str
    process: str
    class_name: str
    pid: int
    handle: int
    # True when the window's process runs with administrator rights (or could not be
    # inspected because it is protected). A normal-rights Jarvis cannot drive it.
    elevated: bool = False
    minimized: bool = False
    rect: tuple[int, int, int, int] | None = None  # left, top, right, bottom in screen pixels


@dataclass
class UiElement:
    """One element of a window's accessibility tree, as shown to the model.

    ``ref`` is stable for a window until the element goes away; it never exposes a pixel
    position. ``rect`` is kept for the tools' own use (fallback clicks, visibility)."""

    ref: int
    role: str  # button, checkbox, edit, link, list item, menu item, tab, text, ...
    name: str = ""
    value: str = ""
    state: str = ""  # on/off, selected, expanded/collapsed, read-only ...
    enabled: bool = True
    automation_id: str = ""
    is_password: bool = False
    rect: tuple[int, int, int, int] | None = None


@dataclass
class UiSnapshot:
    window: WindowInfo
    elements: list[UiElement] = field(default_factory=list)  # tree order, visible and useful only
    total: int = 0  # elements seen in the whole tree (before filtering)


class ElementGone(OSError):
    """The element a ref pointed to no longer exists; inspect the window again."""


@runtime_checkable
class Platform(Protocol):
    """What tools may ask of the operating system."""

    name: str  # "windows" | "linux" | other
    app_dir_name: str  # folder name under the per-user data/config roots
    shell_kind: str  # "powershell" | "bash"
    shell_name: str  # for prompts and tool descriptions
    reserved_device_names: bool  # CON, NUL, ... are devices, not files
    trash_name: str  # "Recycle Bin" / "Trash"
    executable_suffixes: frozenset[str]  # extra file types whose opening runs code

    def describe(self) -> str: ...
    def shell_argv(self, command: str) -> list[str]: ...
    def subprocess_kwargs(self) -> dict[str, Any]: ...
    def kill_tree(self, pid: int) -> None: ...
    def open_path(self, path: str) -> None: ...
    def open_url(self, url: str) -> bool: ...
    def known_folder(self, name: str) -> Path: ...
    def scan_apps(self) -> list[AppEntry]: ...
    def launch_app(self, entry: AppEntry) -> None: ...
    def windows(self) -> list[WindowRow]: ...
    def close_window(self, handle: int) -> None: ...
    def never_close_titles(self) -> set[str]: ...
    def foreground_window_title(self) -> str: ...
    def get_volume(self) -> tuple[int, bool]: ...
    def set_volume(self, percent: int) -> tuple[int, bool]: ...
    def set_mute(self, muted: bool) -> tuple[int, bool]: ...
    def media_key(self, action: str) -> None: ...
    def lock_screen(self) -> None: ...
    def clipboard_problem(self) -> str | None: ...
    def protected_path_reason(self, real_path: str) -> str | None: ...
    def ollama_extra_paths(self) -> list[Path]: ...
    def detached_popen_kwargs(self) -> dict[str, Any]: ...
    def screen_grab(self) -> "Image | None": ...
    def cursor_position(self) -> tuple[int, int] | None: ...
    def prepare_capture(self) -> None: ...
    def has_excel_com(self) -> bool: ...
    def prepare_gpu_libs(self) -> None: ...
    def helper_report(self) -> list[tuple[str, bool, str]]: ...
    # -- senses and hands (camera, desktop control) --
    def camera_backends(self) -> list[str]: ...
    def camera_access_problem(self) -> str | None: ...
    def camera_devices(self) -> list[str] | None: ...
    def desktop_problem(self) -> str | None: ...
    def can_inspect_elements(self) -> bool: ...
    def self_elevated(self) -> bool: ...
    def input_desktop_available(self) -> bool: ...
    def active_window(self) -> WindowInfo | None: ...
    def window_info(self, handle: int) -> WindowInfo | None: ...
    def window_at(self, x: int, y: int) -> WindowInfo | None: ...
    def focus_window(self, handle: int) -> bool: ...
    def ui_snapshot(self, handle: int) -> UiSnapshot: ...
    def ui_read(self, handle: int, ref: int) -> UiElement: ...
    def ui_act(self, handle: int, ref: int, action: str) -> str: ...
    def ui_clear(self, handle: int, ref: int) -> bool: ...
    def ui_focus_element(self, handle: int, ref: int) -> str: ...
    def ui_focused(self) -> UiElement | None: ...
    def input_click(self, x: int, y: int, double: bool = False) -> None: ...
    def input_text(self, text: str) -> None: ...
    def input_keys(self, keys: list[str]) -> None: ...
    def input_scroll(self, x: int, y: int, direction: str, amount: int) -> None: ...


class BasePlatform:
    """Defaults: every optional operation fails clearly. Backends override what they support."""

    name = "other"
    app_dir_name = "jarvis"
    shell_kind = "bash"
    shell_name = "bash"
    reserved_device_names = False
    trash_name = "Trash"
    executable_suffixes: frozenset[str] = frozenset()

    # -- identity ---------------------------------------------------------
    def describe(self) -> str:
        return f"{_stdlib_platform.system()} {_stdlib_platform.release()}"

    def unsupported(self, what: str) -> PlatformUnsupported:
        return PlatformUnsupported(f"{what} is not supported on {self.describe()}")

    # -- shell ------------------------------------------------------------
    def shell_argv(self, command: str) -> list[str]:
        return ["bash", "-lc", command]

    def subprocess_kwargs(self) -> dict[str, Any]:
        """Extra Popen/create_subprocess_exec arguments so the command gets its own
        process group (killed as a whole on timeout or cancel)."""
        return {"start_new_session": True}

    def kill_tree(self, pid: int) -> None:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError, AttributeError):
            pass
        kill_with_psutil(pid)

    # -- opening things ---------------------------------------------------
    def open_path(self, path: str) -> None:
        raise self.unsupported("opening files")

    def open_url(self, url: str) -> bool:
        import webbrowser

        return webbrowser.open(url)

    def known_folder(self, name: str) -> Path:
        return Path.home() / name

    # -- apps and windows -------------------------------------------------
    def scan_apps(self) -> list[AppEntry]:
        return []

    def launch_app(self, entry: AppEntry) -> None:
        raise self.unsupported("opening apps")

    def windows(self) -> list[WindowRow]:
        raise self.unsupported("listing windows")

    def close_window(self, handle: int) -> None:
        raise self.unsupported("closing windows")

    def never_close_titles(self) -> set[str]:
        return set()

    def foreground_window_title(self) -> str:
        return ""

    # -- audio, media, session -------------------------------------------
    def get_volume(self) -> tuple[int, bool]:
        raise self.unsupported("volume control")

    def set_volume(self, percent: int) -> tuple[int, bool]:
        raise self.unsupported("volume control")

    def set_mute(self, muted: bool) -> tuple[int, bool]:
        raise self.unsupported("volume control")

    def media_key(self, action: str) -> None:
        raise self.unsupported("media keys")

    def lock_screen(self) -> None:
        raise self.unsupported("locking the screen")

    def clipboard_problem(self) -> str | None:
        return None

    # -- files and processes ---------------------------------------------
    def protected_path_reason(self, real_path: str) -> str | None:
        return None

    def ollama_extra_paths(self) -> list[Path]:
        return []

    def detached_popen_kwargs(self) -> dict[str, Any]:
        return {"start_new_session": True}

    # -- screen -----------------------------------------------------------
    def screen_grab(self) -> "Image | None":
        """A full-screen image from an OS-specific tool, or None to let mss do it."""
        return None

    def cursor_position(self) -> tuple[int, int] | None:
        return None

    def prepare_capture(self) -> None:
        return None

    # -- office / voice ---------------------------------------------------
    def has_excel_com(self) -> bool:
        return False

    def prepare_gpu_libs(self) -> None:
        return None

    # -- doctor -----------------------------------------------------------
    def helper_report(self) -> list[tuple[str, bool, str]]:
        """(helper, present, one-line fix) for programs Jarvis uses on this OS."""
        return []

    # -- camera ------------------------------------------------------------
    def camera_backends(self) -> list[str]:
        """OpenCV capture backends to try, best first: names from {'dshow','msmf','v4l2','any'}."""
        return ["any"]

    def camera_access_problem(self) -> str | None:
        """Why the OS would refuse camera access (a privacy switch, device permissions), or None."""
        return None

    def camera_devices(self) -> list[str] | None:
        """Names of the cameras the system reports, [] if it reports none, None if it cannot tell."""
        return None

    # -- desktop control ----------------------------------------------------
    def desktop_problem(self) -> str | None:
        """Why input synthesis is unavailable on this machine right now, or None if it works."""
        return f"desktop control is not supported on {self.describe()}"

    def can_inspect_elements(self) -> bool:
        """True where the accessibility tree of a window can be listed (UI Automation on Windows)."""
        return False

    def self_elevated(self) -> bool:
        return False

    def input_desktop_available(self) -> bool:
        """False while a secure desktop (UAC prompt, lock screen) owns the input."""
        return True

    def active_window(self) -> WindowInfo | None:
        raise self.unsupported("identifying the foreground window")

    def window_info(self, handle: int) -> WindowInfo | None:
        raise self.unsupported("reading window details")

    def window_at(self, x: int, y: int) -> WindowInfo | None:
        """The top-level window under a screen point, or None when unknown."""
        return None

    def focus_window(self, handle: int) -> bool:
        raise self.unsupported("focusing windows")

    def ui_snapshot(self, handle: int) -> UiSnapshot:
        raise self.unsupported("listing the elements of a window")

    def ui_read(self, handle: int, ref: int) -> UiElement:
        raise self.unsupported("reading a window element")

    def ui_act(self, handle: int, ref: int, action: str) -> str:
        raise self.unsupported("acting on a window element")

    def ui_clear(self, handle: int, ref: int) -> bool:
        raise self.unsupported("editing a window element")

    def ui_focus_element(self, handle: int, ref: int) -> str:
        raise self.unsupported("focusing a window element")

    def ui_focused(self) -> UiElement | None:
        return None

    def input_click(self, x: int, y: int, double: bool = False) -> None:
        raise self.unsupported("clicking")

    def input_text(self, text: str) -> None:
        raise self.unsupported("typing")

    def input_keys(self, keys: list[str]) -> None:
        raise self.unsupported("pressing keys")

    def input_scroll(self, x: int, y: int, direction: str, amount: int) -> None:
        raise self.unsupported("scrolling")


def kill_with_psutil(pid: int) -> None:
    """Kill a process and all its descendants."""
    try:
        import psutil

        parent = psutil.Process(pid)
    except Exception:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=3)


def run_quiet(argv: list[str], timeout: float = 10.0) -> "subprocess.CompletedProcess[str]":
    """Run a helper program, capturing text output. Raises OSError/TimeoutExpired upward."""
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL
    )
