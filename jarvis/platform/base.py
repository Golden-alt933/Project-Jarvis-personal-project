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
