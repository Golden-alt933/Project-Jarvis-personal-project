"""OS backends. ``current()`` returns the one for this machine, chosen by ``sys.platform``.

    from jarvis import platform as plat
    plat.current().open_url("https://example.com")

Tests replace the backend with ``set_backend(fake)`` (and put the real one back with
``set_backend(None)``).
"""

from __future__ import annotations

import sys

from jarvis.platform.base import (
    AppEntry,
    BasePlatform,
    ElementGone,
    Platform,
    PlatformUnsupported,
    UiElement,
    UiSnapshot,
    WindowInfo,
    WindowRow,
)

__all__ = [
    "AppEntry", "BasePlatform", "Platform", "PlatformUnsupported", "WindowRow",
    "ElementGone", "UiElement", "UiSnapshot", "WindowInfo",
    "current", "set_backend", "create",
]

_backend: Platform | None = None


def create(platform_name: str | None = None) -> Platform:
    """A fresh backend for ``platform_name`` (a ``sys.platform`` value; default: this machine)."""
    name = platform_name or sys.platform
    if name == "win32":
        from jarvis.platform.windows import WindowsPlatform

        return WindowsPlatform()
    if name.startswith("linux"):
        from jarvis.platform.linux import LinuxPlatform

        return LinuxPlatform()
    return BasePlatform()  # macOS and others: imports work, operations fail clearly


def current() -> Platform:
    global _backend
    if _backend is None:
        _backend = create()
    return _backend


def set_backend(backend: Platform | None) -> None:
    """Install ``backend`` (or reset to automatic detection with None). For tests."""
    global _backend
    _backend = backend
