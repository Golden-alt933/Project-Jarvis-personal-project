"""Windows-only helpers (ctypes): capture exclusion, acrylic, global hotkey. All fail quietly."""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Callable

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
WDA_NONE = 0x0
WDA_EXCLUDEFROMCAPTURE = 0x11
WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x1, 0x2, 0x4000
VK_J = 0x4A
HOTKEY_ID = 0x4A41


def set_capture_excluded(hwnd: int, excluded: bool) -> bool:
    if not IS_WINDOWS or not hwnd:
        return False
    try:
        ok = ctypes.windll.user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE)
        return bool(ok)
    except Exception:
        log.debug("SetWindowDisplayAffinity failed", exc_info=True)
        return False


def display_affinity(hwnd: int) -> int | None:
    if not IS_WINDOWS or not hwnd:
        return None
    value = wintypes.DWORD(0)
    try:
        if ctypes.windll.user32.GetWindowDisplayAffinity(wintypes.HWND(hwnd), ctypes.byref(value)):
            return int(value.value)
    except Exception:
        log.debug("GetWindowDisplayAffinity failed", exc_info=True)
    return None


class _AccentPolicy(ctypes.Structure):
    _fields_ = [("AccentState", ctypes.c_int), ("AccentFlags", ctypes.c_int),
                ("GradientColor", ctypes.c_uint), ("AnimationId", ctypes.c_int)]


class _WinCompAttr(ctypes.Structure):
    _fields_ = [("Attribute", ctypes.c_int), ("Data", ctypes.POINTER(_AccentPolicy)), ("SizeOfData", ctypes.c_size_t)]


def set_acrylic(hwnd: int, enabled: bool, abgr_tint: int = 0x99140B07) -> bool:
    """ACCENT_ENABLE_ACRYLICBLURBEHIND (4) via SetWindowCompositionAttribute (undocumented)."""
    if not IS_WINDOWS or not hwnd:
        return False
    try:
        policy = _AccentPolicy(4 if enabled else 0, 2, abgr_tint, 0)
        data = _WinCompAttr(19, ctypes.pointer(policy), ctypes.sizeof(policy))
        return bool(ctypes.windll.user32.SetWindowCompositionAttribute(wintypes.HWND(hwnd), ctypes.byref(data)))
    except Exception:
        log.debug("acrylic unavailable", exc_info=True)
        return False


def register_hotkey(hwnd: int, hotkey_id: int = HOTKEY_ID) -> bool:
    """Ctrl+Alt+J. False (quietly) if another program already owns it."""
    if not IS_WINDOWS:
        return False
    try:
        return bool(ctypes.windll.user32.RegisterHotKey(wintypes.HWND(hwnd), hotkey_id, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_J))
    except Exception:
        return False


def unregister_hotkey(hwnd: int, hotkey_id: int = HOTKEY_ID) -> None:
    if IS_WINDOWS:
        try:
            ctypes.windll.user32.UnregisterHotKey(wintypes.HWND(hwnd), hotkey_id)
        except Exception:
            pass


def make_hotkey_filter(callback: Callable[[], None], hotkey_id: int = HOTKEY_ID):
    """A QAbstractNativeEventFilter that calls `callback` on our WM_HOTKEY."""
    from PyQt6.QtCore import QAbstractNativeEventFilter

    class _Filter(QAbstractNativeEventFilter):
        def nativeEventFilter(self, event_type, message):  # noqa: N802
            if event_type == b"windows_generic_MSG":
                try:
                    msg = wintypes.MSG.from_address(int(message))
                    if msg.message == WM_HOTKEY and msg.wParam == hotkey_id:
                        callback()
                except Exception:
                    log.debug("native event decode failed", exc_info=True)
            return False, 0

    return _Filter()
