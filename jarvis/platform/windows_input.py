"""Windows input synthesis and window plumbing (ctypes, no extra packages).

Typing uses ``SendInput`` with ``KEYEVENTF_UNICODE``: each UTF-16 code unit is sent as a
character, so any text works whatever keyboard layout is active. Chords use virtual-key
codes. Mouse input moves the cursor with ``SetCursorPos`` in physical pixels (the process
is made per-monitor DPI aware first, so the coordinates agree with the screenshots and
with UI Automation) and then sends button events.

Imports on any OS; ``ctypes.windll`` is only touched inside functions.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from ctypes import wintypes

from jarvis.platform.base import PlatformUnsupported

log = logging.getLogger(__name__)

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x0001, 0x0002, 0x0004
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_WHEEL, MOUSEEVENTF_HWHEEL = 0x0800, 0x1000
WHEEL_DELTA = 120
VK_RETURN, VK_TAB = 0x0D, 0x09

# Virtual-key codes for canonical key names (see jarvis.platform.keys).
VK: dict[str, int] = {
    "ctrl": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B,
    "enter": 0x0D, "tab": 0x09, "space": 0x20, "escape": 0x1B, "backspace": 0x08,
    "delete": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23, "pageup": 0x21,
    "pagedown": 0x22, "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "capslock": 0x14, "printscreen": 0x2C,
    "plus": 0xBB, "equals": 0xBB, "minus": 0xBD, "comma": 0xBC, "period": 0xBE,
    "slash": 0xBF, "semicolon": 0xBA, "quote": 0xDE, "backquote": 0xC0,
    "bracketleft": 0xDB, "bracketright": 0xDD, "backslash": 0xDC,
}
VK.update({chr(c): c for c in range(ord("0"), ord("9") + 1)})
VK.update({chr(c): c for c in range(ord("a"), ord("z") + 1)})
VK.update({f"f{n}": 0x6F + n for n in range(1, 25)})
_EXTENDED = {"insert", "delete", "home", "end", "pageup", "pagedown", "up", "down", "left", "right", "win", "printscreen"}
# "plus" is shift+equals on a US keyboard; the VK for '=' alone types '+' only with shift held
_NEEDS_SHIFT = {"plus"}


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _user32() -> ctypes.WinDLL:  # type: ignore[name-defined]
    if os.name != "nt":
        raise PlatformUnsupported("desktop input needs Windows")
    return ctypes.windll.user32  # type: ignore[attr-defined]


def _send(events: list[INPUT]) -> None:
    """Inject events in chunks. Raises OSError if Windows rejected any (blocked by UIPI, or
    a secure desktop owns the input)."""
    user32 = _user32()
    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.SendInput.restype = wintypes.UINT
    for start in range(0, len(events), 100):
        chunk = events[start : start + 100]
        array = (INPUT * len(chunk))(*chunk)
        sent = user32.SendInput(len(chunk), array, ctypes.sizeof(INPUT))
        if sent != len(chunk):
            code = ctypes.get_last_error() or ctypes.GetLastError()
            raise OSError(
                f"Windows accepted {sent} of {len(chunk)} input events (error {code}). The window "
                "may be running as administrator, or a secure prompt (UAC, lock screen) has the input."
            )
        if start + 100 < len(events):
            time.sleep(0.005)


def _key_event(vk: int, up: bool = False, scan: int = 0, flags: int = 0) -> INPUT:
    event = INPUT(type=INPUT_KEYBOARD)
    event.ki = _KEYBDINPUT(vk, scan, flags | (KEYEVENTF_KEYUP if up else 0), 0, 0)
    return event


def _mouse_event(flags: int, data: int = 0) -> INPUT:
    event = INPUT(type=INPUT_MOUSE)
    event.mi = _MOUSEINPUT(0, 0, data & 0xFFFFFFFF, flags, 0, 0)
    return event


def text_events(text: str) -> list[INPUT]:
    """Key events that type ``text``: Unicode characters, with Enter and Tab as real keys."""
    events: list[INPUT] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for char in normalized:
        if char == "\n":
            events += [_key_event(VK_RETURN), _key_event(VK_RETURN, up=True)]
        elif char == "\t":
            events += [_key_event(VK_TAB), _key_event(VK_TAB, up=True)]
        else:
            data = char.encode("utf-16-le")
            for i in range(0, len(data), 2):
                unit = int.from_bytes(data[i : i + 2], "little")
                events += [
                    _key_event(0, scan=unit, flags=KEYEVENTF_UNICODE),
                    _key_event(0, up=True, scan=unit, flags=KEYEVENTF_UNICODE),
                ]
    return events


def type_text(text: str) -> None:
    _send(text_events(text))


def chord_events(keys: list[str]) -> list[INPUT]:
    """Down events for each key in order, then up events in reverse."""
    for name in keys:
        if name not in VK:
            raise ValueError(f"no virtual-key code for '{name}'")
    expanded: list[str] = []
    for name in keys:
        if name in _NEEDS_SHIFT and "shift" not in keys:
            expanded.append("shift")
        expanded.append(name)

    def flags(name: str) -> int:
        return KEYEVENTF_EXTENDEDKEY if name in _EXTENDED else 0

    down = [_key_event(VK[n], flags=flags(n)) for n in expanded]
    up = [_key_event(VK[n], up=True, flags=flags(n)) for n in reversed(expanded)]
    return down + up


def press_chord(keys: list[str]) -> None:
    _send(chord_events(keys))


def set_cursor(x: int, y: int) -> None:
    if not _user32().SetCursorPos(int(x), int(y)):
        raise OSError("could not move the mouse pointer (a secure prompt may be showing)")


def click(x: int, y: int, double: bool = False) -> None:
    set_cursor(x, y)
    time.sleep(0.03)
    pair = [_mouse_event(MOUSEEVENTF_LEFTDOWN), _mouse_event(MOUSEEVENTF_LEFTUP)]
    _send(pair * (2 if double else 1))


def scroll(x: int, y: int, direction: str, amount: int) -> None:
    set_cursor(x, y)
    time.sleep(0.03)
    notches = max(1, int(amount))
    if direction in ("up", "down"):
        delta = WHEEL_DELTA * (1 if direction == "up" else -1)
        flag = MOUSEEVENTF_WHEEL
    else:
        delta = WHEEL_DELTA * (1 if direction == "right" else -1)
        flag = MOUSEEVENTF_HWHEEL
    _send([_mouse_event(flag, delta) for _ in range(notches)])


# ---------------------------------------------------------------- windows and processes
def _title(user32: object, hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)  # type: ignore[attr-defined]
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)  # type: ignore[attr-defined]
    return buf.value


def _class_name(user32: object, hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)  # type: ignore[attr-defined]
    return buf.value


def process_elevated(pid: int) -> bool:
    """True if the process runs elevated, or is protected from inspection (which in practice
    means it is more privileged than this process)."""
    if os.name != "nt":
        return False
    kernel32, advapi32 = ctypes.windll.kernel32, ctypes.windll.advapi32  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return kernel32.GetLastError() == 5  # access denied
    token = wintypes.HANDLE()
    try:
        if not advapi32.OpenProcessToken(handle, 0x0008, ctypes.byref(token)):  # TOKEN_QUERY
            return kernel32.GetLastError() == 5
        elevation = wintypes.DWORD(0)
        size = wintypes.DWORD(0)
        ok = advapi32.GetTokenInformation(token, 20, ctypes.byref(elevation), ctypes.sizeof(elevation), ctypes.byref(size))  # TokenElevation
        return bool(ok and elevation.value)
    finally:
        if token:
            kernel32.CloseHandle(token)
        kernel32.CloseHandle(handle)


def self_elevated() -> bool:
    return process_elevated(os.getpid())


def input_desktop_available() -> bool:
    """False when a secure desktop (UAC consent prompt, lock screen) is the input desktop."""
    if os.name != "nt":
        return True
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.OpenInputDesktop.restype = wintypes.HANDLE
    desk = user32.OpenInputDesktop(0, False, 0x0100)  # DESKTOP_SWITCHDESKTOP
    if not desk:
        return False
    user32.CloseDesktop(desk)
    return True


def window_details(hwnd: int) -> tuple[str, str, int, bool, tuple[int, int, int, int] | None]:
    """(title, class name, pid, minimized, rect) of a window handle."""
    user32 = _user32()
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    rect = wintypes.RECT()
    ok = user32.GetWindowRect(hwnd, ctypes.byref(rect))
    box = (rect.left, rect.top, rect.right, rect.bottom) if ok else None
    return _title(user32, hwnd), _class_name(user32, hwnd), int(pid.value), bool(user32.IsIconic(hwnd)), box


def foreground_handle() -> int:
    return int(_user32().GetForegroundWindow() or 0)


def root_window_at(x: int, y: int) -> int:
    """Handle of the top-level window under a screen point (0 if none)."""
    user32 = _user32()
    user32.WindowFromPoint.argtypes = (wintypes.POINT,)
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetAncestor.restype = wintypes.HWND
    hwnd = user32.WindowFromPoint(wintypes.POINT(x, y))
    if not hwnd:
        return 0
    return int(user32.GetAncestor(hwnd, 2) or hwnd)  # GA_ROOT


def bring_to_front(hwnd: int) -> bool:
    """Make a window the foreground window. Windows only lets the foreground thread do that,
    so this attaches to its input queue (and taps Alt, which lifts the lock) when needed."""
    user32 = _user32()
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    if foreground_handle() == hwnd:
        return True
    current = kernel32.GetCurrentThreadId()
    fore = user32.GetForegroundWindow()
    fore_thread = user32.GetWindowThreadProcessId(fore, None) if fore else 0
    attached = False
    try:
        if fore_thread and fore_thread != current:
            attached = bool(user32.AttachThreadInput(current, fore_thread, True))
        user32.BringWindowToTop(hwnd)
        if not user32.SetForegroundWindow(hwnd):
            _send([_key_event(0x12), _key_event(0x12, up=True)])  # an Alt tap unlocks SetForegroundWindow
            user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(current, fore_thread, False)
    time.sleep(0.15)
    return foreground_handle() == hwnd
