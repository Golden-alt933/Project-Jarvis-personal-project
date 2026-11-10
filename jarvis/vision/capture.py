"""Screen capture. Ported from Friday's vision/screen_capture.py.

Targets the monitor under the mouse pointer (the one the user is looking at),
using physical pixel coordinates so it stays correct at 125%/150% display scaling.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

MonitorSpec = Literal["active", "all"] | int

# Longest edge when a capture is meant for a model: a trade between small-text
# legibility and cost. Measured on a 1920x1080 screen: OCR takes 5.4 s at 1568 and
# 15.8 s at native size for the same text (871 vs 892 chars), so native is not worth it.
DEFAULT_MAX_EDGE = 1568
# Pass max_edge=FULL_RES to keep native resolution.
FULL_RES = 0
JPEG_QUALITY = 80


class CaptureError(RuntimeError):
    """The screen (or an image) could not be captured or read."""


# -- geometry ----------------------------------------------------------------
def scaled_size(size: tuple[int, int], max_edge: int) -> tuple[int, int]:
    """Size after shrinking so the longest edge is at most ``max_edge`` (never enlarges)."""
    width, height = size
    longest = max(width, height)
    if max_edge <= 0 or longest <= max_edge:
        return width, height
    ratio = max_edge / longest
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def downscale(image: "Image", max_edge: int) -> "Image":
    """Shrink to ``max_edge`` on the longest side. An integer box ``reduce`` does the
    bulk (fast, area-averaged); a real resampler handles the fractional remainder."""
    from PIL import Image

    target = scaled_size(image.size, max_edge)
    if target == image.size:
        return image
    factor = max(image.size) // max(target)
    if factor >= 2:
        image = image.reduce(factor)
    if image.size != target:
        image = image.resize(target, Image.Resampling.LANCZOS)
    return image


# -- win32 -------------------------------------------------------------------
_dpi_set = False


def _ensure_dpi_aware() -> None:
    """Without this, cursor and monitor coordinates disagree at non-100% scaling."""
    global _dpi_set
    if _dpi_set or sys.platform != "win32":
        return
    _dpi_set = True
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
    except Exception:  # already set (e.g. by mss) or unavailable
        log.debug("could not set DPI awareness", exc_info=True)


def _cursor_position() -> tuple[int, int] | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        point = wintypes.POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return int(point.x), int(point.y)
    except Exception:
        log.debug("GetCursorPos failed", exc_info=True)
    return None


def pick_monitor(monitors: list[dict], spec: MonitorSpec, cursor: tuple[int, int] | None) -> dict:
    """Choose from mss's list (index 0 is the union of all displays)."""
    if spec == "all":
        return monitors[0]
    if isinstance(spec, int):
        if not 1 <= spec < len(monitors):
            raise CaptureError(f"monitor {spec} does not exist; this PC has {len(monitors) - 1}")
        return monitors[spec]
    primary = monitors[1] if len(monitors) > 1 else monitors[0]
    if cursor is None:
        return primary
    x, y = cursor
    for mon in monitors[1:]:
        if mon["left"] <= x < mon["left"] + mon["width"] and mon["top"] <= y < mon["top"] + mon["height"]:
            return mon
    return primary


def capture_window_title() -> str:
    """Title of the foreground window, or "" if unknown."""
    if sys.platform != "win32":
        return ""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value.strip()
    except Exception:
        log.debug("window title unavailable", exc_info=True)
        return ""


# -- capture -----------------------------------------------------------------
def capture_screen(monitor: MonitorSpec = "active", max_edge: int | None = None) -> "Image":
    """Grab a monitor as an RGB image, longest edge limited to ``max_edge``
    (default ``DEFAULT_MAX_EDGE``). Blocking: call it via ``asyncio.to_thread``."""
    try:
        import mss
        from PIL import Image
    except ImportError as exc:
        raise CaptureError(f"screen capture needs mss and Pillow ({exc.name})") from exc

    _ensure_dpi_aware()
    factory = getattr(mss, "MSS", None) or mss.mss
    try:
        with factory() as sct:
            chosen = pick_monitor(list(sct.monitors), monitor, _cursor_position())
            raw = sct.grab(chosen)
            image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"could not read the screen: {exc}") from exc
    return downscale(image, DEFAULT_MAX_EDGE if max_edge is None else max_edge)


# -- encoding ----------------------------------------------------------------
def to_jpeg_bytes(image: "Image", quality: int = JPEG_QUALITY) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def to_png_bytes(image: "Image") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def jpeg_base64(image: "Image", quality: int = JPEG_QUALITY) -> str:
    return to_base64(to_jpeg_bytes(image, quality))


def data_uri(image: "Image", quality: int = JPEG_QUALITY) -> str:
    return "data:image/jpeg;base64," + jpeg_base64(image, quality)
