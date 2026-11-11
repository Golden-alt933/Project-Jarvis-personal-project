"""Find on-screen text for a click when an app exposes no usable elements.

``match_text`` ranks OCR lines against the wanted text: exact, then case-insensitive, then
contained as whole words, then fuzzy. Only the best tier counts. One hit in that tier is
the target; several are an error that lists them (by line text and area of the screen, never
by pixel) so the model can ask again with a longer, unambiguous phrase.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis import platform as plat
from jarvis.vision.capture import CaptureError, downscale, pick_monitor
from jarvis.vision.ocr import OcrLine

if TYPE_CHECKING:
    from PIL.Image import Image

FUZZY_CUTOFF = 0.82
OCR_MAX_EDGE = 1568
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Candidate:
    line: OcrLine
    tier: str  # exact | case-insensitive | contained | fuzzy
    x: int  # click point in image pixels
    y: int


@dataclass(frozen=True)
class ScreenGrab:
    image: "Image"  # what was OCR'd (possibly downscaled)
    left: int  # screen position of the grab's top-left corner
    top: int
    scale: float  # native pixels per image pixel

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return round(self.left + x * self.scale), round(self.top + y * self.scale)

    @property
    def size(self) -> tuple[int, int]:
        return round(self.image.width * self.scale), round(self.image.height * self.scale)


def norm(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def _contains_words(haystack: str, needle: str) -> int:
    """Index of ``needle`` in ``haystack`` as whole words, or -1."""
    match = re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack)
    return match.start() if match else -1


def _fuzzy_ratio(line: str, wanted: str) -> float:
    best = difflib.SequenceMatcher(None, line, wanted).ratio()
    words, count = line.split(), len(wanted.split())
    for i in range(max(0, len(words) - count + 1)):
        window = " ".join(words[i : i + count])
        best = max(best, difflib.SequenceMatcher(None, window, wanted).ratio())
    return best


def _point(line: OcrLine, wanted: str, tier: str) -> tuple[int, int]:
    """Where to click: the middle of the line, or of the wanted words when they are part of it."""
    cy = round((line.y0 + line.y1) / 2)
    if tier in ("contained", "fuzzy"):
        text = norm(line.text)
        start = _contains_words(text, norm(wanted))
        if start >= 0 and text:
            mid = (start + len(wanted) / 2) / len(text)
            return round(line.x0 + (line.x1 - line.x0) * mid), cy
    return round((line.x0 + line.x1) / 2), cy


def match_text(lines: list[OcrLine], wanted: str) -> list[Candidate]:
    """Candidates in the best tier that has any (empty when nothing matches)."""
    target = norm(wanted)
    if not target:
        return []
    tiers: list[tuple[str, list[OcrLine]]] = [
        ("exact", [ln for ln in lines if ln.text.strip() == wanted.strip()]),
        ("case-insensitive", [ln for ln in lines if norm(ln.text) == target]),
        ("contained", [ln for ln in lines if _contains_words(norm(ln.text), target) >= 0]),
        ("fuzzy", [ln for ln in lines if _fuzzy_ratio(norm(ln.text), target) >= FUZZY_CUTOFF]),
    ]
    for tier, found in tiers:
        if found:
            return [Candidate(ln, tier, *_point(ln, wanted, tier)) for ln in found]
    return []


def region_name(x: float, y: float, width: float, height: float) -> str:
    """'top left', 'middle', 'bottom right' ... for a point on a screen of the given size."""
    col = "left" if x < width / 3 else "right" if x > 2 * width / 3 else "centre"
    row = "top" if y < height / 3 else "bottom" if y > 2 * height / 3 else "middle"
    if row == "middle" and col == "centre":
        return "middle"
    return f"{row} {col}"


def describe_candidates(cands: list[Candidate], size: tuple[int, int]) -> str:
    width, height = size
    ordered = sorted(cands, key=lambda c: (c.y, c.x))
    return "; ".join(
        f"'{c.line.text.strip()[:60]}' ({region_name(c.x, c.y, width, height)})" for c in ordered[:8]
    )


def grab_for_ocr(window_rect: tuple[int, int, int, int] | None = None) -> ScreenGrab:
    """The monitor showing the foreground window (else the one under the pointer), shrunk for
    OCR. Blocking. Wayland goes through the platform's screenshot tool (whole screen)."""
    try:
        import mss
        from PIL import Image
    except ImportError as exc:
        raise CaptureError(f"screen capture needs mss and Pillow ({exc.name})") from exc
    backend = plat.current()
    backend.prepare_capture()
    try:
        native = backend.screen_grab()
    except OSError as exc:
        raise CaptureError(str(exc)) from exc
    left = top = 0
    if native is None:
        anchor = backend.cursor_position()
        if window_rect is not None:
            anchor = ((window_rect[0] + window_rect[2]) // 2, (window_rect[1] + window_rect[3]) // 2)
        factory = getattr(mss, "MSS", None) or mss.mss
        try:
            with factory() as sct:
                monitor = pick_monitor(list(sct.monitors), "active", anchor)
                raw = sct.grab(monitor)
                native = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                left, top = int(monitor["left"]), int(monitor["top"])
        except CaptureError:
            raise
        except Exception as exc:
            raise CaptureError(f"could not read the screen: {exc}") from exc
    image = downscale(native, OCR_MAX_EDGE)
    return ScreenGrab(image=image, left=left, top=top, scale=native.width / image.width)
