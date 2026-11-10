"""OCR with RapidOCR (bundled ONNX models, no system binary). Ported from Friday's vision/ocr.py.

Building the engine loads three ONNX models (a couple of seconds), so it is created
once, lazily, and never on the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

_engine: Any = None
_engine_lock = threading.Lock()


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def height(self) -> int:
        return max(1, self.y1 - self.y0)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


def get_engine() -> Any:
    """The shared RapidOCR instance. Blocking on first call."""
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
        return _engine


async def warm_up() -> None:
    """Build the engine off the event loop (e.g. at startup)."""
    await asyncio.to_thread(get_engine)


def ocr_image(image: "Image") -> list[OcrLine]:
    """Recognise text. Blocking (CPU): call via ``asyncio.to_thread``."""
    import numpy as np

    result, _timings = get_engine()(np.array(image.convert("RGB")))
    lines: list[OcrLine] = []
    for box, text, confidence in result or []:
        text = str(text).strip()
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append(
            OcrLine(text, float(confidence), int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        )
    return lines


async def ocr_image_async(image: "Image") -> list[OcrLine]:
    return await asyncio.to_thread(ocr_image, image)


def _same_row(line: OcrLine, row: list[OcrLine]) -> bool:
    top = min(item.y0 for item in row)
    bottom = max(item.y1 for item in row)
    overlap = min(line.y1, bottom) - max(line.y0, top)
    return overlap >= 0.5 * min(line.height, bottom - top)


def layout_text(lines: list[OcrLine]) -> str:
    """Rebuild reading order: boxes at the same height form a row, rows run top to
    bottom, boxes inside a row run left to right. Wide horizontal gaps (columns,
    table cells) become a run of spaces so the result reads like the screen."""
    if not lines:
        return ""
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: (item.center_y, item.x0)):
        if rows and _same_row(line, rows[-1]):
            rows[-1].append(line)
        else:
            rows.append([line])
    out: list[str] = []
    for row in rows:
        row.sort(key=lambda item: item.x0)
        parts = [row[0].text]
        for prev, cur in zip(row, row[1:]):
            char_w = max(1.0, (prev.x1 - prev.x0) / max(1, len(prev.text)))
            parts.append("    " if cur.x0 - prev.x1 > 3 * char_w else " ")
            parts.append(cur.text)
        out.append("".join(parts))
    return "\n".join(out)
