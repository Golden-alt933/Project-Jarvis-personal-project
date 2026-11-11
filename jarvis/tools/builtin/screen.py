"""Screen tools: read the screen with OCR, ask a vision model about it, save screenshots.

Ported from Friday's vision_skill / screen_text_skill. The Skill/Task wrappers and
the sentence parsing are gone: the model chooses the tool and supplies the arguments.

Privacy: nothing is captured unless a tool is called, and images are never written
to disk except by take_screenshot. With JARVIS_VISION_PROVIDER=local an image never
leaves this machine; otherwise look_at_screen / describe_image send the image (plus
its OCR text) to the configured hosted vision model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from jarvis.tools.base import ToolContext, ToolResult, tool
from jarvis.tools.paths import not_found, resolve_path
from jarvis.vision.capture import (
    FULL_RES,
    CaptureError,
    capture_screen,
    capture_window_title,
    to_png_bytes,
)
from jarvis.vision.client import VisionError, build_vision_client
from jarvis.vision.ocr import layout_text, ocr_image

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

GROUP = "screen"
MAX_OCR_CHARS = 10000
MAX_PROMPT_OCR_CHARS = 3000
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def _pictures_dir() -> Path:
    return Path.home() / "Pictures"


def _ocr_text(image: "Image") -> str:
    return layout_text(ocr_image(image))


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters cut]"


async def _grab(monitor: str | int, max_edge: int | None = None) -> tuple["Image", str]:
    image = await asyncio.to_thread(capture_screen, monitor, max_edge)
    return image, await asyncio.to_thread(capture_window_title)


@tool(group=GROUP, timeout=60)
async def read_screen_text(monitor: Literal["active", "all"] = "active") -> ToolResult:
    """Read the text currently on screen with OCR, in reading order, plus the foreground
    window title. Fast and runs locally with no model call, and it returns the exact
    characters. Prefer this over look_at_screen for error messages, text in apps,
    code, logs, tables or anything the user wants read or copied. It cannot describe
    pictures, charts or layout; use look_at_screen for that.

    Args:
        monitor: "active" = the monitor the mouse is on (default); "all" = every monitor stitched together.
    """
    try:
        image, title = await _grab(monitor)
        text = await asyncio.to_thread(_ocr_text, image)
    except CaptureError as exc:
        return ToolResult.fail(str(exc))
    except ImportError as exc:
        return ToolResult.fail(f"OCR is not installed ({exc.name})", "Install rapidocr-onnxruntime.")
    header = f"Foreground window: {title or '(unknown)'}\n"
    if not text:
        return ToolResult.success(header + "No text was recognised on screen.")
    return ToolResult.success(header + "Screen text (OCR, reading order):\n" + _clip(text, MAX_OCR_CHARS))


async def _ask_vision(ctx: ToolContext, image: "Image", question: str) -> ToolResult:
    client = build_vision_client(ctx.settings)
    started = time.perf_counter()
    try:
        answer = await client.ask(image, question)
    except VisionError as exc:
        return ToolResult.fail(
            f"vision model unavailable: {exc}",
            "For text on screen use read_screen_text, which needs no model.",
        )
    seconds = time.perf_counter() - started
    where = "locally" if answer.path == "local" else f"via {answer.path}"
    return ToolResult.success(f"[answered by {answer.model}, {where}, {seconds:.1f}s]\n{answer.text}")


@tool(group=GROUP, timeout=120)
async def look_at_screen(question: str, monitor: Literal["active", "all"] = "active", ctx: ToolContext = None) -> ToolResult:  # type: ignore[assignment]
    """Take a screenshot and answer a question about it with a vision model. Use it for
    layout, images, charts, colours, UI state or "what am I looking at". For reading
    text (errors, logs, code) prefer read_screen_text: it is faster and exact. The
    answer states which model produced it. The screenshot is sent to that model.

    Args:
        question: What to find out about the screen, e.g. "what is the dialog asking?".
        monitor: "active" = the monitor the mouse is on (default); "all" = every monitor.
    """
    try:
        image, title = await _grab(monitor)
        try:  # OCR text makes small print reliable for the vision model; it is optional
            text = await asyncio.to_thread(_ocr_text, image)
        except Exception:
            log.warning("OCR unavailable for look_at_screen", exc_info=True)
            text = ""
    except CaptureError as exc:
        return ToolResult.fail(str(exc))
    prompt = f"Foreground window title: {title or '(unknown)'}\n"
    if text:
        prompt += f"Text read from the screen by OCR (may contain errors):\n{_clip(text, MAX_PROMPT_OCR_CHARS)}\n\n"
    prompt += f"Question about the screenshot: {question}"
    return await _ask_vision(ctx, image, prompt)


@tool(group=GROUP, timeout=30)
async def take_screenshot(save_to: str | None = None, monitor: Literal["active", "all"] = "active") -> ToolResult:
    """Save a full-resolution PNG screenshot and return its path and size. It does not
    describe or read the screenshot; use look_at_screen or read_screen_text for that.

    Args:
        save_to: File path (or an existing folder) to save into. Default: Pictures\\Screenshots\\jarvis-<timestamp>.png. An existing file is never overwritten.
        monitor: "active" = the monitor the mouse is on (default); "all" = every monitor.
    """
    try:
        image, _ = await _grab(monitor, FULL_RES)
        name = time.strftime("jarvis-%Y%m%d-%H%M%S.png")
        if save_to:
            target = resolve_path(save_to)
            if target.is_dir():
                target = target / name
            elif target.suffix.lower() != ".png":
                target = target.with_name(target.name + ".png")
        else:
            target = _pictures_dir() / "Screenshots" / name
    except (CaptureError, ValueError) as exc:
        return ToolResult.fail(str(exc))
    if target.exists():
        return ToolResult.fail(f"{target} already exists", "Choose another file name.")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, to_png_bytes(image))
    except OSError as exc:
        return ToolResult.fail(f"could not save the screenshot: {exc}")
    return ToolResult.success(f"Saved screenshot to {target} ({image.width}x{image.height} px).", data=str(target))


def _load_image(path: Path) -> "Image":
    from PIL import Image

    with Image.open(path) as opened:
        return opened.convert("RGB")


@tool(group=GROUP, timeout=120)
async def describe_image(path: str, question: str = "Describe this image.", ctx: ToolContext = None) -> ToolResult:  # type: ignore[assignment]
    """Answer a question about an image file (png, jpg, webp, bmp) with a vision model.
    The image is sent to that model. For text in an image the model may misread
    small print, so quote numbers back to the user as approximate.

    Args:
        path: Path of the image file.
        question: What to find out about the image.
    """
    try:
        file = resolve_path(path)
    except ValueError as exc:
        return ToolResult.fail(str(exc))
    if file.suffix.lower() not in IMAGE_SUFFIXES:
        return ToolResult.fail(
            f"unsupported image type '{file.suffix or '(none)'}'",
            f"Supported: {', '.join(sorted(IMAGE_SUFFIXES))}.",
        )
    if not file.exists():
        return not_found(file, "file")
    if not file.is_file():
        return ToolResult.fail(f"{file} is not a file")
    if file.stat().st_size > MAX_IMAGE_BYTES:
        return ToolResult.fail(f"{file.name} is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        image = await asyncio.to_thread(_load_image, file)
    except Exception as exc:
        return ToolResult.fail(f"could not open {file.name} as an image: {exc}")
    return await _ask_vision(ctx, image, question)
