"""Camera tools: look through the webcam, take a photo, list cameras.

Privacy rules, in order of strength:
- JARVIS_CAMERA=off: no tool here touches a device.
- JARVIS_CAMERA=ask (default): the first use in each process asks the user. A yes is
  remembered until the process exits; a no is not, so the next attempt asks again.
  Sending a frame to a model needs its own yes (a yes to a plain photo does not cover it).
- JARVIS_CAMERA=on: never asks.
Each call grabs a single frame and releases the camera at once; nothing is streamed or
recorded. look_through_camera keeps the last frame it sent at ``<data dir>/camera/last.jpg``
(overwritten every time) so a UI can show a thumbnail. With JARVIS_VISION_PROVIDER=local
the frame never leaves this machine.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from jarvis import platform as plat
from jarvis.tools.base import ToolContext, ToolResult, tool
from jarvis.tools.paths import resolve_path
from jarvis.vision import camera as cam
from jarvis.vision.camera import CameraError
from jarvis.vision.capture import to_jpeg_bytes
from jarvis.vision.client import VisionError, build_vision_client

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

GROUP = "camera"
CONFIRM_SEND = "Turn on the camera and send one frame to the vision model"
CONFIRM_LOCAL = "Turn on the camera"
OFF_MESSAGE = (
    "camera access is turned off (JARVIS_CAMERA=off)"
)
OFF_HINT = "Set JARVIS_CAMERA=ask (or on) in the Jarvis config file and restart; jarvis --config-path shows where it is."

_device_lock = threading.Lock()  # one open camera at a time
# What the user has approved in this process: "send" covers "local" too.
_approved: set[str] = set()

VISION_PROMPT = (
    "This is a single frame from the user's webcam. Answer the question about it concretely: "
    "name the objects you can see, quote any text on them exactly as printed, give colours and "
    "counts, and say where things are. If something is blurry, dark, cut off or you are unsure, "
    "say so instead of guessing.\n\nQuestion: "
)


def reset_consent() -> None:
    """Forget this process's approvals (tests, and a user who wants to be asked again)."""
    _approved.clear()


def _mode(ctx: ToolContext) -> str:
    mode = str(getattr(ctx.settings, "camera", "ask") or "ask").strip().lower()
    return mode if mode in ("ask", "on", "off") else "ask"


async def _permission(ctx: ToolContext, level: str) -> ToolResult | None:
    """None when the camera may be used; otherwise the failure to return."""
    mode = _mode(ctx)
    if mode == "off":
        return ToolResult.fail(OFF_MESSAGE, OFF_HINT)
    if mode == "on" or level in _approved or "send" in _approved:
        return None
    question = CONFIRM_SEND if level == "send" else CONFIRM_LOCAL
    try:
        approved = await ctx.confirm(question)
    except Exception:  # a broken frontend must not approve by accident
        log.exception("camera confirm failed")
        approved = False
    if not approved:
        return ToolResult.fail(
            "the user did not allow camera access",
            "Do not retry. Tell the user the camera was not used, or ask what they want instead.",
        )
    _approved.add(level)
    return None


def _index(ctx: ToolContext, camera: int | None) -> int | None:
    if camera is not None:
        return int(camera)
    raw = str(getattr(ctx.settings, "camera_index", "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"JARVIS_CAMERA_INDEX must be a number, not '{raw}'") from None


def _grab(index: int | None, max_edge: int | None) -> tuple["Image", cam.FrameStats]:
    with _device_lock:
        return cam.capture_with_stats(index, max_edge=max_edge)


def _last_frame_path(ctx: ToolContext) -> Path:
    return ctx.settings.data_dir / "camera" / "last.jpg"


def _save_last(ctx: ToolContext, image: "Image") -> Path:
    path = _last_frame_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(to_jpeg_bytes(image, 85))
    tmp.replace(path)  # readers never see a half-written file
    return path


@tool(group=GROUP, timeout=120)
async def look_through_camera(
    question: str = "What do you see?", camera: int | None = None, ctx: ToolContext = None  # type: ignore[assignment]
) -> ToolResult:
    """Take one picture with the webcam and answer a question about it with a vision model:
    real-world objects, a document held up to the lens, the room, the user's surroundings.
    The frame is sent to the vision model (the answer says which one). The camera is on only
    for about a second. Use look_at_screen for what is on the monitor instead.

    Args:
        question: What to find out, e.g. "what is this object?" or "read the label on the bottle".
        camera: Camera number if the PC has several (see list_cameras); default is the configured or first camera.
    """
    if (denied := await _permission(ctx, "send")) is not None:
        return denied
    try:
        index = _index(ctx, camera)
        image, stats = await asyncio.to_thread(_grab, index, cam.MAX_EDGE)
    except (CameraError, ValueError) as exc:
        return ToolResult.fail(str(exc))
    try:
        path = await asyncio.to_thread(_save_last, ctx, image)
    except OSError as exc:
        log.warning("could not save the camera frame: %s", exc)
        path = None
    client = build_vision_client(ctx.settings)
    started = time.perf_counter()
    try:
        answer = await client.ask(image, VISION_PROMPT + question)
    except VisionError as exc:
        return ToolResult.fail(
            f"the camera worked but the vision model is unavailable: {exc}",
            "No image was kept beyond the last frame file; try again in a moment.",
        )
    seconds = time.perf_counter() - started
    if answer.path == "local":
        where = f"{answer.model}, locally (the frame stayed on this machine)"
    else:
        where = f"{answer.model} via {answer.path} (one frame was sent to it)"
    header = f"[camera {stats.index}, {stats.width}x{stats.height}; answered by {where}; {seconds:.1f}s]"
    data = {
        "image_path": str(path) if path else None,
        "camera": stats.index,
        "model": answer.model,
        "provider": answer.path,
        "width": stats.width,
        "height": stats.height,
        "sharpness": round(stats.sharpness, 1),
    }
    return ToolResult.success(f"{header}\n{answer.text}", data=data)


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    for n in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"too many files named like {path.name}")


@tool(group=GROUP, timeout=60)
async def take_photo(save_to: str | None = None, ctx: ToolContext = None) -> ToolResult:  # type: ignore[assignment]
    """Take a full-resolution photo with the webcam and save it as a JPEG. Nothing is sent to
    any model. Use look_through_camera to find out what is in front of the camera instead.

    Args:
        save_to: File path (or an existing folder). Default: Pictures\\Jarvis\\photo-<timestamp>.jpg. An existing file is never overwritten.
    """
    if (denied := await _permission(ctx, "local")) is not None:
        return denied
    name = time.strftime("photo-%Y%m%d-%H%M%S.jpg")
    try:
        if save_to:
            target = resolve_path(save_to)
            if target.is_dir():
                target = target / name
            elif target.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                target = target.with_name(target.name + ".jpg")
            if target.exists():
                return ToolResult.fail(f"{target} already exists", "Choose another file name.")
        else:
            target = _unique(plat.current().known_folder("Pictures") / "Jarvis" / name)
        index = _index(ctx, None)
        image, stats = await asyncio.to_thread(_grab, index, None)
        target.parent.mkdir(parents=True, exist_ok=True)

        def write() -> None:
            if target.suffix.lower() == ".png":
                image.save(target, format="PNG")
            else:
                target.write_bytes(to_jpeg_bytes(image, 92))

        await asyncio.to_thread(write)
    except (CameraError, ValueError) as exc:
        return ToolResult.fail(str(exc))
    except OSError as exc:
        return ToolResult.fail(f"could not save the photo: {exc}")
    return ToolResult.success(
        f"Saved photo to {target} ({stats.width}x{stats.height} px, camera {stats.index}).",
        data={"path": str(target), "width": stats.width, "height": stats.height},
    )


@tool(group=GROUP, timeout=60)
async def list_cameras(ctx: ToolContext = None) -> ToolResult:  # type: ignore[assignment]
    """List the cameras attached to this computer (number and resolution). Each is switched on
    for an instant to find out, so the camera light may blink. Use before look_through_camera
    when the PC has more than one camera.
    """
    if (denied := await _permission(ctx, "local")) is not None:
        return denied
    try:
        def probe() -> list[cam.CameraInfo]:
            with _device_lock:
                return cam.list_cameras()

        found = await asyncio.to_thread(probe)
    except CameraError as exc:
        return ToolResult.fail(str(exc))
    if not found:
        return ToolResult.fail(str(cam.why_not_open(None)))
    names = plat.current().camera_devices()
    lines = [f"camera {c.index}: {c.width}x{c.height} ({c.backend})" for c in found]
    if names:
        lines.append("The system also reports: " + ", ".join(names))
    return ToolResult.success("\n".join(lines), data=[c.index for c in found])
