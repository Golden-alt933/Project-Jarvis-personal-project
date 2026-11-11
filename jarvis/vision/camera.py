"""Camera capture with OpenCV.

Everything here is blocking: call it through ``asyncio.to_thread``. The device is opened,
given a moment to settle auto-exposure, a short burst is read and the sharpest frame
kept, and the device is released straight away (in a ``finally``, so the camera LED goes
off even when something fails). Nothing is recorded or streamed: one frame per call.

OS specifics (which OpenCV backend, privacy switches, device names) come from
``jarvis.platform``; this module has no ``sys.platform`` checks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis import platform as plat

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

SETTLE_FRAMES = 8  # frames discarded while auto-exposure and white balance converge
SETTLE_SECONDS = 0.6  # ... or this long, whichever is longer
BURST_FRAMES = 5  # frames compared for sharpness
MAX_EDGE = 1280  # longest edge of the image handed to a model
PROBE_INDEXES = range(4)  # camera indexes tried when listing
BLACK_MEAN = 3.0  # a frame this dark on average (0-255) with a dark peak is a covered lens
BLACK_PEAK = 24
WANTED_SIZE = (1280, 720)  # asked of the driver; it may give less


class CameraError(RuntimeError):
    """The camera could not be used; the message says why and what to try."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    width: int
    height: int
    backend: str


@dataclass(frozen=True)
class FrameStats:
    width: int
    height: int  # of the returned image
    native_width: int
    native_height: int
    sharpness: float  # variance of the Laplacian of the kept frame (higher = sharper)
    seconds: float  # open to release
    index: int
    backend: str


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise CameraError(
            "camera support needs OpenCV (the 'screen' extra): uv sync --extra screen"
        ) from exc
    return cv2


def _backend_ids(cv2: Any) -> list[tuple[str, int]]:
    ids = {
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "v4l2": getattr(cv2, "CAP_V4L2", None),
        "any": getattr(cv2, "CAP_ANY", 0),
    }
    wanted = plat.current().camera_backends()
    return [(name, ids[name]) for name in wanted if ids.get(name) is not None] or [("any", 0)]


def _open(cv2: Any, index: int) -> tuple[Any, str] | None:
    """The first backend that opens camera ``index``: (capture, backend name), or None."""
    for name, backend in _backend_ids(cv2):
        cap = cv2.VideoCapture(index, backend)
        try:
            if cap.isOpened():
                return cap, name
        except Exception:
            pass
        cap.release()
    return None


def _size(cv2: Any, cap: Any) -> tuple[int, int]:
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)


def list_cameras() -> list[CameraInfo]:
    """Cameras that open, by index. Opens each briefly (the LED may flash) and releases it."""
    cv2 = _cv2()
    found: list[CameraInfo] = []
    for index in PROBE_INDEXES:
        opened = _open(cv2, index)
        if opened is None:
            continue
        cap, backend = opened
        try:
            width, height = _size(cv2, cap)
        finally:
            cap.release()
        found.append(CameraInfo(index, width, height, backend))
    return found


def sharpness(cv2: Any, frame: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if getattr(frame, "ndim", 2) == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_black(frame: Any) -> bool:
    return float(frame.mean()) < BLACK_MEAN and int(frame.max()) < BLACK_PEAK


def why_not_open(index: int | None) -> CameraError:
    """Turn 'it would not open' into the most useful explanation available."""
    backend = plat.current()
    which = "any camera" if index is None else f"camera {index}"
    problem = backend.camera_access_problem()
    if problem:
        return CameraError(f"the system blocks camera access: {problem}")
    names = backend.camera_devices()
    if names == []:
        return CameraError(
            "no camera was found. Plug one in, or check that the laptop's camera is not disabled "
            "(Device Manager, a BIOS switch, a privacy shutter)."
        )
    if names:
        return CameraError(
            f"could not open {which}, although the system lists: {', '.join(names)}. Another app "
            "is probably using the camera (a video call, the Camera app, a browser tab). Close it and try again."
        )
    return CameraError(
        f"could not open {which}. Either no camera is connected, or another app is using it "
        "(a video call, the Camera app, a browser tab)."
    )


def _read_burst(cv2: Any, cap: Any, settle_frames: int, settle_seconds: float, burst: int) -> list[Any]:
    started = time.monotonic()
    discarded = 0
    while discarded < settle_frames or time.monotonic() - started < settle_seconds:
        cap.read()  # the result does not matter while settling
        discarded += 1
        if discarded > settle_frames * 20:  # a camera that returns instantly must not loop for long
            break
    frames: list[Any] = []
    for _ in range(burst):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0):
            frames.append(frame)
    return frames


def _to_image(cv2: Any, frame: Any, max_edge: int | None) -> "Image":
    from PIL import Image

    from jarvis.vision.capture import downscale

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return downscale(image, max_edge) if max_edge else image


def capture_with_stats(
    index: int | None = None,
    *,
    max_edge: int | None = MAX_EDGE,
    settle_frames: int = SETTLE_FRAMES,
    settle_seconds: float = SETTLE_SECONDS,
    burst: int = BURST_FRAMES,
) -> tuple["Image", FrameStats]:
    """Capture one sharp frame. ``index`` None tries 0, 1, ... until a camera opens.
    ``max_edge`` None keeps the camera's native resolution. Raises CameraError."""
    cv2 = _cv2()
    started = time.monotonic()
    candidates = [index] if index is not None else list(PROBE_INDEXES)
    cap = None
    chosen = -1
    backend = ""
    for candidate in candidates:
        opened = _open(cv2, candidate)
        if opened is not None:
            (cap, backend), chosen = opened, candidate
            break
    if cap is None:
        raise why_not_open(index)
    try:
        try:  # ask for HD; many webcams default to 640x480
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WANTED_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WANTED_SIZE[1])
        except Exception:
            log.debug("camera ignored the size request", exc_info=True)
        frames = _read_burst(cv2, cap, settle_frames, settle_seconds, burst)
    except CameraError:
        raise
    except Exception as exc:  # a driver error mid-read: say so, and still release below
        raise CameraError(f"camera {chosen} failed while reading a frame ({type(exc).__name__}: {exc})") from exc
    finally:
        cap.release()  # always: the camera LED must go off
    if not frames:
        raise CameraError(
            f"camera {chosen} opened but delivered no frames: another app has taken it, "
            "or the system is blocking it. Close other apps that use the camera and try again."
        )
    scored = [(sharpness(cv2, f), f) for f in frames]
    best_score, best = max(scored, key=lambda item: item[0])
    if _is_black(best):
        raise CameraError(
            "the camera image is completely black: the lens cover or privacy shutter is closed, "
            "or the room is dark."
        )
    height, width = int(best.shape[0]), int(best.shape[1])
    image = _to_image(cv2, best, max_edge)
    stats = FrameStats(
        width=image.width, height=image.height, native_width=width, native_height=height,
        sharpness=best_score, seconds=time.monotonic() - started, index=chosen, backend=backend,
    )
    return image, stats


def capture_frame(index: int | None = None, *, max_edge: int | None = MAX_EDGE) -> "Image":
    """One sharp frame as an RGB PIL image, longest edge at most ``max_edge`` (None: native)."""
    return capture_with_stats(index, max_edge=max_edge)[0]
