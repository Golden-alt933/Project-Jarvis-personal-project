"""Camera capture and the camera tools. No test touches a real camera, a model or the network:
OpenCV's VideoCapture is replaced with a scripted fake and the vision client with a stub."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
from PIL import Image  # noqa: E402

from jarvis.llm.types import ToolCall  # noqa: E402
from jarvis.tools.base import ToolContext  # noqa: E402
from jarvis.tools.builtin import camera as camera_tool  # noqa: E402
from jarvis.tools.registry import ToolRegistry  # noqa: E402
from jarvis.vision import camera as cam  # noqa: E402
from jarvis.vision.camera import CameraError  # noqa: E402
from jarvis.vision.client import VisionAnswer, VisionError  # noqa: E402


# ---------------------------------------------------------------- fake capture device
def noisy(amplitude: int, size=(48, 64)) -> "np.ndarray":
    """A BGR frame: a checkerboard of 128 +/- amplitude. Higher amplitude = sharper."""
    h, w = size
    board = (np.indices((h, w)).sum(axis=0) % 2) * 2 - 1
    gray = np.clip(128 + board * amplitude, 0, 255).astype(np.uint8)
    return np.dstack([gray, gray, gray])


class Fake:
    """Scripted VideoCapture. ``devices`` maps an index to the frames that camera returns."""

    devices: dict[int, list] = {}
    opened: list[tuple[int, int]] = []
    instances: list["Fake"] = []
    read_error: Exception | None = None
    sets: list = []

    def __init__(self, index, backend=None):
        self.index, self.backend = index, backend
        self.frames = list(Fake.devices.get(index, [])) if index in Fake.devices else None
        self.released = 0
        self.reads = 0
        Fake.instances.append(self)
        Fake.opened.append((index, backend))

    def isOpened(self):
        return self.frames is not None

    def set(self, prop, value):
        Fake.sets.append((prop, value))
        return True

    def get(self, prop):
        if self.frames and prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            shape = self.frames[0].shape
            return shape[1] if prop == cv2.CAP_PROP_FRAME_WIDTH else shape[0]
        return 0

    def read(self):
        if Fake.read_error:
            raise Fake.read_error
        self.reads += 1
        if not self.frames:
            return False, None
        frame = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        return frame is not None, frame

    def release(self):
        self.released += 1


@pytest.fixture(autouse=True)
def fake_camera(monkeypatch, fake_platform):
    Fake.devices, Fake.opened, Fake.instances, Fake.read_error, Fake.sets = {}, [], [], None, []
    monkeypatch.setattr(cv2, "VideoCapture", Fake)
    fake_platform.camera_backends = lambda: ["dshow", "msmf"]
    fake_platform.camera_access_problem = lambda: None
    fake_platform.camera_devices = lambda: None
    camera_tool.reset_consent()
    yield fake_platform
    camera_tool.reset_consent()


def fast(**kw):
    return dict(settle_frames=2, settle_seconds=0.0, burst=5, **kw)


# ---------------------------------------------------------------- capture
def test_settle_frames_are_discarded_and_the_sharpest_of_the_burst_is_kept():
    decoys = [noisy(120), noisy(120)]  # sharpest of all, but taken while exposure settles
    burst = [noisy(5), noisy(20), noisy(60), noisy(20), noisy(5)]
    Fake.devices[0] = decoys + burst
    image, stats = cam.capture_with_stats(0, **fast())
    low, high = image.convert("L").getextrema()
    assert (low, high) == (68, 188), "the amplitude-60 frame, not a settle decoy, must be returned"
    assert stats.sharpness == pytest.approx(cam.sharpness(cv2, burst[2]), rel=0.2)
    assert stats.index == 0 and stats.backend == "dshow"


def test_release_is_called_after_a_good_capture():
    Fake.devices[0] = [noisy(40)] * 8
    cam.capture_frame(0, **{})  # default settle time is short enough for the test
    assert [c.released for c in Fake.instances if c.frames is not None] == [1]


def test_release_is_called_when_reading_raises():
    Fake.devices[0] = [noisy(40)] * 8
    Fake.read_error = RuntimeError("driver exploded")
    with pytest.raises(CameraError, match="failed while reading"):
        cam.capture_with_stats(0, **fast())
    assert all(c.released == 1 for c in Fake.instances if c.frames is not None)


def test_release_is_called_when_no_frames_arrive():
    Fake.devices[0] = [None]
    with pytest.raises(CameraError, match="delivered no frames"):
        cam.capture_with_stats(0, **fast())
    assert Fake.instances[0].released == 1


def test_black_frame_means_a_covered_lens():
    Fake.devices[0] = [np.zeros((48, 64, 3), np.uint8)] * 8
    with pytest.raises(CameraError, match="completely black.*(lens cover|shutter)"):
        cam.capture_with_stats(0, **fast())
    assert Fake.instances[0].released == 1


def test_a_dark_but_real_scene_is_not_called_black():
    Fake.devices[0] = [noisy(15)] * 8  # mean ~128, only low contrast
    image, _ = cam.capture_with_stats(0, **fast())
    assert image.size == (64, 48)


def test_big_frames_are_downscaled_and_full_resolution_is_available():
    Fake.devices[0] = [np.full((1080, 1920, 3), 90, np.uint8)] * 8
    image, stats = cam.capture_with_stats(0, **fast())
    assert max(image.size) == 1280 and image.size == (1280, 720)
    assert (stats.native_width, stats.native_height) == (1920, 1080)
    Fake.instances.clear()
    full, stats2 = cam.capture_with_stats(0, max_edge=None, **fast())
    assert full.size == (1920, 1080) and stats2.width == 1920


def test_returns_an_rgb_pil_image():
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[:, :, 2] = 200  # BGR: red channel
    Fake.devices[0] = [frame] * 8
    image, _ = cam.capture_with_stats(0, **fast())
    assert isinstance(image, Image.Image) and image.getpixel((1, 1)) == (200, 0, 0)


def test_hd_is_requested_from_the_driver():
    Fake.devices[0] = [noisy(40)] * 8
    cam.capture_with_stats(0, **fast())
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in Fake.sets and (cv2.CAP_PROP_FRAME_HEIGHT, 720) in Fake.sets


def test_without_an_index_the_first_camera_that_opens_is_used():
    Fake.devices[1] = [noisy(40)] * 8
    _, stats = cam.capture_with_stats(None, **fast())
    assert stats.index == 1
    assert [i for i, _ in Fake.opened if i == 0], "camera 0 was tried first"


def test_an_explicit_index_is_the_only_one_tried():
    Fake.devices[0] = [noisy(40)] * 8
    with pytest.raises(CameraError):
        cam.capture_with_stats(2, **fast())
    assert {i for i, _ in Fake.opened} == {2}


def test_backends_are_tried_in_the_platforms_order(fake_camera):
    fake_camera.camera_backends = lambda: ["v4l2"]
    Fake.devices[0] = [noisy(40)] * 8
    cam.capture_with_stats(0, **fast())
    assert Fake.opened == [(0, cv2.CAP_V4L2)]


def test_second_backend_is_the_fallback():
    class OnlyMsmf(Fake):
        def isOpened(self):
            return self.backend == cv2.CAP_MSMF and self.frames is not None

    cv2.VideoCapture = OnlyMsmf  # the autouse fixture restores the attribute afterwards
    Fake.devices[0] = [noisy(40)] * 8
    _, stats = cam.capture_with_stats(0, **fast())
    assert stats.backend == "msmf"


# ---------------------------------------------------------------- failure messages
def test_no_camera_at_all(fake_camera):
    fake_camera.camera_devices = lambda: []
    with pytest.raises(CameraError, match="no camera was found"):
        cam.capture_with_stats(None, **fast())


def test_camera_busy_names_the_device_and_the_cause(fake_camera):
    fake_camera.camera_devices = lambda: ["Integrated Camera"]
    with pytest.raises(CameraError) as err:
        cam.capture_with_stats(0, **fast())
    assert "Integrated Camera" in str(err.value) and "another app" in str(err.value).lower()


def test_permission_denied_names_the_windows_setting(fake_camera):
    fake_camera.camera_access_problem = lambda: (
        "'Let desktop apps access your camera' is off. Turn it on in Settings > Privacy & security > Camera."
    )
    with pytest.raises(CameraError) as err:
        cam.capture_with_stats(0, **fast())
    assert "blocks camera access" in str(err.value) and "Privacy & security > Camera" in str(err.value)


def test_unknown_cause_covers_absent_and_busy():
    with pytest.raises(CameraError, match="no camera is connected, or another app"):
        cam.capture_with_stats(0, **fast())


def test_missing_opencv_says_which_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *a, **k):
        if name == "cv2":
            raise ImportError("no cv2", name="cv2")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(CameraError, match="screen"):
        cam.capture_with_stats(0)


def test_list_cameras_reports_each_and_releases_them():
    Fake.devices[0] = [np.zeros((720, 1280, 3), np.uint8)]
    Fake.devices[2] = [np.zeros((480, 640, 3), np.uint8)]
    found = cam.list_cameras()
    assert [(c.index, c.width, c.height) for c in found] == [(0, 1280, 720), (2, 640, 480)]
    assert all(c.released >= 1 for c in Fake.instances), "every probed device is released, opened or not"


# ---------------------------------------------------------------- tools
class StubVision:
    def __init__(self, answer=None, error=None):
        self.answer = answer or VisionAnswer("A red mug on a desk.", "nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia")
        self.error, self.asked = error, []

    async def ask(self, image, question):
        self.asked.append((image.size, question))
        if self.error:
            raise self.error
        return self.answer


def make_ctx(tmp_path, mode="ask", confirm=None, index=""):
    asked: list[str] = []

    async def default_confirm(text: str) -> bool:
        asked.append(text)
        return True

    async def wrapped(text: str) -> bool:
        asked.append(text)
        return await confirm(text)

    settings = SimpleNamespace(camera=mode, camera_index=index, data_dir=tmp_path, vision_provider="auto")
    ctx = ToolContext(settings=settings, confirm=wrapped if confirm else default_confirm)
    return ctx, asked


@pytest.fixture
def stub(monkeypatch):
    stub = StubVision()
    monkeypatch.setattr(camera_tool, "build_vision_client", lambda settings: stub)
    Fake.devices[0] = [noisy(40, (72, 128))] * 12
    monkeypatch.setattr(cam, "SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(cam, "SETTLE_FRAMES", 2)
    return stub


async def test_look_through_camera_answers_names_the_model_and_saves_the_frame(tmp_path, stub):
    ctx, asked = make_ctx(tmp_path, "on")
    result = await camera_tool.look_through_camera("what is on the desk?", ctx=ctx)
    assert result.ok
    assert "A red mug on a desk." in result.content
    assert "nvidia/nemotron-3.5-super-vl-120b-a12b" in result.content and "sent to it" in result.content
    saved = tmp_path / "camera" / "last.jpg"
    assert result.data["image_path"] == str(saved) and saved.is_file()
    with Image.open(saved) as shown:
        assert shown.size == (128, 72)
    question = stub.asked[0][1]
    assert "what is on the desk?" in question and "say so" in question and "text" in question
    assert asked == []


async def test_the_last_frame_file_is_overwritten_each_time(tmp_path, stub):
    ctx, _ = make_ctx(tmp_path, "on")
    await camera_tool.look_through_camera(ctx=ctx)
    Fake.devices[0] = [noisy(90, (36, 64))] * 12
    result = await camera_tool.look_through_camera(ctx=ctx)
    with Image.open(result.data["image_path"]) as shown:
        assert shown.size == (64, 36)
    assert [p.name for p in (tmp_path / "camera").iterdir()] == ["last.jpg"]


async def test_local_vision_says_the_frame_stayed_here(tmp_path, stub):
    stub.answer = VisionAnswer("A mug.", "qwen2.5vl:3b", "local")
    ctx, _ = make_ctx(tmp_path, "on")
    result = await camera_tool.look_through_camera(ctx=ctx)
    assert "locally" in result.content and "stayed on this machine" in result.content


async def test_vision_failure_is_reported_after_the_camera_worked(tmp_path, stub):
    stub.error = VisionError("hosted model timed out")
    ctx, _ = make_ctx(tmp_path, "on")
    result = await camera_tool.look_through_camera(ctx=ctx)
    assert not result.ok and "vision model is unavailable" in result.content


async def test_camera_failure_is_a_clean_tool_failure(tmp_path, stub):
    Fake.devices.clear()
    ctx, _ = make_ctx(tmp_path, "on")
    result = await camera_tool.look_through_camera(ctx=ctx)
    assert not result.ok and result.content.startswith("Error:") and "camera" in result.content


async def test_camera_index_setting_and_argument(tmp_path, stub):
    Fake.devices[1] = [noisy(40)] * 12
    ctx, _ = make_ctx(tmp_path, "on", index="1")
    assert (await camera_tool.look_through_camera(ctx=ctx)).data["camera"] == 1
    ctx, _ = make_ctx(tmp_path, "on", index="1")
    assert (await camera_tool.look_through_camera(camera=0, ctx=ctx)).data["camera"] == 0
    ctx, _ = make_ctx(tmp_path, "on", index="front")
    bad = await camera_tool.look_through_camera(ctx=ctx)
    assert not bad.ok and "JARVIS_CAMERA_INDEX" in bad.content


# ---------------------------------------------------------------- consent: ask / on / off
async def test_ask_mode_asks_once_and_remembers_a_yes(tmp_path, stub):
    ctx, asked = make_ctx(tmp_path, "ask")
    assert (await camera_tool.look_through_camera(ctx=ctx)).ok
    assert (await camera_tool.look_through_camera(ctx=ctx)).ok
    assert asked == ["Turn on the camera and send one frame to the vision model"]


async def test_ask_mode_does_not_remember_a_no(tmp_path, stub):
    async def no(_: str) -> bool:
        return False

    ctx, asked = make_ctx(tmp_path, "ask", confirm=no)
    first = await camera_tool.look_through_camera(ctx=ctx)
    second = await camera_tool.look_through_camera(ctx=ctx)
    assert not first.ok and not second.ok and "did not allow" in first.content
    assert len(asked) == 2, "a refusal must be asked about again"
    assert Fake.instances == [], "the camera was never opened"
    assert stub.asked == []


async def test_a_broken_confirm_callback_counts_as_no(tmp_path, stub):
    async def broken(_: str) -> bool:
        raise RuntimeError("frontend died")

    ctx, _ = make_ctx(tmp_path, "ask", confirm=broken)
    assert not (await camera_tool.look_through_camera(ctx=ctx)).ok
    assert Fake.instances == []


async def test_on_mode_never_asks(tmp_path, stub):
    ctx, asked = make_ctx(tmp_path, "on")
    for _ in range(2):
        assert (await camera_tool.look_through_camera(ctx=ctx)).ok
    assert (await camera_tool.take_photo(save_to=str(tmp_path / "a.jpg"), ctx=ctx)).ok
    assert asked == []


async def test_off_mode_fails_with_how_to_enable_and_touches_nothing(tmp_path, stub):
    ctx, asked = make_ctx(tmp_path, "off")
    for call in (
        camera_tool.look_through_camera(ctx=ctx),
        camera_tool.take_photo(ctx=ctx),
        camera_tool.list_cameras(ctx=ctx),
    ):
        result = await call
        assert not result.ok and "JARVIS_CAMERA=ask" in result.content
    assert asked == [] and Fake.instances == []


async def test_unknown_mode_is_treated_as_ask(tmp_path, stub):
    ctx, asked = make_ctx(tmp_path, "sometimes")
    assert (await camera_tool.look_through_camera(ctx=ctx)).ok
    assert len(asked) == 1


async def test_a_yes_to_a_photo_does_not_cover_sending_to_a_model(tmp_path, stub):
    ctx, asked = make_ctx(tmp_path, "ask")
    assert (await camera_tool.take_photo(save_to=str(tmp_path / "p.jpg"), ctx=ctx)).ok
    assert (await camera_tool.look_through_camera(ctx=ctx)).ok
    assert asked == ["Turn on the camera", "Turn on the camera and send one frame to the vision model"]
    assert (await camera_tool.take_photo(save_to=str(tmp_path / "q.jpg"), ctx=ctx)).ok
    assert len(asked) == 2, "a yes to sending covers the lesser photo"


async def test_works_through_the_real_registry(tmp_path, stub):
    registry = ToolRegistry().load_builtin()
    assert {"look_through_camera", "take_photo", "list_cameras"} <= {s.name for s in registry.specs(["camera"])}
    ctx, asked = make_ctx(tmp_path, "ask")
    result = await registry.execute(ToolCall(id="1", name="look_through_camera", arguments={"question": "what is this?"}), ctx)
    assert result.ok and result.data["image_path"].endswith("last.jpg") and len(asked) == 1
    bad = await registry.execute(ToolCall(id="2", name="look_through_camera", arguments={"nope": 1}), ctx)
    assert not bad.ok and "invalid arguments" in bad.content


# ---------------------------------------------------------------- take_photo, list_cameras
async def test_take_photo_saves_full_resolution_and_never_overwrites(tmp_path, stub, fake_camera, monkeypatch):
    Fake.devices[0] = [np.full((1080, 1920, 3), 120, np.uint8)] * 12
    fake_camera.known_folder = lambda name: tmp_path / name
    monkeypatch.setattr(camera_tool.time, "strftime", lambda fmt: "photo-fixed.jpg")
    ctx, _ = make_ctx(tmp_path, "on")
    first = await camera_tool.take_photo(ctx=ctx)
    second = await camera_tool.take_photo(ctx=ctx)
    assert first.ok and second.ok and first.data["path"] != second.data["path"]
    assert first.data["path"].endswith("Pictures\\Jarvis\\photo-fixed.jpg") or first.data["path"].endswith("Pictures/Jarvis/photo-fixed.jpg")
    assert second.data["path"].endswith("photo-fixed-1.jpg")
    with Image.open(first.data["path"]) as shown:
        assert shown.size == (1920, 1080), "photos are not downscaled"
    assert stub.asked == [], "a photo is never sent to a model"


async def test_take_photo_refuses_an_existing_file(tmp_path, stub):
    target = tmp_path / "mine.jpg"
    target.write_bytes(b"precious")
    ctx, _ = make_ctx(tmp_path, "on")
    result = await camera_tool.take_photo(save_to=str(target), ctx=ctx)
    assert not result.ok and "already exists" in result.content and target.read_bytes() == b"precious"


async def test_take_photo_into_a_folder_and_default_extension(tmp_path, stub):
    ctx, _ = make_ctx(tmp_path, "on")
    folder = tmp_path / "shots"
    folder.mkdir()
    in_folder = await camera_tool.take_photo(save_to=str(folder), ctx=ctx)
    named = await camera_tool.take_photo(save_to=str(tmp_path / "selfie"), ctx=ctx)
    assert in_folder.data["path"].startswith(str(folder)) and in_folder.data["path"].endswith(".jpg")
    assert named.data["path"].endswith("selfie.jpg")


async def test_list_cameras_tool(tmp_path, stub, fake_camera):
    Fake.devices[1] = [np.zeros((480, 640, 3), np.uint8)]
    fake_camera.camera_devices = lambda: ["Integrated Camera"]
    ctx, _ = make_ctx(tmp_path, "on")
    result = await camera_tool.list_cameras(ctx=ctx)
    assert result.ok and "camera 0: 128x72" in result.content
    assert "camera 1: 640x480" in result.content and "Integrated Camera" in result.content
    assert result.data == [0, 1]


async def test_list_cameras_when_there_is_none(tmp_path, stub, fake_camera):
    Fake.devices.clear()
    fake_camera.camera_devices = lambda: []
    ctx, _ = make_ctx(tmp_path, "on")
    result = await camera_tool.list_cameras(ctx=ctx)
    assert not result.ok and "no camera was found" in result.content
