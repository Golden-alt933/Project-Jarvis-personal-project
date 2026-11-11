"""Tests for the screen tools. No test captures the real screen or touches a network/model."""

from __future__ import annotations

import os
import random
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image, ImageDraw, ImageFont

from jarvis.llm.types import ToolCall
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import screen
from jarvis.tools.registry import ToolRegistry
from jarvis.vision import capture, ocr
from jarvis.vision.client import NVIDIA_VISION_MODEL, VisionClient, VisionError, build_vision_client
from jarvis.vision.ocr import OcrLine, layout_text


# ---------------------------------------------------------------- helpers
def make_settings(**over):
    base = dict(
        provider="nvidia", vision_provider="auto", vision_model="", nvidia_api_key="k-test",
        groq_api_key="", openrouter_api_key="", gemini_api_key="", custom_base_url="",
        custom_api_key="", ollama_base_url="http://localhost:11434",
        ollama_vision_model="qwen2.5vl:3b", request_timeout=90.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def line(text, x0, y0, x1=None, y1=None):
    return OcrLine(text, 0.9, x0, y0, x1 if x1 is not None else x0 + 9 * len(text), y1 if y1 is not None else y0 + 20)


def make_image(size=(400, 300)):
    return Image.new("RGB", size, "white")


class Recorder:
    """Fake HTTP layer: records requests and replies per URL fragment."""

    def __init__(self, cloud=(200, "cloud says hi"), local=(200, "local says hi")):
        self.cloud, self.local, self.requests = cloud, local, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        import json

        self.requests.append((str(request.url), json.loads(request.content), dict(request.headers)))
        if request.url.path.endswith("/chat/completions"):
            status, text = self.cloud
            body = {"choices": [{"message": {"content": text}}]} if status == 200 else {"error": "x"}
        else:
            status, text = self.local
            body = {"message": {"content": text}} if status == 200 else {"error": "x"}
        return httpx.Response(status, json=body)


def client_with(recorder, **settings_over):
    http = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return build_vision_client(make_settings(**settings_over), http=http)


# ---------------------------------------------------------------- layout_text
def test_layout_text_rows_and_reading_order_from_shuffled_boxes():
    boxes = [
        line("Header title", 10, 10),
        line("first line of paragraph", 10, 50),
        line("second line of paragraph", 10, 76),
        line("third line", 10, 102),
    ]
    random.Random(3).shuffle(boxes)
    assert layout_text(boxes).splitlines() == [
        "Header title", "first line of paragraph", "second line of paragraph", "third line",
    ]


def test_layout_text_two_columns_stay_on_their_rows():
    boxes = [
        line("Name", 10, 10), line("Price", 600, 11),
        line("Apple", 10, 40), line("1.20", 600, 41),
        line("Banana", 10, 70), line("0.50", 600, 69),
    ]
    random.Random(7).shuffle(boxes)
    rows = layout_text(boxes).splitlines()
    assert [r.split() for r in rows] == [["Name", "Price"], ["Apple", "1.20"], ["Banana", "0.50"]]
    assert "    " in rows[0]  # the wide gap is kept visible


def test_layout_text_joins_adjacent_boxes_with_one_space_and_handles_empty():
    assert layout_text([line("world", 70, 10, 115), line("Hello", 10, 12, 60)]) == "Hello world"
    assert layout_text([]) == ""


# ---------------------------------------------------------------- capture maths
def test_scaled_size_keeps_aspect_and_never_enlarges():
    assert capture.scaled_size((3840, 2160), 1568) == (1568, 882)
    assert capture.scaled_size((1000, 500), 1568) == (1000, 500)
    assert capture.scaled_size((1000, 2000), 500) == (250, 500)
    assert capture.scaled_size((3840, 2160), capture.FULL_RES) == (3840, 2160)


def test_downscale_hits_the_target_size_via_reduce_and_resize():
    big = Image.new("RGB", (3840, 2160), "white")
    assert capture.downscale(big, 896).size == (896, 504)
    assert capture.downscale(big, 1568).size == (1568, 882)
    small = make_image((200, 100))
    assert capture.downscale(small, 896) is small


def test_pick_monitor_prefers_cursor_then_primary():
    mons = [
        {"left": 0, "top": 0, "width": 4480, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": 0, "width": 2560, "height": 1080},
    ]
    assert capture.pick_monitor(mons, "active", (2000, 10)) is mons[2]
    assert capture.pick_monitor(mons, "active", None) is mons[1]
    assert capture.pick_monitor(mons, "all", (5, 5)) is mons[0]
    assert capture.pick_monitor(mons, 2, None) is mons[2]
    with pytest.raises(capture.CaptureError):
        capture.pick_monitor(mons, 5, None)


def test_encoding_helpers_round_trip():
    img = make_image((64, 32))
    assert capture.to_jpeg_bytes(img)[:2] == b"\xff\xd8"
    assert capture.to_png_bytes(img)[:4] == b"\x89PNG"
    assert capture.data_uri(img).startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------- client payloads
async def test_cloud_payload_has_image_url_part_and_bearer_key():
    rec = Recorder()
    answer = await client_with(rec).ask(make_image((2000, 1000)), "what is this?")
    url, body, headers = rec.requests[0]
    assert url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert headers["authorization"] == "Bearer k-test"
    assert body["model"] == NVIDIA_VISION_MODEL
    parts = body["messages"][-1]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert (answer.text, answer.model, answer.path) == ("cloud says hi", NVIDIA_VISION_MODEL, "nvidia")


async def test_local_payload_uses_native_ollama_chat_with_images():
    rec = Recorder()
    answer = await client_with(rec, vision_provider="local").ask(make_image(), "q")
    url, body, _ = rec.requests[0]
    assert url == "http://localhost:11434/api/chat"
    assert body["model"] == "qwen2.5vl:3b" and body["stream"] is False
    assert body["keep_alive"] == "5m" and body["options"]["num_predict"] == 400
    user = body["messages"][-1]
    assert user["content"] == "q" and len(user["images"]) == 1 and not user["images"][0].startswith("data:")
    assert (answer.text, answer.path) == ("local says hi", "local")


async def test_local_refuses_non_loopback_host():
    rec = Recorder()
    with pytest.raises(VisionError, match="non-loopback"):
        await client_with(rec, vision_provider="local", ollama_base_url="http://example.com:11434").ask(make_image(), "q")
    assert rec.requests == []


# ---------------------------------------------------------------- selection and fallback
def test_auto_picks_cloud_when_key_and_model_exist_else_local():
    assert build_vision_client(make_settings()).order() == ["cloud", "local"]
    assert build_vision_client(make_settings(nvidia_api_key="")).order() == ["local"]
    # other providers have no known-good default: they need JARVIS_VISION_MODEL
    assert build_vision_client(make_settings(provider="groq", groq_api_key="g")).order() == ["local"]
    assert build_vision_client(make_settings(provider="groq", groq_api_key="g", vision_model="m")).order() == ["cloud", "local"]
    assert build_vision_client(make_settings(provider="ollama")).order() == ["local"]


def test_forced_modes_and_local_never_leaks_to_cloud():
    assert build_vision_client(make_settings(vision_provider="cloud")).order() == ["cloud", "local"]
    assert build_vision_client(make_settings(vision_provider="local")).order() == ["local"]


def test_vision_model_setting_overrides_default():
    assert build_vision_client(make_settings(vision_model="my/model")).cloud_model == "my/model"


async def test_cloud_failure_falls_back_to_local_and_says_so():
    rec = Recorder(cloud=(404, ""))
    answer = await client_with(rec).ask(make_image(), "q")
    assert answer.path == "local" and answer.text == "local says hi"
    assert [u.endswith("/api/chat") for u, _, _ in rec.requests] == [False, True]


async def test_a_busy_hosted_model_is_tried_once_more_before_the_local_one():
    rec = Recorder(cloud=(503, ""))
    answer = await client_with(rec).ask(make_image(), "q")
    assert answer.path == "local"
    assert [u.endswith("/api/chat") for u, _, _ in rec.requests] == [False, False, True]  # hosted twice, then local


async def test_no_local_model_is_loaded_when_local_fallback_is_off():
    assert build_vision_client(make_settings(local_fallback=False)).order() == ["cloud"]
    assert build_vision_client(make_settings(local_fallback=False, vision_provider="local")).order() == ["local"]
    rec = Recorder(cloud=(404, ""))
    with pytest.raises(VisionError):
        await client_with(rec, local_fallback=False).ask(make_image(), "q")
    assert not any(u.endswith("/api/chat") for u, _, _ in rec.requests)


async def test_local_mode_does_not_try_cloud_when_local_fails():
    rec = Recorder(local=(404, ""))
    with pytest.raises(VisionError, match="local"):
        await client_with(rec, vision_provider="local").ask(make_image(), "q")
    assert all(u.endswith("/api/chat") for u, _, _ in rec.requests)


async def test_both_paths_failing_reports_both():
    rec = Recorder(cloud=(429, ""), local=(404, ""))
    with pytest.raises(VisionError) as info:
        await client_with(rec).ask(make_image(), "q")
    assert "cloud" in str(info.value) and "local" in str(info.value) and "k-test" not in str(info.value)


async def test_empty_cloud_answer_counts_as_failure():
    rec = Recorder(cloud=(200, "  "))
    assert (await client_with(rec).ask(make_image(), "q")).path == "local"


# ---------------------------------------------------------------- tools via the registry
class FakeVisionClient:
    def __init__(self):
        self.calls = []

    async def ask(self, image, question):
        from jarvis.vision.client import VisionAnswer

        self.calls.append((image.size, question))
        return VisionAnswer("a red circle", "fake/model", "nvidia")


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register_module(screen)
    return reg


@pytest.fixture
def ctx():
    return ToolContext(settings=make_settings())


@pytest.fixture
def fake_screen(monkeypatch):
    monkeypatch.setattr(screen, "capture_screen", lambda monitor="active", max_edge=None: make_image((800, 600)))
    monkeypatch.setattr(screen, "capture_window_title", lambda: "Notepad - todo.txt")
    monkeypatch.setattr(screen, "ocr_image", lambda image: [line("World", 65, 10), line("Hello", 10, 12, 60)])


async def run(registry, ctx, name, **args):
    return await registry.execute(ToolCall(id="1", name=name, arguments=args), ctx)


def test_load_builtin_registers_screen_tools():
    names = {s.name for s in ToolRegistry().load_builtin().specs(["screen"])}
    assert names == {"read_screen_text", "look_at_screen", "take_screenshot", "describe_image"}


def test_schema_hides_ctx_and_timeouts_cover_vision(registry):
    look = registry.get("look_at_screen")
    assert "ctx" not in look.parameters["properties"] and look.parameters["required"] == ["question"]
    assert look.timeout >= 120 and registry.get("describe_image").timeout >= 120
    assert "prefer read_screen_text" in look.description


async def test_read_screen_text_returns_title_and_reading_order(registry, ctx, fake_screen):
    result = await run(registry, ctx, "read_screen_text")
    assert result.ok
    assert "Foreground window: Notepad - todo.txt" in result.content
    assert "Hello World" in result.content


async def test_read_screen_text_reports_empty_screen(registry, ctx, fake_screen, monkeypatch):
    monkeypatch.setattr(screen, "ocr_image", lambda image: [])
    assert "No text was recognised" in (await run(registry, ctx, "read_screen_text")).content


async def test_read_screen_text_capture_error_is_a_failed_result(registry, ctx, monkeypatch):
    def boom(monitor="active", max_edge=None):
        raise capture.CaptureError("no display")

    monkeypatch.setattr(screen, "capture_screen", boom)
    monkeypatch.setattr(screen, "capture_window_title", lambda: "")
    result = await run(registry, ctx, "read_screen_text")
    assert not result.ok and "no display" in result.content


async def test_look_at_screen_puts_title_ocr_and_question_in_prompt_and_names_model(
    registry, ctx, fake_screen, monkeypatch
):
    fake = FakeVisionClient()
    monkeypatch.setattr(screen, "build_vision_client", lambda settings: fake)
    result = await run(registry, ctx, "look_at_screen", question="what shape?")
    assert result.ok
    assert "fake/model" in result.content and "nvidia" in result.content and "a red circle" in result.content
    prompt = fake.calls[0][1]
    assert "Notepad - todo.txt" in prompt and "Hello World" in prompt and "what shape?" in prompt


async def test_look_at_screen_still_works_when_ocr_breaks(registry, ctx, fake_screen, monkeypatch):
    def broken(image):
        raise RuntimeError("no engine")

    monkeypatch.setattr(screen, "ocr_image", broken)
    fake = FakeVisionClient()
    monkeypatch.setattr(screen, "build_vision_client", lambda settings: fake)
    assert (await run(registry, ctx, "look_at_screen", question="q")).ok
    assert "OCR" not in fake.calls[0][1]


async def test_look_at_screen_failure_points_to_ocr(registry, ctx, fake_screen, monkeypatch):
    class Down:
        async def ask(self, image, question):
            raise VisionError("cloud: down; local: down")

    monkeypatch.setattr(screen, "build_vision_client", lambda settings: Down())
    result = await run(registry, ctx, "look_at_screen", question="q")
    assert not result.ok and "read_screen_text" in result.content


async def test_take_screenshot_default_path_uses_pictures_screenshots(registry, ctx, fake_screen, monkeypatch, tmp_path):
    monkeypatch.setattr(screen, "_pictures_dir", lambda: tmp_path)
    result = await run(registry, ctx, "take_screenshot")
    assert result.ok and "800x600" in result.content
    saved = list((tmp_path / "Screenshots").glob("jarvis-*.png"))
    assert len(saved) == 1 and saved[0].read_bytes()[:4] == b"\x89PNG"
    assert str(saved[0]) in result.content


async def test_take_screenshot_custom_path_folder_and_no_overwrite(registry, ctx, fake_screen, tmp_path):
    target = tmp_path / "shot.png"
    assert (await run(registry, ctx, "take_screenshot", save_to=str(target))).ok and target.exists()
    again = await run(registry, ctx, "take_screenshot", save_to=str(target))
    assert not again.ok and "already exists" in again.content
    into_dir = await run(registry, ctx, "take_screenshot", save_to=str(tmp_path))
    assert into_dir.ok and len(list(tmp_path.glob("jarvis-*.png"))) == 1


async def test_describe_image_uses_vision_path(registry, ctx, tmp_path, monkeypatch):
    path = tmp_path / "pic.png"
    make_image((320, 200)).save(path)
    fake = FakeVisionClient()
    monkeypatch.setattr(screen, "build_vision_client", lambda settings: fake)
    result = await run(registry, ctx, "describe_image", path=str(path), question="colour?")
    assert result.ok and "fake/model" in result.content
    assert fake.calls == [((320, 200), "colour?")]


async def test_describe_image_rejects_unsupported_and_missing(registry, ctx, tmp_path):
    gif = tmp_path / "a.gif"
    gif.write_bytes(b"GIF89a")
    bad = await run(registry, ctx, "describe_image", path=str(gif))
    assert not bad.ok and "unsupported image type" in bad.content
    missing = await run(registry, ctx, "describe_image", path=str(tmp_path / "nope.png"))
    assert not missing.ok and "does not exist" in missing.content
    fake_png = tmp_path / "broken.png"
    fake_png.write_bytes(b"not an image")
    assert "could not open" in (await run(registry, ctx, "describe_image", path=str(fake_png))).content


# ---------------------------------------------------------------- real OCR (slow)
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("JARVIS_SLOW_TESTS") != "1", reason="set JARVIS_SLOW_TESTS=1 to run real OCR")
def test_real_ocr_reads_clear_text():
    img = Image.new("RGB", (900, 200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 48)
    draw.text((30, 30), "INVOICE 4172", fill="black", font=font)
    draw.text((30, 110), "Total due today", fill="black", font=font)
    text = layout_text(ocr.ocr_image(img))
    assert "4172" in text and "Total" in text
    assert text.index("INVOICE") < text.index("Total")
