"""HUD tests: offscreen Qt, deterministic time, no audio, no model."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QRect, Qt  # noqa: E402
from PyQt6.QtGui import QImage, QPainter  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from jarvis.hud.codepanel import MAX_HEIGHT  # noqa: E402
from jarvis.hud.demo import DemoSession  # noqa: E402
from jarvis.hud.orb import OrbWidget  # noqa: E402
from jarvis.hud.sysmon import Sample, take_sample  # noqa: E402
from jarvis.hud.window import COMPACT, HudWindow, clamp_rect  # noqa: E402
from jarvis.interfaces import cli  # noqa: E402
from jarvis.voice.events import (  # noqa: E402
    CodeShown,
    ConfirmRequested,
    ConfirmResolved,
    Heard,
    Info,
    Levels,
    MicMuted,
    Spoke,
    StateChanged,
    ToolActivity,
    TurnFinished,
    UserText,
    VoiceState,
)

_app = QApplication.instance() or QApplication([])


class FakeSession:
    has_audio = True

    def __init__(self) -> None:
        self.muted = False
        self.calls: list[tuple[str, Any]] = []

    def submit_text(self, text: str) -> None:
        self.calls.append(("submit", text))

    def cancel(self) -> None:
        self.calls.append(("cancel", None))

    def answer_confirm(self, approved: bool) -> None:
        self.calls.append(("confirm", approved))

    def set_muted(self, muted: bool) -> None:
        self.calls.append(("mute", muted))
        self.muted = muted


@pytest.fixture
def hud(tmp_path: Path):
    w = HudWindow(tmp_path, native=False, auto_tick=False, sysmon=False)
    w.show()
    w.set_mode("expanded", animate=False)
    session = FakeSession()
    w.controller.session = session
    yield w, session
    w.close()


def run(w: HudWindow, seconds: float, step: float = 0.016) -> None:
    for _ in range(int(seconds / step)):
        w.advance(step)


# -- controller ----------------------------------------------------------------------
def test_state_events_drive_orb_label_and_input(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    assert w.state_label.text == "STANDING BY"
    for state, word in ((VoiceState.LISTENING, "LISTENING"), (VoiceState.THINKING, "THINKING"),
                        (VoiceState.SPEAKING, "SPEAKING")):
        h(StateChanged(state))
        assert w.orb.state is state and w.state_label.text == word and w.input.busy
    h(StateChanged(VoiceState.IDLE))
    assert w.state_label.text == "STANDING BY" and not w.input.busy


def test_loading_label_and_levels(hud) -> None:
    w, _ = hud
    w.controller.set_loading(True)
    assert w.state_label.text == "WAKING UP" and w.orb.loading
    w.controller.set_loading(False)
    w.controller.handle(Levels(0.7, 0.2))
    assert (w.orb.mic_in, w.orb.spk_in) == (0.7, 0.2)


def test_heard_addressed_starts_exchange_unaddressed_whispers(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(Heard("the weather is nice", False, 0.8))
    assert w.reply.exchanges == [] and w.reply.whisper_text
    h(Heard("jarvis what time is it", True, 0.9))
    assert w.reply.current().user == "jarvis what time is it"
    h(UserText("typed question"))
    assert w.reply.current().user == "typed question" and len(w.reply.exchanges) == 2


def test_reveal_accumulates_spoke_sentences(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(Heard("hi", True, 0.9))
    h(Spoke("Good morning."))
    h(Spoke("You have two meetings today."))
    assert w.reply.revealed_text() == ""  # nothing shown before time passes
    w.advance(0.2)
    assert 0 < len(w.reply.revealed_text().split()) < 7
    run(w, 3.0)
    assert w.reply.revealed_text() == "Good morning. You have two meetings today."
    assert w.reply.reply_text() == w.reply.revealed_text()


def test_turn_finished_footer_and_fallback_text(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(UserText("q"))
    h(TurnFinished("Answer without spoken sentences.", 6.24, 3, 3, "nemotron-3-ultra", "done"))
    assert w.footer.text == "nemotron-3-ultra · 6.2 s · 3 tools"
    assert w.reply.reply_text() == "Answer without spoken sentences."
    h(UserText("q2"))
    h(Spoke("Spoken."))
    h(TurnFinished("Spoken.", 1.0, 1, 0, "m", "done"))
    assert w.reply.reply_text() == "Spoken."  # not duplicated
    h(TurnFinished("", 1.0, 1, 0, "m", "error"))
    assert w.orb.error_left > 0


def test_info_confirm_and_mute_events(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(Info("heads up"))
    assert w.reply.whisper_text == "heads up"
    h(ConfirmRequested("Delete the file?"))
    assert w.confirm.open and w.confirm.description == "Delete the file?" and w.orb.confirming
    h(ConfirmResolved(False))
    assert not w.confirm.open and not w.orb.confirming
    h(MicMuted(True))
    assert w.header.mic.muted and w.orb.muted and w.state_label.text == "MUTED"


# -- ticker --------------------------------------------------------------------------
def test_ticker_chip_lifecycle(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(ToolActivity("web_search", False, summary="query=cats"))
    assert w.ticker.running_count() == 1 and w.orb.tool_running
    run(w, 0.5)
    h(ToolActivity("web_search", True, True, "query=cats", 1.4))
    chip = w.ticker.visible_chips()[0]
    assert chip.state == "ok" and chip.seconds == 1.4 and not w.orb.tool_running
    run(w, 4.3)
    assert w.ticker.visible_chips() == []
    run(w, 0.6)
    assert w.ticker.chips == []


def test_ticker_failure_and_cap(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(ToolActivity("a", False))
    h(ToolActivity("a", True, False, "", 0.2))
    assert w.ticker.visible_chips()[0].state == "fail"
    for i in range(7):
        h(ToolActivity(f"t{i}", False))
    assert len(w.ticker.visible_chips()) == 4
    assert [c.name for c in w.ticker.visible_chips()] == ["t3", "t4", "t5", "t6"]


# -- confirm card, input, mic ----------------------------------------------------------
def test_confirm_buttons_call_answer_confirm(hud) -> None:
    w, session = hud
    w.controller.handle(ConfirmRequested("Send it?"))
    w.confirm.allow.clicked.emit()
    w.confirm.deny.clicked.emit()
    assert session.calls == [("confirm", True), ("confirm", False)]


def test_input_enter_submits_and_clears(hud) -> None:
    w, session = hud
    w.input.edit.setText("  what is the weather  ")
    w.input.edit.returnPressed.emit()
    assert session.calls == [("submit", "what is the weather")]
    assert w.input.edit.text() == ""
    w.input.edit.returnPressed.emit()  # empty: nothing is sent
    assert len(session.calls) == 1


def test_stop_button_and_escape_cancel(hud) -> None:
    w, session = hud
    w.controller.handle(StateChanged(VoiceState.THINKING))
    run(w, 0.5)
    assert w.input.stop.isVisible()
    w.input.stop_requested.emit()
    w.controller.cancel()
    assert [c for c, _ in session.calls] == ["cancel", "cancel"]


def test_mic_toggle_calls_set_muted(hud) -> None:
    w, session = hud
    w.header.mic.clicked.emit()
    assert session.calls == [("mute", True)]
    w.header.mic.clicked.emit()
    assert session.calls[-1] == ("mute", False)


# -- window modes and persistence ----------------------------------------------------------
def test_mode_toggle_and_state_file(tmp_path: Path) -> None:
    w = HudWindow(tmp_path, native=False, auto_tick=False, sysmon=False)
    w.show()
    assert w.mode == "compact" and w.size() == COMPACT
    w.toggle_mode()
    assert w.mode == "expanded"
    for _ in range(40):
        _app.processEvents()
        if w._anim is None:
            break
        import time

        time.sleep(0.02)
    assert w.mode_t == 1.0 and w.width() > COMPACT.width()
    saved = json.loads((tmp_path / "hud.json").read_text())
    assert saved["mode"] == "expanded"
    w.close()
    again = HudWindow(tmp_path, native=False, auto_tick=False, sysmon=False)
    assert again.mode == "expanded"
    again.close()


def test_position_restore_is_clamped_to_screen(tmp_path: Path) -> None:
    (tmp_path / "hud.json").write_text(json.dumps({"mode": "compact", "x": 99999, "y": -5000}))
    w = HudWindow(tmp_path, native=False, auto_tick=False, sysmon=False)
    screen = _app.primaryScreen().availableGeometry()
    assert screen.contains(w.geometry())
    w.close()


def test_clamp_rect_multi_screen() -> None:
    screens = [QRect(0, 0, 1920, 1080), QRect(1920, 0, 1280, 1024)]
    assert clamp_rect(QRect(-50, 10, 200, 200), screens) == QRect(0, 10, 200, 200)
    assert clamp_rect(QRect(3000, 900, 400, 300), screens) == QRect(2800, 724, 400, 300)
    assert clamp_rect(QRect(100, 100, 200, 200), screens) == QRect(100, 100, 200, 200)
    assert clamp_rect(QRect(5000, 5000, 100, 100), screens) == QRect(3100, 924, 100, 100)


def test_drag_saves_position(tmp_path: Path) -> None:
    w = HudWindow(tmp_path, native=False, auto_tick=False, sysmon=False)
    w.show()
    before = w.anchor
    w.anchor = before
    w._press, w._press_origin, w._dragging = before, before, True
    w.anchor = before - type(before)(30, 20)
    w._apply_geometry(clamp=True)
    w.save()
    saved = json.loads((tmp_path / "hud.json").read_text())
    assert (saved["x"], saved["y"]) == (w.anchor.x(), w.anchor.y())
    w.close()


# -- gauges and sampling ---------------------------------------------------------------
def test_gauge_eases_toward_value(hud) -> None:
    w, _ = hud
    g = w.gauges["CPU"]
    g.set_value(0.8)
    shown = []
    for _ in range(60):
        g.advance(0.016)
        shown.append(g.shown)
    assert shown == sorted(shown) and 0.0 < shown[0] < shown[-1] < 0.8
    for _ in range(300):
        g.advance(0.016)
    assert g.shown == pytest.approx(0.8, abs=0.01)


def test_missing_gpu_and_battery_are_hidden(hud) -> None:
    w, _ = hud
    w.apply_sample(Sample(0.2, 0.4, None, None))
    assert w.gauges["CPU"].isVisible() and not w.gauges["GPU"].isVisible() and not w.gauges["BATTERY"].isVisible()
    w.apply_sample(Sample(0.2, 0.4, 0.5, 0.9))
    assert w.gauges["GPU"].isVisible() and w.gauges["BATTERY"].isVisible()


def test_take_sample_with_fake_psutil() -> None:
    fake = SimpleNamespace(
        cpu_percent=lambda interval=None: 40.0,
        virtual_memory=lambda: SimpleNamespace(percent=55.0),
        sensors_battery=lambda: SimpleNamespace(percent=80.0, power_plugged=True),
    )
    s = take_sample(fake, None)
    assert (s.cpu, s.memory, s.gpu, s.battery, s.charging) == (0.4, 0.55, None, 0.8, True)


# -- orb -----------------------------------------------------------------------------------
def render_orb(orb: OrbWidget, scale: float = 1.0) -> QImage:
    size = int(orb.width() * scale)
    img = QImage(size, int(orb.height() * scale), QImage.Format.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(scale)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHints(QPainter.RenderHint.Antialiasing)
    orb.paint_orb(p, orb.width(), orb.height())
    p.end()
    return img


def lit_pixels(img: QImage) -> int:
    return sum(1 for y in range(0, img.height(), 4) for x in range(0, img.width(), 4) if img.pixelColor(x, y).alpha() > 20)


@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.parametrize("state", list(VoiceState))
def test_orb_paints_every_state(state: VoiceState, scale: float) -> None:
    orb = OrbWidget()
    orb.resize(200, 200)
    orb.set_state(state)
    orb.set_levels(0.6, 0.6)
    orb.set_tool_running(state is VoiceState.THINKING)
    for _ in range(80):
        orb.advance(0.016)
    img = render_orb(orb, scale)
    assert img.width() == int(200 * scale)
    assert lit_pixels(img) > 200


def test_orb_moves_between_frames_and_tracks_level() -> None:
    orb = OrbWidget()
    orb.resize(200, 200)
    for _ in range(80):
        orb.advance(0.016)
    first = render_orb(orb)
    for _ in range(5):
        orb.advance(0.016)
    assert first != render_orb(orb)

    def tick_energy(level: float) -> float:
        o = OrbWidget()
        o.set_state(VoiceState.LISTENING)
        o.set_levels(level, 0.0)
        for _ in range(120):
            o.advance(0.016)
        return sum(o.tick_values)

    assert tick_energy(0.9) > 2.0 * tick_energy(0.05)


def test_orb_entrance_draws_ring_over_time() -> None:
    orb = OrbWidget()
    orb.resize(160, 160)
    orb.advance(0.05)
    early = lit_pixels(render_orb(orb))
    for _ in range(70):
        orb.advance(0.016)
    assert lit_pixels(render_orb(orb)) > early


def test_orb_palette_blends_smoothly() -> None:
    orb = OrbWidget()
    start = orb.palette.primary
    orb.set_state(VoiceState.THINKING)
    orb.advance(0.016)
    mid = orb.palette.primary
    for _ in range(60):
        orb.advance(0.016)
    end = orb.palette.primary
    assert start != mid != end and abs(mid[0] - start[0]) < abs(end[0] - start[0])


def test_window_renders_nonblank_in_both_modes(hud) -> None:
    w, _ = hud
    run(w, 1.0)
    img = w.grab().toImage()
    assert lit_pixels(img) > 1000
    w.set_mode("compact", animate=False)
    run(w, 0.5)
    assert lit_pixels(w.grab().toImage()) > 500


# -- demo session ------------------------------------------------------------------------------
def test_demo_session_emits_sane_sequence() -> None:
    events: list[Any] = []
    demo = DemoSession(events.append, speed=500.0, loop=False)
    asyncio.run(demo.run())
    kinds = [type(e).__name__ for e in events]
    for needed in ("StateChanged", "Heard", "ToolActivity", "Spoke", "Levels", "ConfirmRequested", "ConfirmResolved", "TurnFinished"):
        assert needed in kinds
    states = [e.state for e in events if isinstance(e, StateChanged)]
    assert states[:3] == [VoiceState.LISTENING, VoiceState.THINKING, VoiceState.SPEAKING] and states[-1] is VoiceState.IDLE
    assert all(0.0 <= e.mic <= 1.0 and 0.0 <= e.speaker <= 1.0 for e in events if isinstance(e, Levels))
    started = [e.name for e in events if isinstance(e, ToolActivity) and not e.done]
    finished = [e.name for e in events if isinstance(e, ToolActivity) and e.done]
    assert sorted(started) == sorted(finished)
    assert kinds.index("ConfirmRequested") < kinds.index("ConfirmResolved")


def test_demo_session_answers_confirm_and_text() -> None:
    async def scenario() -> list[Any]:
        events: list[Any] = []
        demo = DemoSession(events.append, speed=500.0, loop=False)
        demo.submit_text("hello")
        await asyncio.sleep(0.3)
        demo.set_muted(True)
        return events

    events = asyncio.run(scenario())
    assert any(isinstance(e, UserText) and e.text == "hello" for e in events)
    assert any(isinstance(e, MicMuted) and e.muted for e in events)


# -- cli ---------------------------------------------------------------------------------------
def patch_cli(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    app = SimpleNamespace(settings=SimpleNamespace(), resume=lambda: False, close=lambda: None)
    monkeypatch.setattr(cli, "get_settings", lambda: cli.Settings(data_dir_override="."))
    monkeypatch.setattr(cli, "build_app", lambda settings, force_local=False: app)
    calls: list[Any] = []
    from jarvis.interfaces import hud

    monkeypatch.setattr(hud, "run_hud", lambda a, c, *, voice=True, demo=False: calls.append((voice, demo)) or 0)
    monkeypatch.setattr(hud, "hud_available", lambda: True)
    return calls


@pytest.mark.parametrize(
    "flags,expected",
    [(["--hud"], (True, False)), (["--hud", "--no-voice"], (False, False)), (["--hud-demo"], (True, True))],
)
def test_cli_hud_flags_route(monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: tuple[bool, bool]) -> None:
    calls = patch_cli(monkeypatch)
    assert cli.main(flags) == 0
    assert calls == [expected]


def test_cli_hud_missing_dependency_exits_2(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = patch_cli(monkeypatch)
    from jarvis.interfaces import hud

    monkeypatch.setattr(hud, "hud_available", lambda: False)
    assert cli.main(["--hud-demo"]) == 2
    assert calls == []
    assert "uv sync --extra hud" in capsys.readouterr().out


def test_cli_hud_flags_conflict_with_once(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_cli(monkeypatch)
    with pytest.raises(SystemExit):
        cli.main(["--hud", "--once", "hi"])


# -- performance structure ---------------------------------------------------------------------
def test_orb_glow_sprites_are_reused_across_frames() -> None:
    orb = OrbWidget()
    orb.resize(250, 250)
    for _ in range(80):
        orb.advance(0.016)
    render_orb(orb, 1.25)
    cache = orb.glow
    before = (cache.mask_builds, cache.tint_builds, cache.sprite_builds)
    assert before[0] > 0 and before[2] > 0
    for _ in range(12):
        orb.advance(0.025)
        render_orb(orb, 1.25)
    assert (cache.mask_builds, cache.tint_builds, cache.sprite_builds) == before


def test_orb_cache_survives_small_size_changes_but_rebuilds_for_a_new_ratio() -> None:
    orb = OrbWidget()
    orb.resize(250, 250)
    render_orb(orb, 1.0)
    builds = orb.glow.mask_builds
    orb.zoom = 1.1  # the window zooms the orb a little in expanded mode
    render_orb(orb, 1.0)
    assert orb.glow.mask_builds == builds
    render_orb(orb, 2.0)  # a different display ratio needs new pixmaps
    assert orb.glow.mask_builds > builds


def test_orb_frame_repaints_only_the_orb(hud, monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = hud
    w.controller.handle(StateChanged(VoiceState.SPEAKING))
    w.controller.handle(Levels(0.0, 0.6))
    run(w, 3.0)  # let the colour transition and the input easing finish
    for _ in range(60):  # drain the first full paints and the input focus easing before measuring
        w.advance(0.025)
        _app.processEvents()
    painted: list[tuple[str, int]] = []
    for cls in (type(w), type(w.orb), type(w.header), type(w.header.dot), type(w.state_label), type(w.reply)):
        original = cls.paintEvent

        def spy(self, event, _o=original):
            r = event.rect()
            painted.append((type(self).__name__, r.width() * r.height()))
            return _o(self, event)

        monkeypatch.setattr(cls, "paintEvent", spy)
    for _ in range(8):
        w.advance(0.025)
        _app.processEvents()
    orb_area = w.orb.width() * w.orb.height()
    assert any(name == "OrbWidget" for name, _ in painted)
    assert all(name == "OrbWidget" or area <= 24 * 24 for name, area in painted), painted
    assert max(area for name, area in painted if name == "OrbWidget") <= orb_area
    assert w.glass_renders >= 1


def test_glass_is_rendered_once_while_nothing_changes(hud) -> None:
    w, _ = hud
    run(w, 1.0)
    count = w.glass_renders
    run(w, 1.0)
    assert w.glass_renders == count


def test_orb_is_an_opaque_widget_that_draws_its_own_backdrop(hud) -> None:
    from PyQt6.QtCore import Qt

    w, _ = hud
    assert w.orb.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
    assert w.orb.backdrop is not None and not w.orb.backdrop().isNull()


def test_frame_interval_per_state(hud) -> None:
    from jarvis.hud import window as win

    w, _ = hud
    run(w, 0.5)
    assert w.frame_interval_ms() == win.FRAME_STANDBY
    w.controller.handle(StateChanged(VoiceState.SPEAKING))
    assert w.frame_interval_ms() == win.FRAME_ACTIVE_EXPANDED
    w.controller.handle(StateChanged(VoiceState.IDLE))
    w.orb.set_tool_running(True)
    assert w.frame_interval_ms() == win.FRAME_ACTIVE_EXPANDED
    w.orb.set_tool_running(False)
    w.controller.handle(StateChanged(VoiceState.LISTENING))
    w.set_mode("compact", animate=False)
    assert w.frame_interval_ms() == win.FRAME_ACTIVE_COMPACT
    w.controller.handle(StateChanged(VoiceState.IDLE))
    w.orb.set_levels(0.0, 0.0)
    run(w, 1.5)
    assert w.frame_interval_ms() == win.FRAME_STANDBY
    w.orb.idle_for = 100.0
    assert w.frame_interval_ms() == win.FRAME_REST
    assert win.FRAME_ACTIVE_EXPANDED < win.FRAME_STANDBY < win.FRAME_REST


def test_kick_wakes_a_resting_window(tmp_path: Path) -> None:
    from jarvis.hud import window as win

    w = HudWindow(tmp_path, native=False, sysmon=False)
    w.show()
    w.orb.idle_for = 100.0
    w.timer.setInterval(win.FRAME_REST)
    w.controller.handle(StateChanged(VoiceState.LISTENING))
    assert w.orb.idle_for == 0.0
    assert w.timer.interval() <= 16
    w.close()


def test_animation_is_time_based_not_frame_based() -> None:
    fast, slow = OrbWidget(), OrbWidget()
    for _ in range(100):
        fast.advance(0.01)
    for _ in range(25):
        slow.advance(0.04)
    assert fast.t == pytest.approx(slow.t)
    assert fast.arc_angles == pytest.approx(slow.arc_angles, abs=1e-6)


def test_hidden_window_stops_its_timer(tmp_path: Path) -> None:
    w = HudWindow(tmp_path, native=False, sysmon=False)
    w.show()
    assert w.timer.isActive()
    w.hide()
    assert not w.timer.isActive()
    w.close()


# -- gauge row ---------------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5])
@pytest.mark.parametrize("labels", [["CPU", "MEMORY", "GPU", "BATTERY"], ["CPU", "MEMORY", "BATTERY"]])
def test_gauge_labels_never_overlap(scale: float, labels: list[str]) -> None:
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QFontMetricsF

    from jarvis.hud.widgets import gauge_label_font, layout_gauge_row

    total = 396.0 * scale
    (pt, tracking), cells = layout_gauge_row(labels, total, scale)
    fm = QFontMetricsF(gauge_label_font(pt, tracking, scale))
    rects = []
    for label, (x, width) in zip(labels, cells):
        text_w = fm.horizontalAdvance(label)
        assert text_w <= width + 0.01
        rects.append(QRectF(x + (width - text_w) / 2, 0, text_w, 10))
    for a, b in zip(rects, rects[1:]):
        assert b.left() - a.right() >= 10.0 * scale, (scale, labels, rects)
    assert rects[0].left() >= 0 and rects[-1].right() <= total + 0.01


def test_gauge_labels_shrink_when_the_row_is_tight() -> None:
    from jarvis.hud.widgets import GAUGE_LABEL_STEPS, layout_gauge_row

    roomy, _ = layout_gauge_row(["CPU", "MEMORY", "GPU", "BATTERY"], 396.0)
    tight, cells = layout_gauge_row(["CPU", "MEMORY", "GPU", "BATTERY"], 250.0)
    assert roomy == GAUGE_LABEL_STEPS[0]
    assert GAUGE_LABEL_STEPS.index(tight) > 0
    assert cells[-1][0] + cells[-1][1] <= 250.0 + 0.01


def test_window_gauge_cells_do_not_overlap(hud) -> None:
    w, _ = hud
    w.apply_sample(Sample(0.3, 0.6, 0.2, 0.8))
    rects = [g.geometry() for g in w.gauges.values() if g.available]
    assert len(rects) == 4
    for a, b in zip(rects, rects[1:]):
        assert b.left() - (a.left() + a.width()) >= 10


# -- idle hint ---------------------------------------------------------------------------------
def test_idle_hint_fades_out_with_content_and_back(hud) -> None:
    w, _ = hud
    run(w, 0.2)
    assert w.reply.hint_alpha > 0.95
    w.controller.handle(UserText("hello"))
    run(w, 1.5)
    assert w.reply.hint_alpha == 0.0
    w.reply.clear()
    run(w, 2.0)
    assert w.reply.hint_alpha > 0.95


def test_idle_hint_respects_audio(hud) -> None:
    w, _ = hud
    w.set_audio(True, "Jarvis")
    assert "Say" in w.reply.hint_text and "Jarvis" in w.reply.hint_text
    w.set_audio(False)
    assert "Say" not in w.reply.hint_text and "speak" not in w.reply.hint_text.lower()
    assert w.header.mic.isHidden()


def test_idle_hint_is_painted_while_visible(hud) -> None:
    w, _ = hud
    run(w, 0.3)
    img = w.reply.grab().toImage()
    assert any(img.pixelColor(x, y).alpha() > 0 for y in range(0, img.height(), 2) for x in range(0, img.width(), 2))


# -- compact capsule ---------------------------------------------------------------------------
def test_compact_capsule_width_is_constant_and_fits_every_word(tmp_path: Path) -> None:
    from PyQt6.QtGui import QFontMetricsF

    from jarvis.hud.widgets import state_font
    from jarvis.hud.window import CAPSULE_WORDS

    w = HudWindow(tmp_path, native=False, sysmon=False, auto_tick=False)
    w.show()
    w.set_mode("compact", animate=False)
    width = w.capsule_width()
    fm = QFontMetricsF(state_font(True))
    assert all(fm.horizontalAdvance(word) < width for word in CAPSULE_WORDS)
    for state in VoiceState:
        w.controller.handle(StateChanged(state))
        assert w.capsule_width() == width
    assert width <= COMPACT.width()
    w.close()


# -- glass ---------------------------------------------------------------------------------------
def test_glass_body_is_nearly_opaque() -> None:
    from jarvis.hud.widgets import GLASS_OPACITY_BOTTOM, GLASS_OPACITY_TOP

    # Opaque enough that bright text behind the window does not ghost through, yet not solid.
    assert 0.94 <= GLASS_OPACITY_BOTTOM <= GLASS_OPACITY_TOP <= 0.99


def test_speaking_palette_is_clearly_cyan() -> None:
    import colorsys

    from jarvis.hud.theme import PALETTES

    r, g, b = PALETTES["speaking"].primary
    hue, sat, _val = colorsys.rgb_to_hsv(r, g, b)
    assert 0.47 <= hue <= 0.56 and sat >= 0.6


# -- wiring --------------------------------------------------------------------------------------
def test_hooks_reflect_and_toggle_the_forced_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import jarvis.llm.providers as providers
    from jarvis.interfaces import hud as hud_iface

    class Agent:
        using_local = False
        primary = SimpleNamespace(is_local=False)
        fallback = SimpleNamespace(is_local=True)

        def use_local(self, on: bool) -> None:
            self.using_local = on

        def new_conversation(self) -> None:
            pass

    seen: list[str] = []
    monkeypatch.setattr(providers, "describe", lambda c: "local-model" if c.is_local else "cloud-model")
    hooks = hud_iface._make_hooks(SimpleNamespace(agent=Agent()), lambda: None, seen.append)
    assert hooks.is_local() is False and hooks.local_available() is True
    hooks.set_local(True)
    assert hooks.is_local() is True
    hooks.set_local(not hooks.is_local())
    assert hooks.is_local() is False
    assert seen == ["local-model", "cloud-model"]


def test_run_hud_uses_the_voice_model_before_building_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jarvis.llm.providers as providers
    import jarvis.proactive.scheduler as scheduler
    from jarvis.hud import window as win_mod
    from jarvis.interfaces import hud as hud_iface

    order: list[str] = []
    windows: list[Any] = []

    class Session:
        has_audio = False
        muted = False

        async def run(self, greeting=None) -> None:
            order.append("run")

        def stop(self) -> None:
            pass

        def say(self, *_a) -> None:
            pass

        def is_idle(self) -> bool:
            return True

    class Watcher:
        def __init__(self, *_a) -> None:
            pass

        def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    class Agent:
        using_local = False
        primary = SimpleNamespace(is_local=False)
        fallback = None

        def new_conversation(self) -> None:
            pass

        def use_local(self, on: bool) -> None:
            self.using_local = on

    class App:
        agent = Agent()
        store = None
        settings = SimpleNamespace(data_dir=tmp_path, wake_word="jarvis")

        def use_voice_model(self) -> str:
            order.append("use_voice_model")
            return "fast-model"

    real_window = win_mod.HudWindow

    def make_window(data_dir, hooks=None, **kw):
        w = real_window(data_dir, hooks, native=False, sysmon=False, auto_tick=False)
        windows.append(w)
        return w

    async def fake_make_session(app, voice, demo, on_event, console):
        order.append("make_session")
        return Session()

    monkeypatch.setattr(win_mod, "HudWindow", make_window)
    monkeypatch.setattr(scheduler, "ReminderWatcher", Watcher)
    monkeypatch.setattr(providers, "describe", lambda c: "cloud-model")
    monkeypatch.setattr(hud_iface, "_make_session", fake_make_session)
    assert asyncio.run(hud_iface._hud_main(App(), None, False, False)) == 0
    assert order == ["use_voice_model", "make_session", "run"]
    assert windows[0].footer.text == "fast-model"
    assert "type below" in windows[0].reply.hint_text.lower() and "Say" not in windows[0].reply.hint_text


# -- code panel ----------------------------------------------------------------------
CODE_A = "def f(x):\n    if x:\n        return 1\n\n    return 2"
CODE_B = "echo hi\n\tindented with tab"


def test_code_shown_routes_to_panel_and_keeps_all_blocks(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(UserText("write code"))
    h(CodeShown(CODE_A, "python"))
    h(CodeShown(CODE_B, ""))
    assert [b.code for b in w.code.blocks] == [CODE_A, CODE_B]
    assert [b.language for b in w.code.blocks] == ["python", ""]
    assert w.code.isVisible() and w.code.blocks[1].title == "CODE"
    assert len(w.code.blocks[0].lines) == 5  # blank line and indentation survive
    assert w.code.blocks[0].lines[2] == "        return 1" and w.code.blocks[0].lines[3] == ""


def test_code_panel_paints_and_scrolls(hud) -> None:
    w, _ = hud
    w.controller.handle(CodeShown("\n".join(f"line {i}" + "x" * 90 for i in range(200)), "python"))
    img = QImage(w.size(), QImage.Format.Format_ARGB32_Premultiplied)
    w.render(img)  # paints without raising
    assert w.code._max_scroll_y() > 0 and w.code._max_scroll_x() > 0
    w.code.scroll_y = 1e9
    w.code._clamp()
    assert w.code.scroll_y == w.code._max_scroll_y()


def test_code_copy_puts_exact_text_on_clipboard(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(CodeShown(CODE_A, "python"))
    h(CodeShown(CODE_B, ""))
    w.code.copy_block(1)
    assert QApplication.clipboard().text() == CODE_B and w.code.copied == 1
    w.code.copy_block(0)
    assert QApplication.clipboard().text() == CODE_A
    w.code._clear_copied()
    assert w.code.copied is None


def test_code_copy_button_click(hud) -> None:
    w, _ = hud
    w.controller.handle(CodeShown(CODE_A, "python"))
    pos = w.code._copy_rect(w.code._block_tops()[0], 0).center()
    QApplication.clipboard().setText("")
    assert w.code._hit(pos) == ("copy", 0)
    w.code.copy_block(0)
    assert QApplication.clipboard().text() == CODE_A
    assert w.code._hit(w.code._close_rect().center()) == ("close", 0)
    w.code.dismiss()
    assert not w.code.blocks and not w.code.isVisible()


def test_new_exchange_clears_code(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(UserText("q"))
    h(Spoke("Here you go."))
    h(CodeShown(CODE_A, "python"))
    h(TurnFinished("", 1.0, 0, 0, "m", "done"))
    assert w.code.blocks  # still there after the turn ends
    h(Heard("next", True, 0.9))
    assert not w.code.blocks and not w.code.isVisible()


def test_code_without_spoken_text_does_not_get_cleared_by_turn_finish(hud) -> None:
    w, _ = hud
    h = w.controller.handle
    h(CodeShown(CODE_A, "python"))
    h(TurnFinished("Some text.", 1.0, 0, 0, "m", "done"))
    assert [b.code for b in w.code.blocks] == [CODE_A]


def test_code_panel_never_takes_focus(hud) -> None:
    w, _ = hud
    w.controller.handle(CodeShown(CODE_A, "python"))
    assert w.code.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert w.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    flags = w.windowFlags()
    assert flags & Qt.WindowType.Tool and flags & Qt.WindowType.WindowStaysOnTopHint
    assert not w.isActiveWindow() and not w.code.hasFocus()
    assert w.code.window() is w  # same top-level: capture exclusion and stay-on-top are the window's


def test_code_expands_compact_window(tmp_path: Path) -> None:
    w = HudWindow(tmp_path, native=False, auto_tick=False, sysmon=False)
    w.show()
    w.set_mode("compact", animate=False)
    w.controller.handle(CodeShown(CODE_A, "python"))
    w.set_mode("expanded", animate=False)  # settle the mode animation
    assert w.mode == "expanded" and w.code.isVisible() and w.code.blocks
    w.close()


def test_long_code_is_height_capped(hud) -> None:
    w, _ = hud
    size = w.size()
    w.controller.handle(CodeShown("\n".join(f"x = {i}" for i in range(3000)), "python"))
    assert w.size() == size
    assert 0 < w.code.height() <= MAX_HEIGHT
    assert w.code.geometry().top() >= 0 and w.code.geometry().bottom() < w.input.geometry().top()
    short = w.code.height()
    w.controller.handle(UserText("again"))
    w.controller.handle(CodeShown("a = 1", "python"))
    assert w.code.height() < short
