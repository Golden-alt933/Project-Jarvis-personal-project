"""The HUD window: frameless glass, compact/expanded modes, drag, tray, hotkey, capture exclusion."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QEasingCurve, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QFontMetricsF, QGuiApplication, QIcon, QPainter, QPen, QPixmap, QRadialGradient
from PyQt6.QtWidgets import QApplication, QGraphicsOpacityEffect, QMenu, QSystemTrayIcon, QWidget

from jarvis.hud import winfx
from jarvis.hud.controller import STATE_WORDS, HudController
from jarvis.hud.orb import OrbWidget
from jarvis.hud.sysmon import Sample, SysMonitor
from jarvis.voice.events import VoiceState
from jarvis.hud.theme import approach, clamp01, label_font, lerp, mix, qcolor, smoothstep
from jarvis.hud.widgets import (
    ConfirmCard,
    FooterLabel,
    Gauge,
    Header,
    PillInput,
    ReplyView,
    StateLabel,
    ToolTicker,
    layout_gauge_row,
    paint_glass,
    state_font,
)

log = logging.getLogger(__name__)

COMPACT = QSize(170, 200)
PANEL = QSize(440, 660)
MARGIN = 22
EXPANDED = QSize(PANEL.width() + 2 * MARGIN, PANEL.height() + 2 * MARGIN)
EDGE_GAP = 24
MODE_SECONDS = 0.28
ORB_COMPACT = QRectF(10, 6, 150, 150)
LABEL_COMPACT = QRectF(0, 158, 170, 22)
ORB_EXPANDED_SIZE = 250.0
ORB_ZOOM_EXPANDED = 1.14  # the orb is drawn this much larger than its widget in expanded mode
# Frame intervals (ms). Animation is time-based, so a longer interval changes smoothness, not speed.
FRAME_ACTIVE_EXPANDED = 25  # 40 fps
FRAME_ACTIVE_COMPACT = 33  # 30 fps
FRAME_STANDBY = 62  # ~16 fps
FRAME_REST = 125  # 8 fps after a long rest (the orb also slows to half speed)
CAPSULE_WORDS = (*STATE_WORDS.values(), "WAKING UP", "MUTED")


def clamp_rect(rect: QRect, screens: list[QRect]) -> QRect:
    """Move `rect` fully inside the screen it overlaps most (nearest screen if none)."""
    if not screens:
        return QRect(rect)

    def overlap(s: QRect) -> int:
        i = s.intersected(rect)
        return max(0, i.width()) * max(0, i.height()) if not i.isEmpty() else 0

    def distance(s: QRect) -> int:
        d = s.center() - rect.center()
        return d.x() ** 2 + d.y() ** 2

    best = max(screens, key=overlap)
    if overlap(best) == 0:
        best = min(screens, key=distance)
    x = min(max(rect.x(), best.left()), best.right() + 1 - rect.width())
    y = min(max(rect.y(), best.top()), best.bottom() + 1 - rect.height())
    return QRect(max(x, best.left()), max(y, best.top()), rect.width(), rect.height())


def load_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(path: Path, state: dict) -> None:
    try:
        path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        log.debug("could not save HUD state", exc_info=True)


@dataclass
class HudHooks:
    """Things the window asks of the app; defaults make the window usable standalone."""

    new_conversation: Callable[[], None] = lambda: None
    set_local: Callable[[bool], None] = lambda on: None
    is_local: Callable[[], bool] = lambda: False
    local_available: Callable[[], bool] = lambda: False
    quit: Callable[[], None] = lambda: QApplication.quit()


def make_tray_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    glow = QRadialGradient(QPointF(32, 32), 30)
    glow.setColorAt(0.0, QColor(94, 230, 255, 110))
    glow.setColorAt(1.0, QColor(94, 230, 255, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(glow))
    p.drawEllipse(QPointF(32, 32), 30, 30)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(94, 230, 255, 70), 9))
    p.drawEllipse(QPointF(32, 32), 16, 16)
    p.setPen(QPen(QColor(190, 245, 255, 255), 4))
    p.drawEllipse(QPointF(32, 32), 16, 16)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255, 235))
    p.drawEllipse(QPointF(32, 32), 4, 4)
    p.end()
    return QIcon(pm)


MENU_STYLE = (
    "QMenu{background:#0B1222;border:1px solid rgba(255,255,255,30);border-radius:10px;padding:6px;"
    "color:#DCEBFA;font-family:'Segoe UI','Inter','Cantarell','Ubuntu','Noto Sans',sans-serif;font-size:12px;}"
    "QMenu::item{padding:7px 22px 7px 14px;border-radius:6px;}"
    "QMenu::item:selected{background:rgba(94,230,255,40);}"
    "QMenu::separator{height:1px;background:rgba(255,255,255,24);margin:5px 8px;}"
)


class HudWindow(QWidget):
    mode_changed = pyqtSignal(str)

    def __init__(
        self,
        data_dir: Path,
        hooks: HudHooks | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        native: bool = True,
        auto_tick: bool = True,
        sysmon: bool = True,
        hide_from_capture: bool = True,
        acrylic: bool = False,
    ) -> None:
        super().__init__(None)
        self.hooks = hooks or HudHooks()
        self.state_path = Path(data_dir) / "hud.json"
        self.clock = clock
        self.native = native
        self.acrylic = acrylic
        self.hide_from_capture = hide_from_capture
        self.setWindowTitle("Jarvis")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.mode = "compact"
        self.mode_t = 0.0
        self.anchor = QPoint(0, 0)
        self._anim: QVariantAnimation | None = None
        self._press: QPoint | None = None
        self._press_origin = QPoint()
        self._dragging = False
        self._ticker_h = 0.0
        self._last_accent = (0.0, 0.0, 0.0)
        self._started = False
        self._last_hotkey = -1.0
        self._glass: QPixmap = QPixmap()
        self._glass_key: tuple | None = None
        self.glass_renders = 0
        self._capsule_w: float | None = None

        self.orb = OrbWidget(self)
        self.orb.backdrop = self._glass_pixmap
        self.state_label = StateLabel(self)
        self.body = QWidget(self)
        self.body.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.header = Header(self.body)
        self.reply = ReplyView(self.body)
        self.ticker = ToolTicker(self.body)
        self.gauges = {name: Gauge(name, self.body) for name in ("CPU", "MEMORY", "GPU", "BATTERY")}
        self.input = PillInput(self.body)
        self.footer = FooterLabel(self.body)
        self.confirm = ConfirmCard(self.body)
        self.controller = HudController(self)
        self._animated = [self.state_label, self.header, self.reply, self.ticker, self.input, self.confirm,
                          *self.gauges.values()]
        self.header.mic.clicked.connect(self.controller.toggle_mute)
        self.header.minimise.clicked.connect(lambda: self.set_mode("compact"))
        self.input.submitted.connect(self.controller.submit)
        self.input.stop_requested.connect(self.controller.cancel)
        self.confirm.answered.connect(self.controller.answer_confirm)
        self.footer.set_text("")
        self.controller.set_loading(False)  # sets the initial state word

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._on_tick)
        self._last_tick = clock()
        self.auto_tick = auto_tick

        self.monitor: SysMonitor | None = None
        if sysmon:
            self.monitor = SysMonitor()
            self.monitor.sampled.connect(self.apply_sample)

        self._tray: QSystemTrayIcon | None = None
        self._hotkey_filter = None
        self._hwnd = 0
        self._restore()
        self._layout()
        if native:
            self._build_tray()

    # -- state persistence ---------------------------------------------------------
    def _screens(self) -> list[QRect]:
        return [s.availableGeometry() for s in QGuiApplication.screens()]

    def _restore(self) -> None:
        saved = load_state(self.state_path)
        mode = saved.get("mode") if saved.get("mode") in ("compact", "expanded") else "compact"
        screens = self._screens()
        if isinstance(saved.get("x"), int) and isinstance(saved.get("y"), int):
            self.anchor = QPoint(saved["x"], saved["y"])
        elif screens:
            self.anchor = screens[0].bottomRight() - QPoint(EDGE_GAP - 1, EDGE_GAP - 1)
        self.mode = mode
        self.mode_t = 1.0 if mode == "expanded" else 0.0
        self._apply_geometry(clamp=True)
        self.body.setVisible(self.mode_t > 0.0)

    def save(self) -> None:
        save_state(self.state_path, {"mode": self.mode, "x": self.anchor.x(), "y": self.anchor.y()})

    def _size_at(self, t: float) -> QSize:
        return QSize(round(lerp(COMPACT.width(), EXPANDED.width(), t)), round(lerp(COMPACT.height(), EXPANDED.height(), t)))

    def _apply_geometry(self, clamp: bool = False) -> None:
        size = self._size_at(self.mode_t)
        rect = QRect(self.anchor.x() - size.width(), self.anchor.y() - size.height(), size.width(), size.height())
        if clamp:
            rect = clamp_rect(rect, self._screens())
            self.anchor = QPoint(rect.x() + rect.width(), rect.y() + rect.height())
        self.setGeometry(rect)

    # -- modes ---------------------------------------------------------------------
    def set_mode(self, mode: str, animate: bool = True) -> None:
        if mode == self.mode and self._anim is None:
            return
        self.mode = mode
        target = 1.0 if mode == "expanded" else 0.0
        end = QRect(QPoint(self.anchor.x() - self._size_at(target).width(), self.anchor.y() - self._size_at(target).height()),
                    self._size_at(target))
        fixed = clamp_rect(end, self._screens())
        self.anchor = QPoint(fixed.x() + fixed.width(), fixed.y() + fixed.height())
        if self._anim is not None:
            self._anim.stop()
            self._anim = None
        if target > 0.0:
            self.body.show()
            self._set_body_opacity(self.body_opacity_for(self.mode_t))
        if not animate:
            self._finish_mode(target)
        else:
            anim = QVariantAnimation(self)
            anim.setStartValue(self.mode_t)
            anim.setEndValue(target)
            anim.setDuration(int(MODE_SECONDS * 1000))
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.valueChanged.connect(lambda v: self._step_mode(float(v)))
            anim.finished.connect(lambda: self._finish_mode(target))
            self._anim = anim
            anim.start()
        self.mode_changed.emit(mode)
        self.save()
        self._update_tray_text()

    def toggle_mode(self) -> None:
        self.set_mode("compact" if self.mode == "expanded" else "expanded")

    @staticmethod
    def body_opacity_for(t: float) -> float:
        return clamp01((t - 0.45) / 0.55)

    def _set_body_opacity(self, value: float) -> None:
        if value >= 0.999:
            if self.body.graphicsEffect() is not None:
                self.body.setGraphicsEffect(None)
            return
        effect = self.body.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(self.body)
            self.body.setGraphicsEffect(effect)
        effect.setOpacity(value)

    def _step_mode(self, t: float) -> None:
        self.mode_t = t
        self._apply_geometry()
        self._set_body_opacity(self.body_opacity_for(t))
        self._layout()
        self.update()

    def _finish_mode(self, target: float) -> None:
        self._anim = None
        self._step_mode(target)
        self.body.setVisible(target > 0.0)
        if target > 0.0 and self.isVisible():
            self.input.edit.setFocus()

    # -- layout --------------------------------------------------------------------
    def _layout(self) -> None:
        t = self.mode_t
        w, h = self.width(), self.height()
        oe = QRectF((w - ORB_EXPANDED_SIZE) / 2, MARGIN + 54, ORB_EXPANDED_SIZE, ORB_EXPANDED_SIZE)
        orb = QRectF(*(lerp(a, b, t) for a, b in zip(
            (ORB_COMPACT.x(), ORB_COMPACT.y(), ORB_COMPACT.width(), ORB_COMPACT.height()),
            (oe.x(), oe.y(), oe.width(), oe.height()))))
        self.orb.setGeometry(orb.toRect())
        self.orb.zoom = lerp(1.0, ORB_ZOOM_EXPANDED, smoothstep(t))
        le = QRectF(MARGIN, MARGIN + 304, PANEL.width(), 22)
        lab = QRectF(*(lerp(a, b, t) for a, b in zip(
            (LABEL_COMPACT.x(), LABEL_COMPACT.y(), LABEL_COMPACT.width(), LABEL_COMPACT.height()),
            (le.x(), le.y(), le.width(), le.height()))))
        self.state_label.setGeometry(lab.toRect())
        self.body.setGeometry(0, 0, w, h)
        x0, cw = MARGIN + 22, PANEL.width() - 44
        bottom = MARGIN + PANEL.height()
        self.header.setGeometry(x0, MARGIN + 12, cw, 40)
        self.footer.setGeometry(x0, bottom - 34, cw, 18)
        self.input.setGeometry(x0 - 3, bottom - 34 - 8 - 46, cw + 6, 46)
        gauge_y = self.input.y() - 10 - 66
        visible = [g for g in self.gauges.values() if g.available]
        for g in self.gauges.values():
            g.setVisible(g.available and self.mode_t > 0.0)
        if visible:
            style, cells = layout_gauge_row([g.label for g in visible], float(cw))
            for g, (cx, cwid) in zip(visible, cells):
                g.label_style = style
                g.setGeometry(x0 + round(cx), gauge_y, round(cwid), 66)
        th = round(self._ticker_h)
        self.ticker.setGeometry(x0, gauge_y - 10 - th, cw, th)
        reply_top = MARGIN + 332
        self.reply.setGeometry(x0, reply_top, cw, max(40, self.ticker.y() - 8 - reply_top))
        self.confirm.setGeometry(MARGIN + 12, gauge_y - 70, PANEL.width() - 24, self.input.y() - gauge_y + 70 - 6)
        self.confirm.raise_()

    def resizeEvent(self, e) -> None:  # noqa: N802
        self._layout()
        super().resizeEvent(e)

    # -- time ----------------------------------------------------------------------
    def _on_tick(self) -> None:
        now = self.clock()
        dt, self._last_tick = now - self._last_tick, now
        self.advance(dt)
        want = self.frame_interval_ms()
        if self.timer.interval() != want:
            self.timer.setInterval(want)

    def is_active(self) -> bool:
        o = self.orb
        return (o.state is not VoiceState.IDLE or o.loading or o.tool_running or o.confirming
                or o.error_left > 0.0 or self._anim is not None or o.drive > 0.05)

    def frame_interval_ms(self) -> int:
        """Per-state frame interval: 40 fps expanded / 30 fps compact while active, 16 fps standing by,
        8 fps after a long rest."""
        if self.orb.is_resting():
            return FRAME_REST
        if self.is_active():
            return FRAME_ACTIVE_EXPANDED if self.mode_t >= 0.5 else FRAME_ACTIVE_COMPACT
        return FRAME_STANDBY

    def kick(self) -> None:
        """Something happened: leave rest and pick up the pace without waiting out a slow frame."""
        self.orb.idle_for = 0.0
        if self.timer.isActive():
            want = self.frame_interval_ms()
            if self.timer.interval() > want:
                self.timer.start(min(want, 16))  # next tick is soon; _on_tick then sets the steady interval

    def advance(self, dt: float) -> None:
        """One animation step for everything. Public so tests can drive time."""
        dt = min(max(dt, 0.0), 0.2)
        self.orb.advance(dt)
        accent = self.orb.palette.primary
        self.state_label.accent = accent
        if self.mode_t > 0.0 and self.body.isVisible():
            for wdg in (self.header, self.ticker, self.input, *self.gauges.values()):
                wdg.accent = accent
            for wdg in self._animated:
                wdg.advance(dt)
            want = self.ticker.wanted_height()
            new = approach(self._ticker_h, want, dt, 0.1)
            if abs(new - self._ticker_h) > 0.4 or (want == 0 and new != 0.0):
                self._ticker_h = new if abs(new - want) > 0.4 else float(want)
                self._layout()
        else:
            self.state_label.advance(dt)
        if self._glass_key != self._glass_key_now():
            # accent tint of the glass moved a visible step: the window re-renders its cached glass
            self.update()
        self.orb.update()

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        self._last_tick = self.clock()
        if self.auto_tick:
            self.timer.start(16)
        if self._started and self.monitor is not None:
            self.monitor.start()
        if not self._started:
            self._started = True
            self._after_first_show()

    def hideEvent(self, e) -> None:  # noqa: N802
        self.timer.stop()  # nothing is painted while hidden, so nothing needs to animate
        if self.monitor is not None:
            self.monitor.stop()  # ...and nothing needs sampling
        super().hideEvent(e)

    def _after_first_show(self) -> None:
        if self.monitor is not None:
            self.monitor.start()
        if not self.native:
            return
        self._hwnd = int(self.winId())
        winfx.set_capture_excluded(self._hwnd, self.hide_from_capture)
        if self.acrylic:
            winfx.set_acrylic(self._hwnd, True)
        if winfx.register_hotkey(self._hwnd):
            self._hotkey_filter = winfx.make_hotkey_filter(self._hotkey)
            QApplication.instance().installNativeEventFilter(self._hotkey_filter)
        else:
            log.info("Ctrl+Alt+J is not available; the HUD hotkey is disabled")

    def _hotkey(self) -> None:
        now = self.clock()
        if now - self._last_hotkey < 0.3:  # Qt can deliver one WM_HOTKEY to the filter twice
            return
        self._last_hotkey = now
        if not self.isVisible():
            self.show_window()
        else:
            self.toggle_mode()

    def show_window(self) -> None:
        self.show()
        self.raise_()

    def set_hide_from_capture(self, on: bool) -> None:
        self.hide_from_capture = on
        if self._hwnd:
            winfx.set_capture_excluded(self._hwnd, on)

    # -- sampling ------------------------------------------------------------------
    def apply_sample(self, s: Sample) -> None:
        values = {"CPU": s.cpu, "MEMORY": s.memory, "GPU": s.gpu, "BATTERY": s.battery}
        changed = False
        calm = not self.is_active() and self.mode_t >= 0.999
        for name, gauge in self.gauges.items():
            before = gauge.available
            gauge.set_value(values[name])
            if calm:
                gauge.snap()  # standing by: one repaint per sample instead of a second of easing frames
            changed |= before != gauge.available
        if changed:
            self._layout()

    # -- mouse / keys --------------------------------------------------------------
    def enterEvent(self, e) -> None:  # noqa: N802
        self.kick()
        super().enterEvent(e)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self.kick()
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.globalPosition().toPoint()
            self._press_origin = self.anchor
            self._dragging = False

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._press is None:
            return
        delta = e.globalPosition().toPoint() - self._press
        if not self._dragging and delta.manhattanLength() > 5:
            self._dragging = True
        if self._dragging:
            self.anchor = self._press_origin + delta
            self._apply_geometry()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._press is None:
            return
        if self._dragging:
            self._apply_geometry(clamp=True)
            self.save()
        elif self.orb.geometry().contains(e.position().toPoint()):
            self.toggle_mode()
        self._press = None
        self._dragging = False

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape:
            self.controller.cancel()
        else:
            super().keyPressEvent(e)

    # -- painting ------------------------------------------------------------------
    def _glass_key_now(self) -> tuple:
        accent = self.orb.palette.primary
        return (self.width(), self.height(), round(self.mode_t, 3), tuple(round(c * 20) for c in accent))

    def _glass_pixmap(self) -> QPixmap:
        """The cached glass; the orb blits its backdrop from here, so the window never repaints
        underneath it on an orb frame."""
        key = self._glass_key_now()
        if self._glass_key != key:
            self._glass = self._render_glass()
            self._glass_key = key
            self.glass_renders += 1
        return self._glass

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        p.drawPixmap(0, 0, self._glass_pixmap())
        p.end()

    def _render_glass(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        pm = QPixmap(round(self.width() * dpr), round(self.height() * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        t = smoothstep(self.mode_t)
        if t < 0.99:
            self._paint_bead(p, 1.0 - t)
        if t > 0.01:
            m = MARGIN * t
            paint_glass(p, QRectF(self.rect()).adjusted(m, m, -m, -m), self.orb.palette.primary, t)
        p.end()
        return pm

    def _paint_bead(self, p: QPainter, k: float) -> None:
        """Compact mode: the orb floats on a small dark glass disc so it reads on any desktop."""
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        o = self.orb.geometry()
        c = QPointF(o.center().x() + 0.5, o.center().y() + 0.5)
        r = o.width() / 2 * 0.93
        accent = self.orb.palette.primary
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(1, 8):
            p.setPen(QPen(qcolor(accent, 0.07 * k * (1 - i / 8.0) ** 1.5), 2.4))
            p.drawEllipse(c, r + i * 2.2, r + i * 2.2)
        grad = QRadialGradient(c, r)
        grad.setColorAt(0.0, QColor.fromRgbF(0.05, 0.08, 0.15, 0.90 * k))
        grad.setColorAt(1.0, QColor.fromRgbF(0.03, 0.045, 0.09, 0.84 * k))
        p.setPen(QPen(QColor.fromRgbF(1, 1, 1, 0.10 * k), 1.0))
        p.setBrush(QBrush(grad))
        p.drawEllipse(c, r, r)
        lab = self.state_label.geometry()
        cap_w = self.capsule_width()
        cap = QRectF(lab.center().x() - cap_w / 2, lab.y() + 1, cap_w, lab.height() - 2)
        p.setPen(QPen(QColor.fromRgbF(1, 1, 1, 0.08 * k), 1.0))
        p.setBrush(QColor.fromRgbF(0.03, 0.05, 0.10, 0.78 * k))
        p.drawRoundedRect(cap, cap.height() / 2, cap.height() / 2)

    def capsule_width(self) -> float:
        """Width of compact mode's state capsule: sized for the widest state word, so it never
        changes width (or jumps) when the word changes."""
        if self._capsule_w is None:
            fm = QFontMetricsF(state_font(True))
            self._capsule_w = min(COMPACT.width() - 10.0, max(fm.horizontalAdvance(w) for w in CAPSULE_WORDS) + 30.0)
        return self._capsule_w

    def set_audio(self, has_audio: bool, wake_word: str = "Jarvis") -> None:
        """Tell the window whether the session has a microphone (hides the mic button, picks the hint)."""
        self.header.mic.setVisible(has_audio)
        self.reply.set_audio(has_audio, wake_word)

    def closeEvent(self, e) -> None:  # noqa: N802
        self.save()
        if self.monitor is not None:
            self.monitor.stop()
        if self._hwnd:
            winfx.unregister_hotkey(self._hwnd)
        if self._tray is not None:
            self._tray.hide()
        super().closeEvent(e)

    # -- tray ----------------------------------------------------------------------
    def _build_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        tray = QSystemTrayIcon(make_tray_icon(), self)
        tray.setToolTip("Jarvis")
        menu = QMenu()
        menu.setStyleSheet(MENU_STYLE)
        self.act_show = QAction("Hide", menu)
        self.act_show.triggered.connect(self._toggle_visible)
        self.act_mode = QAction("Expand", menu)
        self.act_mode.triggered.connect(self.toggle_mode)
        self.act_mute = QAction("Mute microphone", menu)
        self.act_mute.setCheckable(True)
        self.act_mute.triggered.connect(self.controller.toggle_mute)
        act_new = QAction("New conversation", menu)
        act_new.triggered.connect(self._new_conversation)
        self.act_model = QAction("Use local model", menu)
        self.act_model.triggered.connect(self._toggle_model)
        self.act_capture = QAction("Hide from screenshots", menu)
        self.act_capture.setCheckable(True)
        self.act_capture.setChecked(self.hide_from_capture)
        self.act_capture.toggled.connect(self.set_hide_from_capture)
        act_quit = QAction("Quit", menu)
        act_quit.triggered.connect(lambda: self.hooks.quit())
        for item in (self.act_show, self.act_mode, None, self.act_mute, act_new, self.act_model, None,
                     self.act_capture, None, act_quit):
            menu.addSeparator() if item is None else menu.addAction(item)
        menu.aboutToShow.connect(self._update_tray_text)
        tray.setContextMenu(menu)
        tray.activated.connect(lambda r: self._toggle_visible() if r == QSystemTrayIcon.ActivationReason.Trigger else None)
        tray.show()
        self._tray, self._menu = tray, menu

    def _update_tray_text(self) -> None:
        if self._tray is None:
            return
        self.act_show.setText("Hide" if self.isVisible() else "Show")
        self.act_mode.setText("Compact" if self.mode == "expanded" else "Expanded")
        self.act_mute.setChecked(self.controller.muted)
        self.act_model.setText("Use cloud model" if self.hooks.is_local() else "Use local model")
        self.act_model.setEnabled(self.hooks.local_available() or self.hooks.is_local())

    def _toggle_visible(self) -> None:
        self.hide() if self.isVisible() else self.show_window()

    def _toggle_model(self) -> None:
        self.hooks.set_local(not self.hooks.is_local())

    def _new_conversation(self) -> None:
        self.hooks.new_conversation()
        self.reply.clear()
        self.ticker.clear()
