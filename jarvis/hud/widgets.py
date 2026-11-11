"""Custom-painted controls: glass, buttons, gauges, tool chips, text reveal, input, confirm card.

Animated widgets expose `advance(dt)`; the window drives them from one timer, so tests and
the render script can step time deterministically.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QFont,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import QLineEdit, QWidget

from jarvis.hud.theme import (
    GLASS_BASE,
    TEXT,
    Rgb,
    approach,
    clamp01,
    label_font,
    mix,
    mono_font,
    qcolor,
    ui_font,
)

ACCENT_DEFAULT: Rgb = (0.37, 0.90, 1.0)
CORAL: Rgb = (1.0, 0.42, 0.42)
AMBER: Rgb = (1.0, 0.71, 0.33)
GREEN: Rgb = (0.50, 0.95, 0.72)
# Base opacity of the glass body. High enough that windows behind it do not compete with the HUD's
# text; the sheen, wash, hairline and outer glow keep it reading as glass.
GLASS_OPACITY_TOP = 0.985
GLASS_OPACITY_BOTTOM = 0.975
GLASS_INNER_EXTRA = 0.55  # extra opacity of the text surface (0.95 + 0.55 of the rest is about 0.98)


def _aa(p: QPainter) -> None:
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)


def paint_glass(p: QPainter, rect: QRectF, accent: Rgb, strength: float, radius: float = 22.0) -> None:
    """The panel: glow tinted by state, deep blue-black body, sheen, accent wash, hairline."""
    if strength <= 0.01:
        return
    _aa(p)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for i in range(1, 10):
        grow = i * 2.4
        p.setPen(QPen(qcolor(accent, 0.075 * strength * (1.0 - i / 10.0) ** 1.6), 2.6))
        p.drawRoundedRect(rect.adjusted(-grow, -grow, grow, grow), radius + grow, radius + grow)
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    body = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    body.setColorAt(0.0, qcolor(mix(GLASS_BASE, (0.10, 0.15, 0.26), 0.55), GLASS_OPACITY_TOP * strength))
    body.setColorAt(1.0, qcolor(GLASS_BASE, GLASS_OPACITY_BOTTOM * strength))
    p.fillPath(path, QBrush(body))
    # slightly more opaque inner surface under the text (conversation, tickers, gauges, input)
    inner = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    inner.setColorAt(0.0, qcolor(GLASS_BASE, 0.35 * GLASS_INNER_EXTRA * strength))
    inner.setColorAt(0.42, qcolor(GLASS_BASE, 0.35 * GLASS_INNER_EXTRA * strength))
    inner.setColorAt(0.55, qcolor(GLASS_BASE, GLASS_INNER_EXTRA * strength))
    inner.setColorAt(1.0, qcolor(GLASS_BASE, GLASS_INNER_EXTRA * strength))
    p.fillPath(path, QBrush(inner))
    p.save()
    p.setClipPath(path)
    wash = QRadialGradient(QPointF(rect.center().x(), rect.top() + 150), rect.width() * 0.8)
    wash.setColorAt(0.0, qcolor(accent, 0.11 * strength))
    wash.setColorAt(0.5, qcolor(accent, 0.035 * strength))
    wash.setColorAt(1.0, qcolor(accent, 0.0))
    p.fillRect(rect, QBrush(wash))
    sheen = QLinearGradient(rect.topLeft(), QPointF(rect.left(), rect.top() + rect.height() * 0.38))
    sheen.setColorAt(0.0, QColor.fromRgbF(1, 1, 1, 0.07 * strength))
    sheen.setColorAt(1.0, QColor.fromRgbF(1, 1, 1, 0.0))
    p.fillRect(rect, QBrush(sheen))
    p.restore()
    edge = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    edge.setColorAt(0.0, QColor.fromRgbF(1, 1, 1, 0.20 * strength))
    edge.setColorAt(0.5, QColor.fromRgbF(1, 1, 1, 0.07 * strength))
    edge.setColorAt(1.0, QColor.fromRgbF(1, 1, 1, 0.05 * strength))
    p.setPen(QPen(QBrush(edge), 1.0))
    p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


class IconButton(QWidget):
    """Round hover-glow button: 'mic' (toggle, `muted`), 'min', or 'stop'."""

    clicked = pyqtSignal()

    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.muted = False
        self.accent: Rgb = ACCENT_DEFAULT
        self.hover = 0.0
        self.hover_target = 0.0
        self._down = False
        self.setFixedSize(30, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        self.update()

    def advance(self, dt: float) -> None:
        new = approach(self.hover, self.hover_target, dt, 0.08)
        moved = abs(new - self.hover) > 0.0005
        self.hover = new
        if moved:
            self.update()

    def enterEvent(self, _) -> None:  # noqa: N802
        self.hover_target = 1.0

    def leaveEvent(self, _) -> None:  # noqa: N802
        self.hover_target = 0.0

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._down = e.button() == Qt.MouseButton.LeftButton

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._down and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        self._down = False

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        c = QPointF(self.width() / 2, self.height() / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor.fromRgbF(1, 1, 1, 0.07 * self.hover))
        p.drawEllipse(c, 14.0, 14.0)
        col = CORAL if (self.kind == "mic" and self.muted) else self.accent
        alpha = 0.55 + 0.4 * self.hover
        pen = QPen(qcolor(col, alpha), 1.5, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.kind == "mic":
            p.drawRoundedRect(QRectF(c.x() - 3.2, c.y() - 8, 6.4, 11), 3.2, 3.2)
            p.drawArc(QRectF(c.x() - 6.5, c.y() - 6, 13, 12), 200 * 16, 140 * 16)
            p.drawLine(QPointF(c.x(), c.y() + 6), QPointF(c.x(), c.y() + 9))
            if self.muted:
                p.setPen(QPen(qcolor(CORAL, 0.95), 1.6, cap=Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(c.x() - 7, c.y() + 8), QPointF(c.x() + 7, c.y() - 8))
        elif self.kind == "min":
            p.drawLine(QPointF(c.x() - 5, c.y() + 2), QPointF(c.x(), c.y() + 6))
            p.drawLine(QPointF(c.x(), c.y() + 6), QPointF(c.x() + 5, c.y() + 2))
            p.drawLine(QPointF(c.x() - 5, c.y() - 5), QPointF(c.x() + 5, c.y() - 5))
        else:  # stop
            p.drawEllipse(c, 10.5, 10.5)
            p.setBrush(qcolor(col, alpha * 0.8))
            p.drawRoundedRect(QRectF(c.x() - 3.5, c.y() - 3.5, 7, 7), 1.8, 1.8)
        p.end()


class PillButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self, text: str, accent: Rgb, filled: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text, self.accent, self.filled = text, accent, filled
        self.hover = 0.0
        self.hover_target = 0.0
        self._down = False
        self.setFixedHeight(34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def advance(self, dt: float) -> None:
        new = approach(self.hover, self.hover_target, dt, 0.08)
        moved = abs(new - self.hover) > 0.0005
        self.hover = new
        if moved:
            self.update()

    def enterEvent(self, _) -> None:  # noqa: N802
        self.hover_target = 1.0

    def leaveEvent(self, _) -> None:  # noqa: N802
        self.hover_target = 0.0

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._down = e.button() == Qt.MouseButton.LeftButton

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._down and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        self._down = False

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = r.height() / 2
        if self.filled:
            p.setBrush(qcolor(self.accent, 0.20 + 0.14 * self.hover))
            p.setPen(QPen(qcolor(self.accent, 0.55 + 0.3 * self.hover), 1.0))
        else:
            p.setBrush(QColor.fromRgbF(1, 1, 1, 0.03 + 0.05 * self.hover))
            p.setPen(QPen(QColor.fromRgbF(1, 1, 1, 0.14 + 0.1 * self.hover), 1.0))
        p.drawRoundedRect(r, radius, radius)
        p.setFont(label_font(9.0))
        p.setPen(qcolor(mix(self.accent, (1, 1, 1), 0.5) if self.filled else TEXT, 0.9))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.text.upper())
        p.end()


def state_font(compact: bool) -> QFont:
    """The state word: wide tracking in the panel, tighter in compact mode where 'STANDING BY' must fit."""
    return ui_font(8.5, QFont.Weight.DemiBold, tracking=1.8) if compact else label_font(9.0)


class StateLabel(QWidget):
    """Tracked-capitals state word that cross-fades when the text changes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text = ""
        self.previous = ""
        self.fade = 1.0
        self.accent: Rgb = ACCENT_DEFAULT
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_text(self, text: str) -> None:
        if text == self.text:
            return
        self.previous, self.text, self.fade = self.text, text, 0.0
        self.update()

    def advance(self, dt: float) -> None:
        if self.fade < 1.0:
            self.fade = min(1.0, self.fade + dt / 0.35)
            self.update()

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        p.setFont(state_font(self.width() < 250))
        col = mix(self.accent, (1, 1, 1), 0.35)
        rect = QRectF(self.rect())
        if self.previous and self.fade < 1.0:
            p.setPen(qcolor(col, 0.62 * (1.0 - self.fade)))
            p.drawText(rect.translated(0, -4 * self.fade), Qt.AlignmentFlag.AlignCenter, self.previous)
        p.setPen(qcolor(col, 0.66 * self.fade))
        p.drawText(rect.translated(0, 4 * (1.0 - self.fade)), Qt.AlignmentFlag.AlignCenter, self.text)
        p.end()


class FooterLabel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text = ""
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_text(self, text: str) -> None:
        self.text = text
        self.update()

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        p.setFont(ui_font(8.5, QFont.Weight.Normal))
        p.setPen(QColor.fromRgbF(*TEXT, 0.45))
        fm = QFontMetricsF(p.font())
        p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter,
                   fm.elidedText(self.text, Qt.TextElideMode.ElideRight, self.width()))
        p.end()


GAUGE_MIN_WIDTH = 56
GAUGE_LABEL_GAP = 14.0
# (point size, tracking px): tried in order until the labels fit with the minimum gap between them
GAUGE_LABEL_STEPS = ((6.5, 1.8), (6.5, 1.2), (6.0, 0.9), (5.5, 0.6), (5.0, 0.3))


def gauge_label_font(pt: float, tracking: float, scale: float = 1.0) -> QFont:
    font = label_font(pt * scale)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, tracking * scale)
    return font


def layout_gauge_row(labels: list[str], total: float, scale: float = 1.0) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    """Lay the gauge row out from measured label widths.

    Returns ((point size, tracking), [(x, width) per gauge]). Each cell is as wide as its label (or the
    arc, if wider); the leftover space becomes equal gaps, so label rects never overlap or touch. When the
    row is tight the tracking, then the size, is reduced. `scale` simulates larger system text (1.25, 1.5)."""
    n = len(labels)
    chosen = GAUGE_LABEL_STEPS[-1]
    widths: list[float] = []
    for pt, trk in GAUGE_LABEL_STEPS:
        fm = QFontMetricsF(gauge_label_font(pt, trk, scale))
        widths = [max(GAUGE_MIN_WIDTH * scale, fm.horizontalAdvance(text) + trk * scale) for text in labels]
        chosen = (pt, trk)
        if sum(widths) + GAUGE_LABEL_GAP * scale * (n - 1) <= total:
            break
    gap = (total - sum(widths)) / (n - 1) if n > 1 else 0.0
    cells, x = [], 0.0 if n > 1 else (total - widths[0]) / 2.0
    for w in widths:
        cells.append((x, w))
        x += w + gap
    return chosen, cells


class Gauge(QWidget):
    """Thin 270-degree arc with a centred value; eases toward each new reading."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        self.target = 0.0
        self.shown = 0.0
        self.available = True
        self.accent: Rgb = ACCENT_DEFAULT
        self.suffix = ""
        self.label_style: tuple[float, float] = GAUGE_LABEL_STEPS[0]
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedHeight(70)

    def set_value(self, value: float | None, suffix: str = "") -> None:
        self.available = value is not None
        if value is not None:
            self.target = clamp01(value)
        self.suffix = suffix

    def snap(self) -> None:
        if self.shown != self.target:
            self.shown = self.target
            self.update()

    def advance(self, dt: float) -> None:
        new = approach(self.shown, self.target, dt, 0.35)
        moved = abs(new - self.shown) > 0.0002
        self.shown = new
        if moved:
            self.update()

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        size = min(self.width(), self.height() - 14) - 6
        cx = self.width() / 2
        rect = QRectF(cx - size / 2, 4, size, size)
        col = self.accent if self.shown < 0.85 else CORAL
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor.fromRgbF(1, 1, 1, 0.09), 2.0, cap=Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 225 * 16, -270 * 16)
        if self.shown > 0.004:
            grad = QConicalGradient(rect.center(), 225)
            grad.setColorAt(0.0, qcolor(col, 0.95))
            grad.setColorAt(0.75, qcolor(mix(col, (1, 1, 1), 0.4), 0.95))
            grad.setColorAt(1.0, qcolor(col, 0.95))
            p.setPen(QPen(QBrush(grad), 2.0, cap=Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, 225 * 16, int(-270 * 16 * self.shown))
        p.setFont(ui_font(9.5, QFont.Weight.Light))
        p.setPen(QColor.fromRgbF(*TEXT, 0.92))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{round(self.shown * 100)}{self.suffix}")
        p.setFont(gauge_label_font(*self.label_style))
        p.setPen(QColor.fromRgbF(*TEXT, 0.5))
        p.drawText(QRectF(0, size + 6, self.width(), 12), Qt.AlignmentFlag.AlignCenter, self.label)
        p.end()


# -- tool ticker ---------------------------------------------------------------------
@dataclass
class Chip:
    name: str
    summary: str
    state: str = "running"  # running | ok | fail
    seconds: float = 0.0
    slide: float = 0.0
    alpha: float = 1.0
    y: float = -1.0
    done_for: float = 0.0
    removing: bool = False


class ToolTicker(QWidget):
    MAX = 4
    ROW = 27
    LINGER = 4.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.chips: list[Chip] = []
        self.spin = 0.0
        self.accent: Rgb = ACCENT_DEFAULT
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def spin_needed(self) -> bool:
        return any(c.state == "running" for c in self.chips)

    def visible_chips(self) -> list[Chip]:
        return [c for c in self.chips if not c.removing]

    def running_count(self) -> int:
        return sum(1 for c in self.chips if c.state == "running" and not c.removing)

    def start(self, name: str, summary: str) -> None:
        self.chips.append(Chip(name, summary))
        for chip in self.visible_chips()[: -self.MAX]:
            chip.removing = True

    def finish(self, name: str, ok: bool, seconds: float) -> None:
        for chip in self.chips:
            if chip.name == name and chip.state == "running":
                chip.state = "ok" if ok else "fail"
                chip.seconds = seconds
                return

    def clear(self) -> None:
        for chip in self.chips:
            chip.removing = True

    def wanted_height(self) -> int:
        return len(self.visible_chips()) * self.ROW

    def advance(self, dt: float) -> None:
        self.spin = (self.spin + 420.0 * dt) % 360.0
        moving = False
        for i, chip in enumerate(self.visible_chips()):
            moving = moving or chip.slide < 1.0 or abs(chip.y - i * self.ROW) > 0.3
            chip.slide = min(1.0, chip.slide + dt / 0.32)
            target = i * self.ROW
            chip.y = target if chip.y < 0 else approach(chip.y, target, dt, 0.12)
            if chip.state != "running":
                chip.done_for += dt
                if chip.done_for > self.LINGER:
                    chip.removing = True
        for chip in self.chips:
            if chip.removing:
                chip.alpha -= dt / 0.4
                moving = True
        self.chips = [c for c in self.chips if c.alpha > 0.0]
        if self.chips and (self.spin_needed() or moving):
            self.update()

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        mono = mono_font(9.0)
        for chip in self.chips:
            ease = 1.0 - (1.0 - chip.slide) ** 3
            a = clamp01(chip.alpha) * ease
            y = max(chip.y, 0.0)
            r = QRectF(2 + (1.0 - ease) * 26, y + 1.5, self.width() - 4, self.ROW - 5)
            p.setPen(QPen(QColor.fromRgbF(1, 1, 1, 0.08 * a), 1.0))
            p.setBrush(QColor.fromRgbF(1, 1, 1, 0.04 * a))
            p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
            self._icon(p, chip, QPointF(r.left() + 13, r.center().y()), a)
            p.setFont(mono)
            fm = QFontMetricsF(mono)
            tail = f"{chip.seconds:.1f}s" if chip.state != "running" else ""
            tail_w = fm.horizontalAdvance(tail) + 14
            x = r.left() + 28
            p.setPen(qcolor(mix(self.accent, (1, 1, 1), 0.3), 0.95 * a))
            p.drawText(QPointF(x, r.center().y() + fm.ascent() / 2 - 1), chip.name)
            x += fm.horizontalAdvance(chip.name) + 9
            room = r.right() - tail_w - x - 4
            if room > 20 and chip.summary:
                p.setPen(QColor.fromRgbF(*TEXT, 0.50 * a))
                p.drawText(QPointF(x, r.center().y() + fm.ascent() / 2 - 1),
                           fm.elidedText(chip.summary, Qt.TextElideMode.ElideRight, room))
            if tail:
                p.setPen(QColor.fromRgbF(*TEXT, 0.55 * a))
                p.drawText(QRectF(r.right() - tail_w, r.top(), tail_w - 8, r.height()),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, tail)
        p.end()

    def _icon(self, p: QPainter, chip: Chip, c: QPointF, a: float) -> None:
        p.setBrush(Qt.BrushStyle.NoBrush)
        if chip.state == "running":
            rect = QRectF(c.x() - 5, c.y() - 5, 10, 10)
            p.setPen(QPen(QColor.fromRgbF(1, 1, 1, 0.12 * a), 1.5))
            p.drawEllipse(rect)
            p.setPen(QPen(qcolor(self.accent, a), 1.6, cap=Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, int(-self.spin * 16), 100 * 16)
        elif chip.state == "ok":
            p.setPen(QPen(qcolor(GREEN, 0.95 * a), 1.6, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin))
            p.drawPolyline([QPointF(c.x() - 4, c.y()), QPointF(c.x() - 1, c.y() + 3), QPointF(c.x() + 4.5, c.y() - 3.5)])
        else:
            p.setPen(QPen(qcolor(CORAL, 0.95 * a), 1.6, cap=Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(c.x() - 3.5, c.y() - 3.5), QPointF(c.x() + 3.5, c.y() + 3.5))
            p.drawLine(QPointF(c.x() - 3.5, c.y() + 3.5), QPointF(c.x() + 3.5, c.y() - 3.5))


# -- conversation --------------------------------------------------------------------
@dataclass
class _Word:
    text: str
    born: float
    x: float = 0.0
    y: float = 0.0


@dataclass
class _Exchange:
    user: str
    words: list[_Word] = field(default_factory=list)
    pending: deque[str] = field(default_factory=deque)
    all_text: list[str] = field(default_factory=list)
    alpha: float = 0.0
    user_lines: list[tuple[str, float]] = field(default_factory=list)
    top: float = 0.0
    height: float = 0.0


class ReplyView(QWidget):
    """The user's line (dim) and the reply revealed word by word; older exchanges fade up."""

    GAP = 20.0
    WORDS_PER_SECOND = 11.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.exchanges: list[_Exchange] = []
        self.clock = 0.0
        self.owed = 0.0
        self.offset = 0.0
        self.offset_target = 0.0
        self.scroll = 0.0
        self.whisper_text = ""
        self.whisper_left = 0.0
        self.hint_text = "Say “Jarvis” or type below"
        self.hint_alpha = 1.0
        self.dirty = True
        self._laid_width = -1
        self.body_font = ui_font(11.5, QFont.Weight.Light)
        self.user_font = ui_font(10.0, QFont.Weight.Normal)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    # -- model ---------------------------------------------------------------------
    def begin_exchange(self, user: str) -> None:
        self.exchanges.append(_Exchange(user=user.strip()))
        self.scroll = 0.0
        self.dirty = True

    def set_audio(self, has_audio: bool, wake_word: str = "Jarvis") -> None:
        """The empty-state hint mentions speaking only when there is a microphone to speak to."""
        self.hint_text = f"Say “{wake_word}” or type below" if has_audio else "Type below to ask Jarvis"
        self.update()

    def hint_visible(self) -> bool:
        return not self.exchanges and self.whisper_left <= 0.0

    def current(self) -> _Exchange | None:
        return self.exchanges[-1] if self.exchanges else None

    def append_reply(self, text: str) -> None:
        if self.current() is None:
            self.begin_exchange("")
        ex = self.exchanges[-1]
        ex.pending.extend(text.split())
        ex.all_text.append(text.strip())

    def reply_text(self) -> str:
        ex = self.current()
        return " ".join(ex.all_text) if ex else ""

    def revealed_text(self) -> str:
        ex = self.current()
        return " ".join(w.text for w in ex.words) if ex else ""

    def whisper(self, text: str, seconds: float = 2.0) -> None:
        self.whisper_text, self.whisper_left = text, seconds

    def clear(self) -> None:
        self.exchanges.clear()
        self.dirty = True
        self.update()

    # -- time ----------------------------------------------------------------------
    def advance(self, dt: float) -> None:
        self.clock += dt
        ex = self.current()
        changed = False
        want = 1.0 if self.hint_visible() else 0.0
        if self.hint_alpha != want:
            self.hint_alpha = approach(self.hint_alpha, want, dt, 0.22)
            if abs(self.hint_alpha - want) < 0.01:
                self.hint_alpha = want
            changed = True
        if ex is not None and ex.pending:
            rate = self.WORDS_PER_SECOND + 1.6 * len(ex.pending)
            self.owed += dt * min(rate, 42.0)
            while self.owed >= 1.0 and ex.pending:
                self.owed -= 1.0
                ex.words.append(_Word(ex.pending.popleft(), self.clock))
                self.dirty = changed = True
        elif ex is not None:
            self.owed = 0.0
        for i, e in enumerate(self.exchanges):
            age = len(self.exchanges) - 1 - i
            target = 1.0 if age == 0 else 0.4 if age == 1 else 0.0
            e.alpha = approach(e.alpha, target, dt, 0.35)
        if self.whisper_left > 0.0:
            self.whisper_left = max(0.0, self.whisper_left - dt)
            changed = True
        if self.dirty or self.width() != self._laid_width:
            self._layout()
        new = approach(self.offset, self.offset_target + self.scroll, dt, 0.16)
        if abs(new - self.offset) > 0.05:
            self.offset = new
            changed = True
        if changed or any(w.born > self.clock - 0.5 for e in self.exchanges for w in e.words[-6:]):
            self.update()

    def _layout(self) -> None:
        width = max(40.0, self.width() - 4.0)
        self._laid_width = self.width()
        self.dirty = False
        body, user = QFontMetricsF(self.body_font), QFontMetricsF(self.user_font)
        space = body.horizontalAdvance(" ")
        line_h = body.height() * 1.32
        y = 0.0
        for ex in self.exchanges:
            ex.top = y
            ex.user_lines = []
            if ex.user:
                uy = y
                for line in _wrap(ex.user, user, width):
                    ex.user_lines.append((line, uy))
                    uy += user.height() * 1.25
                y = uy + 6.0
            x = 0.0
            for w in ex.words:
                ww = body.horizontalAdvance(w.text)
                if x > 0 and x + ww > width:
                    x, y = 0.0, y + line_h
                w.x, w.y = x, y
                x += ww + space
            ex.height = y + line_h - ex.top
            y += line_h + self.GAP
        total = max(0.0, y - self.GAP)
        self.content_height = total
        self.offset_target = self.height() - total
        self.scroll = min(self.scroll, max(0.0, total - self.height()))
        self.exchanges = [e for e in self.exchanges if not (e.alpha < 0.01 and e is not self.exchanges[-1]
                                                            and e.top + e.height + self.offset < -10
                                                            and len(self.exchanges) > 2)]

    content_height = 0.0

    def wheelEvent(self, e) -> None:  # noqa: N802
        delta = e.angleDelta().y() / 120.0 * 36.0
        limit = max(0.0, self.content_height - self.height())
        self.scroll = min(limit, max(0.0, self.scroll + delta))
        e.accept()

    # -- painting ------------------------------------------------------------------
    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        h = self.height()
        body = QFontMetricsF(self.body_font)
        fade_top = 46.0
        for ex in self.exchanges:
            base = ex.alpha
            if base < 0.01:
                continue
            p.setFont(self.user_font)
            for line, ly in ex.user_lines:
                y = ly + self.offset
                a = base * 0.52 * self._edge(y, h, fade_top)
                if a > 0.01:
                    p.setPen(QColor.fromRgbF(*TEXT, a))
                    p.drawText(QPointF(2, y + QFontMetricsF(self.user_font).ascent()), line)
            p.setFont(self.body_font)
            for w in ex.words:
                y = w.y + self.offset
                a = base * 0.94 * clamp01((self.clock - w.born) / 0.4) * self._edge(y, h, fade_top)
                if a > 0.01:
                    p.setPen(QColor.fromRgbF(*TEXT, a))
                    p.drawText(QPointF(2 + w.x, y + body.ascent() + (1.0 - clamp01((self.clock - w.born) / 0.4)) * 3), w.text)
        if self.hint_alpha > 0.01:
            p.setFont(ui_font(10.0, QFont.Weight.Light, tracking=0.6))
            p.setPen(QColor.fromRgbF(*TEXT, 0.34 * self.hint_alpha))
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, self.hint_text)
        if self.whisper_left > 0.0 and self.whisper_text:
            p.setFont(ui_font(9.0, QFont.Weight.Normal))
            p.setPen(QColor.fromRgbF(*TEXT, 0.28 * clamp01(self.whisper_left / 0.6)))
            p.drawText(QRectF(0, h - 16, self.width(), 16), Qt.AlignmentFlag.AlignCenter, self.whisper_text)
        p.end()

    @staticmethod
    def _edge(y: float, height: float, fade: float) -> float:
        if y > height:
            return 0.0
        return clamp01((y + 14.0) / fade)


def _wrap(text: str, fm: QFontMetricsF, width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and fm.horizontalAdvance(trial) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines[:6]


# -- confirmation --------------------------------------------------------------------
class ConfirmCard(QWidget):
    answered = pyqtSignal(bool)
    TIMEOUT = 15.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.description = ""
        self.slide = 1.0
        self.slide_target = 1.0
        self.remaining = self.TIMEOUT
        self.open = False
        self.allow = PillButton("Allow", AMBER, True, self)
        self.deny = PillButton("Deny", AMBER, False, self)
        self.allow.clicked.connect(lambda: self.answered.emit(True))
        self.deny.clicked.connect(lambda: self.answered.emit(False))
        self.hide()

    def request(self, description: str) -> None:
        self.description = description
        self.remaining = self.TIMEOUT
        self.open = True
        self.slide_target = 0.0
        self.show()
        self.raise_()
        self.update()

    def resolve(self) -> None:
        self.open = False
        self.slide_target = 1.0

    def resizeEvent(self, _) -> None:  # noqa: N802
        w = self.width()
        self.allow.setGeometry(w - 22 - 96 - 10 - 96, self.height() - 52, 96, 34)
        self.deny.setGeometry(w - 22 - 96, self.height() - 52, 96, 34)

    def advance(self, dt: float) -> None:
        if not self.isVisible():
            return
        self.slide = approach(self.slide, self.slide_target, dt, 0.09)
        if self.open:
            self.remaining = max(0.0, self.remaining - dt)
        self.allow.advance(dt)
        self.deny.advance(dt)
        if not self.open and self.slide > 0.985:
            self.hide()
        self.update()

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        a = clamp01(1.0 - self.slide)
        r = QRectF(self.rect()).adjusted(1, 1 + self.slide * 40, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(r, 18, 18)
        fill = QLinearGradient(r.topLeft(), r.bottomLeft())
        fill.setColorAt(0.0, qcolor(mix(GLASS_BASE, AMBER, 0.10), a))
        fill.setColorAt(1.0, qcolor(GLASS_BASE, a))
        p.fillPath(path, QBrush(fill))
        p.setPen(QPen(qcolor(AMBER, 0.55 * a), 1.0))
        p.drawPath(path)
        p.setFont(label_font(8.0))
        p.setPen(qcolor(AMBER, 0.85 * a))
        p.drawText(QRectF(r.left() + 22, r.top() + 16, 200, 14), Qt.AlignmentFlag.AlignLeft, "CONFIRM")
        ring = QRectF(r.right() - 40, r.top() + 14, 18, 18)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor.fromRgbF(1, 1, 1, 0.12 * a), 1.6))
        p.drawEllipse(ring)
        p.setPen(QPen(qcolor(AMBER, 0.95 * a), 1.6, cap=Qt.PenCapStyle.RoundCap))
        p.drawArc(ring, 90 * 16, int(-360 * 16 * clamp01(self.remaining / self.TIMEOUT)))
        p.setFont(ui_font(11.0, QFont.Weight.Normal))
        p.setPen(QColor.fromRgbF(*TEXT, 0.95 * a))
        text_rect = QRectF(r.left() + 22, r.top() + 40, r.width() - 44, r.height() - 100)
        fm = QFontMetricsF(p.font())
        lines = _wrap(self.description, fm, text_rect.width())[:4]
        p.drawText(text_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop), "\n".join(lines))
        p.end()


# -- input ---------------------------------------------------------------------------
class PillInput(QWidget):
    submitted = pyqtSignal(str)
    stop_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.accent: Rgb = ACCENT_DEFAULT
        self.focus = 0.0
        self.busy = False
        self.busy_alpha = 0.0
        self.edit = QLineEdit(self)
        self.edit.setFrame(False)
        self.edit.setPlaceholderText("Ask Jarvis…")
        self.edit.setFont(ui_font(11.0, QFont.Weight.Normal))
        self.edit.setStyleSheet(
            "QLineEdit{background:transparent;border:none;color:#E6F2FF;"
            "selection-background-color:rgba(94,230,255,80);}"
        )
        pal = self.edit.palette()
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(190, 210, 235, 110))
        self.edit.setPalette(pal)
        self.edit.returnPressed.connect(self._submit)
        self.edit.installEventFilter(self)
        self.stop = IconButton("stop", self)
        self.stop.clicked.connect(self.stop_requested)
        self.stop.hide()
        self.setFixedHeight(46)

    def _submit(self) -> None:
        text = self.edit.text().strip()
        if text:
            self.edit.clear()
            self.submitted.emit(text)

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        if busy:
            self.stop.show()

    def eventFilter(self, obj, event):  # noqa: N802
        return super().eventFilter(obj, event)

    def resizeEvent(self, _) -> None:  # noqa: N802
        self.stop.move(self.width() - 38, (self.height() - 30) // 2)
        self.edit.setGeometry(22, 4, self.width() - 22 - 46, self.height() - 8)

    def advance(self, dt: float) -> None:
        self.stop.accent = self.accent
        self.stop.advance(dt)
        want = 1.0 if self.edit.hasFocus() else 0.0
        new = approach(self.focus, want, dt, 0.1)
        ba = approach(self.busy_alpha, 1.0 if self.busy else 0.0, dt, 0.12)
        moved = abs(new - self.focus) > 0.0005 or abs(ba - self.busy_alpha) > 0.0005
        self.focus, self.busy_alpha = new, ba
        if moved:
            self.stop.setVisible(ba > 0.02)
            self.update()

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        r = QRectF(self.rect()).adjusted(3, 3, -3, -3)
        rad = r.height() / 2
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(1, 5):
            p.setPen(QPen(qcolor(self.accent, 0.10 * self.focus * (1 - i / 5.0)), 2.0))
            g = i * 1.2
            p.drawRoundedRect(r.adjusted(-g, -g, g, g), rad + g, rad + g)
        p.setBrush(QColor.fromRgbF(1, 1, 1, 0.045 + 0.02 * self.focus))
        p.setPen(QPen(mix_q(self.accent, self.focus), 1.0))
        p.drawRoundedRect(r, rad, rad)
        p.end()


def mix_q(accent: Rgb, focus: float) -> QColor:
    base = QColor.fromRgbF(1, 1, 1, 0.11)
    tint = qcolor(accent, 0.55)
    return QColor.fromRgbF(
        base.redF() + (tint.redF() - base.redF()) * focus,
        base.greenF() + (tint.greenF() - base.greenF()) * focus,
        base.blueF() + (tint.blueF() - base.blueF()) * focus,
        base.alphaF() + (tint.alphaF() - base.alphaF()) * focus,
    )


class StateDot(QWidget):
    """The breathing dot beside the wordmark. A widget of its own so that its repaint is 24 x 24
    pixels, not the whole header (text shaping is the expensive part of the header)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.accent: Rgb = ACCENT_DEFAULT
        self.pulse = 0.0
        self.setFixedSize(24, 24)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        _aa(p)
        c = QPointF(6, 12)
        glow = QRadialGradient(c, 11)
        glow.setColorAt(0.0, qcolor(self.accent, 0.45 + 0.2 * self.pulse))
        glow.setColorAt(1.0, qcolor(self.accent, 0.0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(c, 11, 11)
        p.setBrush(qcolor(mix(self.accent, (1, 1, 1), 0.4), 1.0))
        p.drawEllipse(c, 2.6, 2.6)
        p.end()


class Header(QWidget):
    """State dot, tracked wordmark, and the mic / minimise controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.accent: Rgb = ACCENT_DEFAULT
        self.t = 0.0
        self._pulse_level = -1
        self._mark: QPixmap | None = None
        self._mark_dpr = 0.0
        self.dot = StateDot(self)
        self.mic = IconButton("mic", self)
        self.minimise = IconButton("min", self)
        self.setFixedHeight(40)

    def resizeEvent(self, _) -> None:  # noqa: N802
        self.dot.move(0, (self.height() - self.dot.height()) // 2)
        self.minimise.move(self.width() - 30, 5)
        self.mic.move(self.width() - 66, 5)

    def advance(self, dt: float) -> None:
        self.t += dt
        for b in (self.mic, self.minimise):
            b.accent = self.accent
            b.advance(dt)
        # The dot breathes in 6 steps, repainting only when a step changes (about 4 times a second):
        # every repaint outside the orb widens the area the window has to present for that frame.
        level = round(3.0 + 3.0 * math.sin(self.t * 2.0))
        if level != self._pulse_level:
            self._pulse_level = level
            self.dot.accent = self.accent
            self.dot.pulse = level / 6.0
            self.dot.update()

    def paintEvent(self, _) -> None:  # noqa: N802
        # The wordmark never changes but the dot under it repaints often (each repaint re-runs every
        # widget beneath it), so the wordmark is shaped once into a pixmap.
        dpr = self.devicePixelRatioF()
        if self._mark is None or self._mark_dpr != dpr:
            self._mark_dpr = dpr
            pm = QPixmap(round(200 * dpr), round(self.height() * dpr))
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            _aa(q)
            q.setFont(ui_font(9.5, QFont.Weight.DemiBold, tracking=6.0))
            q.setPen(QColor.fromRgbF(*TEXT, 0.78))
            q.drawText(QRectF(0, 0, 200, self.height()), Qt.AlignmentFlag.AlignVCenter, "JARVIS")
            q.end()
            self._mark = pm
        p = QPainter(self)
        p.drawPixmap(24, 0, self._mark)
        p.end()
