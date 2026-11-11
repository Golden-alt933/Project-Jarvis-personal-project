"""The code panel: fenced code blocks of the current reply, monospace, indentation kept, one Copy each.

A child of the HUD window (so it shares the window's capture exclusion, always-on-top state and position,
and showing it never activates anything). It paints only on events: show, wheel, hover, click, and one
single-shot timer after a copy. There is no animation clock.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QGuiApplication, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from jarvis.hud.theme import GLASS_BASE, TEXT, Rgb, label_font, mix, mono_font, qcolor, ui_font

ACCENT_DEFAULT: Rgb = (0.37, 0.90, 1.0)
GREEN: Rgb = (0.50, 0.95, 0.72)
MAX_HEIGHT = 420  # hard cap, whatever room the window offers
PAD = 12.0
HEADER_H = 26.0
BLOCK_GAP = 10.0
RIGHT_GUTTER = 32.0  # the close button lives here; Copy buttons stop short of it
COPIED_SECONDS = 1.4
TAB_WIDTH = 4


@dataclass
class CodeBlock:
    code: str  # exactly as written
    language: str = ""

    @property
    def lines(self) -> list[str]:
        """Display lines: a trailing newline does not add an empty last line."""
        return self.code.rstrip("\n").split("\n")

    @property
    def title(self) -> str:
        return (self.language or "code").upper()


class CodePanel(QWidget):
    """Scrollable stack of code blocks. Never takes focus."""

    closed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.blocks: list[CodeBlock] = []
        self.accent: Rgb = ACCENT_DEFAULT
        self.scroll_y = 0.0
        self.scroll_x = 0.0
        self.copied: int | None = None
        self._hover: tuple[str, int] | None = None
        self._down: tuple[str, int] | None = None
        self._accent_key = (0, 0, 0)
        self.code_font = mono_font(9.5)
        self._cell = QFontMetricsF(self.code_font)
        self.line_h = self._cell.height() * 1.15
        self._reset = QTimer(self)
        self._reset.setSingleShot(True)
        self._reset.timeout.connect(self._clear_copied)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.hide()

    # -- model ---------------------------------------------------------------------
    def add_block(self, code: str, language: str = "") -> None:
        self.blocks.append(CodeBlock(code, language))
        self.copied = None
        self.scroll_x = 0.0
        if len(self.blocks) == 1:
            self.scroll_y = 0.0
        self.setVisible(True)
        self.update()

    def clear(self) -> None:
        if not self.blocks:
            return
        self.blocks.clear()
        self.scroll_y = self.scroll_x = 0.0
        self.copied = None
        self._reset.stop()
        self.hide()

    def dismiss(self) -> None:
        self.clear()
        self.closed.emit()

    def block_text(self, index: int) -> str:
        return self.blocks[index].code

    def copy_block(self, index: int) -> None:
        """Put block `index`'s exact text on the clipboard and show 'Copied' for a moment."""
        if not 0 <= index < len(self.blocks):
            return
        QGuiApplication.clipboard().setText(self.blocks[index].code)
        self.copied = index
        self._reset.start(int(COPIED_SECONDS * 1000))
        self.update()

    def _clear_copied(self) -> None:
        self.copied = None
        self.update()

    def set_accent(self, accent: Rgb) -> None:
        key = tuple(round(c * 12) for c in accent)
        self.accent = accent
        if key != self._accent_key:
            self._accent_key = key
            if self.isVisible():
                self.update()

    # -- geometry ------------------------------------------------------------------
    def _block_height(self, b: CodeBlock) -> float:
        return HEADER_H + len(b.lines) * self.line_h + PAD * 0.8

    def content_height(self) -> float:
        if not self.blocks:
            return 0.0
        return sum(self._block_height(b) for b in self.blocks) + BLOCK_GAP * (len(self.blocks) - 1)

    def wanted_height(self, available: int = MAX_HEIGHT) -> int:
        """Height the panel asks for: its content plus padding, never above `available` or MAX_HEIGHT."""
        return int(min(self.content_height() + PAD * 1.4, available, MAX_HEIGHT)) if self.blocks else 0

    def _view_h(self) -> float:
        return max(0.0, self.height() - PAD * 0.7 * 2)

    def _gutter_w(self, b: CodeBlock) -> float:
        return self._cell.horizontalAdvance("0") * max(2, len(str(len(b.lines)))) + 14.0

    def _text_w(self, line: str) -> float:
        return self._cell.horizontalAdvance(line.expandtabs(TAB_WIDTH))

    def _max_scroll_y(self) -> float:
        return max(0.0, self.content_height() - self._view_h())

    def _max_scroll_x(self) -> float:
        if not self.blocks:
            return 0.0
        widest = max(self._gutter_w(b) + max(self._text_w(ln) for ln in b.lines) for b in self.blocks)
        return max(0.0, widest + 8.0 - (self.width() - 2 * PAD))

    def _clamp(self) -> None:
        self.scroll_y = min(self._max_scroll_y(), max(0.0, self.scroll_y))
        self.scroll_x = min(self._max_scroll_x(), max(0.0, self.scroll_x))

    def resizeEvent(self, e) -> None:  # noqa: N802
        self._clamp()
        super().resizeEvent(e)

    def _block_tops(self) -> list[float]:
        tops, y = [], PAD * 0.7
        for b in self.blocks:
            tops.append(y - self.scroll_y)
            y += self._block_height(b) + BLOCK_GAP
        return tops

    def _copy_rect(self, top: float, index: int) -> QRectF:
        label = "COPIED" if self.copied == index else "COPY"
        w = QFontMetricsF(label_font(7.5)).horizontalAdvance(label) + 26.0
        return QRectF(self.width() - PAD - RIGHT_GUTTER - w + 10.0, top + 4.0, w, 20.0)

    def _close_rect(self) -> QRectF:
        return QRectF(self.width() - PAD - 22.0, PAD * 0.7 + 3.0, 22.0, 22.0)

    def _hit(self, pos: QPointF) -> tuple[str, int] | None:
        if self.blocks and self._close_rect().contains(pos):
            return ("close", 0)
        for i, top in enumerate(self._block_tops()):
            if self._copy_rect(top, i).contains(pos):
                return ("copy", i)
        return None

    # -- input ---------------------------------------------------------------------
    def wheelEvent(self, e) -> None:  # noqa: N802
        dx, dy = e.angleDelta().x(), e.angleDelta().y()
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier and not dx:
            dx, dy = dy, 0
        self.scroll_y -= dy / 120.0 * 3 * self.line_h
        self.scroll_x -= dx / 120.0 * 6 * self._cell.horizontalAdvance("0")
        self._clamp()
        self.update()
        e.accept()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        hit = self._hit(e.position())
        if hit != self._hover:
            self._hover = hit
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, e) -> None:  # noqa: N802
        if self._hover is not None:
            self._hover = None
            self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._down = self._hit(e.position()) if e.button() == Qt.MouseButton.LeftButton else None
        e.accept()  # never fall through to the window's drag handling

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        hit = self._hit(e.position())
        if hit is not None and hit == self._down:
            self.dismiss() if hit[0] == "close" else self.copy_block(hit[1])
        self._down = None
        e.accept()

    # -- painting ------------------------------------------------------------------
    def paintEvent(self, _) -> None:  # noqa: N802
        if not self.blocks:
            return
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        card = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(card, 14, 14)
        p.fillPath(path, qcolor(mix(GLASS_BASE, (0.0, 0.0, 0.0), 0.35), 1.0))
        p.setPen(QPen(QColor.fromRgbF(1, 1, 1, 0.11), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setClipPath(path)
        for i, (b, top) in enumerate(zip(self.blocks, self._block_tops())):
            if top > self.height() or top + self._block_height(b) < 0:
                continue
            self._paint_block(p, i, b, top)
        self._paint_close(p)
        self._paint_scrollbar(p)
        p.end()

    def _paint_block(self, p: QPainter, index: int, b: CodeBlock, top: float) -> None:
        left, width = PAD, self.width() - 2 * PAD
        height = self._block_height(b)
        body = QRectF(left, top, width, height)
        p.setPen(QPen(QColor.fromRgbF(1, 1, 1, 0.07), 1.0))
        p.setBrush(QColor.fromRgbF(1, 1, 1, 0.035))
        p.drawRoundedRect(body, 9, 9)
        # header: language and line count, then Copy
        p.setFont(label_font(7.5))
        p.setPen(qcolor(mix(self.accent, (1, 1, 1), 0.3), 0.85))
        p.drawText(QRectF(left + 12, top, width, HEADER_H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, b.title)
        n = len(b.lines)
        p.setFont(ui_font(8.0, QFont.Weight.Normal))
        p.setPen(QColor.fromRgbF(*TEXT, 0.40))
        title_w = QFontMetricsF(label_font(7.5)).horizontalAdvance(b.title)
        p.drawText(QRectF(left + 12 + title_w + 10, top, width, HEADER_H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   f"{n} line{'s' if n != 1 else ''}")
        self._paint_copy(p, index, top)
        # code: clipped to the block body, scrolled horizontally, only visible lines drawn
        p.save()
        p.setClipRect(QRectF(left, top + HEADER_H, width, height - HEADER_H - 2), Qt.ClipOperation.IntersectClip)
        gutter = self._gutter_w(b)
        asc = self._cell.ascent()
        first = max(0, int((-(top + HEADER_H)) / self.line_h))
        for li in range(first, n):
            y = top + HEADER_H + li * self.line_h
            if y > self.height():
                break
            base = y + (self.line_h - self._cell.height()) / 2 + asc
            p.setFont(self.code_font)
            p.setPen(QColor.fromRgbF(*TEXT, 0.22))
            p.drawText(QPointF(left + 12 - self.scroll_x, base), str(li + 1).rjust(max(2, len(str(n)))))
            p.setPen(QColor.fromRgbF(*TEXT, 0.92))
            p.drawText(QPointF(left + 12 + gutter - self.scroll_x, base), b.lines[li].expandtabs(TAB_WIDTH))
        p.restore()

    def _paint_copy(self, p: QPainter, index: int, top: float) -> None:
        r = self._copy_rect(top, index)
        done = self.copied == index
        hot = self._hover == ("copy", index)
        col = GREEN if done else self.accent
        p.setBrush(qcolor(col, 0.22 if done else 0.12 + 0.12 * hot))
        p.setPen(QPen(qcolor(col, 0.65 if done else 0.38 + 0.25 * hot), 1.0))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setFont(label_font(7.5))
        p.setPen(qcolor(mix(col, (1, 1, 1), 0.5), 0.95))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "COPIED" if done else "COPY")

    def _paint_close(self, p: QPainter) -> None:
        r = self._close_rect()
        hot = self._hover == ("close", 0)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor.fromRgbF(1, 1, 1, 0.04 + 0.08 * hot))
        p.drawEllipse(r)
        p.setPen(QPen(qcolor(self.accent, 0.55 + 0.4 * hot), 1.5, cap=Qt.PenCapStyle.RoundCap))
        c, d = r.center(), 3.6
        p.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
        p.drawLine(QPointF(c.x() - d, c.y() + d), QPointF(c.x() + d, c.y() - d))

    def _paint_scrollbar(self, p: QPainter) -> None:
        limit = self._max_scroll_y()
        if limit <= 0:
            return
        view = self._view_h()
        total = view + limit
        h = max(24.0, view * view / total)
        y = PAD * 0.7 + (view - h) * (self.scroll_y / limit)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor.fromRgbF(1, 1, 1, 0.20))
        p.drawRoundedRect(QRectF(self.width() - 5.5, y, 3.0, h), 1.5, 1.5)
