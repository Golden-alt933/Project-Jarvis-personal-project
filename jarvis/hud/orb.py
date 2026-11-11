"""The orb: one QPainter-drawn widget, layered, always moving a little.

Layers, back to front: halo, level ticks, orbiting arcs (+ tool comet), living ring,
particles, core with ripples. All motion derives from a single time value advanced by
`advance(dt)`, so a caller can drive it with real or fake time.

Cost model (this runs every frame on a laptop): the soft glows are radial gradients, the
most expensive thing a raster painter does, so each is rendered ONCE into a white alpha
mask, tinted per quantised colour, and then blitted scaled with an opacity. The thin lines
are drawn with `drawLines` and flat caps, which Qt strokes about twice as fast as polylines
and four to seven times faster than round caps. Painting is in device-independent pixels
(no large painter scale, which makes the stroker slower).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

from jarvis.hud.theme import (
    PaletteBlend,
    Rgb,
    Smoother,
    approach,
    clamp01,
    ease_out_cubic,
    mix,
    qcolor,
)
from jarvis.voice.events import VoiceState

TAU = math.tau
TICKS = 72
RING_POINTS = 40
PARTICLES = 28
ENTRANCE_SECONDS = 0.9
WHITE: Rgb = (1.0, 1.0, 1.0)
CLEAR = QColor(0, 0, 0, 0)

# Per-state motion multipliers: (arc speed, particle speed, breath period seconds).
MOTION = {
    VoiceState.IDLE: (1.0, 0.6, 4.6),
    VoiceState.LISTENING: (1.3, 0.8, 3.8),
    VoiceState.THINKING: (3.4, 1.9, 2.6),
    VoiceState.SPEAKING: (1.7, 1.2, 3.2),
}
PALETTE_KEY = {
    VoiceState.IDLE: "idle",
    VoiceState.LISTENING: "listening",
    VoiceState.THINKING: "thinking",
    VoiceState.SPEAKING: "speaking",
}


@dataclass
class _Mote:
    radius: float
    ratio: float
    tilt: float
    angle: float
    speed: float
    depth: float
    drift: float
    drift_rate: float
    size: float
    ct: float = 1.0
    st: float = 0.0


@dataclass(frozen=True)
class _Arc:
    radius: float
    spans: tuple[tuple[float, float], ...]  # (start offset deg, span deg)
    speed: float  # deg/s at idle
    width: float
    alpha: float


ARCS = (
    _Arc(0.745, ((0.0, 118.0), (170.0, 52.0)), 13.0, 0.012, 0.85),
    _Arc(0.815, ((40.0, 76.0), (200.0, 130.0)), -21.0, 0.009, 0.65),
    _Arc(0.885, ((90.0, 40.0), (190.0, 24.0), (270.0, 70.0)), 8.0, 0.007, 0.5),
)
HARMONICS = (  # (k, drift deg/s, idle amplitude, level gain)
    (2, 17.0, 0.010, 0.060),
    (3, -23.0, 0.008, 0.050),
    (5, 31.0, 0.006, 0.040),
    (7, -41.0, 0.004, 0.028),
)

GLOW_QUANT = 24  # colour steps per channel for tinted glow pixmaps (about 10/255 apart)
CACHE_LIMIT = 72
REST_SLOWDOWN = 0.5  # motion speed lost at full rest
RING_RADIUS = 0.405  # the living ring's mean radius, orb units
RING_GLOW_R = 0.52  # extent of its baked glow
RING_GLOW = 0.38  # opacity of that glow inside the ambient sprite
AMBIENT_SECOND = 0.5  # opacity of the secondary glow baked into the ambient sprite
AMBIENT_OFFSET = 0.12  # its offset from the centre, in orb units
CORE_REF = 0.215  # core sprite radius in orb units: the largest the core ever gets
GLOW_RES = 0.5  # soft glows are rendered at this fraction of device resolution
FLAT = Qt.PenCapStyle.FlatCap
BEVEL = Qt.PenJoinStyle.BevelJoin
SOLID = Qt.PenStyle.SolidLine
NO_PEN = Qt.PenStyle.NoPen
NO_BRUSH = Qt.BrushStyle.NoBrush
_TICK_ANGLES = [-math.pi / 2 + i * TAU / TICKS for i in range(TICKS)]
TICK_COS = [math.cos(a) for a in _TICK_ANGLES]
TICK_SIN = [math.sin(a) for a in _TICK_ANGLES]
_RING_ANGLES = [-math.pi / 2 + TAU * i / RING_POINTS for i in range(RING_POINTS + 1)]
RING_COS = [math.cos(a) for a in _RING_ANGLES]
RING_SIN = [math.sin(a) for a in _RING_ANGLES]
# sin/cos of k*angle for each harmonic, so a ring vertex costs a few multiplies per frame
HARM_SIN = [[math.sin(k * a) for a in _RING_ANGLES] for (k, _, _, _) in HARMONICS]
HARM_COS = [[math.cos(k * a) for a in _RING_ANGLES] for (k, _, _, _) in HARMONICS]

# name -> (reference radius in orb units, gaussian hardness, reach). A mask is drawn scaled to any
# radius. `reach` < 1 stops the mask where the gaussian tail is already invisible, which cuts the
# blitted area (the cost of a blit is its destination pixels).
GLOWS = {
    "halo": (1.04, 2.6, 0.70),
    "halo2": (0.72, 2.2, 0.75),
    "shell": (0.66, 2.0, 0.85),
    "head": (0.075, 1.8, 1.0),
    "dot": (0.036, 0.0, 1.0),
}
UNIT_CIRCLE = [(math.cos(TAU * i / 40), math.sin(TAU * i / 40)) for i in range(41)]


def _conical(angle_deg: float, stops: list[tuple[float, QColor]]) -> QBrush:
    grad = QConicalGradient(0.0, 0.0, angle_deg)
    for pos, color in stops:
        grad.setColorAt(clamp01(pos), color)
    return QBrush(grad)


def _quant(rgb: Rgb) -> tuple[int, int, int]:
    q = GLOW_QUANT
    return (round(clamp01(rgb[0]) * q), round(clamp01(rgb[1]) * q), round(clamp01(rgb[2]) * q))


class GlowCache:
    """White alpha masks for the soft glows, and their tinted copies, for one (size, dpr).

    `mask_builds` / `tint_builds` count gradient renders, so a test can assert that painting
    consecutive frames reuses the cache."""

    def __init__(self) -> None:
        self.key: int | None = None
        self.sprites: dict[tuple, QPixmap] = {}
        self.sprite_builds = 0
        self.u = 1.0
        self.dpr = 1.0
        self.masks: dict[str, QPixmap] = {}
        self.tinted: dict[tuple[str, tuple[int, int, int]], QPixmap] = {}
        self.mask_builds = 0
        self.tint_builds = 0

    def prepare(self, u: float, dpr: float) -> None:
        # Masks are scaled when blitted, so a small size change (the orb growing while the window
        # opens) keeps the cache; only a real change in scale or display ratio rebuilds it.
        ratio = u / self.u if self.u else 0.0
        if self.key is None or self.key != round(dpr * 100) or not 0.85 < ratio < 1.2:
            self.key = round(dpr * 100)
            self.masks.clear()
            self.tinted.clear()
            self.sprites.clear()
            self.u, self.dpr = u, dpr

    def _mask(self, name: str) -> QPixmap:
        pm = self.masks.get(name)
        if pm is None:
            ref, hard, reach = GLOWS[name]
            mdpr = self.dpr * GLOW_RES
            side = max(2, math.ceil(2.0 * ref * reach * self.u * mdpr))
            pm = QPixmap(side, side)
            pm.setDevicePixelRatio(mdpr)
            pm.fill(CLEAR)
            half = side / mdpr / 2.0
            grad = QRadialGradient(QPointF(half, half), half)
            steps = 12  # many stops keep the 8-bit gradient from banding
            for i in range(steps + 1):
                x = i / steps
                alpha = math.exp(-((x * reach * hard) ** 2)) * (1.0 - (x * reach) ** 6) * (1.0 - x**4)
                grad.setColorAt(x, QColor.fromRgbF(1.0, 1.0, 1.0, alpha))
            q = QPainter(pm)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            q.setPen(NO_PEN)
            q.setBrush(QBrush(grad))
            q.drawEllipse(QPointF(half, half), half, half)
            q.end()
            self.masks[name] = pm
            self.mask_builds += 1
        return pm

    def core(self, a: Rgb, b: Rgb) -> QPixmap:
        """The glassy core sphere with its specular dot, rendered once per quantised colour pair."""
        key = (_quant(a), _quant(b))
        pm = self.sprites.get(key)
        if pm is not None:
            return pm
        n = float(GLOW_QUANT)
        qa = (key[0][0] / n, key[0][1] / n, key[0][2] / n)
        qb = (key[1][0] / n, key[1][1] / n, key[1][2] / n)
        ru = CORE_REF * self.u
        side = max(4, math.ceil(2.0 * ru * self.dpr))
        pm = QPixmap(side, side)
        pm.setDevicePixelRatio(self.dpr)
        pm.fill(CLEAR)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        half = side / self.dpr / 2.0
        painter.translate(half, half)
        painter.setPen(NO_PEN)
        grad = QRadialGradient(QPointF(0, 0), ru, QPointF(-ru * 0.25, -ru * 0.3))
        grad.setColorAt(0.0, qcolor(mix(qa, WHITE, 0.78), 1.0))
        grad.setColorAt(0.5, qcolor(mix(qa, WHITE, 0.35), 0.97))
        grad.setColorAt(0.85, qcolor(qb, 0.85))
        grad.setColorAt(1.0, qcolor(qb, 0.0))
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(QPointF(0, 0), ru, ru)
        spec = QRadialGradient(QPointF(-ru * 0.32, -ru * 0.38), ru * 0.5)
        spec.setColorAt(0.0, QColor.fromRgbF(1, 1, 1, 0.75))
        spec.setColorAt(1.0, QColor.fromRgbF(1, 1, 1, 0.0))
        painter.setBrush(QBrush(spec))
        painter.drawEllipse(QPointF(-ru * 0.32, -ru * 0.38), ru * 0.5, ru * 0.42)
        painter.end()
        if len(self.sprites) >= CACHE_LIMIT:
            self.sprites.pop(next(iter(self.sprites)))
        self.sprites[key] = pm
        self.sprite_builds += 1
        return pm

    def ambient(self, a: Rgb, b: Rgb) -> QPixmap:
        """Halo plus the offset secondary glow, baked into one sprite per quantised colour pair, so
        the two big soft blits of the old design are one (rotating it gives the glow its wander)."""
        key = ("ambient", _quant(a), _quant(b))
        pm = self.sprites.get(key)
        if pm is not None:
            return pm
        halo = self.get("halo", mix(a, b, 0.6))  # blitted 1:1 by the orb: its mask size is its drawn size
        halo2 = self.get("halo2", b)
        side = halo.width() / halo.devicePixelRatio()  # logical size
        pm = QPixmap(max(2, round(side * self.dpr)), max(2, round(side * self.dpr)))
        pm.setDevicePixelRatio(self.dpr)  # full device resolution: the orb blits it 1:1
        pm.fill(CLEAR)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(QRectF(0, 0, side, side), halo, QRectF(halo.rect()))
        r2 = GLOWS["halo2"][0] * GLOWS["halo2"][2] * self.u  # blitted radius of the secondary glow
        c = side / 2.0
        painter.setOpacity(AMBIENT_SECOND)
        painter.drawPixmap(QRectF(c + AMBIENT_OFFSET * self.u - r2, c - r2, 2 * r2, 2 * r2), halo2, QRectF(halo2.rect()))
        # the soft body of the living ring: a static annulus (the ring wobbles by a couple of pixels,
        # far less than the glow is wide), which is much cheaper than a wide stroke every frame
        ring = QRadialGradient(QPointF(c, c), RING_GLOW_R * self.u)
        tint = mix(a, b, 0.35)
        peak = RING_RADIUS / RING_GLOW_R
        for pos, alpha in ((0.0, 0.0), (peak - 0.20, 0.0), (peak - 0.10, 0.35), (peak, 1.0), (peak + 0.07, 0.35),
                           (peak + 0.13, 0.0), (1.0, 0.0)):
            ring.setColorAt(clamp01(pos), qcolor(tint, alpha))
        painter.setOpacity(RING_GLOW)
        painter.setPen(NO_PEN)
        painter.setBrush(QBrush(ring))
        painter.drawEllipse(QPointF(c, c), RING_GLOW_R * self.u, RING_GLOW_R * self.u)
        painter.end()
        if len(self.sprites) >= CACHE_LIMIT:
            self.sprites.pop(next(iter(self.sprites)))
        self.sprites[key] = pm
        self.sprite_builds += 1
        return pm

    def get(self, name: str, rgb: Rgb) -> QPixmap:
        q = _quant(rgb)
        key = (name, q)
        pm = self.tinted.get(key)
        if pm is not None:
            return pm
        mask = self._mask(name)
        pm = QPixmap(mask.size())
        pm.setDevicePixelRatio(mask.devicePixelRatio())
        pm.fill(CLEAR)
        painter = QPainter(pm)
        painter.drawPixmap(0, 0, mask)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        n = float(GLOW_QUANT)
        painter.fillRect(QRectF(0, 0, pm.width() / pm.devicePixelRatio(), pm.height() / pm.devicePixelRatio()),
                         QColor.fromRgbF(q[0] / n, q[1] / n, q[2] / n, 1.0))
        painter.end()
        if len(self.tinted) >= CACHE_LIMIT:
            self.tinted.pop(next(iter(self.tinted)))  # oldest first: colour steps pass quickly
        self.tinted[key] = pm
        self.tint_builds += 1
        return pm


class OrbWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, seed: int = 11) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        rng = random.Random(seed)
        self.t = 0.0
        self.age = 0.0
        self.state = VoiceState.IDLE
        self.weights = {s: 1.0 if s is VoiceState.IDLE else 0.0 for s in VoiceState}
        self.palette = PaletteBlend("idle")
        self.loading = False
        self.confirming = False
        self.muted = False
        self.error_left = 0.0
        self.tool_running = False
        self.tool_w = 0.0
        self.mic = Smoother(0.045, 0.32)
        self.spk = Smoother(0.045, 0.30)
        self.mic_in = 0.0
        self.spk_in = 0.0
        self.drive = 0.0
        self.motion_arc = 1.0
        self.motion_part = 0.6
        self.breath_period = MOTION[VoiceState.IDLE][2]
        self.arc_angles = [rng.uniform(0, 360) for _ in ARCS]
        self.comet = 0.0
        self.burst = 0.0
        self.ripples: list[float] = []
        self._slow_spk = 0.0
        self._peak_cooldown = 0.0
        self.tick_values = [0.0] * TICKS
        self.tick_phase = [rng.uniform(0, TAU) for _ in range(TICKS)]
        self.ring_phase = [rng.uniform(0, 360) for _ in HARMONICS]
        self.motes = [
            _Mote(
                radius=rng.uniform(0.30, 0.93),
                ratio=rng.uniform(0.55, 1.0),
                tilt=rng.uniform(0, math.pi),
                angle=rng.uniform(0, TAU),
                speed=rng.uniform(0.12, 0.5) * rng.choice((-1, 1)),
                depth=rng.uniform(0, TAU),
                drift=rng.uniform(0, TAU),
                drift_rate=rng.uniform(0.2, 0.7),
                size=rng.uniform(0.006, 0.016),
            )
            for _ in range(PARTICLES)
        ]
        for m in self.motes:
            m.ct, m.st = math.cos(m.tilt), math.sin(m.tilt)
        self.idle_for = 0.0
        self.glow = GlowCache()
        self.rest_w = 0.0
        self.zoom = 1.0  # drawn size relative to the widget (the window grows it in expanded mode)
        self.backdrop = None  # callable returning the window's cached glass pixmap, or None
        self._tick_geo: tuple = (0.0, [], [])
        self._arc_key: tuple | None = None
        self._arc_lines: list = []
        self._comet_key: int | None = None
        self._comet_lines: list = []
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    # -- inputs --------------------------------------------------------------------
    def set_state(self, state: VoiceState) -> None:
        self.state = state

    def set_levels(self, mic: float, speaker: float) -> None:
        self.mic_in = clamp01(mic)
        self.spk_in = clamp01(speaker)

    def set_tool_running(self, running: bool) -> None:
        self.tool_running = running

    def set_loading(self, loading: bool) -> None:
        self.loading = loading

    def set_confirming(self, confirming: bool) -> None:
        self.confirming = confirming

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def flash_error(self, seconds: float = 2.4) -> None:
        self.error_left = seconds

    def is_resting(self) -> bool:
        """True after a long quiet stretch; the window may then lower its frame rate."""
        return self.idle_for > 45.0

    # -- time ----------------------------------------------------------------------
    def _palette_key(self) -> str:
        if self.error_left > 0.0:
            return "error"
        if self.confirming:
            return "confirm"
        return PALETTE_KEY[VoiceState.THINKING if self.loading else self.state]

    def advance(self, dt: float) -> None:
        real_dt = dt = min(max(dt, 0.0), 0.25)
        # after a long rest the orb drifts at about half speed: it is drawn at ~8 fps then, and slow
        # motion is what keeps those frames from reading as steps
        self.rest_w = approach(self.rest_w, 1.0 if self.is_resting() else 0.0, real_dt, 1.5)
        dt *= 1.0 - REST_SLOWDOWN * self.rest_w
        self.t += dt
        self.age += dt
        self.error_left = max(0.0, self.error_left - dt)
        active = VoiceState.THINKING if self.loading else self.state
        for s in VoiceState:
            self.weights[s] = approach(self.weights[s], 1.0 if s is active else 0.0, dt, 0.22)
        self.palette.set(self._palette_key())
        self.palette.advance(dt)
        self.tool_w = approach(self.tool_w, 1.0 if self.tool_running else 0.0, dt, 0.25)

        self.mic.update(0.0 if self.muted else self.mic_in, dt)
        self.spk.update(self.spk_in, dt)
        w = self.weights
        pulse = 0.5 + 0.5 * math.sin(self.t * 3.1)
        self.drive = clamp01(
            self.mic.value * w[VoiceState.LISTENING]
            + self.spk.value * w[VoiceState.SPEAKING]
            + (0.06 + 0.05 * pulse) * w[VoiceState.THINKING]
            + 0.03 * w[VoiceState.IDLE]
        )
        self.motion_arc = sum(MOTION[s][0] * w[s] for s in VoiceState)
        self.motion_part = sum(MOTION[s][1] * w[s] for s in VoiceState)
        self.breath_period = sum(MOTION[s][2] * w[s] for s in VoiceState)

        for i, arc in enumerate(ARCS):
            self.arc_angles[i] = (self.arc_angles[i] + arc.speed * self.motion_arc * dt) % 360.0
        self.comet = (self.comet + 210.0 * dt) % 360.0
        self._update_ticks(dt)
        self._update_ripples(dt)
        for m in self.motes:
            m.angle += m.speed * self.motion_part * dt
            m.drift += m.drift_rate * dt
        quiet = self.drive < 0.06 and self.state is VoiceState.IDLE and not self.tool_running
        self.idle_for = self.idle_for + real_dt if quiet else 0.0

    def _update_ticks(self, dt: float) -> None:
        level = clamp01(self.drive**0.6 * 1.35)
        t = self.t
        rise = 1.0 - math.exp(-dt / 0.035)
        fall = 1.0 - math.exp(-dt / 0.22)
        sin = math.sin
        values = self.tick_values
        phase = self.tick_phase
        half = TICKS // 2
        for j in range(half + 1):  # spokes are mirrored left/right: compute one side
            envelope = 0.55 + 0.45 * (1.0 - j / half) ** 0.7
            noise = 0.5 + 0.5 * sin(t * (4.0 + (j % 7) * 0.9) + j * 2.399 + phase[j])
            wave = 0.5 + 0.5 * sin(j * 0.52 - t * 2.4)
            target = level * (0.22 + 0.55 * noise + 0.23 * wave) * envelope
            target += 0.045 + 0.03 * sin(t * 0.9 + j * 0.35)
            cur = values[j]
            values[j] = cur + (target - cur) * (rise if target > cur else fall)
            if 0 < j < half:
                values[TICKS - j] = values[j]

    def _update_ripples(self, dt: float) -> None:
        self._peak_cooldown = max(0.0, self._peak_cooldown - dt)
        self._slow_spk = approach(self._slow_spk, self.spk.value, dt, 0.25)
        speaking = self.weights[VoiceState.SPEAKING] > 0.5
        if speaking and self._peak_cooldown <= 0.0 and self.spk.value - self._slow_spk > 0.09:
            self.ripples.append(0.0)
            self.burst = 1.0
            self._peak_cooldown = 0.28
        self.burst = approach(self.burst, 0.0, dt, 0.35)
        self.ripples = [a + dt for a in self.ripples if a + dt < 1.3][-4:]

    # -- painting ------------------------------------------------------------------
    def paintEvent(self, _: QPaintEvent) -> None:
        painter = QPainter(self)
        if self.backdrop is not None:
            # Opaque paint event: Qt does not repaint the window underneath, so the glass behind the
            # orb is blitted here from the window's cached pixmap (clipped to this widget's rect).
            pm = self.backdrop()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            if pm is None or pm.isNull():
                painter.fillRect(self.rect(), CLEAR)
            else:
                painter.drawPixmap(-self.x(), -self.y(), pm)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.paint_orb(painter, self.width(), self.height())
        painter.end()

    def paint_orb(self, p: QPainter, width: float, height: float) -> None:
        p.save()
        p.translate(width / 2.0, height / 2.0)
        u = min(width, height) / 2.0 * self.zoom
        self.glow.prepare(u, p.device().devicePixelRatioF())
        p.setBrush(NO_BRUSH)
        ent = ease_out_cubic(self.age / ENTRANCE_SECONDS)
        a, b = self.palette.primary, self.palette.secondary
        breath = 0.5 + 0.5 * math.sin(self.t * TAU / max(self.breath_period, 0.5))
        self._paint_halo(p, u, a, b, breath, ent)
        self._paint_ticks(p, u, a, ent)
        self._paint_arcs(p, u, a, b, ent)
        self._paint_ring(p, u, a, b)
        self._paint_motes(p, u, a, b, ent)
        self._paint_core(p, u, a, b, breath, ent)
        p.restore()

    def _blit(self, p: QPainter, name: str, rgb: Rgb, cx: float, cy: float, radius: float, alpha: float) -> None:
        pm = self.glow.get(name, rgb)
        radius *= GLOWS[name][2]
        p.setOpacity(clamp01(alpha))
        p.drawPixmap(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius), pm, QRectF(pm.rect()))
        p.setOpacity(1.0)

    def _paint_halo(self, p: QPainter, u: float, a: Rgb, b: Rgb, breath: float, ent: float) -> None:
        """One unscaled blit of the baked ambient glow. Its brightness (not its size) follows the
        breath and the level: an unscaled blit costs half a scaled one, and this is the largest
        thing the orb draws."""
        d = self.drive
        alpha = (0.30 + 0.12 * breath + 0.42 * d) * ent
        pm = self.glow.ambient(a, b)
        half = pm.width() / pm.devicePixelRatio() / 2.0
        p.setOpacity(clamp01(alpha))
        p.drawPixmap(QPointF(-half, -half), pm)
        p.setOpacity(1.0)

    def _paint_ticks(self, p: QPainter, u: float, a: Rgb, ent: float) -> None:
        """72 spokes in three brightness bands (taller is brighter): one drawLines call per band."""
        geo = self._tick_geo
        if geo[0] != u:
            inner = 0.535 * u
            geo = self._tick_geo = (u, [c * inner for c in TICK_COS], [s * inner for s in TICK_SIN])
        _, ix, iy = geo
        base_len = 0.018 * u
        reach = 0.135 * u * ent
        b0: list[QLineF] = []
        b1: list[QLineF] = []
        b2: list[QLineF] = []
        line = QLineF
        for c, s, x0, y0, v in zip(TICK_COS, TICK_SIN, ix, iy, self.tick_values):
            length = base_len + reach * v
            seg = line(x0, y0, x0 + c * length, y0 + s * length)
            (b2 if v > 0.30 else b1 if v > 0.12 else b0).append(seg)
        mute = 0.45 if self.muted else 1.0
        hot = mix(a, WHITE, 0.35)
        w = 0.008 * u
        for lines, col, alpha in ((b0, a, 0.42), (b1, mix(a, hot, 0.5), 0.66), (b2, hot, 0.92)):
            if lines:
                p.setPen(QPen(QBrush(qcolor(col, alpha * ent * mute)), w, SOLID, FLAT))
                p.drawLines(lines)

    def _arc_geometry(self, u: float, ent: float) -> list[list[list[QLineF]]]:
        """Per arc, three alpha bands of short chords, built for base angle 0 and cached.

        Rotating the painter then moves the arc, so a frame costs no trigonometry and no gradient."""
        key = (round(u * 2), round(ent * 24))
        if self._arc_key == key:
            return self._arc_lines
        out: list[list[list[QLineF]]] = []
        for arc in ARCS:
            r = arc.radius * u
            bands: list[list[QLineF]] = [[], [], []]
            for off, span in arc.spans:
                sweep = span * ent
                if sweep < 1.0:
                    continue
                n = max(3, int(sweep / 5.0) + 1)
                step = sweep / n
                px, py = r * math.cos(math.radians(off)), -r * math.sin(math.radians(off))
                for k in range(1, n + 1):
                    ang = math.radians(off + step * k)
                    x, y = r * math.cos(ang), -r * math.sin(ang)
                    pos = (k - 0.5) / n
                    if pos < 0.35:
                        level = pos / 0.35 * 0.7
                    elif pos < 0.75:
                        level = 0.7 + 0.3 * (pos - 0.35) / 0.4
                    else:
                        level = (1.0 - pos) / 0.25
                    if level > 0.08:
                        bands[min(2, int(clamp01(level) * 3.0))].append(QLineF(px, py, x, y))
                    px, py = x, y
            out.append(bands)
        self._arc_key, self._arc_lines = key, out
        return out

    def _paint_arcs(self, p: QPainter, u: float, a: Rgb, b: Rgb, ent: float) -> None:
        tight = 1.0 - 0.095 * self.weights[VoiceState.THINKING]
        mid = mix(a, b, 0.25)
        geo = self._arc_geometry(u, ent)
        for arc, base, bands in zip(ARCS, self.arc_angles, geo):
            p.save()
            p.rotate(-base)
            if tight != 1.0:
                p.scale(tight, tight)
            pen_w = arc.width * u
            for i, lines in enumerate(bands):
                if lines:
                    col = qcolor(mix(b, mid, i / 2.0), arc.alpha * ent * (0.42 + 0.29 * i))
                    p.setPen(QPen(QBrush(col), pen_w, SOLID, FLAT))
                    p.drawLines(lines)
            p.restore()
        if self.tool_w > 0.02:
            self._paint_comet(p, u, a, tight)

    def _comet_geometry(self, u: float) -> list[list[QLineF]]:
        if self._comet_key == round(u * 2):
            return self._comet_lines
        r = 0.78 * u
        tail, n = 95.0, 19
        bands: list[list[QLineF]] = [[], [], [], []]
        px, py = r * math.cos(math.radians(-tail)), -r * math.sin(math.radians(-tail))
        for k in range(1, n + 1):
            ang = math.radians(-tail + tail * k / n)
            x, y = r * math.cos(ang), -r * math.sin(ang)
            pos = (k - 0.5) / n
            bands[min(3, int(pos**1.6 * 4.0))].append(QLineF(px, py, x, y))
            px, py = x, y
        self._comet_key, self._comet_lines = round(u * 2), bands
        return bands

    def _paint_comet(self, p: QPainter, u: float, a: Rgb, tight: float) -> None:
        w = self.tool_w
        hot = mix(a, WHITE, 0.7)
        bands = self._comet_geometry(u)
        p.save()
        p.rotate(-self.comet)
        p.scale(tight, tight)
        for i, lines in enumerate(bands):
            p.setPen(QPen(QBrush(qcolor(mix(a, hot, i / 3.0), (0.22 + 0.24 * i) * w)), 0.016 * u, SOLID, FLAT))
            p.drawLines(lines)
        p.restore()
        r = 0.78 * tight * u
        hx, hy = r * math.cos(math.radians(self.comet)), -r * math.sin(math.radians(self.comet))
        self._blit(p, "head", hot, hx, hy, 0.075 * u, 0.9 * w)

    def _ring_lines(self, segments: int, r0: float) -> list[QLineF]:
        d = self.drive
        think = self.weights[VoiceState.THINKING]
        speed_mul = 1.0 + 1.2 * think
        t = self.t
        amps = []
        for (_, deg, idle, gain), ph, hs, hc in zip(HARMONICS, self.ring_phase, HARM_SIN, HARM_COS):
            phase = (ph + deg * t) * math.pi / 180.0 * speed_mul
            amps.append((idle * (1.0 + 0.6 * think) + gain * d, math.cos(phase), math.sin(phase), hs, hc))
        pts: list[tuple[float, float]] = []
        for i in range(segments + 1):
            wob = 0.0
            for amp, cp, sp, hs, hc in amps:
                wob += amp * (hs[i] * cp + hc[i] * sp)  # sin(k * angle + phase)
            r = r0 * (1.0 + wob)
            pts.append((r * RING_COS[i], r * RING_SIN[i]))
        return [QLineF(x0, y0, x1, y1) for (x0, y0), (x1, y1) in zip(pts, pts[1:])]

    def _paint_ring(self, p: QPainter, u: float, a: Rgb, b: Rgb) -> None:
        frac = ease_out_cubic(self.age / ENTRANCE_SECONDS)
        segments = max(2, int(RING_POINTS * frac))
        lines = self._ring_lines(segments, (RING_RADIUS + 0.012 * self.drive) * u)
        spin = (self.t * 24.0) % 360.0
        light = mix(a, WHITE, 0.45)
        # the soft glow body is baked into the ambient sprite; only the thin bright line is stroked
        alpha = 0.95
        stops = [
            (0.0, qcolor(a, alpha)),
            (0.35, qcolor(b, alpha * 0.75)),
            (0.7, qcolor(light, alpha)),
            (1.0, qcolor(a, alpha)),
        ]
        p.setPen(QPen(_conical(spin, stops), 0.011 * u, SOLID, FLAT, BEVEL))
        p.drawLines(lines)

    def _paint_motes(self, p: QPainter, u: float, a: Rgb, b: Rgb, ent: float) -> None:
        """One opacity per depth band (cheaper than a colour per mote)."""
        pull = self.weights[VoiceState.THINKING]
        scale = 1.0 - 0.42 * pull + 0.30 * self.burst
        grow = 1.0 + 0.5 * pull
        bands: list[list[tuple[float, float, float]]] = [[], [], []]
        sin, cos = math.sin, math.cos
        for m in self.motes:
            r = m.radius * scale * (1.0 + 0.05 * sin(m.drift))
            x, y = r * cos(m.angle), r * m.ratio * sin(m.angle)
            ct, st = m.ct, m.st
            z = 0.5 + 0.5 * sin(m.angle + m.depth)
            size = m.size * (0.55 + 0.9 * z) * grow
            bands[min(2, int(z * 3))].append(((x * ct - y * st) * u, (x * st + y * ct) * u, size * u * 1.12))
        pm = self.glow.get("dot", mix(a, b, 0.4))
        src = QRectF(pm.rect())
        rect = QRectF
        for i, motes in enumerate(bands):
            p.setOpacity(clamp01((0.22 + 0.6 * (i + 0.5) / 3) * ent))
            for x, y, s in motes:
                p.drawPixmap(rect(x - s, y - s, 2 * s, 2 * s), pm, src)
        p.setOpacity(1.0)

    def _paint_core(self, p: QPainter, u: float, a: Rgb, b: Rgb, breath: float, ent: float) -> None:
        d = self.drive
        r = (0.145 + 0.012 * breath + 0.05 * d) * (0.4 + 0.6 * ent)
        for age in self.ripples:
            k = age / 1.3
            rr = (r + k * (0.82 - r)) * u
            p.setPen(QPen(QBrush(qcolor(a, 0.5 * (1.0 - k) ** 2)), (0.010 + 0.01 * (1 - k)) * u, SOLID, FLAT))
            p.drawLines([QLineF(x0 * rr, y0 * rr, x1 * rr, y1 * rr)
                         for (x0, y0), (x1, y1) in zip(UNIT_CIRCLE, UNIT_CIRCLE[1:])])
        self._blit(p, "shell", mix(a, b, 0.55), 0.0, 0.0, r * 3.2 * u, 0.55 + 0.25 * d)
        ru = r * u
        pm = self.glow.core(a, b)
        p.drawPixmap(QRectF(-ru, -ru, 2 * ru, 2 * ru), pm, QRectF(pm.rect()))
