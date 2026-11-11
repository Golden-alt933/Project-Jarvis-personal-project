"""Colours, fonts, easing and the smoothing helpers every animated widget shares."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from PyQt6.QtGui import QColor, QFont, QFontDatabase

Rgb = tuple[float, float, float]


def hex_rgb(value: str) -> Rgb:
    value = value.lstrip("#")
    return (int(value[0:2], 16) / 255.0, int(value[2:4], 16) / 255.0, int(value[4:6], 16) / 255.0)


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix(a: Rgb, b: Rgb, t: float) -> Rgb:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))


def qcolor(rgb: Rgb, alpha: float = 1.0) -> QColor:
    return QColor.fromRgbF(clamp01(rgb[0]), clamp01(rgb[1]), clamp01(rgb[2]), clamp01(alpha))


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1.0 - (1.0 - t) ** 3


def smoothstep(t: float) -> float:
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def approach(current: float, target: float, dt: float, tau: float) -> float:
    """Exponential approach: frame-rate independent easing toward a target."""
    if tau <= 0.0:
        return target
    return current + (target - current) * (1.0 - math.exp(-dt / tau))


class Smoother:
    """Fast attack, slow release: a level that rises with the signal and falls gently."""

    def __init__(self, attack: float = 0.05, release: float = 0.35, value: float = 0.0) -> None:
        self.attack, self.release, self.value = attack, release, value

    def update(self, target: float, dt: float) -> float:
        tau = self.attack if target > self.value else self.release
        self.value = approach(self.value, target, dt, tau)
        return self.value


# -- palettes -----------------------------------------------------------------------
@dataclass(frozen=True)
class Palette:
    primary: Rgb
    secondary: Rgb


PALETTES: dict[str, Palette] = {
    "idle": Palette(hex_rgb("#5EE6FF"), hex_rgb("#3B82F6")),
    "listening": Palette(hex_rgb("#D4F8FF"), hex_rgb("#5EE6FF")),
    "thinking": Palette(hex_rgb("#8B7CFF"), hex_rgb("#4FD8F0")),
    "speaking": Palette(hex_rgb("#2EE0F8"), hex_rgb("#10B8DC")),
    "error": Palette(hex_rgb("#FF6B6B"), hex_rgb("#FF9F8A")),
    "confirm": Palette(hex_rgb("#FFB454"), hex_rgb("#FFD79A")),
}

GLASS_BASE = hex_rgb("#070B14")
TEXT = (0.90, 0.95, 1.0)


class PaletteBlend:
    """The current accent, easing toward the palette of the active state (~450 ms)."""

    TAU = 0.15

    def __init__(self, key: str = "idle") -> None:
        self.key = key
        pal = PALETTES[key]
        self.primary: Rgb = pal.primary
        self.secondary: Rgb = pal.secondary

    def set(self, key: str) -> None:
        self.key = key

    def advance(self, dt: float) -> None:
        target = PALETTES[self.key]
        k = 1.0 - math.exp(-dt / self.TAU)
        self.primary = mix(self.primary, target.primary, k)
        self.secondary = mix(self.secondary, target.secondary, k)

    def snap(self) -> None:
        pal = PALETTES[self.key]
        self.primary, self.secondary = pal.primary, pal.secondary


# -- fonts --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _family() -> str:
    families = set(QFontDatabase.families())
    for name in ("Segoe UI Variable Display", "Segoe UI Variable", "Segoe UI"):
        if name in families:
            return name
    return QFont().family()


@lru_cache(maxsize=1)
def _mono_family() -> str:
    families = set(QFontDatabase.families())
    for name in ("Cascadia Code", "Cascadia Mono", "Consolas"):
        if name in families:
            return name
    return "monospace"


def ui_font(pt: float, weight: QFont.Weight = QFont.Weight.Light, tracking: float = 0.0) -> QFont:
    font = QFont(_family())
    font.setPointSizeF(pt)
    font.setWeight(weight)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    if tracking:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, tracking)
    return font


def label_font(pt: float = 9.0) -> QFont:
    """Small letter-spaced capitals used for every label."""
    return ui_font(pt, QFont.Weight.DemiBold, tracking=3.0)


def mono_font(pt: float = 9.0) -> QFont:
    font = QFont(_mono_family())
    font.setPointSizeF(pt)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font
