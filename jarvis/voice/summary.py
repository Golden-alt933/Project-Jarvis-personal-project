"""Compact, display-safe one-line views of tool arguments (no heavy imports)."""

from __future__ import annotations

import re

SUMMARY_MAX = 80
VALUE_MAX = 30
_SECRET_NAME = re.compile(r"password|passwd|token|secret|key", re.IGNORECASE)


def summarize_args(arguments: dict | None) -> str:
    """One short line like `path=a.txt, count=3`; secret-looking values are never shown."""
    parts: list[str] = []
    for name, value in (arguments or {}).items():
        if _SECRET_NAME.search(str(name)):
            shown = "***"
        else:
            shown = " ".join(str(value).split())
            if len(shown) > VALUE_MAX:
                shown = shown[: VALUE_MAX - 1] + "…"
        parts.append(f"{' '.join(str(name).split())}={shown}")
    line = ", ".join(parts)
    return line if len(line) <= SUMMARY_MAX else line[: SUMMARY_MAX - 1] + "…"
