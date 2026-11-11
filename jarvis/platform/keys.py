"""Key chords: one strict parser shared by every backend.

``parse_chord("Ctrl+Shift+T")`` returns canonical key names (``["ctrl", "shift", "t"]``),
modifiers first. Backends map those names to virtual-key codes or X keysyms. Anything not
in the tables is rejected with the list of valid names, so a model that invents a key
name gets a useful error instead of a silently wrong keystroke.
"""

from __future__ import annotations

import string

MODIFIERS: tuple[str, ...] = ("ctrl", "alt", "shift", "win")

_ALIASES = {
    "control": "ctrl", "ctl": "ctrl", "option": "alt", "windows": "win", "super": "win",
    "meta": "win", "cmd": "win", "command": "win",
    "return": "enter", "esc": "escape", "del": "delete", "ins": "insert",
    "pgup": "pageup", "page_up": "pageup", "pgdn": "pagedown", "pgdown": "pagedown",
    "page_down": "pagedown", "spacebar": "space", "bksp": "backspace", "caps": "capslock",
    "prtsc": "printscreen", "printscr": "printscreen", "arrowup": "up", "arrowdown": "down",
    "arrowleft": "left", "arrowright": "right",
    # punctuation typed as itself
    "-": "minus", "=": "equals", ",": "comma", ".": "period", "/": "slash",
    ";": "semicolon", "'": "quote", "`": "backquote", "[": "bracketleft", "]": "bracketright",
    "\\": "backslash",
}

NAMED_KEYS: tuple[str, ...] = (
    "enter", "tab", "space", "escape", "backspace", "delete", "insert", "home", "end",
    "pageup", "pagedown", "up", "down", "left", "right", "capslock", "printscreen",
    "plus", "minus", "equals", "comma", "period", "slash", "semicolon", "quote",
    "backquote", "bracketleft", "bracketright", "backslash",
)
FUNCTION_KEYS: tuple[str, ...] = tuple(f"f{n}" for n in range(1, 25))
CHAR_KEYS: tuple[str, ...] = tuple(string.ascii_lowercase + string.digits)
MAIN_KEYS = frozenset(NAMED_KEYS + FUNCTION_KEYS + CHAR_KEYS)


class ChordError(ValueError):
    """The text is not a valid key chord; the message lists what is valid."""


def valid_key_names() -> str:
    return (
        "modifiers: " + ", ".join(MODIFIERS)
        + "; keys: a-z, 0-9, f1-f24, " + ", ".join(NAMED_KEYS)
    )


def _canonical(token: str) -> str:
    key = token.strip().lower()
    return _ALIASES.get(key, key)


def parse_chord(text: str) -> list[str]:
    """Parse "ctrl+s", "alt+tab", "enter", "win+d" into canonical names, modifiers first.

    Exactly one non-modifier key is allowed (or a lone modifier such as "win"); modifiers
    may not repeat. Raises ChordError otherwise."""
    raw = (text or "").strip()
    if not raw:
        raise ChordError(f"no keys given. Valid names: {valid_key_names()}")
    if raw == "+":
        parts = ["plus"]
    elif raw.endswith("++"):  # "ctrl++" means ctrl and the plus key
        head = raw[:-2]
        parts = (head.split("+") if head else []) + ["plus"]
    else:
        parts = raw.split("+")
    tokens = [_canonical(p) for p in parts]
    if any(not t for t in tokens):
        raise ChordError(f"'{text}' has an empty key name (use 'plus' for the + key)")
    mods: list[str] = []
    mains: list[str] = []
    for token in tokens:
        if token in MODIFIERS:
            if token in mods:
                raise ChordError(f"'{token}' appears twice in '{text}'")
            mods.append(token)
        elif token in MAIN_KEYS:
            mains.append(token)
        else:
            raise ChordError(f"unknown key '{token}'. Valid names: {valid_key_names()}")
    if len(mains) > 1:
        raise ChordError(
            f"'{text}' has more than one non-modifier key ({', '.join(mains)}); "
            "a chord is modifiers plus one key. Send separate press_keys calls for a sequence"
        )
    return [m for m in MODIFIERS if m in mods] + mains
