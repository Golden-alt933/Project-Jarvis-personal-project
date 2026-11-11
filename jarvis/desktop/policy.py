"""What the desktop-control tools may do without asking.

Screen content is data, never instructions: a web page or document on screen can contain
text written to steer an agent. So every action is judged from what the OS reports about
the target (element name and kind, window, key chord), never from the text on screen
alone, and the consequential ones go to the user.

Verdicts reuse ``jarvis.browser.policy`` (``safe`` / ``ask`` / ``refuse``):

- clicking or pressing a control whose name says pay, buy, send, post, delete, sign in,
  accept, or an account or security change: ask (the browser classifier decides);
- typing into a password, credential or payment field: refuse;
- typing into, or pressing Enter in, a terminal, console, the Run dialog, or an elevated
  window: ask, with the text shown (that is running a command);
- elevated windows cannot be driven from a normal process, Jarvis's own windows and the
  Windows security desktop are never touched: refuse;
- key chords that close, delete or switch context irreversibly: ask; Win+L goes to the lock
  tool; Ctrl+Alt+Delete cannot be sent.

Everything here is pure (strings and small objects in, a Verdict out) so it is table-tested.
"""

from __future__ import annotations

import os
import re
from typing import Iterable

from jarvis.browser.policy import (
    SAFE,
    Verdict,
    classify_click,
    classify_submit,
    classify_typing,
    credential_reason,
)
from jarvis.platform.base import UiElement, WindowInfo

# ---------------------------------------------------------------- windows
TERMINAL_PROCESSES = frozenset({
    "cmd.exe", "powershell.exe", "pwsh.exe", "windowsterminal.exe", "wt.exe", "conhost.exe",
    "openconsole.exe", "bash.exe", "wsl.exe", "wslhost.exe", "mintty.exe", "alacritty.exe",
    "wezterm-gui.exe", "kitty.exe", "conemu.exe", "conemu64.exe", "cmder.exe", "putty.exe",
    "gnome-terminal", "gnome-terminal-server", "gnome-terminal-", "konsole", "xterm", "uxterm",
    "alacritty", "kitty", "wezterm-gui", "terminator", "tilix", "xfce4-terminal", "lxterminal",
    "urxvt", "st", "foot", "ptyxis", "kgx", "guake", "yakuake",
})
TERMINAL_CLASSES = frozenset({
    "consolewindowclass", "cascadia_hosting_window_class", "pseudoconsolewindow", "mintty",
    "putty", "xterm", "urxvt", "gnome-terminal-server", "konsole", "alacritty", "kitty",
})
# Windows surfaces that handle credentials or consent for the whole machine.
SECURITY_PROCESSES = frozenset({"consent.exe", "logonui.exe", "winlogon.exe", "lockapp.exe", "credentialuibroker.exe"})
SECURITY_TITLES = re.compile(r"^(user account control|windows security|windows hello|sign in to)\b", re.IGNORECASE)

WindowKind = str  # "terminal" | "run dialog" | "security" | "normal"


def own_process_ids(extra: Iterable[int] = ()) -> set[int]:
    """This process and the terminal that hosts it. Jarvis may be running in the very console
    the user is looking at; typing there would feed Jarvis's own prompt."""
    own = {os.getpid(), *extra}
    try:
        import psutil

        process = psutil.Process()
        for parent in process.parents():
            if parent.name().lower() in TERMINAL_PROCESSES:
                own.add(parent.pid)
    except Exception:  # best effort
        pass
    return own


def window_kind(window: WindowInfo) -> WindowKind:
    process = window.process.lower()
    klass = window.class_name.lower()
    title = window.title.strip().lower()
    if process in SECURITY_PROCESSES or SECURITY_TITLES.match(window.title.strip()):
        return "security"
    if process in TERMINAL_PROCESSES or klass in TERMINAL_CLASSES:
        return "terminal"
    if title == "run" and klass == "#32770" and process in ("explorer.exe", "?"):
        return "run dialog"
    return "normal"


def window_block(
    window: WindowInfo | None,
    *,
    own_pids: set[int],
    self_elevated: bool = False,
    input_desktop_ok: bool = True,
) -> Verdict | None:
    """A refusal when this window must never be driven, else None."""
    if not input_desktop_ok:
        return Verdict(
            "refuse",
            "A secure prompt (a UAC consent window or the lock screen) has the screen. "
            "Windows does not let any program reach it; the user must answer it themselves.",
        )
    if window is None:
        return Verdict("refuse", "There is no foreground window to act on (the desktop may be locked or empty).")
    if window.pid in own_pids:
        return Verdict(
            "refuse",
            "That is Jarvis's own window (or the terminal Jarvis runs in). Bring the app you mean to the front first.",
        )
    if window_kind(window) == "security":
        return Verdict(
            "refuse",
            f"'{window.title}' is a Windows security or sign-in surface; the user must handle it themselves.",
        )
    if window.elevated and not self_elevated:
        return Verdict(
            "refuse",
            f"'{window.title}' ({window.process}) runs as administrator. Jarvis runs with normal "
            "rights, and Windows blocks a normal program from controlling an elevated one. "
            "The user must act in that window themselves.",
        )
    return None


def _command_surface(window: WindowInfo, self_elevated: bool) -> str | None:
    """Why input here counts as running commands, or None."""
    kind = window_kind(window)
    if kind == "terminal":
        return "a terminal or console"
    if kind == "run dialog":
        return "the Windows Run dialog"
    if window.elevated and self_elevated:
        return "an administrator window"
    return None


# ---------------------------------------------------------------- elements
def element_dict(element: UiElement) -> dict[str, object]:
    """The element in the shape ``jarvis.browser.policy`` classifies."""
    role = element.role
    shaped: dict[str, object] = {
        "label": element.name,
        "id": element.automation_id,
        "title": "",
    }
    if element.is_password:
        shaped["type"] = "password"
    if role in ("button", "split button"):
        shaped["tag"] = "button"
        shaped["type"] = shaped.get("type") or "button"
    elif role == "checkbox":
        shaped["type"] = "checkbox"
        shaped["role"] = "checkbox"
    elif role == "radio button":
        shaped["type"] = "radio"
    elif role == "link":
        shaped["tag"] = "a"
        shaped["role"] = "link"
    elif role in ("edit", "combobox", "spinner", "document"):
        shaped["tag"] = "input"
    return shaped


def classify_element_click(element: UiElement) -> Verdict:
    return classify_click(element_dict(element))


def classify_element_typing(element: UiElement | None, text: str) -> Verdict:
    """Refuse credential and payment fields (and card-number-looking text anywhere)."""
    if element is None:
        return classify_typing({}, text)
    return classify_typing(element_dict(element), text)


_MESSAGEY = re.compile(r"\b(message|chat|reply|comment|post|compose|tweet|caption|prompt|say something)\b", re.IGNORECASE)


def classify_text_entry(
    window: WindowInfo,
    text: str,
    *,
    element: UiElement | None = None,
    submit: bool = False,
    self_elevated: bool = False,
) -> Verdict:
    """Typing text (optionally followed by Enter) into a window."""
    surface = _command_surface(window, self_elevated)
    enters = submit or "\n" in text or "\r" in text
    if surface:
        shown = text if len(text) <= 200 else text[:200] + "..."
        tail = " and press Enter" if enters else ""
        return Verdict("ask", f"typing into {surface} runs commands: type {shown!r}{tail}")
    if submit and element is not None and _MESSAGEY.search(element.name or ""):
        return classify_submit({"label": element.name, "tag": "input"})
    return SAFE


# ---------------------------------------------------------------- key chords
_ASK_CHORDS: dict[frozenset[str], str] = {
    frozenset({"alt", "f4"}): "closing the current window or app",
    frozenset({"ctrl", "w"}): "closing the current tab or document",
    frozenset({"ctrl", "shift", "w"}): "closing the whole window",
    frozenset({"ctrl", "f4"}): "closing the current document or tab",
    frozenset({"ctrl", "q"}): "quitting the app",
    frozenset({"delete"}): "deleting the selection or the selected file",
    frozenset({"shift", "delete"}): "permanently deleting the selection or file",
    frozenset({"ctrl", "shift", "delete"}): "clearing browsing data or deleting for good",
    frozenset({"win", "r"}): "opening the Run dialog, which launches programs",
}
_ENTER_KEYS = frozenset({"enter"})
_ACTIVATE_KEYS = frozenset({"enter", "space"})


def classify_chord(
    keys: list[str],
    window: WindowInfo | None,
    *,
    focused: UiElement | None = None,
    self_elevated: bool = False,
) -> Verdict:
    """Decide whether this chord (canonical names from ``parse_chord``) needs approval.
    ``window`` is the foreground window, ``focused`` the element holding keyboard focus."""
    combo = frozenset(keys)
    if combo == frozenset({"win", "l"}):
        return Verdict("ask", "locking the screen (the user must sign in again)")
    if combo == frozenset({"ctrl", "alt", "delete"}):
        return Verdict("refuse", "Ctrl+Alt+Delete is a secure attention sequence; no program can send it.")
    reason = _ASK_CHORDS.get(combo)
    if reason:
        return Verdict("ask", reason)
    main = [k for k in keys if k not in ("ctrl", "alt", "shift", "win")]
    if main and main[0] in _ENTER_KEYS and window is not None:
        surface = _command_surface(window, self_elevated)
        if surface:
            return Verdict("ask", f"pressing Enter in {surface} runs what is typed there")
    if main and main[0] in _ACTIVATE_KEYS and len(keys) == 1 and focused is not None:
        verdict = classify_element_click(focused)
        if not verdict.safe:
            return Verdict("ask", f"pressing {main[0].title()} activates the focused control '{focused.name}' ({verdict.reason})")
    return SAFE


def is_win_l(keys: list[str]) -> bool:
    return frozenset(keys) == frozenset({"win", "l"})


def credential_field(element: UiElement) -> str | None:
    """Why this element is a credential or payment field, or None."""
    return credential_reason(element_dict(element))
