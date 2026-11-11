"""Desktop control tools: inspect, click, type, press keys, scroll, focus windows.

Order of preference, most reliable first:
1. the accessibility tree (Windows UI Automation): ``desktop_inspect`` lists the elements of
   a window with stable ref numbers; actions use the element's own patterns (Invoke, Toggle,
   Select, ...) and only fall back to a real click at its centre;
2. OCR text (``desktop_click(text=...)``) for apps that expose no usable elements: games,
   canvases, remote desktops. This is also the only route on Linux, where listing elements
   is not implemented;
3. raw coordinates are never offered to the model.

Safety is in ``jarvis.desktop.policy`` and applied here before any input is sent: purchases,
sends, deletions, sign-ins and the like ask the user; credential fields, elevated windows,
Jarvis's own windows and the security desktop are refused; terminals and the Run dialog ask
with the text shown. Screen content is data, never instructions.

Every action reports what changed (foreground window before and after, the element's new
value or state) so the model can check the effect instead of assuming it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from jarvis import platform as plat
from jarvis.desktop import policy
from jarvis.desktop.ocr_target import (
    describe_candidates,
    grab_for_ocr,
    match_text,
    region_name,
)
from jarvis.platform.base import ElementGone, PlatformUnsupported, UiElement, WindowInfo
from jarvis.platform.keys import ChordError, parse_chord
from jarvis.tools.base import ToolContext, ToolResult, tool
from jarvis.tools.builtin import system as sysmod
from jarvis.vision.capture import CaptureError
from jarvis.vision.ocr import ocr_image

log = logging.getLogger(__name__)

GROUP = "desktop"
CONFIRM_TIMEOUT = 180
SETTLE = 0.25  # seconds an app gets to react before the result is read back
MAX_TEXT = 5000
MAX_REPEAT = 50


class _State:
    """The window whose refs the model is using: the one most recently inspected."""

    target: int | None = None


_state = _State()


# ---------------------------------------------------------------- shared steps
def _backend() -> plat.Platform:
    return plat.current()


def _unavailable() -> ToolResult | None:
    problem = _backend().desktop_problem()
    return ToolResult.fail(problem) if problem else None


def _block(info: WindowInfo | None, *, allow_own: bool = False) -> ToolResult | None:
    """Refuse windows that must never be driven (own, elevated, security, secure desktop)."""
    backend = _backend()
    own = policy.own_process_ids()
    if allow_own and info is not None and info.pid in own:
        own = set()
    verdict = policy.window_block(
        info, own_pids=own, self_elevated=backend.self_elevated(),
        input_desktop_ok=backend.input_desktop_available(),
    )
    return ToolResult.fail(verdict.reason) if verdict is not None else None


async def _approve(ctx: ToolContext, verdict: policy.Verdict, what: str) -> ToolResult | None:
    """None when the action may go ahead; otherwise the failure to return."""
    if verdict.kind == "safe":
        return None
    if verdict.kind == "refuse":
        return ToolResult.fail(verdict.reason)
    try:
        approved = await ctx.confirm(f"{what} - {verdict.reason}")
    except Exception:  # a broken frontend must not approve by accident
        log.exception("desktop confirm failed")
        approved = False
    if not approved:
        return ToolResult.fail(
            "the user declined this action",
            "Do not retry it. Tell the user it was not done, or ask what they want instead.",
        )
    return None


def _fg_title() -> str:
    try:
        return _backend().foreground_window_title()
    except Exception:
        return ""


def _title_change(before: str, after: str) -> str:
    if before == after:
        return f"Foreground window unchanged: '{after or '(none)'}'."
    return f"Foreground window changed: '{before or '(none)'}' -> '{after or '(none)'}'."


def _element_text(element: UiElement) -> str:
    bits = [f"{element.role} '{element.name}'" if element.name else element.role]
    if element.value:
        bits.append(f"value '{element.value}'")
    if element.state:
        bits.append(f"state {element.state}")
    if not element.enabled:
        bits.append("disabled")
    return ", ".join(bits)


def _element_change(before: UiElement, after: UiElement | None) -> str:
    if after is None:
        return f"Element [{before.ref}] is gone now (the click probably closed or replaced it)."
    changes = []
    if after.value != before.value:
        changes.append(f"value '{before.value}' -> '{after.value}'")
    if after.state != before.state:
        changes.append(f"state '{before.state or '-'}' -> '{after.state or '-'}'")
    if after.name != before.name:
        changes.append(f"name '{before.name}' -> '{after.name}'")
    if after.enabled != before.enabled:
        changes.append("now enabled" if after.enabled else "now disabled")
    if changes:
        return f"Element [{before.ref}] changed: {'; '.join(changes)}."
    return f"Element [{before.ref}] shows no change ({_element_text(after)})."


async def _reread(handle: int, ref: int) -> UiElement | None:
    try:
        return await asyncio.to_thread(_backend().ui_read, handle, ref)
    except ElementGone:
        return None
    except (OSError, TimeoutError) as exc:
        log.debug("could not re-read element %s: %s", ref, exc)
        return None


async def _ensure_foreground(info: WindowInfo) -> ToolResult | None:
    """Real input goes to the foreground window, so bring the target there first."""
    backend = _backend()
    current = await asyncio.to_thread(backend.active_window)
    if current is not None and current.handle == info.handle:
        return None
    try:
        ok = await asyncio.to_thread(backend.focus_window, info.handle)
    except OSError as exc:
        return ToolResult.fail(f"could not bring '{info.title}' to the front: {exc}")
    if not ok:
        return ToolResult.fail(
            f"could not bring '{info.title}' to the front (Windows may be blocking it)",
            "Use focus_window to try again, or ask the user to click it once.",
        )
    return None


def _target_info() -> tuple[WindowInfo | None, ToolResult | None]:
    """The window the refs belong to, or why there is none."""
    handle = _state.target
    if handle is None:
        return None, ToolResult.fail(
            "no window has been inspected yet", "Call desktop_inspect first to get element refs."
        )
    info = _backend().window_info(handle)
    if info is None:
        return None, ToolResult.fail(
            "the inspected window no longer exists", "Call desktop_inspect again for the current window."
        )
    return info, None


def _find_window(name: str) -> tuple[WindowInfo | None, ToolResult | None]:
    """The window an app name refers to (same matching as close_app), top-most first."""
    try:
        rows = sysmod._windows()
    except OSError as exc:
        return None, ToolResult.fail(f"cannot list windows: {exc}")
    matches = sysmod._match_windows(name, rows)
    if not matches:
        open_now = sorted({f"{t} ({p})" for p, t, _, _ in rows})[:12]
        return None, ToolResult.fail(
            f"no open window belongs to an app called '{name}'",
            "Open windows: " + ("; ".join(open_now) or "none"),
        )
    info = _backend().window_info(matches[0][3])
    if info is None:
        return None, ToolResult.fail(f"the window '{matches[0][1]}' is gone")
    return info, None


# ---------------------------------------------------------------- inspect
def _matches(element: UiElement, wanted: str) -> bool:
    haystack = " ".join((element.name, element.value, element.automation_id, element.role)).casefold()
    words = wanted.casefold().split()
    return all(word in haystack for word in words)


def _line(element: UiElement) -> str:
    parts = [f"[{element.ref}] {element.role}"]
    if element.name:
        parts.append(f'"{element.name}"')
    if element.value:
        parts.append(f'value="{element.value}"')
    if element.state:
        parts.append(f"({element.state})")
    if not element.enabled:
        parts.append("(disabled)")
    return " ".join(parts)


@tool(group=GROUP, timeout=60)
async def desktop_inspect(window: str | None = None, filter: str | None = None, max_items: int = 60) -> ToolResult:
    """List the controls of a desktop application window so they can be clicked or typed
    into by number: buttons, fields, checkboxes, menu items, tabs, links, list items, and
    on-screen readouts (such as a calculator display). Each line is "[ref] role name value
    state". The refs stay the same for the same control until it disappears. Defaults to the
    foreground window; name an app to inspect another. Use this before desktop_click or
    desktop_type, and again after the screen changes. Windows only; on Linux use
    read_screen_text and desktop_click with text.

    Args:
        window: App or window name such as "calculator" or "notepad" (default: the foreground window).
        filter: Only elements whose name, value or id contains these words, e.g. "save" or "display".
        max_items: How many elements to list (default 60, at most 200).
    """
    if (bad := _unavailable()) is not None:
        return bad
    backend = _backend()
    if not backend.can_inspect_elements():
        return ToolResult.fail(
            "listing the controls of a window is not implemented on this system",
            "Use read_screen_text to see the text on screen and desktop_click with text= to click it (OCR).",
        )
    try:
        if window:
            info, err = await asyncio.to_thread(_find_window, window)
            if err is not None:
                return err
        else:
            info = await asyncio.to_thread(backend.active_window)
        if (blocked := _block(info)) is not None:
            return blocked
        assert info is not None
        snap = await asyncio.to_thread(backend.ui_snapshot, info.handle)
    except PlatformUnsupported as exc:
        return ToolResult.fail(str(exc))
    except OSError as exc:
        return ToolResult.fail(f"could not read the window: {exc}")
    _state.target = info.handle
    wanted = (filter or "").strip()
    elements = [e for e in snap.elements if not wanted or _matches(e, wanted)]
    limit = max(1, min(int(max_items), 200))
    shown = elements[:limit]
    foreground = await asyncio.to_thread(backend.active_window)
    is_front = foreground is not None and foreground.handle == info.handle
    head = f'Window: "{info.title}" ({info.process}){"" if is_front else " [not the foreground window]"}'
    if not snap.elements:
        return ToolResult.success(
            head + "\nThis window exposes no usable controls to the accessibility tree "
            "(games, canvases and remote desktops do this). Use read_screen_text and desktop_click with text= instead.",
            data={"window": info.title, "elements": []},
        )
    if wanted and not elements:
        return ToolResult.success(
            head + f"\nNo control matches '{wanted}' ({len(snap.elements)} controls in the window). Try fewer words, or no filter.",
            data={"window": info.title, "elements": []},
        )
    summary = f"Showing {len(shown)} of {len(elements)}"
    summary += f" matching '{wanted}'" if wanted else " controls"
    if len(elements) > len(shown):
        summary += f"; {len(elements) - len(shown)} more not shown (raise max_items or use filter)"
    lines = [head, summary + "."] + [_line(e) for e in shown]
    data = {
        "window": info.title,
        "process": info.process,
        "elements": [
            {"ref": e.ref, "role": e.role, "name": e.name, "value": e.value, "state": e.state, "enabled": e.enabled}
            for e in shown
        ],
    }
    return ToolResult.success("\n".join(lines), data=data)


# ---------------------------------------------------------------- click
async def _click_ref(ctx: ToolContext, ref: int, double: bool) -> ToolResult:
    backend = _backend()
    info, err = await asyncio.to_thread(_target_info)
    if err is not None:
        return err
    assert info is not None
    if (blocked := _block(info)) is not None:
        return blocked
    try:
        element = await asyncio.to_thread(backend.ui_read, info.handle, ref)
    except ElementGone as exc:
        return ToolResult.fail(str(exc))
    except OSError as exc:
        return ToolResult.fail(f"could not read element {ref}: {exc}")
    if not element.enabled:
        return ToolResult.fail(f"element [{ref}] ({_element_text(element)}) is disabled and cannot be clicked")
    verdict = policy.classify_element_click(element)
    if (stop := await _approve(ctx, verdict, f"Click '{element.name or element.role}' in {info.title}")) is not None:
        return stop
    if (fg := await _ensure_foreground(info)) is not None:
        return fg
    before = await asyncio.to_thread(_fg_title)
    try:
        method = await asyncio.to_thread(backend.ui_act, info.handle, ref, "double_click" if double else "click")
    except ElementGone as exc:
        return ToolResult.fail(str(exc))
    except OSError as exc:
        return ToolResult.fail(f"could not click element {ref}: {exc}")
    await asyncio.sleep(SETTLE)
    after_title = await asyncio.to_thread(_fg_title)
    after = await _reread(info.handle, ref)
    text = (
        f"Clicked [{ref}] {_element_text(element)} (via {method}).\n"
        f"{_title_change(before, after_title)}\n{_element_change(element, after)}"
    )
    return ToolResult.success(text, data={"method": method, "ref": ref})


async def _click_text(ctx: ToolContext, text: str, double: bool) -> ToolResult:
    backend = _backend()
    info = await asyncio.to_thread(backend.active_window)
    if (blocked := _block(info)) is not None:
        return blocked
    assert info is not None
    try:
        grab = await asyncio.to_thread(grab_for_ocr, info.rect)
        lines = await asyncio.to_thread(ocr_image, grab.image)
    except CaptureError as exc:
        return ToolResult.fail(str(exc))
    except ImportError as exc:
        return ToolResult.fail(f"OCR is not installed ({exc.name})", "Install rapidocr-onnxruntime (the 'screen' extra).")
    candidates = match_text(lines, text)
    if not candidates:
        sample = ", ".join(f"'{ln.text.strip()[:30]}'" for ln in lines[:12])
        return ToolResult.fail(
            f"no text on the screen matches '{text}'",
            "Text OCR saw includes: " + (sample or "nothing") + ". Check the spelling, or use read_screen_text.",
        )
    if len(candidates) > 1:
        return ToolResult.fail(
            f"'{text}' matches {len(candidates)} places on screen: {describe_candidates(candidates, grab.size)}",
            "Call again with more of the line's text so only one place matches.",
        )
    hit = candidates[0]
    x, y = grab.to_screen(hit.x, hit.y)
    under = await asyncio.to_thread(backend.window_at, x, y)
    if under is not None and (blocked := _block(under)) is not None:
        return blocked
    label = hit.line.text.strip()
    probe = UiElement(ref=0, role="button", name=label)
    verdict = policy.classify_element_click(probe)
    if (stop := await _approve(ctx, verdict, f"Click the on-screen text '{label}' in {info.title}")) is not None:
        return stop
    before = await asyncio.to_thread(_fg_title)
    try:
        await asyncio.to_thread(backend.input_click, x, y, double)
    except OSError as exc:
        return ToolResult.fail(f"could not click: {exc}")
    await asyncio.sleep(SETTLE)
    after = await asyncio.to_thread(_fg_title)
    where = region_name(hit.x, hit.y, *grab.size)
    return ToolResult.success(
        f"Clicked the on-screen text '{label}' ({hit.tier} match, {where}, found by OCR).\n{_title_change(before, after)}\n"
        "OCR clicks have no element to read back; look at the screen again to check the effect."
    )


@tool(group=GROUP, timeout=CONFIRM_TIMEOUT)
async def desktop_click(
    ref: int | None = None, text: str | None = None, double: bool = False, ctx: ToolContext = None  # type: ignore[assignment]
) -> ToolResult:
    """Click a control in a desktop application. Give a ref from desktop_inspect (reliable: it
    uses the control's own action, or clicks its centre), or text= to click the on-screen text
    found by OCR (for apps whose controls desktop_inspect cannot see). Exactly one of the two.
    Buying, paying, sending, posting, deleting, signing in and accepting terms ask the user
    first. The result says what changed: the foreground window and the control's new state.

    Args:
        ref: Element number from the latest desktop_inspect.
        text: On-screen text to click when there is no usable ref. If several places match, the call fails and lists them.
        double: Double-click (open a file or item) instead of a single click.
    """
    if (bad := _unavailable()) is not None:
        return bad
    if (ref is None) == (not text):
        return ToolResult.fail("give exactly one of ref or text", "Use ref from desktop_inspect, or text= for OCR.")
    try:
        if ref is not None:
            return await _click_ref(ctx, int(ref), bool(double))
        return await _click_text(ctx, str(text), bool(double))
    except PlatformUnsupported as exc:
        return ToolResult.fail(str(exc))


# ---------------------------------------------------------------- type
@tool(group=GROUP, timeout=CONFIRM_TIMEOUT)
async def desktop_type(
    text: str, ref: int | None = None, clear: bool = False, submit: bool = False, ctx: ToolContext = None  # type: ignore[assignment]
) -> ToolResult:
    """Type text into a desktop application, as real keystrokes (any language or symbol). With
    ref (from desktop_inspect) the field is focused first; without it the text goes to whatever
    has focus. Password, credential and payment-card fields are refused: the user types those.
    Typing into a terminal or the Run dialog asks the user first, because it runs commands.

    Args:
        text: The text to type. A newline presses Enter.
        ref: Element number of the field to focus first.
        clear: Empty the field before typing.
        submit: Press Enter after typing.
    """
    if (bad := _unavailable()) is not None:
        return bad
    if not text:
        return ToolResult.fail("text is empty")
    if len(text) > MAX_TEXT:
        return ToolResult.fail(f"text is longer than {MAX_TEXT} characters", "Type it in smaller parts.")
    backend = _backend()
    try:
        element: UiElement | None = None
        if ref is not None:
            info, err = await asyncio.to_thread(_target_info)
            if err is not None:
                return err
            assert info is not None
            if (blocked := _block(info)) is not None:
                return blocked
            element = await asyncio.to_thread(backend.ui_read, info.handle, int(ref))
            if not element.enabled:
                return ToolResult.fail(f"element [{ref}] ({_element_text(element)}) is disabled")
        else:
            info = await asyncio.to_thread(backend.active_window)
            if (blocked := _block(info)) is not None:
                return blocked
            assert info is not None
            element = await asyncio.to_thread(backend.ui_focused)  # may be None (unknown)
        verdict = policy.classify_element_typing(element, text)
        if verdict.kind == "refuse":
            return ToolResult.fail(verdict.reason)
        entry = policy.classify_text_entry(
            info, text, element=element, submit=submit, self_elevated=backend.self_elevated()
        )
        shown = text if len(text) <= 60 else text[:60] + "..."
        if (stop := await _approve(ctx, entry, f"Type '{shown}' into {info.title}")) is not None:
            return stop
        if (fg := await _ensure_foreground(info)) is not None:
            return fg
        if ref is not None:
            await asyncio.to_thread(backend.ui_focus_element, info.handle, int(ref))
            if clear and not await asyncio.to_thread(backend.ui_clear, info.handle, int(ref)):
                await asyncio.to_thread(backend.input_keys, ["ctrl", "a"])
                await asyncio.to_thread(backend.input_keys, ["delete"])
        elif clear:
            await asyncio.to_thread(backend.input_keys, ["ctrl", "a"])
            await asyncio.to_thread(backend.input_keys, ["delete"])
        before = await asyncio.to_thread(_fg_title)
        await asyncio.to_thread(backend.input_text, text)
        if submit:
            await asyncio.to_thread(backend.input_keys, ["enter"])
        await asyncio.sleep(SETTLE)
        after_title = await asyncio.to_thread(_fg_title)
    except ElementGone as exc:
        return ToolResult.fail(str(exc))
    except PlatformUnsupported as exc:
        return ToolResult.fail(str(exc))
    except OSError as exc:
        return ToolResult.fail(f"could not type: {exc}")
    lines = [f"Typed {len(text)} characters{' and pressed Enter' if submit else ''}.", _title_change(before, after_title)]
    if ref is not None and element is not None:
        after = await _reread(info.handle, int(ref))
        if after is None:
            lines.append(f"Element [{ref}] is gone now.")
        elif after.value and after.value != "(hidden)":
            lines.append(f"Element [{ref}] now holds '{after.value[:200]}'.")
        else:
            lines.append(f"Element [{ref}] shows no readable value ({_element_text(after)}).")
    else:
        lines.append("No field was named, so the typed text cannot be read back; look at the screen to check.")
    return ToolResult.success("\n".join(lines))


# ---------------------------------------------------------------- keys
@tool(group=GROUP, timeout=CONFIRM_TIMEOUT)
async def press_keys(keys: str, repeat: int = 1, ctx: ToolContext = None) -> ToolResult:  # type: ignore[assignment]
    """Press a key or key combination in the foreground window, such as "ctrl+s", "alt+tab",
    "enter", "escape", "win+d", "f5", "ctrl+shift+t". Modifiers: ctrl, alt, shift, win. One
    non-modifier key per call. Closing, deleting and quitting shortcuts (alt+f4, ctrl+w,
    delete, ctrl+q, win+r) and Enter inside a terminal ask the user first; win+l locks the
    screen after asking. Unknown key names are rejected with the list of valid ones.

    Args:
        keys: The chord, joined with +, for example "ctrl+c" or "enter".
        repeat: Press it this many times (1 to 50), for example to step through a list.
    """
    if (bad := _unavailable()) is not None:
        return bad
    try:
        parsed = parse_chord(keys)
    except ChordError as exc:
        return ToolResult.fail(str(exc))
    if not 1 <= int(repeat) <= MAX_REPEAT:
        return ToolResult.fail(f"repeat must be between 1 and {MAX_REPEAT}, got {repeat}")
    backend = _backend()
    chord_text = "+".join(parsed)
    try:
        info = await asyncio.to_thread(backend.active_window)
        if policy.is_win_l(parsed):
            if (stop := await _approve(ctx, policy.classify_chord(parsed, info), "Lock the screen")) is not None:
                return stop
            await asyncio.to_thread(backend.lock_screen)
            return ToolResult.success("Screen locked.")
        # Chords that act on the system rather than the window may be sent while Jarvis's own
        # window is in front (alt+tab away from the HUD, win+d to show the desktop).
        system_chord = "win" in parsed or frozenset(parsed) == frozenset({"alt", "tab"})
        if (blocked := _block(info, allow_own=system_chord)) is not None:
            return blocked
        assert info is not None
        focused = None
        if parsed in (["enter"], ["space"]) and backend.can_inspect_elements():
            focused = await asyncio.to_thread(backend.ui_focused)
        verdict = policy.classify_chord(parsed, info, focused=focused, self_elevated=backend.self_elevated())
        if (stop := await _approve(ctx, verdict, f"Press {chord_text} in {info.title}")) is not None:
            return stop
        before = await asyncio.to_thread(_fg_title)
        for i in range(int(repeat)):
            await asyncio.to_thread(backend.input_keys, parsed)
            if i + 1 < int(repeat):
                await asyncio.sleep(0.05)
        await asyncio.sleep(SETTLE)
        after_title = await asyncio.to_thread(_fg_title)
    except PlatformUnsupported as exc:
        return ToolResult.fail(str(exc))
    except OSError as exc:
        return ToolResult.fail(f"could not press {chord_text}: {exc}")
    times = f" x{int(repeat)}" if int(repeat) > 1 else ""
    return ToolResult.success(f"Pressed {chord_text}{times}.\n{_title_change(before, after_title)}")


# ---------------------------------------------------------------- scroll
@tool(group=GROUP, timeout=60)
async def desktop_scroll(
    direction: Literal["up", "down", "left", "right"], amount: int = 3, ref: int | None = None
) -> ToolResult:
    """Scroll with the mouse wheel in a desktop application, over a control (ref from
    desktop_inspect) or the middle of the foreground window. Use to reach list items or text
    that is off-screen, then desktop_inspect again.

    Args:
        direction: up, down, left or right.
        amount: Wheel notches, 1 to 50 (3 is about one short step).
        ref: Element number to scroll over; default is the centre of the foreground window.
    """
    if (bad := _unavailable()) is not None:
        return bad
    if not 1 <= int(amount) <= 50:
        return ToolResult.fail(f"amount must be between 1 and 50, got {amount}")
    backend = _backend()
    try:
        if ref is not None:
            info, err = await asyncio.to_thread(_target_info)
            if err is not None:
                return err
            assert info is not None
            if (blocked := _block(info)) is not None:
                return blocked
            element = await asyncio.to_thread(backend.ui_read, info.handle, int(ref))
            rect = element.rect
        else:
            info = await asyncio.to_thread(backend.active_window)
            if (blocked := _block(info)) is not None:
                return blocked
            assert info is not None
            rect = info.rect
        if rect is None:
            return ToolResult.fail("cannot tell where that control is on screen")
        if (fg := await _ensure_foreground(info)) is not None:
            return fg
        x, y = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
        before = await asyncio.to_thread(_fg_title)
        await asyncio.to_thread(backend.input_scroll, x, y, direction, int(amount))
        await asyncio.sleep(SETTLE)
        after = await asyncio.to_thread(_fg_title)
    except ElementGone as exc:
        return ToolResult.fail(str(exc))
    except PlatformUnsupported as exc:
        return ToolResult.fail(str(exc))
    except OSError as exc:
        return ToolResult.fail(f"could not scroll: {exc}")
    return ToolResult.success(
        f"Scrolled {direction} by {int(amount)} notches.\n{_title_change(before, after)}\n"
        "Scrolling cannot be read back; call desktop_inspect again to see what is now visible."
    )


# ---------------------------------------------------------------- windows
@tool(group=GROUP, timeout=30)
async def focus_window(name: str) -> ToolResult:
    """Bring an open application's window to the front so it can be inspected or typed into.
    Matches the way close_app does: "notepad" finds "Untitled - Notepad", "chrome" finds the
    Chrome window. If the app has several windows the top-most is used.

    Args:
        name: App name as a person would say it.
    """
    if (bad := _unavailable()) is not None:
        return bad
    backend = _backend()
    info, err = await asyncio.to_thread(_find_window, name)
    if err is not None:
        return err
    assert info is not None
    try:
        ok = await asyncio.to_thread(backend.focus_window, info.handle)
    except OSError as exc:
        return ToolResult.fail(f"could not focus '{info.title}': {exc}")
    now = await asyncio.to_thread(backend.active_window)
    front = now.title if now is not None else "(unknown)"
    if not ok:
        extra = " It runs as administrator, which Windows protects." if info.elevated and not backend.self_elevated() else ""
        return ToolResult.fail(
            f"asked for '{info.title}' but the foreground window is '{front}'.{extra}",
            "Windows sometimes refuses focus changes; ask the user to click the window once.",
        )
    return ToolResult.success(f"'{info.title}' ({info.process}) is now the foreground window.")


@tool(group=GROUP, timeout=15)
async def get_active_window() -> ToolResult:
    """Say which window is in the foreground: its title, program, window class, and whether it
    is a terminal, an administrator window or Jarvis's own. Use to check where input would go.
    """
    if (bad := _unavailable()) is not None:
        return bad
    backend = _backend()
    try:
        info = await asyncio.to_thread(backend.active_window)
    except PlatformUnsupported as exc:
        return ToolResult.fail(str(exc))
    if info is None:
        secure = not await asyncio.to_thread(backend.input_desktop_available)
        why = " A secure prompt (UAC or the lock screen) may be showing." if secure else ""
        return ToolResult.success("There is no foreground window." + why)
    notes = []
    kind = policy.window_kind(info)
    if kind != "normal":
        notes.append(kind)
    if info.pid in policy.own_process_ids():
        notes.append("Jarvis's own window")
    if info.elevated:
        notes.append("administrator window")
    suffix = f" [{', '.join(notes)}]" if notes else ""
    return ToolResult.success(
        f"Foreground window: '{info.title}' (process {info.process}, class {info.class_name}){suffix}",
        data={"title": info.title, "process": info.process, "class": info.class_name, "elevated": info.elevated},
    )
