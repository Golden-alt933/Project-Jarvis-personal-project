"""The desktop tools against an in-memory backend: no real window, input, screen or model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from jarvis import platform as plat
from jarvis.desktop import ocr_target, policy
from jarvis.llm.types import ToolCall
from jarvis.platform.base import BasePlatform, ElementGone, PlatformUnsupported, UiElement, UiSnapshot, WindowInfo
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import desktop as tools
from jarvis.tools.registry import ToolRegistry
from jarvis.vision.ocr import OcrLine


# ---------------------------------------------------------------- the fake desktop
def winfo(handle, title, process, klass="Frame", pid=None, elevated=False, rect=(0, 0, 400, 300)):
    return WindowInfo(title=title, process=process, class_name=klass, pid=pid or handle + 1000, handle=handle,
                      elevated=elevated, rect=rect)


def el(ref, role, name, **kw):
    return UiElement(ref=ref, role=role, name=name, rect=kw.pop("rect", (10 * ref, 10, 10 * ref + 50, 40)), **kw)


class FakeDesktop(BasePlatform):
    name = "fake"
    app_dir_name = "jarvis"

    def __init__(self):
        self.wins = [
            winfo(1, "Calculator", "ApplicationFrameHost.exe", "ApplicationFrameWindow"),
            winfo(2, "Untitled - Notepad", "notepad.exe", "Notepad"),
            winfo(3, "Command Prompt", "cmd.exe", "ConsoleWindowClass"),
            winfo(4, "Jarvis", "python.exe", "Qt", pid=777),
            winfo(5, "Registry Editor", "regedit.exe", "RegEdit_RegEdit", elevated=True),
        ]
        self.active = 1
        self.elements: dict[int, list[UiElement]] = {}
        self.log: list[tuple] = []
        self.problem: str | None = None
        self.inspectable = True
        self.elevated_self = False
        self.input_ok = True
        self.focused: UiElement | None = None
        self.on_act = None  # callback(handle, ref, action) -> method string
        self.focus_works = True
        self.under: WindowInfo | None = None
        self.clear_works = True
        self.lock_calls = 0
        self.screen = Image.new("RGB", (400, 300), "white")

    # windows
    def windows(self):
        return [(w.process, w.title, w.pid, w.handle) for w in self.wins]

    def never_close_titles(self):
        return {"program manager"}

    def foreground_window_title(self):
        w = self.window_info(self.active)
        return w.title if w else ""

    def window_info(self, handle):
        return next((w for w in self.wins if w.handle == handle), None)

    def active_window(self):
        return self.window_info(self.active)

    def window_at(self, x, y):
        return self.under

    def focus_window(self, handle):
        self.log.append(("focus_window", handle))
        if self.focus_works:
            self.active = handle
        return self.focus_works

    def desktop_problem(self):
        return self.problem

    def can_inspect_elements(self):
        return self.inspectable

    def self_elevated(self):
        return self.elevated_self

    def input_desktop_available(self):
        return self.input_ok

    def lock_screen(self):
        self.lock_calls += 1

    # elements
    def ui_snapshot(self, handle):
        self.log.append(("snapshot", handle))
        items = self.elements.get(handle, [])
        return UiSnapshot(window=self.window_info(handle), elements=list(items), total=len(items) + 3)

    def _find(self, handle, ref):
        for e in self.elements.get(handle, []):
            if e.ref == ref:
                return e
        raise ElementGone(f"there is no element {ref} in that window right now. Run desktop_inspect again for fresh refs.")

    def ui_read(self, handle, ref):
        e = self._find(handle, ref)
        return UiElement(**{**e.__dict__})

    def ui_act(self, handle, ref, action):
        self._find(handle, ref)
        self.log.append(("act", handle, ref, action))
        return self.on_act(handle, ref, action) if self.on_act else "invoke"

    def ui_focus_element(self, handle, ref):
        self.log.append(("focus_element", handle, ref))
        return "focused"

    def ui_clear(self, handle, ref):
        self.log.append(("clear", handle, ref))
        return self.clear_works

    def ui_focused(self):
        return self.focused

    # input
    def input_click(self, x, y, double=False):
        self.log.append(("click", x, y, double))

    def input_text(self, text):
        self.log.append(("text", text))

    def input_keys(self, keys):
        self.log.append(("keys", list(keys)))

    def input_scroll(self, x, y, direction, amount):
        self.log.append(("scroll", x, y, direction, amount))

    def screen_grab(self):
        return self.screen

    def cursor_position(self):
        return (5, 5)

    def prepare_capture(self):
        pass

    def actions(self):
        return [entry for entry in self.log if entry[0] in ("act", "click", "text", "keys", "scroll")]


@pytest.fixture
def desk(monkeypatch):
    backend = FakeDesktop()
    plat.set_backend(backend)
    monkeypatch.setattr(tools.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(policy, "own_process_ids", lambda extra=(): {777})
    tools._state.target = None
    backend.elements[1] = [
        el(1, "button", "Seven"), el(2, "button", "Plus"), el(3, "button", "Two"), el(4, "button", "Equals"),
        el(5, "text", "Display is 0", automation_id="CalculatorResults"),
        el(6, "button", "Memory recall", enabled=False),
        el(7, "button", "Clear all memory"),
    ]
    backend.elements[2] = [
        el(1, "edit", "Text editor", value="", automation_id="15"),
        el(2, "edit", "Password", is_password=True),
        el(3, "button", "Delete"),
        el(4, "checkbox", "Word wrap", state="off"),
        el(5, "edit", "Search"),
    ]
    yield backend
    plat.set_backend(None)
    tools._state.target = None


async def _no_sleep(_):
    return None


class Asker:
    def __init__(self, answer=True):
        self.answer, self.asked = answer, []

    async def __call__(self, text):
        self.asked.append(text)
        return self.answer


def ctx_with(answer=True):
    asker = Asker(answer)
    return ToolContext(settings=SimpleNamespace(), confirm=asker), asker


# ---------------------------------------------------------------- inspect
async def test_inspect_lists_the_foreground_window_with_refs(desk):
    result = await tools.desktop_inspect()
    assert result.ok
    lines = result.content.splitlines()
    assert lines[0] == 'Window: "Calculator" (ApplicationFrameHost.exe)'
    assert "Showing 7 of 7 controls." in lines[1]
    assert '[1] button "Seven"' in result.content and '[5] text "Display is 0"' in result.content
    assert "[6] button \"Memory recall\" (disabled)" in result.content
    assert tools._state.target == 1
    assert "rect" not in str(result.data) and "x=" not in result.content, "no pixel positions are ever shown"


async def test_inspect_by_app_name_uses_the_close_app_matching(desk):
    result = await tools.desktop_inspect(window="notepad")
    assert result.ok and 'Window: "Untitled - Notepad" (notepad.exe) [not the foreground window]' in result.content
    assert tools._state.target == 2


async def test_inspect_filter_and_limit(desk):
    result = await tools.desktop_inspect(filter="display")
    assert 'Showing 1 of 1 matching \'display\'.' in result.content and "Seven" not in result.content
    limited = await tools.desktop_inspect(max_items=3)
    assert "Showing 3 of 7 controls; 4 more not shown" in limited.content
    assert limited.content.count("\n[") == 3
    nothing = await tools.desktop_inspect(filter="zebra")
    assert nothing.ok and "No control matches 'zebra'" in nothing.content


async def test_inspect_says_so_when_a_window_has_no_controls(desk):
    desk.elements[1] = []
    result = await tools.desktop_inspect()
    assert result.ok and "exposes no usable controls" in result.content and "desktop_click with text=" in result.content


async def test_inspect_refuses_unreachable_windows(desk):
    desk.active = 4
    own = await tools.desktop_inspect()
    assert not own.ok and "Jarvis's own window" in own.content
    elevated = await tools.desktop_inspect(window="regedit")
    assert not elevated.ok and "administrator" in elevated.content
    desk.input_ok = False
    secure = await tools.desktop_inspect()
    assert not secure.ok and "UAC" in secure.content


async def test_inspect_unknown_app_lists_open_windows(desk):
    result = await tools.desktop_inspect(window="photoshop")
    assert not result.ok and "photoshop" in result.content and "Calculator" in result.content


async def test_inspect_where_element_listing_is_not_implemented(desk):
    desk.inspectable = False
    result = await tools.desktop_inspect()
    assert not result.ok and "read_screen_text" in result.content and "text=" in result.content


async def test_every_desktop_tool_reports_an_unavailable_backend(desk):
    desk.problem = "desktop input is not supported in a Wayland session"
    ctx, _ = ctx_with()
    for result in (
        await tools.desktop_inspect(), await tools.desktop_click(ref=1, ctx=ctx), await tools.desktop_type("x", ctx=ctx),
        await tools.press_keys("enter", ctx=ctx), await tools.desktop_scroll("down"), await tools.focus_window("calc"),
        await tools.get_active_window(),
    ):
        assert not result.ok and "Wayland" in result.content
    assert desk.actions() == []


# ---------------------------------------------------------------- click by ref
async def test_click_by_ref_acts_and_reports_what_changed(desk):
    ctx, asker = ctx_with()
    await tools.desktop_inspect()

    def on_act(handle, ref, action):
        desk.elements[1][4].name = "Display is 7"  # the display is not the clicked element
        desk.elements[1][0].state = "pressed"
        return "invoke"

    desk.on_act = on_act
    result = await tools.desktop_click(ref=1, ctx=ctx)
    assert result.ok
    assert "Clicked [1] button 'Seven' (via invoke)" in result.content
    assert "Foreground window unchanged: 'Calculator'" in result.content
    assert "Element [1] changed: state '-' -> 'pressed'" in result.content
    assert desk.actions() == [("act", 1, 1, "click")] and asker.asked == []


async def test_click_reports_a_window_change(desk):
    ctx, _ = ctx_with()
    await tools.desktop_inspect()

    def on_act(handle, ref, action):
        desk.active = 2
        return "invoke"

    desk.on_act = on_act
    result = await tools.desktop_click(ref=2, ctx=ctx)
    assert "Foreground window changed: 'Calculator' -> 'Untitled - Notepad'" in result.content


async def test_click_brings_a_background_window_forward_first(desk):
    ctx, _ = ctx_with()
    await tools.desktop_inspect(window="notepad")
    assert desk.active == 1
    await tools.desktop_click(ref=5, ctx=ctx)
    assert ("focus_window", 2) in desk.log and desk.active == 2


async def test_double_click_is_passed_through(desk):
    ctx, _ = ctx_with()
    await tools.desktop_inspect()
    await tools.desktop_click(ref=1, double=True, ctx=ctx)
    assert desk.actions() == [("act", 1, 1, "double_click")]


async def test_click_a_vanished_element(desk):
    ctx, _ = ctx_with()
    await tools.desktop_inspect()
    result = await tools.desktop_click(ref=99, ctx=ctx)
    assert not result.ok and "desktop_inspect" in result.content and desk.actions() == []


async def test_click_a_disabled_element(desk):
    ctx, _ = ctx_with()
    await tools.desktop_inspect()
    result = await tools.desktop_click(ref=6, ctx=ctx)
    assert not result.ok and "disabled" in result.content and desk.actions() == []


async def test_click_needs_an_inspected_window_and_exactly_one_target(desk):
    ctx, _ = ctx_with()
    first = await tools.desktop_click(ref=1, ctx=ctx)
    assert not first.ok and "desktop_inspect" in first.content
    neither = await tools.desktop_click(ctx=ctx)
    both = await tools.desktop_click(ref=1, text="Seven", ctx=ctx)
    assert not neither.ok and not both.ok and "exactly one" in both.content


async def test_click_on_a_dangerous_control_asks_and_a_no_stops_it(desk):
    await tools.desktop_inspect(window="notepad")
    ctx, asker = ctx_with(False)
    result = await tools.desktop_click(ref=3, ctx=ctx)
    assert not result.ok and "declined" in result.content and desk.actions() == []
    assert len(asker.asked) == 1 and "Delete" in asker.asked[0] and "deleting" in asker.asked[0]


async def test_click_on_a_dangerous_control_goes_ahead_on_a_yes(desk):
    await tools.desktop_inspect(window="notepad")
    ctx, asker = ctx_with(True)
    result = await tools.desktop_click(ref=3, ctx=ctx)
    assert result.ok and desk.actions() == [("act", 2, 3, "click")] and len(asker.asked) == 1


async def test_a_broken_confirm_counts_as_no(desk):
    async def broken(_):
        raise RuntimeError("frontend died")

    await tools.desktop_inspect(window="notepad")
    result = await tools.desktop_click(ref=3, ctx=ToolContext(settings=SimpleNamespace(), confirm=broken))
    assert not result.ok and desk.actions() == []


async def test_clear_all_memory_style_names_ask(desk):
    await tools.desktop_inspect()
    ctx, asker = ctx_with(False)
    assert not (await tools.desktop_click(ref=7, ctx=ctx)).ok and asker.asked


async def test_click_in_an_elevated_or_own_window_is_refused(desk):
    await tools.desktop_inspect()
    desk.wins[0] = winfo(1, "Calculator", "ApplicationFrameHost.exe", elevated=True)
    ctx, _ = ctx_with()
    result = await tools.desktop_click(ref=1, ctx=ctx)
    assert not result.ok and "administrator" in result.content and desk.actions() == []


# ---------------------------------------------------------------- click by on-screen text (OCR)
def ocr_lines(*items):
    return [OcrLine(text, 0.95, x0, y0, x1, y1) for text, x0, y0, x1, y1 in items]


@pytest.fixture
def ocr(monkeypatch):
    holder = SimpleNamespace(lines=[])
    monkeypatch.setattr(tools, "ocr_image", lambda image: holder.lines)
    return holder


async def test_ocr_click_hits_the_middle_of_the_matching_line(desk, ocr):
    ocr.lines = ocr_lines(("File", 10, 10, 50, 30), ("Start game", 100, 200, 200, 230), ("Quit", 100, 250, 140, 270))
    ctx, _ = ctx_with()
    result = await tools.desktop_click(text="Start game", ctx=ctx)
    assert result.ok and "found by OCR" in result.content and "exact match" in result.content
    assert desk.actions() == [("click", 150, 215, False)]
    assert "Foreground window unchanged" in result.content


async def test_ocr_match_tiers(desk, ocr):
    ocr.lines = ocr_lines(("Settings", 10, 10, 90, 30))
    ctx, _ = ctx_with()
    for wanted, tier in (("Settings", "exact"), ("settings", "case-insensitive"), ("Setings", "fuzzy")):
        desk.log.clear()
        result = await tools.desktop_click(text=wanted, ctx=ctx)
        assert result.ok and f"{tier} match" in result.content, wanted


async def test_ocr_exact_beats_case_insensitive_beats_contained(desk, ocr):
    ocr.lines = ocr_lines(("Open file", 10, 10, 90, 30), ("OPEN", 10, 100, 60, 120), ("Open", 10, 200, 60, 220))
    ctx, _ = ctx_with()
    result = await tools.desktop_click(text="Open", ctx=ctx)
    assert result.ok and "exact match" in result.content and desk.actions() == [("click", 35, 210, False)]


async def test_ocr_several_matches_fail_with_positions_and_no_click(desk, ocr):
    ocr.lines = ocr_lines(("Save", 10, 10, 60, 30), ("Save", 300, 270, 360, 290), ("Cancel", 200, 270, 250, 290))
    ctx, _ = ctx_with()
    result = await tools.desktop_click(text="Save", ctx=ctx)
    assert not result.ok and desk.actions() == []
    assert "matches 2 places" in result.content and "top left" in result.content and "bottom right" in result.content
    assert "more of the line's text" in result.content
    assert "x=" not in result.content and "310" not in result.content, "positions are words, never pixels"


async def test_ocr_no_match_lists_what_was_seen(desk, ocr):
    ocr.lines = ocr_lines(("Play", 10, 10, 60, 30), ("Options", 10, 50, 80, 70))
    ctx, _ = ctx_with()
    result = await tools.desktop_click(text="Quit", ctx=ctx)
    assert not result.ok and "no text on the screen matches 'Quit'" in result.content and "'Play'" in result.content


async def test_ocr_coordinates_follow_the_downscale_and_monitor_origin(desk, ocr, monkeypatch):
    desk.screen = Image.new("RGB", (3136, 1764), "white")  # twice the OCR size of 1568 wide
    real_grab = ocr_target.grab_for_ocr

    def grab_with_offset(rect=None):
        grab = real_grab(rect)
        return ocr_target.ScreenGrab(grab.image, left=1920, top=-100, scale=grab.scale)

    monkeypatch.setattr(tools, "grab_for_ocr", grab_with_offset)
    ocr.lines = ocr_lines(("Start", 100, 200, 200, 240))  # image pixels
    ctx, _ = ctx_with()
    await tools.desktop_click(text="Start", ctx=ctx)
    assert desk.actions() == [("click", 1920 + 150 * 2, -100 + 220 * 2, False)]


async def test_ocr_click_checks_the_window_it_would_hit(desk, ocr):
    ocr.lines = ocr_lines(("Start", 100, 200, 200, 240))
    desk.under = desk.wins[3]  # Jarvis's own HUD sits on top of that spot
    ctx, _ = ctx_with()
    result = await tools.desktop_click(text="Start", ctx=ctx)
    assert not result.ok and "Jarvis's own window" in result.content and desk.actions() == []


async def test_ocr_click_asks_for_dangerous_text(desk, ocr):
    ocr.lines = ocr_lines(("Send", 100, 200, 160, 240), ("Next", 10, 10, 60, 30))
    ctx, asker = ctx_with(False)
    result = await tools.desktop_click(text="Send", ctx=ctx)
    assert not result.ok and "declined" in result.content and "Send" in asker.asked[0] and desk.actions() == []
    ctx, asker = ctx_with(True)
    assert (await tools.desktop_click(text="Next", ctx=ctx)).ok and asker.asked == []


async def test_ocr_click_in_a_blocked_foreground_window(desk, ocr):
    desk.active = 4
    ocr.lines = ocr_lines(("Start", 100, 200, 200, 240))
    ctx, _ = ctx_with()
    assert not (await tools.desktop_click(text="Start", ctx=ctx)).ok and desk.actions() == []


def test_contained_text_clicks_where_the_words_are():
    lines = ocr_lines(("Press Start to play", 0, 100, 190, 120))
    (hit,) = ocr_target.match_text(lines, "Start")
    assert hit.tier == "contained" and 60 <= hit.x <= 100 and hit.y == 110


def test_region_names():
    assert ocr_target.region_name(10, 10, 300, 300) == "top left"
    assert ocr_target.region_name(290, 290, 300, 300) == "bottom right"
    assert ocr_target.region_name(150, 150, 300, 300) == "middle"
    assert ocr_target.region_name(150, 10, 300, 300) == "top centre"
    assert ocr_target.region_name(10, 150, 300, 300) == "middle left"


# ---------------------------------------------------------------- type
async def test_type_into_a_field_focuses_clears_types_and_reads_back(desk):
    ctx, asker = ctx_with()
    await tools.desktop_inspect(window="notepad")

    result = await tools.desktop_type("hello world", ref=1, clear=True, ctx=ctx)
    assert result.ok
    assert desk.actions() == [("text", "hello world")]
    assert ("focus_element", 2, 1) in desk.log and ("clear", 2, 1) in desk.log
    assert "Typed 11 characters." in result.content and "Element [1]" in result.content
    assert asker.asked == [] and desk.active == 2, "the window was brought forward"


async def test_clear_falls_back_to_select_all_and_delete(desk):
    ctx, _ = ctx_with()
    desk.clear_works = False
    await tools.desktop_inspect(window="notepad")
    await tools.desktop_type("x", ref=1, clear=True, ctx=ctx)
    assert desk.actions() == [("keys", ["ctrl", "a"]), ("keys", ["delete"]), ("text", "x")]


async def test_submit_presses_enter(desk):
    ctx, _ = ctx_with()
    await tools.desktop_inspect(window="notepad")
    result = await tools.desktop_type("query", ref=5, submit=True, ctx=ctx)
    assert result.ok and desk.actions() == [("text", "query"), ("keys", ["enter"])] and "pressed Enter" in result.content


async def test_password_fields_are_refused_outright(desk):
    ctx, asker = ctx_with(True)
    await tools.desktop_inspect(window="notepad")
    result = await tools.desktop_type("hunter2", ref=2, ctx=ctx)
    assert not result.ok and "password" in result.content and "user must enter it themselves" in result.content
    assert desk.actions() == [] and asker.asked == [], "refused, not asked: even a yes cannot override it"


async def test_a_focused_password_field_is_refused_when_no_ref_is_given(desk):
    desk.active = 2
    desk.focused = el(0, "edit", "Password", is_password=True)
    ctx, _ = ctx_with()
    result = await tools.desktop_type("hunter2", ctx=ctx)
    assert not result.ok and desk.actions() == []
    desk.focused = el(0, "edit", "Card number")
    assert not (await tools.desktop_type("1234", ctx=ctx)).ok


async def test_card_numbers_are_refused_anywhere(desk):
    desk.active = 2
    ctx, _ = ctx_with()
    result = await tools.desktop_type("4111 1111 1111 1111", ctx=ctx)
    assert not result.ok and "card number" in result.content and desk.actions() == []


async def test_typing_into_a_terminal_asks_with_the_text_shown(desk):
    desk.active = 3
    ctx, asker = ctx_with(False)
    result = await tools.desktop_type("rm -rf /tmp/x", ctx=ctx)
    assert not result.ok and "declined" in result.content and desk.actions() == []
    assert "rm -rf /tmp/x" in asker.asked[0] and "terminal" in asker.asked[0]
    ctx, asker = ctx_with(True)
    assert (await tools.desktop_type("dir", submit=True, ctx=ctx)).ok
    assert desk.actions() == [("text", "dir"), ("keys", ["enter"])] and "press Enter" in asker.asked[0]


async def test_typing_into_a_background_terminal_is_judged_by_the_target_window(desk):
    desk.elements[3] = [el(1, "edit", "Terminal")]
    await tools.desktop_inspect(window="cmd")
    ctx, asker = ctx_with(False)
    result = await tools.desktop_type("whoami", ref=1, ctx=ctx)
    assert not result.ok and asker.asked and desk.actions() == []


async def test_typing_into_own_or_elevated_windows_is_refused(desk):
    ctx, _ = ctx_with()
    desk.active = 4
    own = await tools.desktop_type("hi", ctx=ctx)
    desk.active = 5
    elevated = await tools.desktop_type("hi", ctx=ctx)
    assert not own.ok and "own window" in own.content and not elevated.ok and "administrator" in elevated.content
    assert desk.actions() == []


async def test_type_validates_its_input(desk):
    ctx, _ = ctx_with()
    assert not (await tools.desktop_type("", ctx=ctx)).ok
    assert "5000" in (await tools.desktop_type("x" * 5001, ctx=ctx)).content


async def test_type_with_a_stale_ref(desk):
    await tools.desktop_inspect(window="notepad")
    ctx, _ = ctx_with()
    result = await tools.desktop_type("x", ref=42, ctx=ctx)
    assert not result.ok and "desktop_inspect" in result.content and desk.actions() == []


# ---------------------------------------------------------------- keys
async def test_press_keys_sends_the_parsed_chord_and_reports_the_window(desk):
    ctx, asker = ctx_with()
    result = await tools.press_keys("Ctrl+S", ctx=ctx)
    assert result.ok and desk.actions() == [("keys", ["ctrl", "s"])] and "Pressed ctrl+s." in result.content
    assert "Foreground window unchanged: 'Calculator'" in result.content and asker.asked == []


async def test_press_keys_repeat(desk):
    ctx, _ = ctx_with()
    result = await tools.press_keys("down", repeat=3, ctx=ctx)
    assert desk.actions() == [("keys", ["down"])] * 3 and "x3" in result.content
    for bad in (0, 51, -1):
        assert not (await tools.press_keys("down", repeat=bad, ctx=ctx)).ok


async def test_press_keys_rejects_unknown_names_with_the_valid_list(desk):
    ctx, _ = ctx_with()
    result = await tools.press_keys("ctrl+banana", ctx=ctx)
    assert not result.ok and "banana" in result.content and "f1-f24" in result.content and desk.actions() == []
    assert not (await tools.press_keys("a+b", ctx=ctx)).ok


@pytest.mark.parametrize("chord", ["alt+f4", "ctrl+w", "ctrl+shift+w", "delete", "shift+delete", "ctrl+q", "win+r"])
async def test_irreversible_chords_ask_and_a_no_stops_them(desk, chord):
    ctx, asker = ctx_with(False)
    result = await tools.press_keys(chord, ctx=ctx)
    assert not result.ok and "declined" in result.content and desk.actions() == [] and len(asker.asked) == 1
    ctx, asker = ctx_with(True)
    assert (await tools.press_keys(chord, ctx=ctx)).ok and len(desk.actions()) == 1


async def test_enter_in_a_terminal_asks(desk):
    desk.active = 3
    ctx, asker = ctx_with(False)
    assert not (await tools.press_keys("enter", ctx=ctx)).ok and desk.actions() == []
    assert "Enter" in asker.asked[0]
    assert (await tools.press_keys("ctrl+c", ctx=ctx)).ok, "plain chords in a terminal do not ask"


async def test_enter_on_a_dangerous_focused_control_asks(desk):
    desk.focused = el(0, "button", "Delete")
    ctx, asker = ctx_with(False)
    result = await tools.press_keys("enter", ctx=ctx)
    assert not result.ok and "Delete" in asker.asked[0]
    desk.focused = el(0, "button", "Seven")
    assert (await tools.press_keys("enter", ctx=ctx)).ok


async def test_win_l_goes_to_the_lock_screen_after_asking(desk):
    ctx, asker = ctx_with(False)
    declined = await tools.press_keys("win+l", ctx=ctx)
    assert not declined.ok and desk.lock_calls == 0 and desk.actions() == []
    ctx, asker = ctx_with(True)
    done = await tools.press_keys("win+l", ctx=ctx)
    assert done.ok and desk.lock_calls == 1 and desk.actions() == [] and "Lock the screen" in asker.asked[0]


async def test_ctrl_alt_delete_cannot_be_sent(desk):
    ctx, asker = ctx_with(True)
    result = await tools.press_keys("ctrl+alt+delete", ctx=ctx)
    assert not result.ok and "secure attention" in result.content and asker.asked == []


async def test_keys_cannot_go_to_own_or_elevated_windows_except_system_chords(desk):
    ctx, _ = ctx_with()
    desk.active = 4  # Jarvis's own window
    assert not (await tools.press_keys("ctrl+s", ctx=ctx)).ok
    assert not (await tools.press_keys("enter", ctx=ctx)).ok
    assert (await tools.press_keys("alt+tab", ctx=ctx)).ok, "alt+tab leaves the HUD"
    assert (await tools.press_keys("win+d", ctx=ctx)).ok
    desk.active = 5  # an elevated window cannot be reached even by system chords
    assert not (await tools.press_keys("alt+tab", ctx=ctx)).ok


# ---------------------------------------------------------------- scroll
async def test_scroll_over_an_element_and_over_the_window(desk):
    await tools.desktop_inspect()
    result = await tools.desktop_scroll("down", 5, ref=1)
    assert result.ok and desk.actions() == [("scroll", 35, 25, "down", 5)] and "desktop_inspect again" in result.content
    desk.log.clear()
    await tools.desktop_scroll("up")
    assert desk.actions() == [("scroll", 200, 150, "up", 3)]


async def test_scroll_validates(desk):
    assert not (await tools.desktop_scroll("down", 0)).ok and not (await tools.desktop_scroll("down", 51)).ok
    desk.active = 5
    assert not (await tools.desktop_scroll("down")).ok


# ---------------------------------------------------------------- windows
async def test_focus_window_matches_like_close_app(desk):
    result = await tools.focus_window("notepad")
    assert result.ok and "Untitled - Notepad" in result.content and desk.active == 2


async def test_focus_window_failure_names_the_window_in_front(desk):
    desk.focus_works = False
    result = await tools.focus_window("notepad")
    assert not result.ok and "foreground window is 'Calculator'" in result.content
    elevated = await tools.focus_window("regedit")
    assert not elevated.ok and "administrator" in elevated.content


async def test_focus_window_unknown_app(desk):
    result = await tools.focus_window("photoshop")
    assert not result.ok and "Open windows" in result.content


async def test_get_active_window_flags(desk):
    plain = await tools.get_active_window()
    assert plain.ok and "'Calculator'" in plain.content and "[" not in plain.content
    desk.active = 3
    assert "[terminal]" in (await tools.get_active_window()).content
    desk.active = 4
    assert "Jarvis's own window" in (await tools.get_active_window()).content
    desk.active = 5
    assert "administrator window" in (await tools.get_active_window()).content
    desk.active = 99
    assert "no foreground window" in (await tools.get_active_window()).content


# ---------------------------------------------------------------- registry
async def test_desktop_tools_through_the_registry(desk):
    registry = ToolRegistry().load_builtin()
    specs = {s.name: s for s in registry.specs(["desktop"])}
    assert set(specs) == {
        "desktop_inspect", "desktop_click", "desktop_type", "press_keys", "desktop_scroll", "focus_window", "get_active_window",
    }
    assert len(specs) <= 8
    for name in ("desktop_click", "desktop_type", "press_keys"):
        assert specs[name].timeout == 180, "tools that ask the user need time for the answer"
    ctx, _ = ctx_with()
    inspected = await registry.execute(ToolCall(id="1", name="desktop_inspect", arguments={}), ctx)
    assert inspected.ok
    clicked = await registry.execute(ToolCall(id="2", name="desktop_click", arguments={"ref": 1}), ctx)
    assert clicked.ok
    bad = await registry.execute(ToolCall(id="3", name="press_keys", arguments={"keys": "enter", "extra": 1}), ctx)
    assert not bad.ok and "invalid arguments" in bad.content
    schema = specs["desktop_click"].parameters
    assert set(schema["properties"]) == {"ref", "text", "double"}
