"""The UI Automation layer against a fake element tree (runs on any OS, no COM, no UI).

The fake mimics the small part of the comtypes API that jarvis.platform.windows_uia uses:
a cached subtree, cached and current properties, patterns reached through QueryInterface.
These tests prove the walking, filtering, ref and pattern-selection logic; they say nothing
about how a real application's accessibility tree behaves (that was checked by hand on
Calculator and Chrome, see the README)."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from jarvis.platform import windows_input as wi
from jarvis.platform import windows_uia as uia
from jarvis.platform.base import ElementGone
from jarvis.platform.keys import MAIN_KEYS, MODIFIERS
from jarvis.platform.windows_uia import UiaSession, _Meta, choose_action

# UIA control type ids (the real values)
BUTTON, CHECKBOX, EDIT, LISTITEM, TEXT, PANE, WINDOW, CUSTOM, LINK, TABITEM = 50000, 50002, 50004, 50007, 50020, 50033, 50032, 50025, 50005, 50019
E_GONE = -2147220991


class ComError(Exception):
    def __init__(self, hresult):
        super().__init__(hresult)
        self.hresult = hresult


_ids = itertools.count(1)
_PROPS = (
    "ControlType Name AutomationId BoundingRectangle IsOffscreen IsEnabled IsPassword RuntimeId "
    "IsInvokePatternAvailable IsTogglePatternAvailable IsSelectionItemPatternAvailable "
    "IsExpandCollapsePatternAvailable IsValuePatternAvailable ValueValue ValueIsReadOnly "
    "ToggleToggleState SelectionItemIsSelected ExpandCollapseExpandCollapseState"
).split()


class FakeUi:
    TreeScope_Subtree = 7
    UIA_InvokePatternId, UIA_TogglePatternId, UIA_SelectionItemPatternId = 10000, 10015, 10010
    UIA_ExpandCollapsePatternId, UIA_ValuePatternId = 10005, 10002
    IUIAutomationInvokePattern, IUIAutomationTogglePattern = "Invoke", "Toggle"
    IUIAutomationSelectionItemPattern, IUIAutomationExpandCollapsePattern = "Select", "Expand"
    IUIAutomationValuePattern = "Value"


for _i, _name in enumerate(_PROPS):
    setattr(FakeUi, f"UIA_{_name}PropertyId", 30000 + _i)
PID = {name: getattr(FakeUi, f"UIA_{name}PropertyId") for name in _PROPS}


class Pattern:
    def __init__(self, node, kind):
        self.node, self.kind = node, kind

    def QueryInterface(self, iface):
        assert iface == self.kind
        return self

    def Invoke(self):
        self.node.calls.append("invoke")
        if self.node.fail == "invoke":
            raise ComError(-2147024809)

    def Toggle(self):
        self.node.calls.append("toggle")

    def Select(self):
        self.node.calls.append("select")

    def Expand(self):
        self.node.calls.append("expand")

    def Collapse(self):
        self.node.calls.append("collapse")

    CurrentExpandCollapseState = 0

    def SetValue(self, text):
        self.node.calls.append(f"set:{text}")
        self.node.value = text


class Node:
    def __init__(self, ctype=BUTTON, name="", aid="", rect=(10, 10, 110, 40), offscreen=False, enabled=True,
                 password=False, rid=None, children=(), invoke=False, toggle=None, select=None, expand=None,
                 value=None, readonly=False, fail=None):
        self.ctype, self.name, self.aid, self.rect = ctype, name, aid, rect
        self.offscreen, self.enabled, self.password = offscreen, enabled, password
        self.rid = rid if rid is not None else (42, next(_ids))
        self.children = list(children)
        self.invoke, self.toggle, self.select, self.expand = invoke, toggle, select, expand
        self.value, self.readonly, self.fail = value, readonly, fail
        self.calls: list[str] = []
        self.gone = False
        self.focused = False
        self.clickable = None

    # cached view
    @property
    def CachedControlType(self):
        return self.ctype

    @property
    def CachedName(self):
        return self.name

    @property
    def CachedAutomationId(self):
        return self.aid

    @property
    def CachedBoundingRectangle(self):
        return SimpleNamespace(left=self.rect[0], top=self.rect[1], right=self.rect[2], bottom=self.rect[3])

    @property
    def CachedIsOffscreen(self):
        return self.offscreen

    @property
    def CachedIsEnabled(self):
        return self.enabled

    @property
    def CachedIsPassword(self):
        return self.password

    def GetCachedPropertyValue(self, pid):
        return self._value(pid)

    def _value(self, pid):
        table = {
            PID["RuntimeId"]: self.rid,
            PID["IsInvokePatternAvailable"]: self.invoke,
            PID["IsTogglePatternAvailable"]: self.toggle is not None,
            PID["IsSelectionItemPatternAvailable"]: self.select is not None,
            PID["IsExpandCollapsePatternAvailable"]: self.expand is not None,
            PID["IsValuePatternAvailable"]: self.value is not None,
            PID["ValueValue"]: self.value if self.value is not None else object(),  # 'not supported' marker
            PID["ValueIsReadOnly"]: self.readonly,
            PID["ToggleToggleState"]: self.toggle or 0,
            PID["SelectionItemIsSelected"]: bool(self.select),
            PID["ExpandCollapseExpandCollapseState"]: self.expand or 0,
        }
        return table[pid]

    def GetCachedChildren(self):
        if not self.children:
            return None
        kids = self.children
        return SimpleNamespace(Length=len(kids), GetElement=lambda i: kids[i])

    # live view
    def _check(self):
        if self.gone:
            raise ComError(E_GONE)

    @property
    def CurrentName(self):
        self._check()
        return self.name

    @property
    def CurrentIsEnabled(self):
        self._check()
        return self.enabled

    @property
    def CurrentIsPassword(self):
        self._check()
        return self.password

    @property
    def CurrentAutomationId(self):
        self._check()
        return self.aid

    @property
    def CurrentBoundingRectangle(self):
        return self.CachedBoundingRectangle

    def GetCurrentPropertyValue(self, pid):
        self._check()
        return self._value(pid)

    def GetCurrentPattern(self, pattern_id):
        self._check()
        kinds = {10000: "Invoke", 10015: "Toggle", 10010: "Select", 10005: "Expand", 10002: "Value"}
        return Pattern(self, kinds[pattern_id])

    def SetFocus(self):
        self._check()
        self.focused = True

    def GetClickablePoint(self):
        return (SimpleNamespace(x=self.clickable[0], y=self.clickable[1]), True) if self.clickable else (None, False)

    def BuildUpdatedCache(self, request):
        return self


class FakeUia:
    RawViewCondition = object()

    def __init__(self, windows):
        self.windows = windows

    def ElementFromHandle(self, hwnd):
        return self.windows[hwnd]


def make_session(windows):
    session = UiaSession()
    session._uia, session._ui, session._cache_request = FakeUia(windows), FakeUi(), object()
    session.guarded = lambda fn, timeout=0: (True, fn())  # run actions inline: no helper thread, no COM
    return session


def window(*children):
    return Node(WINDOW, "Calc", children=children)


def snap(session, hwnd=1):
    return session.run(session.snapshot, hwnd)


# ---------------------------------------------------------------- what is listed
def test_only_useful_visible_elements_are_listed():
    tree = window(
        Node(PANE, "layout"),  # a container, not interactive
        Node(BUTTON, "Seven", aid="num7Button", invoke=True),
        Node(BUTTON, "Hidden", offscreen=True, invoke=True),  # scrolled out of view
        Node(BUTTON, "Flat", rect=(10, 10, 10, 40), invoke=True),  # zero width
        Node(BUTTON, "Collapsed", rect=(0, 0, 0, 0), invoke=True),
        Node(TEXT, "Display is 7", aid="CalculatorResults"),  # a readout
        Node(TEXT, "Just a caption"),  # plain text is not listed
        Node(CUSTOM, "Fancy tile", invoke=True),  # custom control that can be invoked
        Node(CUSTOM, "", invoke=True),  # ... but nameless ones are not
        Node(CUSTOM, "Decor"),  # custom without any pattern
        Node(EDIT, "Search", value=""),
    )
    session = make_session({1: tree})
    elements, total = snap(session)
    assert [(e.role, e.name) for e in elements] == [
        ("button", "Seven"), ("text", "Display is 7"), ("custom", "Fancy tile"), ("edit", "Search"),
    ]
    assert total == 12  # the window plus eleven children


def test_disabled_elements_are_listed_but_marked():
    session = make_session({1: window(Node(BUTTON, "Memory recall", enabled=False, invoke=True))})
    (element,), _ = snap(session)
    assert element.enabled is False and element.rect == (10, 10, 110, 40)


def test_nested_children_come_in_document_order():
    tree = window(
        Node(PANE, "toolbar", children=[Node(BUTTON, "One", invoke=True), Node(BUTTON, "Two", invoke=True)]),
        Node(BUTTON, "Three", invoke=True),
    )
    elements, _ = snap(make_session({1: tree}))
    assert [e.name for e in elements] == ["One", "Two", "Three"]


def test_states_and_values():
    tree = window(
        Node(CHECKBOX, "Dark mode", toggle=1),
        Node(CHECKBOX, "Notifications", toggle=0),
        Node(LISTITEM, "Inbox", select=True),
        Node(LISTITEM, "Sent", select=False),
        Node(BUTTON, "Menu", expand=1, invoke=True),
        Node(EDIT, "Title", value="Hello"),
        Node(EDIT, "Read only box", value="x", readonly=True),
    )
    by_name = {e.name: e for e in snap(make_session({1: tree}))[0]}
    assert by_name["Dark mode"].state == "on" and by_name["Notifications"].state == "off"
    assert by_name["Inbox"].state == "selected" and by_name["Sent"].state == ""
    assert by_name["Menu"].state == "expanded"
    assert by_name["Title"].value == "Hello"
    assert "read-only" in by_name["Read only box"].state


def test_password_and_credential_values_are_never_returned():
    tree = window(
        Node(EDIT, "Secret", password=True, value="hunter2"),
        Node(EDIT, "Card number", value="4111111111111111"),
        Node(EDIT, "Empty password", aid="passwordBox", value=""),
        Node(EDIT, "Nickname", value="Tony"),
    )
    by_name = {e.name: e for e in snap(make_session({1: tree}))[0]}
    assert by_name["Secret"].value == "(hidden)" and by_name["Secret"].is_password
    assert by_name["Card number"].value == "(hidden)"
    assert by_name["Empty password"].value == ""
    assert by_name["Nickname"].value == "Tony"


def test_long_names_and_whitespace_are_tidied():
    session = make_session({1: window(Node(BUTTON, "  Save \n  everything  " + "x" * 300, invoke=True))})
    (element,), _ = snap(session)
    assert element.name.startswith("Save everything xxx") and len(element.name) <= 120


# ---------------------------------------------------------------- refs
def test_refs_are_stable_across_snapshots_and_survive_reordering():
    seven, plus = Node(BUTTON, "Seven", invoke=True), Node(BUTTON, "Plus", invoke=True)
    tree = window(seven, plus)
    session = make_session({1: tree})
    first = {e.name: e.ref for e in snap(session)[0]}
    assert first == {"Seven": 1, "Plus": 2}
    tree.children.reverse()
    fresh = Node(BUTTON, "Equals", invoke=True)
    tree.children.append(fresh)
    second = {e.name: e.ref for e in snap(session)[0]}
    assert second["Seven"] == 1 and second["Plus"] == 2, "an element keeps its number while it exists"
    assert second["Equals"] == 3, "a new element gets the next number"


def test_refs_are_per_window():
    a, b = Node(BUTTON, "A", invoke=True), Node(BUTTON, "B", invoke=True)
    session = make_session({1: window(a), 2: window(b)})
    assert snap(session, 1)[0][0].ref == 1 and snap(session, 2)[0][0].ref == 1


def test_a_vanished_element_cannot_be_acted_on():
    seven, plus = Node(BUTTON, "Seven", invoke=True), Node(BUTTON, "Plus", invoke=True)
    tree = window(seven, plus)
    session = make_session({1: tree})
    snap(session)
    tree.children.remove(seven)
    snap(session)
    with pytest.raises(ElementGone, match="desktop_inspect"):
        session.run(session.read, 1, 1)
    assert session.run(session.read, 1, 2).name == "Plus"
    with pytest.raises(ElementGone):
        session.run(session.read, 99, 1)  # a window that was never inspected


def test_an_element_destroyed_after_the_snapshot_is_reported_gone():
    node = Node(BUTTON, "Seven", invoke=True)
    session = make_session({1: window(node)})
    snap(session)
    node.gone = True
    with pytest.raises(ElementGone):
        session.run(session.read, 1, 1)
    with pytest.raises(ElementGone):
        session.run(session.act, 1, 1, "click", lambda *a: None)


def test_elements_without_a_runtime_id_still_get_stable_refs():
    first = window(Node(BUTTON, "Save", invoke=True, rid=()))
    session = make_session({1: first})
    assert snap(session)[0][0].ref == 1
    assert snap(session)[0][0].ref == 1


# ---------------------------------------------------------------- read back
def test_read_reports_live_state():
    box = Node(CHECKBOX, "Dark mode", toggle=0)
    session = make_session({1: window(box)})
    snap(session)
    assert session.run(session.read, 1, 1).state == "off"
    box.toggle = 1
    assert session.run(session.read, 1, 1).state == "on"
    box.enabled = False
    assert session.run(session.read, 1, 1).enabled is False


def test_read_of_a_password_field_hides_the_value():
    field = Node(EDIT, "Secret", password=True, value="hunter2")
    session = make_session({1: window(field)})
    snap(session)
    assert session.run(session.read, 1, 1).value == "(hidden)"


# ---------------------------------------------------------------- how to act
@pytest.mark.parametrize(
    "meta,expected",
    [
        (_Meta("button", invoke=True), ["invoke"]),
        (_Meta("button", invoke=True, toggle=True), ["invoke", "toggle"]),
        (_Meta("checkbox", toggle=True), ["toggle"]),
        (_Meta("list item", select=True), ["select"]),
        (_Meta("list item", select=True, invoke=True), ["invoke", "select"]),
        (_Meta("combobox", expand=True), ["expand"]),
        (_Meta("tab item", select=True, expand=True), ["select", "expand"]),
        (_Meta("edit", value=True), []),  # a text box has no click pattern: fall back to a real click
        (_Meta("button"), []),
    ],
)
def test_pattern_order(meta, expected):
    assert choose_action(meta) == expected


def clicks():
    log = []
    return log, lambda x, y, double: log.append((x, y, double))


def act(session, ref=1, action="click", real=None):
    return session.run(session.act, 1, ref, action, real or (lambda *a: None))


def test_invoke_is_preferred_over_a_real_click():
    button = Node(BUTTON, "Seven", invoke=True)
    session = make_session({1: window(button)})
    snap(session)
    log, real = clicks()
    assert act(session, real=real) == "invoke"
    assert button.calls == ["invoke"] and log == []


def test_toggle_select_and_expand_are_used_when_nothing_invokes():
    nodes = [Node(CHECKBOX, "Dark", toggle=0), Node(LISTITEM, "Inbox", select=False), Node(BUTTON, "Menu", expand=0, aid="m")]
    session = make_session({1: window(*nodes)})
    snap(session)
    assert [act(session, i + 1) for i in range(3)] == ["toggle", "select", "expand"]
    assert [n.calls for n in nodes] == [["toggle"], ["select"], ["expand"]]


def test_a_real_click_at_the_centre_when_no_pattern_applies():
    edit = Node(EDIT, "Search", rect=(100, 200, 300, 240), value="")
    session = make_session({1: window(edit)})
    snap(session)
    log, real = clicks()
    assert act(session, real=real) == "click"
    assert log == [(200, 220, False)]


def test_the_clickable_point_is_preferred_to_the_centre():
    edit = Node(EDIT, "Search", rect=(100, 200, 300, 240), value="")
    edit.clickable = (130, 215)
    session = make_session({1: window(edit)})
    snap(session)
    log, real = clicks()
    act(session, real=real)
    assert log == [(130, 215, False)]


def test_double_click_is_always_a_real_click():
    button = Node(LISTITEM, "report.docx", select=False, invoke=True)
    session = make_session({1: window(button)})
    snap(session)
    log, real = clicks()
    assert act(session, action="double_click", real=real) == "double click"
    assert log == [(60, 25, True)] and button.calls == []


def test_a_failing_pattern_falls_through_to_the_next_then_to_a_click():
    button = Node(BUTTON, "Flaky", invoke=True, select=False, fail="invoke")
    session = make_session({1: window(button)})
    snap(session)
    log, real = clicks()
    assert act(session, real=real) == "select"  # invoke raised, select worked
    only_invoke = Node(BUTTON, "Flaky2", invoke=True, fail="invoke")
    session = make_session({1: window(only_invoke)})
    snap(session)
    log, real = clicks()
    assert act(session, real=real) == "click" and len(log) == 1


def test_a_pattern_that_blocks_is_reported_not_hung():
    button = Node(BUTTON, "Save as...", invoke=True)
    session = make_session({1: window(button)})
    snap(session)
    session.guarded = lambda fn, timeout=0: (False, None)  # the helper thread did not return in time
    result = act(session)
    assert result.startswith("invoke (sent") and "dialog" in result


def test_disabled_elements_are_not_clicked():
    button = Node(BUTTON, "Memory recall", enabled=False, invoke=True)
    session = make_session({1: window(button)})
    snap(session)
    with pytest.raises(OSError, match="disabled"):
        act(session)
    assert button.calls == []


def test_focus_and_clear():
    edit = Node(EDIT, "Search", value="old text")
    ro = Node(EDIT, "Locked", value="x", readonly=True)
    session = make_session({1: window(edit, ro)})
    snap(session)
    assert session.run(session.focus, 1, 1, lambda *a: None) == "focused" and edit.focused
    assert session.run(session.clear, 1, 1) is True and edit.value == ""
    assert session.run(session.clear, 1, 2) is False, "a read-only field cannot be cleared"


def test_focus_falls_back_to_a_click():
    class Stubborn(Node):
        def SetFocus(self):
            raise ComError(-2147467259)

    edit = Stubborn(EDIT, "Search", rect=(0, 0, 100, 20), value="")
    session = make_session({1: window(edit)})
    snap(session)
    log, real = clicks()
    assert session.run(session.focus, 1, 1, real) == "clicked to focus" and log == [(50, 10, False)]


# ---------------------------------------------------------------- input synthesis (pure parts)
def test_text_events_are_unicode_code_units_with_enter_and_tab_as_keys():
    events = wi.text_events("a\nb\t")
    kinds = [(e.ki.wVk, e.ki.wScan, bool(e.ki.dwFlags & wi.KEYEVENTF_UNICODE), bool(e.ki.dwFlags & wi.KEYEVENTF_KEYUP)) for e in events]
    assert kinds == [
        (0, ord("a"), True, False), (0, ord("a"), True, True),
        (wi.VK_RETURN, 0, False, False), (wi.VK_RETURN, 0, False, True),
        (0, ord("b"), True, False), (0, ord("b"), True, True),
        (wi.VK_TAB, 0, False, False), (wi.VK_TAB, 0, False, True),
    ]


def test_characters_outside_the_bmp_are_sent_as_surrogate_pairs():
    events = wi.text_events("\U0001F600")
    scans = [e.ki.wScan for e in events if not e.ki.dwFlags & wi.KEYEVENTF_KEYUP]
    assert scans == [0xD83D, 0xDE00]


def test_crlf_is_one_enter():
    assert len(wi.text_events("x\r\ny")) == 6


def test_chord_events_press_in_order_and_release_in_reverse():
    events = wi.chord_events(["ctrl", "shift", "t"])
    seq = [(e.ki.wVk, bool(e.ki.dwFlags & wi.KEYEVENTF_KEYUP)) for e in events]
    assert seq == [(0x11, False), (0x10, False), (ord("T"), False), (ord("T"), True), (0x10, True), (0x11, True)]


def test_extended_keys_carry_the_extended_flag():
    (down, up) = wi.chord_events(["delete"])
    assert down.ki.dwFlags & wi.KEYEVENTF_EXTENDEDKEY and up.ki.dwFlags & wi.KEYEVENTF_EXTENDEDKEY
    (plain, _) = wi.chord_events(["enter"])
    assert not plain.ki.dwFlags & wi.KEYEVENTF_EXTENDEDKEY


def test_plus_adds_shift_when_it_is_not_already_held():
    names = [e.ki.wVk for e in wi.chord_events(["ctrl", "plus"]) if not e.ki.dwFlags & wi.KEYEVENTF_KEYUP]
    assert names == [0x11, 0x10, 0xBB]


def test_every_parseable_key_has_a_virtual_key_code():
    for name in (*MODIFIERS, *MAIN_KEYS):
        assert name in wi.VK, name
    with pytest.raises(ValueError):
        wi.chord_events(["nonsense"])
