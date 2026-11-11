"""Windows UI Automation through ``comtypes`` against UIAutomationCore.

Why comtypes directly. Measured on the dev machine (Python 3.13, Windows 11), enumerating
a whole window, seconds per enumeration after a warm-up, plus import time:

    library          import   Calculator          Chrome window
    uiautomation     0.3-0.5  0.08 (126 el.)       0.35 (300 el., raw view)
    pywinauto (uia)  1.0-1.2  0.07 (73 el.)        0.19 (76 el.: control view only)
    comtypes direct  0.5      0.03-0.10 (73-126)   0.07-0.09 (300 el., raw view)

(The element counts differ because the control view hides most of Chrome's page content
and Calculator's raw tree is larger; comtypes was timed in both views.) With the library
choice made, the final implementation measures 0.08-0.10 s for Calculator (126 raw elements,
56 listed) and 0.22 s for a Chrome window with a loaded page (640 raw, 128 listed); the
first call in a process takes about 1 s (type library wrapper, COM start-up). comtypes is
one pure-Python package that uiautomation itself builds on; pywinauto is slower to import
and adds pywin32 and six.

The fast path is one ``BuildUpdatedCache`` call with ``TreeScope_Subtree``: UIA fetches the
whole subtree and every property we read in one cross-process round trip, and the walk is
then local. The control view hides most of Chrome's page content, so the raw view is used.

All UIA objects live on one dedicated COM thread (multithreaded apartment); callers on any
thread get results through a future. Slow actions run on a helper thread with a timeout so
an element that opens a modal dialog from ``Invoke`` cannot hang the tools.

Refs: each window handle has a table mapping an element's UIA RuntimeId to a small integer.
The same element keeps its number for as long as it exists; a new element gets the next one.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from jarvis.browser.policy import credential_reason
from jarvis.platform.base import ElementGone, UiElement

log = logging.getLogger(__name__)

MAX_NODES = 6000
MAX_DEPTH = 60
CALL_TIMEOUT = 60.0
ACTION_TIMEOUT = 4.0
E_ELEMENTNOTAVAILABLE = -2147220991  # 0x80040201 as a signed 32-bit int
E_NOTSUPPORTED = -2147220992  # 0x80040200

# ControlType id -> label shown to the model
_ROLE_NAMES = {
    50000: "button", 50001: "calendar", 50002: "checkbox", 50003: "combobox", 50004: "edit",
    50005: "link", 50006: "image", 50007: "list item", 50008: "list", 50009: "menu",
    50010: "menu bar", 50011: "menu item", 50012: "progress bar", 50013: "radio button",
    50014: "scroll bar", 50015: "slider", 50016: "spinner", 50017: "status bar", 50018: "tab",
    50019: "tab item", 50020: "text", 50021: "toolbar", 50022: "tooltip", 50023: "tree",
    50024: "tree item", 50025: "custom", 50026: "group", 50027: "thumb", 50028: "data grid",
    50029: "data item", 50030: "document", 50031: "split button", 50032: "window", 50033: "pane",
    50034: "header", 50035: "header item", 50036: "table", 50037: "title bar", 50038: "separator",
}
# Roles a person would click or type into, whatever patterns they report.
_INTERACTIVE = {
    "button", "checkbox", "combobox", "edit", "link", "list item", "menu item",
    "radio button", "slider", "spinner", "tab item", "tree item", "split button", "data item",
}
_EDITABLE = {"edit", "combobox", "spinner"}
_WS = re.compile(r"\s+")


def clean(text: Any, limit: int = 120) -> str:
    value = _WS.sub(" ", str(text or "")).strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


@dataclass
class _Meta:
    """What the snapshot learned about one element, kept for choosing how to act."""

    role: str
    invoke: bool = False
    toggle: bool = False
    select: bool = False
    expand: bool = False
    value: bool = False
    read_only: bool = False


@dataclass
class _WindowRefs:
    next_ref: int = 1
    by_key: dict[Any, int] = field(default_factory=dict)
    elements: dict[int, Any] = field(default_factory=dict)
    meta: dict[int, _Meta] = field(default_factory=dict)

    def ref_for(self, key: Any) -> int:
        ref = self.by_key.get(key)
        if ref is None:
            ref = self.next_ref
            self.next_ref += 1
            self.by_key[key] = ref
        return ref


def toggle_text(state: int) -> str:
    return {0: "off", 1: "on", 2: "indeterminate"}.get(state, "")


def expand_text(state: int) -> str:
    return {0: "collapsed", 1: "expanded", 2: "partly expanded"}.get(state, "")


def choose_action(meta: _Meta) -> list[str]:
    """Order in which UI Automation patterns are tried for a click; a real click is the
    last resort and is added by the caller. Pure, so it is table-tested."""
    order: list[str] = []
    if meta.invoke:
        order.append("invoke")
    if meta.toggle:
        order.append("toggle")
    if meta.select:
        order.append("select")
    if meta.expand:
        order.append("expand")
    return order


class UiaSession:
    """The COM thread plus per-window ref tables. One per process."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-uia")
        self._wins: dict[int, _WindowRefs] = {}
        self._uia: Any = None
        self._ui: Any = None  # comtypes.gen.UIAutomationClient
        self._cache_request: Any = None
        self._ready_error: Exception | None = None

    # -- thread plumbing ----------------------------------------------------------
    def run(self, fn: Callable[..., Any], *args: Any, timeout: float = CALL_TIMEOUT) -> Any:
        """Run ``fn`` on the COM thread and wait for it."""
        return self._pool.submit(self._guarded_init, fn, *args).result(timeout)

    def _guarded_init(self, fn: Callable[..., Any], *args: Any) -> Any:
        self._ensure_ready()
        return fn(*args)

    def _ensure_ready(self) -> None:
        if self._uia is not None:
            return
        if self._ready_error is not None:
            raise self._ready_error
        try:
            # comtypes initialises COM on import using sys.coinit_flags; 0 = multithreaded
            # apartment, so elements can also be used from the short helper threads below.
            if not hasattr(sys, "coinit_flags"):
                sys.coinit_flags = 0  # type: ignore[attr-defined]
            import comtypes
            import comtypes.client

            try:  # comtypes only initialises the thread that first imported it; this is another
                comtypes.CoInitializeEx(0)
            except OSError:
                pass  # this thread is already initialised (possibly in another apartment mode)
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as ui

            self._ui = ui
            self._uia = comtypes.client.CreateObject(
                "{ff48dba4-60ef-4201-aa87-54103eef594e}", interface=ui.IUIAutomation
            )
            request = self._uia.CreateCacheRequest()
            for name in (
                "ControlType", "Name", "AutomationId", "BoundingRectangle", "IsOffscreen",
                "IsEnabled", "IsPassword", "RuntimeId",
                "IsInvokePatternAvailable", "IsTogglePatternAvailable",
                "IsSelectionItemPatternAvailable", "IsExpandCollapsePatternAvailable",
                "IsValuePatternAvailable", "ValueValue", "ValueIsReadOnly", "ToggleToggleState",
                "SelectionItemIsSelected", "ExpandCollapseExpandCollapseState",
            ):
                request.AddProperty(getattr(ui, f"UIA_{name}PropertyId"))
            request.TreeScope = ui.TreeScope_Subtree
            request.TreeFilter = self._uia.RawViewCondition
            self._cache_request = request
        except Exception as exc:
            self._ready_error = OSError(f"UI Automation is not available: {exc}")
            raise self._ready_error from exc

    def guarded(self, fn: Callable[[], Any], timeout: float = ACTION_TIMEOUT) -> tuple[bool, Any]:
        """Run ``fn`` on a helper thread; (finished, result). An exception inside is re-raised.
        If it does not finish in time the thread is left to finish by itself."""
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                import comtypes

                comtypes.CoInitializeEx()
                try:
                    box["value"] = fn()
                finally:
                    comtypes.CoUninitialize()
            except BaseException as exc:  # carried back to the caller
                box["error"] = exc

        thread = threading.Thread(target=work, name="jarvis-uia-action", daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return False, None
        if "error" in box:
            raise box["error"]
        return True, box.get("value")

    # -- reading ----------------------------------------------------------------
    @staticmethod
    def _plain(value: Any) -> Any:
        """A property value, or None when UIA reported 'not supported' (an IUnknown marker)."""
        return value if isinstance(value, (str, int, float, bool, tuple)) else None

    def _prop(self, element: Any, name: str) -> Any:
        try:
            return self._plain(element.GetCachedPropertyValue(getattr(self._ui, f"UIA_{name}PropertyId")))
        except Exception:
            return None

    def snapshot(self, hwnd: int) -> tuple[list[UiElement], int]:
        root = self._uia.ElementFromHandle(hwnd)
        top = root.BuildUpdatedCache(self._cache_request)
        refs = self._wins.setdefault(hwnd, _WindowRefs())
        refs.elements.clear()
        refs.meta.clear()
        seen: set[int] = set()
        result: list[UiElement] = []
        total = 0
        stack: list[tuple[Any, int]] = [(top, 0)]
        while stack and total < MAX_NODES:
            node, depth = stack.pop()
            total += 1
            try:
                role = _ROLE_NAMES.get(int(node.CachedControlType), "element")
            except Exception:
                continue
            if depth < MAX_DEPTH:
                try:
                    kids = node.GetCachedChildren()
                except Exception:
                    kids = None
                if kids:
                    count = int(kids.Length)
                    stack.extend((kids.GetElement(i), depth + 1) for i in range(count - 1, -1, -1))
            if depth == 0:
                continue  # the window itself
            name = clean(self._safe(node, "CachedName"))
            automation_id = clean(self._safe(node, "CachedAutomationId"), 80)
            flags = _Meta(
                role=role,
                invoke=bool(self._prop(node, "IsInvokePatternAvailable")),
                toggle=bool(self._prop(node, "IsTogglePatternAvailable")),
                select=bool(self._prop(node, "IsSelectionItemPatternAvailable")),
                expand=bool(self._prop(node, "IsExpandCollapsePatternAvailable")),
                value=bool(self._prop(node, "IsValuePatternAvailable")),
                read_only=bool(self._prop(node, "ValueIsReadOnly")),
            )
            actionable = bool(choose_action(flags))
            readout = role == "text" and bool(automation_id) and bool(name)
            if not (role in _INTERACTIVE or readout or (actionable and name)):
                continue
            if self._safe(node, "CachedIsOffscreen"):
                continue
            rect = self._rect(node)
            if rect is None or rect[2] - rect[0] <= 0 or rect[3] - rect[1] <= 0:
                continue
            password = bool(self._safe(node, "CachedIsPassword"))
            value = ""
            if flags.value:
                raw = self._prop(node, "ValueValue")
                value = clean(raw if isinstance(raw, str) else "", 200)
            state = self._state(node, flags)
            element = UiElement(
                ref=0, role=role, name=name, value=value, state=state,
                enabled=bool(self._safe(node, "CachedIsEnabled", True)),
                automation_id=automation_id, is_password=password, rect=rect,
            )
            if password or credential_reason(_policy_dict(element)):
                element.value = "(hidden)" if (element.value or password) else ""
            key = self._runtime_key(node, element)
            ref = refs.ref_for(key)
            if ref in seen:  # two nodes with one identity: keep the first
                continue
            seen.add(ref)
            element.ref = ref
            refs.elements[ref] = node
            refs.meta[ref] = flags
            result.append(element)
        return result, total

    def _runtime_key(self, node: Any, element: UiElement) -> Any:
        runtime = self._prop(node, "RuntimeId")
        if isinstance(runtime, tuple) and runtime:
            return ("rid", runtime)
        return ("fallback", element.role, element.name, element.automation_id, element.rect)

    @staticmethod
    def _safe(node: Any, attr: str, default: Any = "") -> Any:
        try:
            return getattr(node, attr)
        except Exception:
            return default

    @staticmethod
    def _rect(node: Any) -> tuple[int, int, int, int] | None:
        try:
            r = node.CachedBoundingRectangle
            return int(r.left), int(r.top), int(r.right), int(r.bottom)
        except Exception:
            return None

    def _state(self, node: Any, flags: _Meta) -> str:
        parts: list[str] = []
        if flags.toggle:
            parts.append(toggle_text(int(self._prop(node, "ToggleToggleState") or 0)))
        if flags.select and self._prop(node, "SelectionItemIsSelected"):
            parts.append("selected")
        if flags.expand:
            parts.append(expand_text(int(self._prop(node, "ExpandCollapseExpandCollapseState") or 0)))
        if flags.value and flags.read_only and flags.role in _EDITABLE:
            parts.append("read-only")
        return ", ".join(p for p in parts if p)

    # -- acting -----------------------------------------------------------------
    def element(self, hwnd: int, ref: int) -> tuple[Any, _Meta]:
        refs = self._wins.get(hwnd)
        if refs is None or ref not in refs.elements:
            raise ElementGone(
                f"there is no element {ref} in that window right now. Run desktop_inspect again for fresh refs."
            )
        return refs.elements[ref], refs.meta[ref]

    def _gone(self, exc: Exception) -> bool:
        code = getattr(exc, "hresult", None)
        if code is None and getattr(exc, "args", None):
            code = exc.args[0] if isinstance(exc.args[0], int) else None
        return code == E_ELEMENTNOTAVAILABLE

    def read(self, hwnd: int, ref: int) -> UiElement:
        node, meta = self.element(hwnd, ref)
        ui = self._ui
        try:
            name = clean(node.CurrentName)
            enabled = bool(node.CurrentIsEnabled)
            password = bool(node.CurrentIsPassword)
            automation_id = clean(node.CurrentAutomationId, 80)
            value = ""
            parts: list[str] = []
            if meta.toggle:
                parts.append(toggle_text(int(self._plain(node.GetCurrentPropertyValue(ui.UIA_ToggleToggleStatePropertyId)) or 0)))
            if meta.select and self._plain(node.GetCurrentPropertyValue(ui.UIA_SelectionItemIsSelectedPropertyId)):
                parts.append("selected")
            if meta.expand:
                parts.append(expand_text(int(self._plain(node.GetCurrentPropertyValue(ui.UIA_ExpandCollapseExpandCollapseStatePropertyId)) or 0)))
            if meta.value:
                raw = self._plain(node.GetCurrentPropertyValue(ui.UIA_ValueValuePropertyId))
                value = clean(raw if isinstance(raw, str) else "", 200)
            rect = self._rect_current(node)
        except Exception as exc:
            if self._gone(exc):
                raise ElementGone("that element no longer exists. Run desktop_inspect again.") from exc
            raise
        element = UiElement(
            ref=ref, role=meta.role, name=name, value=value, state=", ".join(p for p in parts if p),
            enabled=enabled, automation_id=automation_id, is_password=password, rect=rect,
        )
        if password or credential_reason(_policy_dict(element)):
            element.value = "(hidden)" if (element.value or password) else ""
        return element

    @staticmethod
    def _rect_current(node: Any) -> tuple[int, int, int, int] | None:
        try:
            r = node.CurrentBoundingRectangle
            return int(r.left), int(r.top), int(r.right), int(r.bottom)
        except Exception:
            return None

    def _pattern(self, node: Any, pattern_id: int, interface: Any) -> Any:
        return node.GetCurrentPattern(pattern_id).QueryInterface(interface)

    def center(self, node: Any) -> tuple[int, int]:
        try:
            point, ok = node.GetClickablePoint()
            if ok:
                return int(point.x), int(point.y)
        except Exception:
            pass
        rect = self._rect_current(node)
        if rect is None or rect[2] <= rect[0]:
            raise OSError("the element has no on-screen position to click")
        return (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2

    def act(self, hwnd: int, ref: int, action: str, real_click: Callable[[int, int, bool], None]) -> str:
        """Click an element. Returns the method used: invoke, toggle, select, expand, collapse
        or 'click' (a real click at the element's centre)."""
        node, meta = self.element(hwnd, ref)
        ui = self._ui
        try:
            if not node.CurrentIsEnabled:
                raise OSError("the element is disabled")
            if action == "double_click":
                x, y = self.center(node)
                real_click(x, y, True)
                return "double click"
            for method in choose_action(meta):
                try:
                    done, _ = self.guarded(lambda m=method: self._do_pattern(node, m, ui))
                except Exception as exc:
                    if self._gone(exc):
                        raise
                    log.debug("pattern %s failed on element %s: %s", method, ref, exc)
                    continue
                if done:
                    return method
                return f"{method} (sent; the app is still busy, possibly showing a dialog)"
            x, y = self.center(node)
            real_click(x, y, False)
            return "click"
        except OSError:
            raise
        except Exception as exc:
            if self._gone(exc):
                raise ElementGone("that element no longer exists. Run desktop_inspect again.") from exc
            raise

    def _do_pattern(self, node: Any, method: str, ui: Any) -> str:
        if method == "invoke":
            self._pattern(node, ui.UIA_InvokePatternId, ui.IUIAutomationInvokePattern).Invoke()
        elif method == "toggle":
            self._pattern(node, ui.UIA_TogglePatternId, ui.IUIAutomationTogglePattern).Toggle()
        elif method == "select":
            self._pattern(node, ui.UIA_SelectionItemPatternId, ui.IUIAutomationSelectionItemPattern).Select()
        else:
            pattern = self._pattern(node, ui.UIA_ExpandCollapsePatternId, ui.IUIAutomationExpandCollapsePattern)
            if pattern.CurrentExpandCollapseState == 0:
                pattern.Expand()
            else:
                pattern.Collapse()
        return method

    def focus(self, hwnd: int, ref: int, real_click: Callable[[int, int, bool], None]) -> str:
        node, _ = self.element(hwnd, ref)
        try:
            node.SetFocus()
            return "focused"
        except Exception as exc:
            if self._gone(exc):
                raise ElementGone("that element no longer exists. Run desktop_inspect again.") from exc
        x, y = self.center(node)
        real_click(x, y, False)
        return "clicked to focus"

    def clear(self, hwnd: int, ref: int) -> bool:
        node, meta = self.element(hwnd, ref)
        if not meta.value or meta.read_only:
            return False
        ui = self._ui
        try:
            self._pattern(node, ui.UIA_ValuePatternId, ui.IUIAutomationValuePattern).SetValue("")
            return True
        except Exception:
            return False

    def focused(self) -> UiElement | None:
        try:
            node = self._uia.GetFocusedElement()
            role = _ROLE_NAMES.get(int(node.CurrentControlType), "element")
            element = UiElement(
                ref=0, role=role, name=clean(node.CurrentName),
                automation_id=clean(node.CurrentAutomationId, 80),
                enabled=bool(node.CurrentIsEnabled), is_password=bool(node.CurrentIsPassword),
                rect=self._rect_current(node),
            )
            return element
        except Exception:
            return None

    def forget(self, hwnd: int) -> None:
        self._wins.pop(hwnd, None)


def _policy_dict(element: UiElement) -> dict[str, Any]:
    """The element in the shape jarvis.browser.policy understands."""
    return {
        "label": element.name,
        "name": element.automation_id,
        "id": element.automation_id,
        "type": "password" if element.is_password else "",
    }


__all__ = ["UiaSession", "choose_action", "clean"]
