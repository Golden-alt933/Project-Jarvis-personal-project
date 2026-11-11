"""Tests for the formula evaluator (excel_eval) and the Excel COM recalculation (excel_calc).

The COM path is tested with win32com fully mocked. One real-Excel test is marked slow.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook, load_workbook

from jarvis.tools import excel_calc
from jarvis.tools.excel_eval import Evaluator, XlError

# ---------------------------------------------------------------- evaluator
DATA: dict[str, dict[str, Any]] = {
    "Sheet1": {
        "A1": 10, "A2": 20, "A3": 30, "A4": "x", "A5": True,
        "B1": "food", "B2": "rent", "B3": "food", "B4": 5, "B5": "Food2",
        "C1": 100, "C2": -5, "C3": 250, "C4": 0,
        "D1": "=D2", "D2": "=D1",  # a cycle
        "E1": "=A1*2", "E2": "=E1+1",  # nested formulas
        "F1": "#N/A",
    },
    "My Data": {"A1": 7, "A2": 8, "B1": "yes"},
    "Empty": {"A1": None},
}


class DictSource:
    def __init__(self, sheets: dict[str, dict[str, Any]]) -> None:
        self.sheets = sheets

    def dims(self, sheet: str) -> tuple[int, int] | None:
        for name, cells in self.sheets.items():
            if name.casefold() == sheet.casefold():
                rows = [int("".join(c for c in k if c.isdigit())) for k in cells] or [1]
                cols = [ord(k[0]) - 64 for k in cells] or [1]
                return max(rows), max(cols)
        return None

    def cell(self, sheet: str, row: int, col: int) -> tuple[Any, Any]:
        for name, cells in self.sheets.items():
            if name.casefold() == sheet.casefold():
                return cells.get(f"{chr(64 + col)}{row}"), None
        raise KeyError(sheet)


def ev(formula: str) -> Any:
    return Evaluator(DictSource(DATA)).formula(formula, "Sheet1")


CASES: list[tuple[str, Any]] = [
    # literals and arithmetic
    ("=1+2", 3),
    ("=2*3+4", 10),
    ("=2+3*4", 14),
    ("=(2+3)*4", 20),
    ("=10/4", 2.5),
    ("=2^10", 1024),
    ("=-2^2", 4),  # Excel: unary minus binds tighter than ^
    ("=-(3+4)", -7),
    ("=--5", 5),
    ("=50%", 0.5),
    ("=200*10%", 20),
    ("=1.5E2", 150),
    ("=.5+.5", 1),
    # strings, booleans, comparison
    ('="a"&"b"', "ab"),
    ('="n="&A1', "n=10"),
    ('="say ""hi"""', 'say "hi"'),
    ("=TRUE", True),
    ("=1<2", True),
    ("=2<=1", False),
    ("=A1<>10", False),
    ('="abc"="ABC"', True),
    ('=A4="x"', True),
    # references and ranges
    ("=A1+A2", 30),
    ("=$A$1+$A$2", 30),
    ("=SUM(A1:A3)", 60),
    ("=SUM(A1:A3,5)", 65),
    ("=SUM(A1:A4)", 60),  # text in a range is ignored
    ("=SUM(A1:A5)", 60),  # and so are booleans
    ("='My Data'!A1+'My Data'!A2", 15),
    ("=SUM('My Data'!A1:A2)", 15),
    ("=Sheet1!A1*2", 20),
    ("=SUM(A1:A2)+SUM('My Data'!A1:A2)", 45),
    ("=E2", 21),  # nested formula chain
    ("=Z9", 0),  # an empty cell reads as 0
    # functions
    ("=AVERAGE(A1:A3)", 20),
    ("=MIN(A1:A3)", 10),
    ("=MAX(A1:A3,99)", 99),
    ("=MIN(Z1:Z3)", 0),
    ("=COUNT(A1:A5)", 3),
    ("=COUNTA(A1:A5)", 5),
    ("=ROUND(2.5,0)", 3),
    ("=ROUND(-2.5,0)", -3),
    ("=ROUND(3.14159,2)", 3.14),
    ("=ROUND(1234,-2)", 1200),
    ("=ABS(-4.5)", 4.5),
    ('=IF(A1>5,"big","small")', "big"),
    ("=IF(A1>50,1)", False),
    ("=IF(1=1,1,1/0)", 1),  # IF is lazy
    ("=AND(A1>5,A2>5)", True),
    ("=OR(A1>50,A2>50)", False),
    ("=NOT(A1>50)", True),
    ("=sum(a1:a2)", 30),  # case-insensitive
    # conditional aggregation
    ('=SUMIF(C1:C4,">100")', 250),
    ('=SUMIF(B1:B3,"food",A1:A3)', 40),
    ('=COUNTIF(B1:B3,"food")', 2),
    ('=COUNTIF(B1:B5,"food*")', 3),
    ('=COUNTIF(C1:C4,"<>0")', 3),
    ('=COUNTIF(C1:C4,"<0")', 1),
    ("=COUNTIF(A1:A3,20)", 1),
    ('=AVERAGEIF(C1:C4,">0")', 175),
    ('=AVERAGEIF(B1:B3,"food",A1:A3)', 20),
    # errors
    ("=1/0", "#DIV/0!"),
    ("=A1/C4", "#DIV/0!"),
    ("=AVERAGE(Z1:Z3)", "#DIV/0!"),
    ("=0^-1", "#DIV/0!"),
    ("=Missing!A1", "#REF!"),
    ("=D1", "#REF!"),  # circular
    ("=1+D1", "#REF!"),
    ('="a"+1', "#VALUE!"),
    ("=A4*2", "#VALUE!"),
    ("=SUM(A1:A2,", "#VALUE!"),
    ("=(1+2", "#VALUE!"),
    ("=F1+1", "#N/A"),  # a stored error propagates
    ("=SUM(F1,1)", "#N/A"),
    ("=10^400", "#NUM!"),
    ("=VLOOKUP(A1,A1:B3,2,FALSE)", "#NAME?"),
    ("=FOO(1)", "#NAME?"),
]


@pytest.mark.parametrize(("formula", "expected"), CASES, ids=[c[0] for c in CASES])
def test_evaluator_table(formula: str, expected: Any) -> None:
    got = ev(formula)
    if isinstance(expected, str) and expected.startswith("#"):
        assert isinstance(got, XlError) and got == expected
    elif isinstance(expected, float):
        assert got == pytest.approx(expected)
    else:
        assert got == expected and type(got) is type(expected)


def test_case_count() -> None:
    assert len(CASES) >= 40


def test_unsupported_function_is_named() -> None:
    got = ev("=XLOOKUP(A1,A:A,B:B)")
    assert got == "#NAME?" and got.detail == "XLOOKUP"
    nested = Evaluator(DictSource({"Sheet1": {"A1": "=FOO(1)", "B1": "=A1+1"}})).value("Sheet1", 1, 2)
    assert nested == "#NAME?" and nested.detail == "FOO"


def test_cycle_does_not_hang_and_is_reported() -> None:
    e = Evaluator(DictSource(DATA))
    got = e.value("Sheet1", 1, 4)  # D1
    assert got == "#REF!" and "circular" in got.detail
    assert e.value("Sheet1", 2, 4) == "#REF!"


def test_cached_value_wins_over_recalculation() -> None:
    class Cached(DictSource):
        def cell(self, sheet: str, row: int, col: int) -> tuple[Any, Any]:
            content, _ = super().cell(sheet, row, col)
            return content, 999 if content == "=A1*2" else None

    assert Evaluator(Cached(DATA)).value("Sheet1", 1, 5) == 999  # E1 keeps Excel's stored result


def test_case_insensitive_sheet_names() -> None:
    assert ev("=SUM('my data'!A1:A2)") == 15


# ------------------------------------------------------------- COM (mocked)
FAKE_PID = 4_000_001  # far above any real Windows PID


class FakeApp:
    def __init__(self, *, behaviour: str = "ok", pid: int = FAKE_PID) -> None:
        self.behaviour, self.pid = behaviour, pid
        self.Hwnd = 1234
        self.settings: dict[str, Any] = {}
        self.calls: list[str] = []
        self.release = threading.Event()
        self.Workbooks = self

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"Visible", "DisplayAlerts", "ScreenUpdating", "AskToUpdateLinks", "AutomationSecurity", "EnableEvents"}:
            self.settings[name] = value
        else:
            super().__setattr__(name, value)

    def Open(self, path: str, *args: Any) -> "FakeBook":
        self.calls.append("open")
        if self.behaviour == "hang":
            self.release.wait(10)
            raise RuntimeError("killed")
        if self.behaviour == "open_fails":
            raise RuntimeError("cannot open")
        return FakeBook(self, Path(path))

    def CalculateFull(self) -> None:
        self.calls.append("calc")

    def Quit(self) -> None:
        self.calls.append("quit")


class FakeBook:
    def __init__(self, app: FakeApp, path: Path) -> None:
        self.app, self.path = app, path

    def Save(self) -> None:
        self.app.calls.append("save")
        if self.app.behaviour == "corrupt":
            self.path.write_bytes(b"not a zip")
        else:  # stand in for Excel's recalculated file
            wb = load_workbook(self.path)
            wb.active["Z1"] = "recalculated"
            wb.save(self.path)

    def Close(self, save: bool = False) -> None:
        self.app.calls.append(f"close:{save}")


class FakeCom:
    def __init__(self, app: FakeApp) -> None:
        self.app = app
        self.threads: list[str] = []
        self.initialised = 0
        self.uninitialised = 0

    def modules(self) -> tuple[Any, Any, Any]:
        outer = self

        class Pythoncom:
            @staticmethod
            def CoInitialize() -> None:
                outer.initialised += 1
                outer.threads.append(threading.current_thread().name)

            @staticmethod
            def CoUninitialize() -> None:
                outer.uninitialised += 1

        class Win32process:
            @staticmethod
            def GetWindowThreadProcessId(hwnd: int) -> tuple[int, int]:
                return (1, outer.app.pid)

        def dispatch_ex(prog_id: str) -> FakeApp:
            assert prog_id == "Excel.Application"  # DispatchEx, never GetActiveObject
            return outer.app

        return Pythoncom, dispatch_ex, Win32process


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    wb = Workbook()
    wb.active["A1"] = 2
    wb.active["B1"] = "=A1*3"
    path = tmp_path / "book.xlsx"
    wb.save(path)
    wb.close()
    return path


@pytest.fixture
def killed(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    pids: list[int] = []
    monkeypatch.setattr(excel_calc, "kill_process", pids.append)
    return pids


def install(monkeypatch: pytest.MonkeyPatch, app: FakeApp) -> FakeCom:
    com = FakeCom(app)
    monkeypatch.setattr(excel_calc, "excel_installed", lambda: True)
    monkeypatch.setattr(excel_calc, "_com_modules", com.modules)
    return com


def test_recalc_success_replaces_file_and_cleans_up(workbook: Path, monkeypatch: pytest.MonkeyPatch, killed: list[int]) -> None:
    app = FakeApp()
    com = install(monkeypatch, app)
    result = excel_calc.recalculate(workbook, timeout=10)
    assert result.ok, result
    assert load_workbook(workbook).active["Z1"].value == "recalculated"  # Excel's file was swapped in
    assert app.calls == ["open", "calc", "save", "close:False", "quit"]
    assert app.settings == {
        "Visible": False, "DisplayAlerts": False, "ScreenUpdating": False,
        "AskToUpdateLinks": False, "EnableEvents": False, "AutomationSecurity": 3,
    }
    assert com.initialised == com.uninitialised == 1
    assert com.threads == ["excel-recalc"]  # COM is used on a worker thread, not the caller's
    assert killed == []  # a process that quit by itself is not killed
    assert [p.name for p in workbook.parent.iterdir()] == ["book.xlsx"]  # no temp copy left


def test_recalc_timeout_kills_only_the_helper_pid(workbook: Path, monkeypatch: pytest.MonkeyPatch, killed: list[int]) -> None:
    app = FakeApp(behaviour="hang")
    com = install(monkeypatch, app)
    original = workbook.read_bytes()
    try:
        result = excel_calc.recalculate(workbook, timeout=0.3)
    finally:
        app.release.set()
    assert result.status == "timeout"
    assert killed == [FAKE_PID]
    assert workbook.read_bytes() == original
    assert [p.name for p in workbook.parent.iterdir()] == ["book.xlsx"]
    assert com.initialised == 1


def test_recalc_skips_locked_workbook(workbook: Path, monkeypatch: pytest.MonkeyPatch, killed: list[int]) -> None:
    app = FakeApp()
    install(monkeypatch, app)
    monkeypatch.setattr(excel_calc, "is_locked", lambda p: True)
    result = excel_calc.recalculate(workbook, timeout=5)
    assert result.status == "locked" and "open in Excel" in result.detail
    assert app.calls == []  # Excel was never even started


def test_recalc_locked_during_swap(workbook: Path, monkeypatch: pytest.MonkeyPatch, killed: list[int]) -> None:
    install(monkeypatch, FakeApp())
    original = workbook.read_bytes()

    def deny(src: Any, dst: Any) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(os, "replace", deny)
    result = excel_calc.recalculate(workbook, timeout=5)
    monkeypatch.undo()
    assert result.status == "locked"
    assert workbook.read_bytes() == original
    assert [p.name for p in workbook.parent.iterdir()] == ["book.xlsx"]


@pytest.mark.parametrize("behaviour", ["open_fails", "corrupt"])
def test_recalc_failure_leaves_original_intact(
    workbook: Path, monkeypatch: pytest.MonkeyPatch, killed: list[int], behaviour: str
) -> None:
    app = FakeApp(behaviour=behaviour)
    install(monkeypatch, app)
    original = workbook.read_bytes()
    result = excel_calc.recalculate(workbook, timeout=5)
    assert result.status == "failed"
    assert workbook.read_bytes() == original
    assert [p.name for p in workbook.parent.iterdir()] == ["book.xlsx"]
    if behaviour == "open_fails":
        assert "quit" in app.calls  # the helper Excel is still shut down after an error


def test_recalc_unavailable_without_excel(workbook: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(excel_calc, "excel_installed", lambda: False)
    assert excel_calc.recalculate(workbook).status == "unavailable"


def test_lingering_helper_is_killed_after_quit(workbook: Path, monkeypatch: pytest.MonkeyPatch, killed: list[int]) -> None:
    install(monkeypatch, FakeApp())
    import psutil

    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)  # Excel "refuses" to exit
    monkeypatch.setattr(excel_calc, "EXIT_GRACE", 0.2)
    monkeypatch.setattr(excel_calc, "_wait_gone", lambda pid, grace=0.2: excel_calc.kill_process(pid) if pid else None)
    assert excel_calc.recalculate(workbook, timeout=5).ok
    assert killed == [FAKE_PID]


def test_kill_process_only_touches_excel(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    events: list[str] = []

    class Proc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def name(self) -> str:
            return "notepad.exe" if self.pid == 1 else "EXCEL.EXE"

        def children(self, recursive: bool = False) -> list[Any]:
            return []

        def kill(self) -> None:
            events.append(f"kill {self.pid}")

    monkeypatch.setattr(psutil, "Process", Proc)
    excel_calc.kill_process(1)  # a reused PID that is not Excel
    assert events == []
    excel_calc.kill_process(2)
    assert events == ["kill 2"]


# ------------------------------------------------------- real Excel (slow)
def excel_pids() -> set[int]:
    import psutil

    return {p.pid for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower() == "excel.exe"}


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("JARVIS_SLOW_TESTS") != "1", reason="set JARVIS_SLOW_TESTS=1 to start real Excel")
def test_real_excel_recalculation(tmp_path: Path) -> None:
    excel_calc.excel_installed.cache_clear()
    if not excel_calc.excel_installed():
        pytest.skip("Excel is not installed")
    wb = Workbook()
    ws = wb.active
    ws.append(["Item", "Cost"])
    for i, cost in enumerate((12000, 8500, 4049, 3000), 2):
        ws.append([f"item{i}", cost])
    ws["B6"] = "=SUM(B2:B5)"
    ws["C6"] = '=IF(B6>20000,"big","small")'
    path = tmp_path / "real.xlsx"
    wb.save(path)
    wb.close()

    before = excel_pids()
    started = time.monotonic()
    result = excel_calc.recalculate(path)
    elapsed = time.monotonic() - started
    print(f"real Excel recalculation: {result.status} in {elapsed:.1f}s")
    assert result.ok, result
    cached = load_workbook(path, data_only=True).active
    assert cached["B6"].value == 27549
    assert cached["C6"].value == "big"
    assert load_workbook(path).active["B6"].value == "=SUM(B2:B5)"  # the formula is still there
    time.sleep(1)
    after = excel_pids()
    assert after == before, f"EXCEL.EXE processes changed: before={before} after={after}"
    assert [p.name for p in tmp_path.iterdir()] == ["real.xlsx"]
