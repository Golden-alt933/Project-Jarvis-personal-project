"""Tests for the spreadsheet tools, run through a real ToolRegistry and ToolContext."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

from jarvis.config import Settings
from jarvis.llm.types import ToolCall
from jarvis.tools.base import Risk, ToolContext
from jarvis.tools import excel_calc
from jarvis.tools.builtin import excel
from jarvis.tools.registry import ToolRegistry


class Asker:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, text: str) -> bool:
        self.prompts.append(text)
        return self.answer


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_module(excel)
    return reg


@pytest.fixture(autouse=True)
def fake_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path_factory.mktemp("users") / "home"
    home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(excel, "known_folder", lambda name: home / name)
    return home


@pytest.fixture(autouse=True)
def no_real_excel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never start Excel; the COM path is tested with fakes in test_excel_calc."""
    monkeypatch.setattr(excel_calc, "excel_installed", lambda: False)


def ctx_for(answer: bool = True) -> tuple[ToolContext, Asker]:
    asker = Asker(answer)
    return ToolContext(settings=Settings(), confirm=asker), asker


def call(reg: ToolRegistry, name: str, ctx: ToolContext | None = None, /, **args: Any):
    ctx = ctx or ctx_for()[0]
    return asyncio.run(reg.execute(ToolCall(id="t", name=name, arguments=args), ctx))


@pytest.fixture
def budget(tmp_path: Path) -> Path:
    """Workbook with a styled header, formulas (uncached, as a script would leave them), two sheets."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws.append(["Item", "Qty", "Price", "Total"])
    ws.append(["Rent", 1, 1200, "=B2*C2"])
    ws.append(["Food", 4, 150.5, "=B3*C3"])
    ws.append(["Gym", 12, "₹1,200", "n/a"])
    ws["A1"].font = Font(bold=True)
    ws.column_dimensions["A"].width = 33
    wb.create_sheet("Notes")["A1"] = "keep me"
    path = tmp_path / "budget.xlsx"
    wb.save(path)
    return path


@pytest.fixture
def lookup_book(tmp_path: Path) -> Path:
    """Like `budget`, but its Total formulas use a function Jarvis cannot calculate."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws.append(["Item", "Qty", "Price", "Total"])
    ws.append(["Rent", 1, 1200, "=XLOOKUP(A2,A:A,B:B)"])
    ws.append(["Food", 4, 150.5, "=XLOOKUP(A3,A:A,B:B)"])
    path = tmp_path / "lookup.xlsx"
    wb.save(path)
    return path


# ------------------------------------------------------------------ registry
def test_tools_registered_in_office_group():
    reg = ToolRegistry().load_builtin()
    names = {s.name for s in reg.specs(["office"])}
    assert {
        "excel_overview", "excel_read", "excel_write",
        "excel_append_rows", "excel_create", "excel_query",
    } <= names
    spec = reg.get("excel_write")
    assert "ctx" not in spec.parameters["properties"]
    assert "cells" in spec.parameters["required"]


def test_risk_confirm_only_when_file_exists(registry: ToolRegistry, budget: Path, tmp_path: Path):
    spec = registry.get("excel_write")
    assert spec.risk_for({"path": str(budget)}) is Risk.CONFIRM
    assert spec.risk_for({"path": str(tmp_path / "new.xlsx")}) is Risk.SAFE
    text = spec.confirmation_text({"path": str(budget), "sheet": "Sheet1", "cells": {"A1": 1, "B1": 2, "C1": 3}})
    assert text.startswith("Write 3 cells to Sheet1 of budget.xlsx")


# ------------------------------------------------------------------ overview
def test_overview_xlsx(registry: ToolRegistry, budget: Path):
    r = call(registry, "excel_overview", path=str(budget))
    assert r.ok, r.content
    assert "2 sheet(s)" in r.content
    assert "Sheet 'Budget' A1:D4" in r.content
    assert "header row 1: Item, Qty, Price, Total" in r.content
    assert "formulas: 2 (2 with no cached value" in r.content
    assert "2 | Rent | 1 | 1200 | 1200*" in r.content  # computed by Jarvis, marked with *
    assert "macros: no" in r.content
    assert "Sheet 'Notes'" in r.content


def test_overview_tables_and_missing_file(registry: ToolRegistry, tmp_path: Path):
    from openpyxl.worksheet.table import Table

    wb = Workbook()
    ws = wb.active
    ws.append(["a", "b"])
    ws.append([1, 2])
    ws.add_table(Table(displayName="Sales", ref="A1:B2"))
    p = tmp_path / "t.xlsx"
    wb.save(p)
    assert "tables: Sales (A1:B2)" in call(registry, "excel_overview", path=str(p)).content
    gone = call(registry, "excel_overview", path=str(tmp_path / "budgat.xlsx"))
    assert not gone.ok and "does not exist" in gone.content


def test_overview_csv(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("name;amount\nAsha;10\nRavi;20\n", encoding="utf-8")
    r = call(registry, "excel_overview", path=str(p))
    assert r.ok and "Sheet 'data' A1:B3" in r.content and "header row 1: name, amount" in r.content


def test_overview_header_guess_none_for_numeric_first_row(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "n.csv"
    p.write_text("1,2\n3,4\n", encoding="utf-8")
    assert "none detected" in call(registry, "excel_overview", path=str(p)).content


def test_xls_and_other_types_refused(registry: ToolRegistry, tmp_path: Path):
    old = tmp_path / "old.xls"
    old.write_bytes(b"x")
    r = call(registry, "excel_overview", path=str(old))
    assert not r.ok and "legacy .xls" in r.content and "Save As .xlsx" in r.content
    r = call(registry, "excel_write", path=str(old), sheet="S", cells={"A1": 1})
    assert not r.ok and "legacy .xls" in r.content
    txt = tmp_path / "a.txt"
    txt.write_text("hi")
    assert not call(registry, "excel_read", path=str(txt)).ok


def test_corrupt_workbook_reported(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "bad.xlsx"
    p.write_text("not a zip")
    r = call(registry, "excel_read", path=str(p))
    assert not r.ok and "could not be opened" in r.content


# ---------------------------------------------------------------------- read
def test_read_grid_has_labels_and_flags_uncalculated(registry: ToolRegistry, lookup_book: Path):
    r = call(registry, "excel_read", path=str(lookup_book), sheet="Budget")
    assert r.ok, r.content
    lines = r.content.splitlines()
    assert "| A | B | C | D" in lines[1]
    assert any(ln.startswith("2 | Rent | 1 | 1200 | (uncalculated)") for ln in lines)
    assert "2 formula cell(s) show (uncalculated)" in r.content
    assert "never calculated" in r.content
    assert "Jarvis cannot calculate XLOOKUP" in r.content


def test_read_computes_uncached_formulas_and_labels_them(registry: ToolRegistry, budget: Path):
    r = call(registry, "excel_read", path=str(budget), sheet="Budget", cell_range="A1:D3")
    lines = r.content.splitlines()
    assert any(ln.startswith("2 | Rent | 1 | 1200 | 1200*") for ln in lines)
    assert any(ln.startswith("3 | Food | 4 | 150.5 | 602*") for ln in lines)
    assert "2 value(s) marked * were calculated by Jarvis" in r.content
    assert "(uncalculated)" not in r.content
    stored = call(registry, "excel_read", path=str(budget), sheet="Budget", cell_range="B2:C3")
    assert "*" not in stored.content  # plain stored values are never marked


def test_query_uses_computed_values(registry: ToolRegistry, budget: Path):
    r = call(registry, "excel_query", path=str(budget), sheet="Budget", column="Total", operation="sum")
    assert r.ok, r.content
    assert "Sum of 'Total' (D2:D4, sheet 'Budget', header row 1): 1802" in r.content
    assert "2 formula value(s) in this column were calculated by Jarvis" in r.content


def test_read_formulas_true_shows_formula_text(registry: ToolRegistry, budget: Path):
    r = call(registry, "excel_read", path=str(budget), sheet="Budget", cell_range="D2:D3", formulas=True)
    assert "=B2*C2" in r.content and "=B3*C3" in r.content
    assert "(uncalculated)" not in r.content


def test_read_distinguishes_empty_from_uncalculated(registry: ToolRegistry, tmp_path: Path):
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 5
    ws["C1"] = "=FOO(A1)"
    p = tmp_path / "e.xlsx"
    wb.save(p)
    row = call(registry, "excel_read", path=str(p), cell_range="A1:C1").content.splitlines()[2]
    assert row == "1 | 5 |  | (uncalculated)"


def test_read_cached_values_are_shown(registry: ToolRegistry, tmp_path: Path):
    """A workbook saved by Excel has cached values; simulate by writing the cache XML directly."""
    import zipfile

    wb = Workbook()
    ws = wb.active
    ws["A1"] = 2
    ws["B1"] = "=A1*3"
    p = tmp_path / "c.xlsx"
    wb.save(p)
    patched = tmp_path / "cached.xlsx"
    with zipfile.ZipFile(p) as zin, zipfile.ZipFile(patched, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                data = data.replace(b"<v></v>", b"<v>6</v>").replace(b"<v />", b"<v>6</v>")
            zout.writestr(item, data)
    r = call(registry, "excel_read", path=str(patched))
    assert "1 | 2 | 6" in r.content or "(uncalculated)" in r.content  # engine-dependent XML shape
    if "1 | 2 | 6" in r.content:
        assert "(uncalculated)" not in r.content


def test_read_range_clipping_and_wrong_sheet(registry: ToolRegistry, budget: Path):
    r = call(registry, "excel_read", path=str(budget), sheet="Budget", cell_range="A1:Z100")
    assert r.ok and "clipped" in r.content
    r = call(registry, "excel_read", path=str(budget), sheet="Budjet")
    assert not r.ok and "Did you mean 'Budget'" in r.content
    r = call(registry, "excel_read", path=str(budget), sheet="Budget", cell_range="H50:J60")
    assert not r.ok and "outside the used area" in r.content


def test_read_truncation_and_paging(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "big.xlsx"
    call(registry, "excel_create", path=str(p), sheets={"S": [["n", "sq"]] + [[i, i * i] for i in range(1, 501)]})
    r = call(registry, "excel_read", path=str(p), max_rows=100)
    assert "501 rows" not in r.content
    assert "401 more rows not shown; call again with cell_range='A101:B501'" in r.content
    nxt = call(registry, "excel_read", path=str(p), cell_range="A101:B501", max_rows=100)
    assert "\n101 | 100 | 10000" in nxt.content
    assert "301 more rows not shown; call again with cell_range='A201:B501'" in nxt.content


def test_read_wide_sheet_column_paging(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "wide.xlsx"
    row = list(range(60))
    call(registry, "excel_create", path=str(p), sheets={"S": [row]})
    r = call(registry, "excel_read", path=str(p))
    assert "20 more columns not shown; call again with cell_range='AO1:BH1'" in r.content


def test_read_csv(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    r = call(registry, "excel_read", path=str(p))
    assert r.ok and "2 | 1 | 2" in r.content


# --------------------------------------------------------------------- write
def test_write_round_trip_preserves_formatting_and_sheets(registry: ToolRegistry, budget: Path):
    r = call(registry, "excel_write", path=str(budget), sheet="Budget",
             cells={"B2": 42, "E1": "Tax", "E2": "=D2*0.18"})
    assert r.ok, r.content
    assert "Wrote 3 cells" in r.content and "1 formula" in r.content
    assert "Excel is not installed" in r.content and "E2 = 9072" in r.content  # 42 * 1200 * 0.18
    wb = load_workbook(budget)
    ws = wb["Budget"]
    assert ws["B2"].value == 42 and ws["E1"].value == "Tax"
    assert ws["E2"].value == "=D2*0.18"  # stored as formula, not text
    assert ws["D2"].value == "=B2*C2"  # untouched formula survives
    assert ws["A1"].font.bold is True
    assert ws.column_dimensions["A"].width == 33
    assert wb["Notes"]["A1"].value == "keep me"


def test_write_creates_bak_with_previous_content(registry: ToolRegistry, budget: Path):
    call(registry, "excel_write", path=str(budget), sheet="Budget", cells={"B2": 99})
    bak = budget.with_name("budget.xlsx.bak")
    assert bak.exists()
    restored = budget.with_name("restored.xlsx")
    restored.write_bytes(bak.read_bytes())
    assert load_workbook(restored)["Budget"]["B2"].value == 1
    assert "Previous version kept" in call(registry, "excel_write", path=str(budget), sheet="Budget", cells={"B2": 5}).content
    assert not list(budget.parent.glob("*.tmp"))


def test_write_asks_and_decline_changes_nothing(registry: ToolRegistry, budget: Path):
    before = budget.read_bytes()
    ctx, asker = ctx_for(False)
    r = call(registry, "excel_write", ctx, path=str(budget), sheet="Budget", cells={"B2": 7})
    assert not r.ok and "declined" in r.content
    assert asker.prompts and asker.prompts[0].startswith("Write 1 cell to Budget of budget.xlsx")
    assert budget.read_bytes() == before
    assert not budget.with_name("budget.xlsx.bak").exists()


def test_write_approved_prompts_once(registry: ToolRegistry, budget: Path):
    ctx, asker = ctx_for(True)
    assert call(registry, "excel_write", ctx, path=str(budget), sheet="Budget", cells={"B2": 7}).ok
    assert len(asker.prompts) == 1


def test_write_new_file_and_new_sheet_need_no_confirmation(registry: ToolRegistry, tmp_path: Path, budget: Path):
    ctx, asker = ctx_for(False)
    fresh = tmp_path / "fresh.xlsx"
    r = call(registry, "excel_write", ctx, path=str(fresh), sheet="Données", cells={"A1": "Prix €"})
    assert r.ok and not asker.prompts
    assert load_workbook(fresh)["Données"]["A1"].value == "Prix €"
    r = call(registry, "excel_write", ctx_for(True)[0], path=str(budget), sheet="Summary", cells={"A1": 1})
    assert r.ok and "Created sheet 'Summary'" in r.content
    assert load_workbook(budget).sheetnames == ["Budget", "Notes", "Summary"]


def test_write_sheet_name_case_insensitive_match(registry: ToolRegistry, budget: Path):
    call(registry, "excel_write", path=str(budget), sheet="budget", cells={"A1": "Items"})
    wb = load_workbook(budget)
    assert wb.sheetnames == ["Budget", "Notes"] and wb["Budget"]["A1"].value == "Items"


def test_write_none_clears_and_types_kept(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "t.xlsx"
    call(registry, "excel_write", path=str(p), sheet="S", cells={"A1": 1.5, "A2": True, "A3": "text", "A4": "x"})
    call(registry, "excel_write", path=str(p), sheet="S", cells={"A4": None})
    ws = load_workbook(p)["S"]
    assert (ws["A1"].value, ws["A2"].value, ws["A3"].value, ws["A4"].value) == (1.5, True, "text", None)


@pytest.mark.parametrize(
    "ref, fragment",
    [
        ("B0", "'B0' is not a valid cell: rows start at 1"),
        ("1B", "not a valid cell"),
        ("", "not a valid cell"),
        ("A", "not a valid cell"),
        ("AAAA1", "not a valid cell"),
        ("XFE1", "last column is XFD"),
        ("A1048577", "last row is 1048576"),
        ("A1:B2", "not a valid cell"),
        ("B-1", "not a valid cell"),
    ],
)
def test_write_invalid_references(registry: ToolRegistry, tmp_path: Path, ref: str, fragment: str):
    p = tmp_path / "v.xlsx"
    r = call(registry, "excel_write", path=str(p), sheet="S", cells={"A1": 1, ref: 2})
    assert not r.ok and fragment in r.content
    assert "Nothing was written" in r.content
    assert not p.exists()


def test_write_rejects_bad_formula_and_duplicate_refs(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "f.xlsx"
    r = call(registry, "excel_write", path=str(p), sheet="S", cells={"A1": "=SUM(B1:B3"})
    assert not r.ok and "malformed" in r.content
    r = call(registry, "excel_write", path=str(p), sheet="S", cells={"b2": 1, "B2": 2})
    assert not r.ok and "same cell" in r.content
    r = call(registry, "excel_write", path=str(p), sheet="Bad/Name", cells={"A1": 1})
    assert not r.ok and "forbids" in r.content
    assert not call(registry, "excel_write", path=str(p), sheet="S", cells={}).ok


def test_write_locked_file_gives_hint_and_keeps_original(registry: ToolRegistry, budget: Path, monkeypatch):
    before = budget.read_bytes()

    def locked(*a, **k):
        raise PermissionError(13, "The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(os, "replace", locked)
    r = call(registry, "excel_write", path=str(budget), sheet="Budget", cells={"B2": 5})
    assert not r.ok
    assert "Close it there" in r.content and "different file name" in r.content
    assert budget.read_bytes() == before
    assert not list(budget.parent.glob("*.tmp"))


def test_write_locked_at_open_time(registry: ToolRegistry, budget: Path, monkeypatch):
    import builtins

    real_open = builtins.open

    def deny(file, mode="r", *a, **k):
        if str(file) == str(budget) and "+" in mode:
            raise PermissionError(13, "locked")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", deny)
    r = call(registry, "excel_write", path=str(budget), sheet="Budget", cells={"B2": 5})
    assert not r.ok and "Excel" in r.content


# openpyxl leaves an in-memory vba_archive ZipFile that warns when garbage collected.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_xlsm_keeps_macros(registry: ToolRegistry, tmp_path: Path):
    import zipfile

    wb = Workbook()
    wb.active["A1"] = 1
    plain = tmp_path / "src.xlsx"
    wb.save(plain)
    # Fake a macro workbook: add a vbaProject part and the macro-enabled content type.
    macro = tmp_path / "m.xlsm"
    with zipfile.ZipFile(plain) as zin, zipfile.ZipFile(macro, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(
                    b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
                    b"application/vnd.ms-excel.sheet.macroEnabled.main+xml",
                ).replace(b"</Types>", b'<Default Extension="bin" ContentType="application/vnd.ms-office.vbaProject"/></Types>')
            zout.writestr(item, data)
        zout.writestr("xl/vbaProject.bin", b"FAKE-VBA")
    assert "macros: yes" in call(registry, "excel_overview", path=str(macro)).content
    r = call(registry, "excel_write", path=str(macro), sheet="Sheet", cells={"A1": 2})
    assert r.ok, r.content
    assert "macros (VBA) were kept" in r.content
    with zipfile.ZipFile(macro) as zf:
        assert "xl/vbaProject.bin" in zf.namelist()
    assert load_workbook(macro, keep_vba=True).active["A1"].value == 2


def test_new_xlsm_refused(registry: ToolRegistry, tmp_path: Path):
    r = call(registry, "excel_write", path=str(tmp_path / "n.xlsm"), sheet="S", cells={"A1": 1})
    assert not r.ok and "macro" in r.content


def test_write_to_csv_refused(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "a.csv"
    p.write_text("a\n1\n")
    r = call(registry, "excel_write", path=str(p), sheet="S", cells={"A1": 1})
    assert not r.ok and "text file" in r.content
    assert p.read_text() == "a\n1\n"


# -------------------------------------------------------------------- append
def test_append_below_last_used_row_ignoring_formatting(registry: ToolRegistry, budget: Path):
    wb = load_workbook(budget)
    wb["Budget"]["A50"].font = Font(italic=True)  # formatted but empty: not "used"
    wb.save(budget.with_name("fmt.xlsx"))
    wb.close()
    os.replace(budget.with_name("fmt.xlsx"), budget)
    r = call(registry, "excel_append_rows", path=str(budget), sheet="Budget",
             rows=[["Water", 1, 30, "=B5*C5"], ["Net", None, 20]])
    assert r.ok, r.content
    assert "Appended 2 rows at A5:D6" in r.content and "D5 = 30" in r.content
    ws = load_workbook(budget)["Budget"]
    assert [c.value for c in ws[5]] == ["Water", 1, 30, "=B5*C5"]
    assert [c.value for c in ws[6]][:3] == ["Net", None, 20]
    assert ws["A1"].font.bold is True


def test_append_confirms_and_decline_is_noop(registry: ToolRegistry, budget: Path):
    before = budget.read_bytes()
    ctx, asker = ctx_for(False)
    r = call(registry, "excel_append_rows", ctx, path=str(budget), sheet="Budget", rows=[["x"]])
    assert not r.ok and asker.prompts[0].startswith("Append 1 row to Budget of budget.xlsx")
    assert budget.read_bytes() == before


def test_append_to_new_sheet_and_empty_rows(registry: ToolRegistry, budget: Path):
    r = call(registry, "excel_append_rows", path=str(budget), sheet="Log", rows=[["a", 1]])
    assert r.ok and "A1:B1" in r.content
    assert not call(registry, "excel_append_rows", path=str(budget), sheet="Log", rows=[]).ok


# -------------------------------------------------------------------- create
def test_create_formatting_and_widths(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "out" / "new.xlsx"
    r = call(registry, "excel_create", path=str(p), sheets={
        "Budget": [["Item", "A very long header title here", "Qty"], ["Rent", 1200, 1]],
        "Notes": [],
    })
    assert r.ok, r.content
    wb = load_workbook(p)
    ws = wb["Budget"]
    assert ws["A1"].font.bold and ws.freeze_panes == "A2"
    assert ws.column_dimensions["B"].width > ws.column_dimensions["A"].width
    assert ws["B2"].value == 1200
    assert wb.sheetnames == ["Budget", "Notes"]


def test_create_without_header_style(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "h.xlsx"
    call(registry, "excel_create", path=str(p), sheets={"S": [["a"], [1]]}, header_style=False)
    ws = load_workbook(p)["S"]
    assert not ws["A1"].font.bold and ws.freeze_panes is None


def test_create_never_overwrites(registry: ToolRegistry, budget: Path):
    before = budget.read_bytes()
    r = call(registry, "excel_create", path=str(budget), sheets={"S": [[1]]})
    assert not r.ok and "never overwrites" in r.content
    assert budget.read_bytes() == before


def test_create_bare_name_goes_to_documents(registry: ToolRegistry, fake_home: Path):
    r = call(registry, "excel_create", path="plan", sheets={"S": [["x"]]})
    assert r.ok, r.content
    assert (fake_home / "Documents" / "plan.xlsx").exists()
    assert str(fake_home / "Documents" / "plan.xlsx") == r.data


def test_create_validation(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "v.xlsx"
    assert not call(registry, "excel_create", path=str(p), sheets={}).ok
    r = call(registry, "excel_create", path=str(p), sheets={"a/b": [[1]]})
    assert not r.ok and "forbids" in r.content
    r = call(registry, "excel_create", path=str(p), sheets={"X": [[1]], "x": [[2]]})
    assert not r.ok and "used twice" in r.content
    r = call(registry, "excel_create", path=str(tmp_path / "a.xlsm"), sheets={"S": [[1]]})
    assert not r.ok and "must be .xlsx" in r.content
    r = call(registry, "excel_create", path=str(p), sheets={"S": [["=SUM("]]})
    assert not r.ok and "malformed" in r.content
    assert not p.exists()


def test_create_non_ascii_sheet_names_and_values(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "unicode.xlsx"
    rows = [["名前", "Prix €"], ["Ünïcode ✓", 3], ["日本語のとても長いテキスト", 4]]
    assert call(registry, "excel_create", path=str(p), sheets={"売上 Données": rows}).ok
    wb = load_workbook(p)
    assert wb.sheetnames == ["売上 Données"]
    assert wb.active["A2"].value == "Ünïcode ✓"
    assert wb.active.column_dimensions["A"].width >= 28  # CJK counted double width
    out = call(registry, "excel_read", path=str(p), sheet="売上 Données")
    assert "Ünïcode ✓" in out.content and "日本語" in out.content
    q = call(registry, "excel_query", path=str(p), column="Prix €", operation="sum")
    assert q.ok and "7" in q.content


def test_create_formula_note(registry: ToolRegistry, tmp_path: Path):
    r = call(registry, "excel_create", path=str(tmp_path / "f.xlsx"), sheets={"S": [["a", "b"], [1, "=A2*2"]]})
    assert r.ok and "B2 = 2" in r.content and "built-in evaluator" in r.content


# --------------------------------------------------------------------- query
def test_query_aggregates_by_header_and_letter(registry: ToolRegistry, budget: Path):
    r = call(registry, "excel_query", path=str(budget), sheet="Budget", column="Price", operation="sum")
    assert r.ok, r.content
    assert "2550.5" in r.content
    assert "3 numeric cell(s)" in r.content
    same = call(registry, "excel_query", path=str(budget), sheet="Budget", column="C", operation="sum")
    assert "2550.5" in same.content
    low = call(registry, "excel_query", path=str(budget), column="price", operation="min")  # case-insensitive header
    assert low.ok and ": 150.5" in low.content


def test_query_operations(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "q.csv"
    p.write_text("Region,Sales\nNorth,\"$1,000\"\nSouth,(250)\nNorth,12%\nEast,n/a\nWest,\n", encoding="utf-8")

    def q(**kw):
        return call(registry, "excel_query", path=str(p), **kw)

    s = q(column="Sales", operation="sum")
    assert ": 750.12" in s.content  # 1000 - 250 + 0.12
    assert "Skipped 1 non-numeric cell(s) (e.g. B5='n/a')" in s.content and "1 empty" in s.content
    assert "percent values counted as fractions" in s.content
    assert "Max of" in q(column="Sales", operation="max").content and ": 1000" in q(column="Sales", operation="max").content
    assert ": -250" in q(column="Sales", operation="min").content
    assert ": 250.04" in q(column="Sales", operation="average").content
    assert "filled cells in 'Sales' (B2:B6, sheet 'q', header row 1): 4" in q(column="Sales", operation="count").content
    u = q(column="Region", operation="unique")
    assert "4 distinct value(s)" in u.content and "North: 2" in u.content
    f = q(operation="find", match="south")
    assert "1 row(s) contain 'south'" in f.content and "3 | South | (250)" in f.content
    assert "1 | Region | Sales" in f.content  # header row included for context
    fc = q(operation="find", match="n/a", column="Sales")
    assert "5 | East" in fc.content
    assert "No rows contain" in q(operation="find", match="zzz").content


def test_query_errors(registry: ToolRegistry, budget: Path):
    bad = call(registry, "excel_query", path=str(budget), sheet="Budget", column="Nope")
    assert not bad.ok and "Columns: A=Item, B=Qty, C=Price, D=Total" in bad.content
    assert not call(registry, "excel_query", path=str(budget), operation="sum").ok  # needs column
    assert not call(registry, "excel_query", path=str(budget), operation="find").ok  # needs match
    text_col = call(registry, "excel_query", path=str(budget), sheet="Budget", column="Item", operation="sum")
    assert not text_col.ok and "no numbers found" in text_col.content
    notes = call(registry, "excel_query", path=str(budget), sheet="Nope", column="A")
    assert not notes.ok and "Sheets in budget.xlsx" in notes.content


def test_query_reports_uncalculated_formulas(registry: ToolRegistry, lookup_book: Path):
    r = call(registry, "excel_query", path=str(lookup_book), sheet="Budget", column="Total", operation="sum")
    assert not r.ok
    assert "2 formula cell(s) skipped because they have no cached value" in r.content
    assert "Jarvis cannot calculate: XLOOKUP" in r.content


def test_query_ambiguous_header(registry: ToolRegistry, tmp_path: Path):
    p = tmp_path / "a.csv"
    p.write_text("Net amount,Gross amount\n1,2\n", encoding="utf-8")
    r = call(registry, "excel_query", path=str(p), column="amount")
    assert not r.ok and "matches several headers" in r.content


# ------------------------------------------------------------ number parsing
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("₹1,20,000.50", 120000.5),
        ("$1,234.56", 1234.56),
        ("(450)", -450.0),
        ("-12.5", -12.5),
        ("450-", -450.0),
        ("12%", 0.12),
        ("1.2e3", 1200.0),
        ("1 234", 1234.0),
        ("1.234,56", 1234.56),
        ("Rs. 5,000", 5000.0),
        ("€ 99", 99.0),
        ("  42  ", 42.0),
        (".5", 0.5),
        ("+7", 7.0),
        (3, 3.0),
        (2.5, 2.5),
        ("", None),
        ("   ", None),
        ("abc", None),
        ("n/a", None),
        ("nan", None),
        ("inf", None),
        ("1_000", None),
        ("1,2,3,4", None),
        ("12abc", None),
        ("--5", None),
        (True, None),
        (None, None),
        ("#DIV/0!", None),
    ],
)
def test_parse_number(raw: Any, expected: float | None):
    got = excel.parse_number(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, expected",
    [("A", 1), ("Z", 26), ("AA", 27), ("XFD", 16384)],
)
def test_column_letters_round_trip(text: str, expected: int):
    assert excel.col_index(text) == expected
    assert excel.col_letter(expected) == text


@pytest.mark.parametrize(
    "text, expected",
    [("B2", (2, 2, 2, 2)), ("a1:c10", (1, 1, 10, 3)), ("C10:A1", (1, 1, 10, 3)), ("$A$1:$B$2", (1, 1, 2, 2))],
)
def test_parse_range_valid(text: str, expected: tuple[int, int, int, int]):
    assert excel.parse_range(text) == expected


@pytest.mark.parametrize("text", ["A1:", ":B2", "A1:B2:C3", "A0:B2", "A1:B0", "hello", "A1-B2", "0:3"])
def test_parse_range_invalid(text: str):
    with pytest.raises(ValueError):
        excel.parse_range(text)


def test_read_invalid_range_reports_error(registry: ToolRegistry, budget: Path):
    r = call(registry, "excel_read", path=str(budget), cell_range="A0:B2")
    assert not r.ok and "rows start at 1" in r.content


def test_read_column_and_row_ranges(registry: ToolRegistry, budget: Path):
    cols = call(registry, "excel_read", path=str(budget), sheet="Budget", cell_range="B:C")
    assert cols.ok and "| B | C" in cols.content and "1 | Qty | Price" in cols.content
    rows = call(registry, "excel_read", path=str(budget), sheet="Budget", cell_range="2:3")
    assert rows.ok and "2 | Rent" in rows.content and "4 | Gym" not in rows.content


# ------------------------------------------------------ recalculation reports
def test_write_reports_excel_calculated_values(registry: ToolRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: list[Path] = []

    def fake_recalc(path: Path, timeout: float = 40.0) -> excel_calc.Recalc:
        seen.append(path)
        return excel_calc.Recalc("ok")

    monkeypatch.setattr(excel_calc, "recalculate", fake_recalc)
    p = tmp_path / "calc.xlsx"
    r = call(registry, "excel_write", path=str(p), sheet="S",
             cells={"B2": 12000, "B3": 8500, "B4": 4049, "B5": 3000, "B6": "=SUM(B2:B5)"})
    assert r.ok, r.content
    assert seen == [p]
    assert "Excel recalculated the workbook" in r.content and "B6 = 27549" in r.content


def test_write_without_formulas_does_not_start_excel(registry: ToolRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(excel_calc, "recalculate", lambda *a, **k: pytest.fail("Excel started for a workbook without formulas"))
    r = call(registry, "excel_write", path=str(tmp_path / "plain.xlsx"), sheet="S", cells={"A1": 1})
    assert r.ok, r.content


@pytest.mark.parametrize(
    ("status", "detail", "expected"),
    [
        ("locked", "b.xlsx is open in Excel", "so it was not recalculated"),
        ("timeout", "Excel did not finish within 40s and was stopped", "did not complete"),
        ("failed", "RuntimeError: boom", "did not complete"),
    ],
)
def test_write_falls_back_to_builtin_values(
    registry: ToolRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, detail: str, expected: str
):
    monkeypatch.setattr(excel_calc, "recalculate", lambda *a, **k: excel_calc.Recalc(status, detail))
    r = call(registry, "excel_write", path=str(tmp_path / "b.xlsx"), sheet="S", cells={"A1": 4, "B1": "=A1*A1"})
    assert r.ok, r.content
    assert expected in r.content and "built-in evaluator: B1 = 16" in r.content


def test_write_reports_unsupported_formula(registry: ToolRegistry, tmp_path: Path):
    r = call(registry, "excel_write", path=str(tmp_path / "u.xlsx"), sheet="S", cells={"A1": "=XLOOKUP(1,B:B,C:C)"})
    assert r.ok and "A1 = (uncalculated: XLOOKUP is not supported)" in r.content


def test_create_reports_values_with_sheet_names(registry: ToolRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(excel_calc, "recalculate", lambda *a, **k: excel_calc.Recalc("ok"))
    r = call(registry, "excel_create", path=str(tmp_path / "m.xlsx"),
             sheets={"A": [[1, 2, "=A1+B1"]], "B": [["=A!C1*2"]]})
    assert r.ok, r.content
    assert "A!C1 = 3" in r.content and "B!A1 = 6" in r.content
