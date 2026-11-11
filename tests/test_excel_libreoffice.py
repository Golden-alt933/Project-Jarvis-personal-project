"""LibreOffice recalculation (the non-Excel engine). ``soffice`` is replaced by a fake that
writes the "converted" file, so this runs anywhere and checks the safety rules: private
profile, temp copy, validation, atomic replace, nothing left behind."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from jarvis.tools import excel_calc
from jarvis.tools.builtin import excel


def make_book(path: Path, marker: str) -> None:
    wb = Workbook()
    wb.active["A1"] = marker
    wb.save(path)
    wb.close()


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    path = tmp_path / "book.xlsx"
    make_book(path, "original")
    return path


@pytest.fixture
def lo(monkeypatch, tmp_path):
    """No Excel, a fake soffice that 'converts' by writing a recalculated copy."""
    seen: dict = {"argv": []}
    monkeypatch.setattr(excel_calc, "excel_installed", lambda: False)
    monkeypatch.setattr(excel_calc, "find_soffice", lambda: "/usr/bin/soffice")

    def fake_run(argv, timeout):
        seen["argv"].append(argv)
        seen["timeout"] = timeout
        outdir = Path(argv[argv.index("--outdir") + 1])
        source = Path(argv[-1])
        seen["input_existed"] = source.exists()
        make_book(outdir / source.name, "recalculated")
        return 0, "convert /tmp/x -> /tmp/y using filter : Calc MS Excel 2007 XML"

    monkeypatch.setattr(excel_calc, "_run_soffice", fake_run)
    return seen


def test_libreoffice_recalculation_replaces_the_file(workbook, lo):
    result = excel_calc.recalculate(workbook, timeout=33)
    assert result.ok and result.engine == "LibreOffice"
    assert load_workbook(workbook).active["A1"].value == "recalculated"
    assert [p.name for p in workbook.parent.iterdir()] == ["book.xlsx"]  # no temp files beside it
    argv = lo["argv"][0]
    assert argv[0] == "/usr/bin/soffice" and "--headless" in argv and "--convert-to" in argv
    assert argv[argv.index("--convert-to") + 1].startswith("xlsx")
    assert any(a.startswith("-env:UserInstallation=file:") for a in argv)  # private profile
    assert lo["timeout"] == 33 and lo["input_existed"]
    # the engine works on a temp copy, never on the user's file
    assert Path(argv[-1]) != workbook and Path(argv[-1]).name == "book.xlsx"


def test_failed_conversion_leaves_the_original(workbook, lo, monkeypatch):
    original = workbook.read_bytes()
    monkeypatch.setattr(excel_calc, "_run_soffice", lambda argv, timeout: (1, "Error: source file could not be loaded"))
    result = excel_calc.recalculate(workbook)
    assert result.status == "failed" and "could not be loaded" in result.detail
    assert workbook.read_bytes() == original
    assert [p.name for p in workbook.parent.iterdir()] == ["book.xlsx"]


def test_invalid_output_is_rejected(workbook, lo, monkeypatch):
    original = workbook.read_bytes()

    def garbage(argv, timeout):
        (Path(argv[argv.index("--outdir") + 1]) / Path(argv[-1]).name).write_bytes(b"not a zip")
        return 0, ""

    monkeypatch.setattr(excel_calc, "_run_soffice", garbage)
    assert excel_calc.recalculate(workbook).status == "failed"
    assert workbook.read_bytes() == original


def test_missing_output_is_rejected(workbook, lo, monkeypatch):
    monkeypatch.setattr(excel_calc, "_run_soffice", lambda argv, timeout: (0, ""))
    assert excel_calc.recalculate(workbook).status == "failed"


def test_timeout_is_reported(workbook, lo, monkeypatch):
    original = workbook.read_bytes()

    def slow(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(excel_calc, "_run_soffice", slow)
    result = excel_calc.recalculate(workbook, timeout=1)
    assert result.status == "timeout" and "LibreOffice" in result.detail
    assert workbook.read_bytes() == original


def test_file_open_in_libreoffice_is_locked(workbook, lo):
    (workbook.parent / f".~lock.{workbook.name}#").write_text("user,host,,1.1.2020;")
    result = excel_calc.recalculate(workbook)
    assert result.status == "locked" and "LibreOffice" in result.detail
    assert lo["argv"] == []  # never started


def test_excel_is_preferred_when_both_exist(workbook, lo, monkeypatch):
    monkeypatch.setattr(excel_calc, "excel_installed", lambda: True)
    sentinel = excel_calc.Recalc("ok", "via excel")
    monkeypatch.setattr(excel_calc, "_recalculate_excel", lambda path, timeout: sentinel)
    assert excel_calc.recalculate(workbook) is sentinel
    assert lo["argv"] == []  # LibreOffice was not started


def test_unavailable_names_both_engines(workbook, monkeypatch):
    monkeypatch.setattr(excel_calc, "excel_installed", lambda: False)
    monkeypatch.setattr(excel_calc, "find_soffice", lambda: None)
    result = excel_calc.recalculate(workbook)
    assert result.status == "unavailable"
    assert "Excel is not installed" in result.detail and "LibreOffice" in result.detail


def test_find_soffice_looks_for_both_launchers(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(excel_calc.shutil, "which", lambda n: "/opt/lo/libreoffice" if n == "libreoffice" else None)
    assert excel_calc.find_soffice() == "/opt/lo/libreoffice"
    monkeypatch.setattr(excel_calc.shutil, "which", lambda n: None)
    assert excel_calc.find_soffice() is None


def test_messages_name_the_engine_that_ran():
    ok = excel_calc.Recalc("ok", engine="LibreOffice")
    failed = excel_calc.Recalc("failed", "boom", engine="LibreOffice")
    assert "LibreOffice recalculation did not complete (boom)" in excel._recalc_reason(failed)
    assert ok.ok and excel_calc.Recalc("ok").engine == "Excel"
