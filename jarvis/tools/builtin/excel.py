"""Spreadsheet tools: overview, read, write, append, create, query (.xlsx/.xlsm/.csv).

Ported from Friday's excel_skill (file path only): load with ``data_only=False`` when
writing so formulas survive, check A1 references before touching the file, handle locked
files. Rewritten for Jarvis: typed arguments, compact grids with A1 labels instead of
spoken prose, atomic saves with a ``.bak``, and a number parser so the model never does
arithmetic itself. The xlwings/live-Excel path is intentionally absent.

openpyxl cannot calculate formulas, and saving a workbook with it drops the cached value of
every formula. After a write that leaves formulas, the file is recalculated by a private hidden
Excel when one is installed (``excel_calc``); otherwise, and for reading files that have no
stored results, ``excel_eval`` computes the values and the output says so.
"""

from __future__ import annotations

import csv
import difflib
import io
import logging
import math
import os
import re
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from jarvis.tools.base import Risk, ToolResult, tool
from jarvis.tools import excel_calc
from jarvis.tools.excel_eval import Evaluator, XlError
from jarvis.tools.paths import known_folder, make_backup, not_found, resolve_path

log = logging.getLogger(__name__)

GROUP = "office"

CellValue = str | int | float | bool | None

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_WRITE_CELLS = 5000
MAX_WRITE_ROWS = 5000
MAX_CREATE_ROWS = 100_000
MAX_COLS = 200
MAX_SHOWN_COLS = 40
MAX_CELL_CHARS = 60
OUTPUT_BUDGET = 9000  # chars; the registry truncates at 12000
MAX_ROW_LIMIT = 1000
MAX_EXCEL_ROW, MAX_EXCEL_COL = 1_048_576, 16_384
FORMULA_SCAN_CELLS = 200_000
READ_SUFFIXES = {".xlsx", ".xlsm", ".csv", ".tsv"}
XLSX_SUFFIXES = {".xlsx", ".xlsm"}
ERROR_VALUES = frozenset(
    {"#DIV/0!", "#N/A", "#NAME?", "#NULL!", "#NUM!", "#REF!", "#VALUE!", "#SPILL!", "#CALC!"}
)

LOCK_HINT = (
    "The file is probably open in Excel. Close it there and try again, or write to a "
    "different file name."
)
UNCALCULATED = "(uncalculated)"  # formula cell that has no cached value


# ------------------------------------------------------------------ references
_CELL_RE = re.compile(r"^\$?([A-Za-z]{1,3})\$?(\d+)$")
_COLS_RE = re.compile(r"^\$?([A-Za-z]{1,3}):\$?([A-Za-z]{1,3})$")
_ROWS_RE = re.compile(r"^\$?(\d+):\$?(\d+)$")


def col_letter(index: int) -> str:
    letters = ""
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def col_index(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + ord(ch) - 64
    return n


def parse_cell(ref: str) -> tuple[int, int]:
    """(row, col), both 1-based. Raises ValueError with a message the model can act on."""
    text = str(ref).strip()
    m = _CELL_RE.match(text)
    if not m:
        raise ValueError(
            f"'{ref}' is not a valid cell: use a column letter then a row number, like B2"
        )
    col, row = col_index(m.group(1)), int(m.group(2))
    if row < 1:
        raise ValueError(f"'{ref}' is not a valid cell: rows start at 1")
    if row > MAX_EXCEL_ROW:
        raise ValueError(f"'{ref}' is not a valid cell: the last row is {MAX_EXCEL_ROW}")
    if col > MAX_EXCEL_COL:
        raise ValueError(f"'{ref}' is not a valid cell: the last column is XFD")
    return row, col


def parse_range(text: str) -> tuple[int, int, int, int]:
    """(row1, col1, row2, col2) from 'B2', 'A1:C10', 'A:C' or '2:5'.

    Whole-column and whole-row ranges extend to the sheet limits and are clipped later.
    """
    raw = str(text).strip()
    if m := _COLS_RE.match(raw):
        c1, c2 = sorted((col_index(m.group(1)), col_index(m.group(2))))
        if c2 > MAX_EXCEL_COL:
            raise ValueError(f"'{text}' is not a valid range: the last column is XFD")
        return 1, c1, MAX_EXCEL_ROW, c2
    if m := _ROWS_RE.match(raw):
        r1, r2 = sorted((int(m.group(1)), int(m.group(2))))
        if r1 < 1 or r2 > MAX_EXCEL_ROW:
            raise ValueError(f"'{text}' is not a valid range: rows run from 1 to {MAX_EXCEL_ROW}")
        return r1, 1, r2, MAX_EXCEL_COL
    parts = raw.split(":")
    if len(parts) > 2 or not all(parts):
        raise ValueError(f"'{text}' is not a valid range: use a cell like B2 or a range like A1:C10")
    try:
        r1, c1 = parse_cell(parts[0])
        r2, c2 = parse_cell(parts[-1])
    except ValueError as exc:
        raise ValueError(f"range '{text}': {exc}") from None
    return min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2)


def valid_sheet_name(name: str) -> str | None:
    """Why this cannot be an Excel sheet name, or None."""
    if not name or not name.strip():
        return "a sheet name cannot be empty"
    if len(name) > 31:
        return f"sheet name '{name}' is longer than Excel's 31-character limit"
    if re.search(r"[\[\]:*?/\\]", name):
        return f"sheet name '{name}' contains one of [ ] : * ? / \\ which Excel forbids"
    if name.startswith("'") or name.endswith("'"):
        return f"sheet name '{name}' cannot start or end with an apostrophe"
    return None


# --------------------------------------------------------------- number parsing
_CURRENCY = re.compile(r"(?i)(₹|\$|€|£|¥|₩|₽|₺|₫|₪|\brs\.?|\binr\b|\busd\b|\beur\b|\bgbp\b)")
_PLAIN_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_GROUPED = re.compile(r"^\d{1,3}(,\d{2,3})+(\.\d+)?$")  # 1,234  1,20,000.50
_EURO = re.compile(r"^\d{1,3}(\.\d{3})*,\d+$")  # 1.234,56


def parse_number(value: Any) -> float | None:
    """A float from a cell, or None if it is not a number.

    Handles currency symbols, thousands separators (including Indian 1,20,000 grouping
    and European 1.234,56), accounting negatives "(450)", trailing minus, and percentages
    ("12%" -> 0.12, as Excel stores it). Booleans and NaN/inf are not numbers.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    text = _CURRENCY.sub("", value).replace(" ", "").replace(" ", "").replace(" ", "").strip()
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1]
    if text.endswith("-") and not text.startswith(("-", "+")):
        negative, text = True, text[:-1]
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    sign = ""
    if text[:1] in ("+", "-"):
        sign, text = text[0], text[1:]
    if _GROUPED.match(text):
        text = text.replace(",", "")
    elif _EURO.match(text):
        text = text.replace(".", "").replace(",", ".")
    text = sign + text
    if not _PLAIN_NUMBER.match(text):
        return None
    number = float(text)
    if not math.isfinite(number):
        return None
    if percent:
        number /= 100
    return -number if negative else number


def _is_percent(value: Any) -> bool:
    return isinstance(value, str) and value.strip().endswith("%")


def _num(x: float) -> str:
    x = round(x, 10)
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else f"{x:.12g}"


# ----------------------------------------------------------------- sheet model
class _NoCache:
    """Marker for a formula cell whose value was never calculated."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NO_CACHE"


NO_CACHE = _NoCache()


def _is_formula(value: Any) -> bool:
    # ArrayFormula / DataTableFormula objects carry .text instead of being a str
    return hasattr(value, "text") or (isinstance(value, str) and value.startswith("="))


def _formula_text(value: Any) -> Any:
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text if text.startswith("=") else "=" + text
    return value


class _Sheet:
    """Read-only view over one worksheet (or a CSV) with a shared interface."""

    name: str
    max_row: int
    max_col: int
    hidden: bool = False
    tables: list[tuple[str, str]]

    def block(self, r1: int, c1: int, r2: int, c2: int, formulas: bool = False) -> list[list[Any]]:
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        return self.max_row == 0 or self.max_col == 0


class _XlsxSource:
    """Gives the formula evaluator access to a workbook loaded twice (values and formulas)."""

    def __init__(self, values: Any, formulas: Any) -> None:
        self.values, self.formulas = values, formulas
        self._names = {n.casefold(): n for n in formulas.sheetnames}

    def dims(self, sheet: str) -> tuple[int, int] | None:
        name = self._names.get(sheet.casefold())
        return None if name is None else (self.formulas[name].max_row, self.formulas[name].max_column)

    def cell(self, sheet: str, row: int, col: int) -> tuple[Any, Any]:
        name = self._names[sheet.casefold()]
        return (
            _formula_text(self.formulas[name].cell(row=row, column=col).value),
            self.values[name].cell(row=row, column=col).value,
        )


class _XlsxSheet(_Sheet):
    def __init__(self, vals: Any, fmls: Any, calc: Evaluator | None = None) -> None:
        self.vals, self.fmls, self.name, self.calc = vals, fmls, vals.title, calc
        self.computed: set[tuple[int, int]] = set()  # (row, col) values calculated by Jarvis
        self.unsupported: dict[tuple[int, int], str] = {}  # (row, col) -> function not supported
        self.hidden = vals.sheet_state != "visible"
        self.tables = [(t.name, t.ref) for t in getattr(fmls, "tables", {}).values()]
        self.max_row, self.max_col = vals.max_row, vals.max_column
        if self.max_row == 1 and self.max_col == 1 and vals["A1"].value is None and fmls["A1"].value is None:
            self.max_row = self.max_col = 0

    def block(self, r1: int, c1: int, r2: int, c2: int, formulas: bool = False) -> list[list[Any]]:
        kw: dict[str, Any] = dict(min_row=r1, max_row=r2, min_col=c1, max_col=c2, values_only=True)
        f_rows = self.fmls.iter_rows(**kw)
        if formulas:
            return [[_formula_text(v) for v in row] for row in f_rows]
        out: list[list[Any]] = []
        for i, (v_row, f_row) in enumerate(zip(self.vals.iter_rows(**kw), f_rows)):
            out.append(
                [
                    self._calculate(r1 + i, c1 + j) if v is None and _is_formula(f) else v
                    for j, (v, f) in enumerate(zip(v_row, f_row))
                ]
            )
        return out

    def _calculate(self, row: int, col: int) -> Any:
        """Value of a formula cell that has no stored result, or NO_CACHE if it cannot be computed."""
        if self.calc is None:
            return NO_CACHE
        result = self.calc.value(self.name, row, col)
        if isinstance(result, XlError) and result == "#NAME?" and result.detail:
            self.unsupported[(row, col)] = result.detail
            return NO_CACHE
        self.computed.add((row, col))
        return result


class _CsvSheet(_Sheet):
    def __init__(self, name: str, rows: list[list[str]]) -> None:
        self.name, self.rows, self.tables = name, rows, []
        self.max_row = len(rows)
        self.max_col = max((len(r) for r in rows), default=0)

    def block(self, r1: int, c1: int, r2: int, c2: int, formulas: bool = False) -> list[list[Any]]:
        out = []
        for r in range(r1, r2 + 1):
            row = self.rows[r - 1] if r <= len(self.rows) else []
            out.append([(row[c - 1] if c <= len(row) and row[c - 1] != "" else None) for c in range(c1, c2 + 1)])
        return out


@dataclass
class _Book:
    path: Path
    kind: str
    sheets: list[_Sheet]
    active: str
    has_macros: bool = False
    _closers: list[Callable[[], None]] = field(default_factory=list)

    def get(self, name: str | None) -> _Sheet | None:
        if name is None:
            return next((s for s in self.sheets if s.name == self.active), self.sheets[0])
        for s in self.sheets:
            if s.name == name:
                return s
        return next((s for s in self.sheets if s.name.casefold() == name.strip().casefold()), None)

    def close(self) -> None:
        for fn in self._closers:
            try:
                fn()
            except Exception:  # closing is best effort
                log.debug("workbook close failed", exc_info=True)


class _OpenError(Exception):
    def __init__(self, result: ToolResult) -> None:
        super().__init__(result.content)
        self.result = result


def _decode_text(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _read_csv(path: Path) -> _Book:
    text = _decode_text(path.read_bytes())
    try:
        dialect: Any = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    return _Book(path, "csv", [_CsvSheet(path.stem, rows)], path.stem)


def _read_xlsx(path: Path) -> _Book:
    from openpyxl import load_workbook

    formulas = load_workbook(path, data_only=False)
    try:
        values = load_workbook(path, data_only=True)
    except BaseException:
        formulas.close()
        raise
    calc = Evaluator(_XlsxSource(values, formulas))
    sheets = [_XlsxSheet(values[n], formulas[n], calc) for n in formulas.sheetnames]
    macros = path.suffix.lower() == ".xlsm" or _zip_has(path, "xl/vbaProject.bin")
    book = _Book(path, path.suffix.lower().lstrip("."), sheets, formulas.active.title, macros)
    book._closers = [formulas.close, values.close]
    return book


def _zip_has(path: Path, *prefixes: str) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return any(n.startswith(p) for n in zf.namelist() for p in prefixes)
    except (zipfile.BadZipFile, OSError):
        return False


# ------------------------------------------------------------ path / open glue
def _suffix_problem(path: Path, allowed: set[str], action: str) -> ToolResult | None:
    suffix = path.suffix.lower()
    if suffix in allowed:
        return None
    if suffix == ".xls":
        return ToolResult.fail(
            f"{path.name} is a legacy .xls file, which these tools cannot {action}",
            "Open it in Excel, use Save As .xlsx, then use the .xlsx copy.",
        )
    if suffix in {".csv", ".tsv"}:
        return ToolResult.fail(
            f"{path.name} is a text file; these tools can only {action} .xlsx/.xlsm workbooks",
            "Use excel_create to build an .xlsx from the rows, or edit it with the file tools.",
        )
    return ToolResult.fail(
        f"{path.name} is not a spreadsheet ({suffix or 'no extension'})",
        "Spreadsheet tools handle .xlsx, .xlsm and .csv.",
    )


def _locate(raw: str) -> Path | ToolResult:
    try:
        path = resolve_path(raw)
    except ValueError as exc:
        return ToolResult.fail(str(exc), "Give a non-empty file path.")
    if not path.exists():
        return not_found(path, "file")
    if path.is_dir():
        return ToolResult.fail(f"{path} is a folder", "Give the path of the workbook file.")
    if (problem := _suffix_problem(path, READ_SUFFIXES, "read")) is not None:
        return problem
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return ToolResult.fail(f"{path.name} is {size // (1024 * 1024)} MB, too large to load")
    return path


def _unreadable(path: Path, detail: str) -> ToolResult:
    return ToolResult.fail(
        f"{path.name} could not be opened as a workbook ({detail})",
        "It may be corrupted, password-protected, or not really an Excel file.",
    )


def _open_book(path: Path) -> _Book:
    """Load a workbook, turning expected failures into _OpenError."""
    try:
        return _read_csv(path) if path.suffix.lower() in {".csv", ".tsv"} else _read_xlsx(path)
    except PermissionError:
        raise _OpenError(ToolResult.fail(f"cannot read {path.name}: access denied", LOCK_HINT)) from None
    except (zipfile.BadZipFile, KeyError, OSError, csv.Error, UnicodeError) as exc:
        raise _OpenError(_unreadable(path, type(exc).__name__)) from None
    except Exception as exc:  # openpyxl raises its own types (InvalidFileException, ...)
        if type(exc).__module__.startswith("openpyxl"):
            raise _OpenError(_unreadable(path, str(exc))) from None
        raise


def _pick_sheet(book: _Book, name: str | None) -> _Sheet | ToolResult:
    sheet = book.get(name)
    if sheet is not None:
        return sheet
    names = [s.name for s in book.sheets]
    close = difflib.get_close_matches(name or "", names, n=2, cutoff=0.5)
    hint = f"Sheets in {book.path.name}: {', '.join(repr(n) for n in names)}."
    if close:
        hint = f"Did you mean {' or '.join(repr(c) for c in close)}? " + hint
    return ToolResult.fail(f"no sheet named '{name}'", hint)


# ------------------------------------------------------------------- rendering
def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if value is NO_CACHE:
        return UNCALCULATED
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == time(0, 0) else value.isoformat(sep=" ", timespec="minutes")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else format(value, ".15g")
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "/")
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 1] + "…"


def _render(
    rows: list[tuple[int, list[Any]]],
    c1: int,
    budget: int = OUTPUT_BUDGET,
    computed: set[tuple[int, int]] | None = None,
) -> tuple[str, int]:
    """Grid with column letters on top and row numbers on the left. Returns (text, rows shown).

    Cells whose (row, col) is in ``computed`` get a trailing ``*``: calculated by Jarvis.
    """
    if not rows:
        return "", 0
    width = len(rows[0][1])
    label_w = max(len(str(r)) for r, _ in rows)
    lines = [" " * label_w + " | " + " | ".join(col_letter(c1 + i) for i in range(width))]
    used, shown = len(lines[0]), 0
    for number, values in rows:
        cells = [
            _cell_text(v) + ("*" if computed and (number, c1 + i) in computed else "")
            for i, v in enumerate(values)
        ]
        line = f"{number:>{label_w}} | " + " | ".join(cells)
        if shown and used + len(line) > budget:
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1
    return "\n".join(lines), shown


def _marks(sheet: _Sheet) -> set[tuple[int, int]]:
    return getattr(sheet, "computed", set())


def _unsupported_names(sheet: _Sheet) -> str:
    return ", ".join(sorted(set(getattr(sheet, "unsupported", {}).values())))


def _computed_note(count: int) -> str:
    return (
        f"{count} value(s) marked * were calculated by Jarvis's built-in formula evaluator because the "
        "file stores no results for them; they equal Excel's for the supported functions."
    )


def _dims(sheet: _Sheet) -> str:
    return "empty" if sheet.is_empty else f"A1:{col_letter(sheet.max_col)}{sheet.max_row}"


def _guess_header(rows: list[list[Any]]) -> int | None:
    """1-based index of the likely header row among the first rows, or None."""
    sample = rows[:10]
    widest = max((sum(v is not None for v in r) for r in sample), default=0)
    for i, row in enumerate(sample):
        filled = [v for v in row if v is not None]
        if not filled or len(filled) * 2 < widest:
            continue
        texts = [v for v in filled if isinstance(v, str) and not v.startswith("=") and parse_number(v) is None]
        if len(texts) == len(filled) and len({t.strip().casefold() for t in texts}) == len(texts):
            return i + 1
        return None  # the first substantial row is data, so there is no header
    return None


# ----------------------------------------------------------------- excel_overview
def _scan_formulas(sheet: _Sheet) -> tuple[int, int]:
    """(formula cells, of which have no cached value) on a worksheet."""
    if not isinstance(sheet, _XlsxSheet):
        return 0, 0
    total = uncached = scanned = 0
    for v_row, f_row in zip(sheet.vals.iter_rows(values_only=True), sheet.fmls.iter_rows(values_only=True)):
        scanned += len(f_row)
        for v, f in zip(v_row, f_row):
            if _is_formula(f):
                total += 1
                uncached += v is None
        if scanned > FORMULA_SCAN_CELLS:
            break
    return total, uncached


@tool(group=GROUP, timeout=60.0)
def excel_overview(path: str) -> ToolResult:
    """Summarise a spreadsheet: each sheet with its size, a guess at the header row, the first
    rows, named tables, and whether the workbook has formulas or macros. Use this first when
    asked about an .xlsx/.xlsm/.csv file, before reading or editing it. Returns text.

    Args:
        path: Workbook (.xlsx, .xlsm) or .csv file.
    """
    located = _locate(path)
    if isinstance(located, ToolResult):
        return located
    try:
        book = _open_book(located)
    except _OpenError as exc:
        return exc.result
    try:
        total_f = total_u = 0
        blocks = []
        for sheet in book.sheets:
            f, u = _scan_formulas(sheet)
            total_f, total_u = total_f + f, total_u + u
            blocks.append(_sheet_overview(sheet, f))
        flags = [
            f"formulas: {total_f}"
            + (f" ({total_u} with no cached value: never calculated by Excel; Jarvis calculates them when read)" if total_u else "")
            if total_f
            else "formulas: none",
            "macros: yes" if book.has_macros else "macros: no",
        ]
        head = f"{located.name}: {book.kind}, {located.stat().st_size:,} bytes, {len(book.sheets)} sheet(s); " + "; ".join(flags)
        return ToolResult.success(
            head + "\n" + "\n".join(blocks),
            data={"path": str(located), "sheets": [s.name for s in book.sheets]},
        )
    finally:
        book.close()


def _sheet_overview(sheet: _Sheet, formulas: int) -> str:
    title = f"Sheet '{sheet.name}' {_dims(sheet)}" + (" [hidden]" if sheet.hidden else "")
    if sheet.is_empty:
        return title
    lines = [title + (f", {formulas} formula cells" if formulas else "")]
    first = sheet.block(1, 1, min(sheet.max_row, 10), min(sheet.max_col, MAX_SHOWN_COLS))
    header = _guess_header(first)
    if header:
        names = [_cell_text(v) for v in first[header - 1] if v is not None]
        lines.append(f"  header row {header}: {', '.join(names)}")
    else:
        lines.append("  header row: none detected (first row looks like data)")
    if sheet.tables:
        lines.append("  tables: " + ", ".join(f"{n} ({ref})" for n, ref in sheet.tables))
    cols = min(sheet.max_col, 12)
    preview = sheet.block(1, 1, min(sheet.max_row, 5), cols)
    text, _ = _render([(i + 1, r) for i, r in enumerate(preview)], 1, computed=_marks(sheet))
    lines.append("  first rows:" + ("" if cols == sheet.max_col else f" (first {cols} of {sheet.max_col} columns)"))
    lines.extend("    " + ln for ln in text.splitlines())
    if any(r <= 5 and c <= cols for r, c in _marks(sheet)):
        lines.append("    (* = calculated by Jarvis; the file stores no result)")
    return "\n".join(lines)


# -------------------------------------------------------------------- excel_read
@tool(group=GROUP, timeout=60.0)
def excel_read(
    path: str,
    sheet: str | None = None,
    cell_range: str | None = None,
    formulas: bool = False,
    max_rows: int = 200,
) -> ToolResult:
    """Read cells from a spreadsheet as a grid with column letters across the top and row
    numbers down the side, so you can cite cells like B7. Use excel_overview first to learn
    the sheets. Shows calculated values by default; formulas=True shows formula text instead.
    A formula Excel never calculated shows "(uncalculated)", which is not the same as empty.
    Long sheets are cut at max_rows: call again with the cell_range given in the note to page.

    Args:
        path: Workbook (.xlsx, .xlsm) or .csv file.
        sheet: Sheet name. Default: the active (first) sheet.
        cell_range: A1-style range such as "A1:D20", a single cell "B2", columns "A:C" or rows "5:30". Default: the whole used area.
        formulas: Show formula text (e.g. =B2*1.18) instead of calculated values.
        max_rows: Maximum rows to return (1-1000).
    """
    located = _locate(path)
    if isinstance(located, ToolResult):
        return located
    try:
        bounds = parse_range(cell_range) if cell_range else None
    except ValueError as exc:
        return ToolResult.fail(str(exc), "Use a range like A1:D20, a cell like B2, columns A:C or rows 5:30.")
    try:
        book = _open_book(located)
    except _OpenError as exc:
        return exc.result
    try:
        picked = _pick_sheet(book, sheet)
        if isinstance(picked, ToolResult):
            return picked
        return _read_result(picked, bounds, formulas, max(1, min(max_rows, MAX_ROW_LIMIT)))
    finally:
        book.close()


def _read_result(
    sheet: _Sheet, bounds: tuple[int, int, int, int] | None, formulas: bool, max_rows: int
) -> ToolResult:
    if sheet.is_empty:
        return ToolResult.success(f"Sheet '{sheet.name}' is empty.")
    r1, c1, r2, c2 = bounds or (1, 1, sheet.max_row, sheet.max_col)
    clipped = bounds is not None and (r2 > sheet.max_row or c2 > sheet.max_col)
    r2, c2 = min(r2, sheet.max_row), min(c2, sheet.max_col)
    if r1 > r2 or c1 > c2:
        return ToolResult.fail(
            f"range is outside the used area of '{sheet.name}' ({_dims(sheet)})",
            "Pick a range inside the used area, or call without cell_range.",
        )
    last_col = min(c2, c1 + MAX_SHOWN_COLS - 1)
    end_row = min(r2, r1 + max_rows - 1)
    data = sheet.block(r1, c1, end_row, last_col, formulas)
    text, shown = _render([(r1 + i, row) for i, row in enumerate(data)], c1, computed=_marks(sheet))
    shown_last = r1 + shown - 1
    uncalculated = sum(v is NO_CACHE for row in data[:shown] for v in row)
    calculated = 0 if formulas else sum(
        r1 <= r < r1 + shown and c1 <= c <= last_col for r, c in _marks(sheet)
    )
    head = f"Sheet '{sheet.name}' {col_letter(c1)}{r1}:{col_letter(last_col)}{shown_last}"
    head += f" of used area {_dims(sheet)}" + (" (range clipped to used area)" if clipped else "")
    notes = []
    if shown_last < r2:
        nxt = f"{col_letter(c1)}{shown_last + 1}:{col_letter(c2)}{r2}"
        notes.append(f"{r2 - shown_last} more rows not shown; call again with cell_range='{nxt}'.")
    if last_col < c2:
        nxt = f"{col_letter(last_col + 1)}{r1}:{col_letter(c2)}{r2}"
        notes.append(f"{c2 - last_col} more columns not shown; call again with cell_range='{nxt}'.")
    if calculated:
        notes.append(_computed_note(calculated))
    if uncalculated:
        names = _unsupported_names(sheet)
        why = f" Jarvis cannot calculate {names}." if names else ""
        notes.append(
            f"{uncalculated} formula cell(s) show (uncalculated): no value is stored because the file was "
            f"never calculated by Excel (or was last saved by a script).{why} Use formulas=True to see the "
            "formulas; values appear once the file is opened and saved in Excel."
        )
    return ToolResult.success("\n".join([head, text, *notes]), data={"rows_shown": shown})


# ------------------------------------------------------------------ writing core
class _BadInput(Exception):
    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


def _write_risk(args: dict[str, Any]) -> Risk:
    try:
        return Risk.CONFIRM if resolve_path(args["path"]).exists() else Risk.SAFE
    except (ValueError, KeyError, OSError):
        return Risk.SAFE


def _file_label(args: dict[str, Any]) -> str:
    try:
        return resolve_path(args["path"]).name
    except (ValueError, KeyError):
        return str(args.get("path"))


def _write_describe(args: dict[str, Any]) -> str:
    n = len(args.get("cells") or {})
    return f"Write {n} cell{'s' * (n != 1)} to {args.get('sheet')} of {_file_label(args)} (the old file is kept as .bak)"


def _append_describe(args: dict[str, Any]) -> str:
    n = len(args.get("rows") or [])
    return f"Append {n} row{'s' * (n != 1)} to {args.get('sheet')} of {_file_label(args)} (the old file is kept as .bak)"


def _check_value(ref: str, value: Any) -> str | None:
    """Problem with a value to be written, or None."""
    if isinstance(value, float) and not math.isfinite(value):
        return f"{ref}: {value} cannot be stored in a cell"
    if isinstance(value, str):
        if len(value) > 32767:
            return f"{ref}: text longer than Excel's 32767-character cell limit"
        if value.startswith("="):
            return _check_formula(ref, value)
    return None


def _check_formula(ref: str, formula: str) -> str | None:
    depth, quoted = 0, False
    for ch in formula:
        if ch == '"':
            quoted = not quoted
        elif not quoted:
            depth += (ch == "(") - (ch == ")")
            if depth < 0:
                break
    if len(formula) < 2 or quoted or depth != 0:
        return (
            f"{ref}: formula {formula!r} is malformed (unbalanced parentheses or quotes); "
            "use English function names and commas, e.g. =SUM(B2:B9)"
        )
    if len(formula) > 8192:
        return f"{ref}: formula longer than Excel's 8192-character limit"
    return None


def _count_formulas(wb: Any) -> int:
    n = scanned = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            scanned += len(row)
            n += sum(1 for v in row if _is_formula(v))
            if scanned > FORMULA_SCAN_CELLS:
                return n
    return n


def _atomic_save(wb: Any, path: Path) -> Path | None:
    """Save beside the target and swap in with os.replace; returns the .bak path if one was made."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    bak = None
    try:
        wb.save(tmp)
        if path.exists():
            bak = make_backup(path)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return bak


MAX_REPORTED_FORMULAS = 12


def _recalc_reason(recalc: excel_calc.Recalc) -> str:
    return {
        "unavailable": f"{recalc.detail}, so the workbook was not recalculated",
        "locked": f"{recalc.detail}, so it was not recalculated (close it and write again to store values)",
    }.get(recalc.status, f"Excel recalculation did not complete ({recalc.detail}); the file is saved as written")


def _formula_report(
    path: Path,
    cells: list[tuple[str, str]],
    recalc: excel_calc.Recalc,
    n_formulas: int,
    qualify: bool = False,
) -> str:
    """Sentence(s) saying how the formulas just written were calculated, with their values."""
    shown = cells[:MAX_REPORTED_FORMULAS]
    values: list[str] = []
    try:
        book = _open_book(path)
    except _OpenError:
        book = None
    if book is not None:
        try:
            for sheet_name, ref in shown:
                sheet = book.get(sheet_name)
                row, col = parse_cell(ref)
                value = sheet.block(row, col, row, col)[0][0] if sheet else NO_CACHE
                label = f"{sheet_name}!{ref}" if qualify else ref
                text = _cell_text(value)
                if value is NO_CACHE and sheet is not None:
                    names = getattr(sheet, "unsupported", {}).get((row, col))
                    text = f"(uncalculated: {names} is not supported)" if names else text
                values.append(f"{label} = {text}")
        finally:
            book.close()
    listing = ", ".join(values) + (f" and {len(cells) - len(shown)} more" if len(cells) > len(shown) else "")
    if recalc.ok:
        head = f"Excel recalculated the workbook ({n_formulas} formula cell(s)); stored values are now current."
        return head + (f" Calculated: {listing}." if listing else "")
    head = f"{_recalc_reason(recalc)}."
    tail = (
        f" Calculated by Jarvis's built-in evaluator: {listing}." if listing else ""
    ) + " Excel will store the values the next time the file is opened and saved there; Jarvis calculates them on read until then."
    return head + tail


def _fidelity_note(path: Path) -> str:
    """Warn when the file holds content openpyxl may not round-trip exactly."""
    if path.suffix.lower() == ".xlsm":
        what = "macros (VBA) were kept, but form controls or shapes may not survive"
    elif _zip_has(path, "xl/drawings/", "xl/charts/", "xl/pivot"):
        what = "this file has charts, shapes or pivots that may lose detail when re-saved"
    else:
        return ""
    return f" Note: {what}; the .bak holds the original."


def _modify(raw_path: str, sheet: str, mutate: Callable[[Any], tuple[str, dict[str, Any]]]) -> ToolResult:
    """Shared pipeline: validate, load (or create), mutate one sheet, atomic save with .bak."""
    from openpyxl import Workbook, load_workbook
    from openpyxl.utils.exceptions import IllegalCharacterError

    try:
        path = resolve_path(raw_path)
    except ValueError as exc:
        return ToolResult.fail(str(exc), "Give a non-empty file path.")
    if problem := valid_sheet_name(sheet):
        return ToolResult.fail(problem, "Give a sheet name of up to 31 characters without [ ] : * ? / \\")
    if path.is_dir():
        return ToolResult.fail(f"{path} is a folder", "Give a file path including the file name.")
    if not path.suffix:
        path = path.with_suffix(".xlsx")
    if (bad_suffix := _suffix_problem(path, XLSX_SUFFIXES, "write to")) is not None:
        return bad_suffix

    existed = path.exists()
    notes: list[str] = []
    try:
        if existed:
            with open(path, "r+b"):  # fails with PermissionError while Excel holds the file
                pass
            wb = load_workbook(path, data_only=False, keep_vba=path.suffix.lower() == ".xlsm")
        else:
            if path.suffix.lower() == ".xlsm":
                return ToolResult.fail("cannot create a new macro-enabled .xlsm", "Use .xlsx for a new workbook.")
            wb = Workbook()
            wb.active.title = sheet
            notes.append(f"Created new workbook {path.name}.")
        try:
            names = wb.sheetnames
            if sheet not in names:
                match = next((n for n in names if n.casefold() == sheet.casefold()), None)
                if match:
                    sheet = match
                else:
                    wb.create_sheet(sheet)
                    notes.append(f"Created sheet '{sheet}' (existing sheets: {', '.join(names)}).")
            summary, data = mutate(wb[sheet])
            n_formulas = _count_formulas(wb)
            bak = _atomic_save(wb, path)
        finally:
            wb.close()
    except _BadInput as exc:
        return ToolResult.fail(str(exc), exc.hint)
    except PermissionError:
        return ToolResult.fail(f"cannot write {path.name}: access denied", LOCK_HINT)
    except IllegalCharacterError as exc:
        return ToolResult.fail(f"a value contains a control character Excel cannot store: {exc}", "Remove it and retry.")
    except (zipfile.BadZipFile, KeyError) as exc:
        return _unreadable(path, type(exc).__name__)
    except OSError as exc:
        return ToolResult.fail(f"cannot write {path}: {exc.strerror or exc}")

    parts = [f"{summary} in sheet '{sheet}' of {path.name}.", *notes]
    if n_formulas:
        recalc = excel_calc.recalculate(path)
        parts.append(_formula_report(path, [(sheet, r) for r in data.get("formulas", [])], recalc, n_formulas))
    if bak:
        parts.append(f"Previous version kept at {bak}.")
    text = " ".join(parts) + (_fidelity_note(path) if existed else "")
    return ToolResult.success(text, data={"path": str(path), "sheet": sheet, **data})


# -------------------------------------------------------------------- excel_write
@tool(group=GROUP, risk=_write_risk, describe=_write_describe, timeout=60.0)
def excel_write(path: str, sheet: str, cells: dict[str, str | int | float | bool | None]) -> ToolResult:
    """Set individual cells in an .xlsx/.xlsm workbook by A1 reference, e.g. {"B2": 42,
    "C2": "=B2*1.18"}. A string starting with "=" is stored as a formula (English function
    names, comma separators). null clears a cell. Creates the sheet, and the file if it does
    not exist. Existing formatting and other sheets are kept and the previous file is saved
    as .bak. Needs user approval when the file exists. After writing formulas the file is
    recalculated with Excel when it is installed, and the result lists the calculated values.
    Fails if the file is open in Excel.

    Args:
        path: Workbook to modify (.xlsx or .xlsm). .xls is not supported.
        sheet: Sheet name; created if missing.
        cells: Map of A1 reference to the new value, e.g. {"A1": "Total", "B1": "=SUM(B2:B9)"}.
    """
    if not cells:
        return ToolResult.fail("cells is empty", 'Give at least one cell, e.g. {"B2": 42}.')
    if len(cells) > MAX_WRITE_CELLS:
        return ToolResult.fail(f"{len(cells)} cells in one call is too many (max {MAX_WRITE_CELLS})", "Split it into several calls.")
    targets: dict[tuple[int, int], tuple[str, Any]] = {}
    problems = []
    for ref, value in cells.items():
        try:
            rc = parse_cell(ref)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        label = f"{col_letter(rc[1])}{rc[0]}"
        if rc in targets:
            problems.append(f"{ref} and {targets[rc][0]} are the same cell")
        elif bad := _check_value(label, value):
            problems.append(bad)
        else:
            targets[rc] = (label, value)
    if problems:
        return ToolResult.fail("; ".join(problems[:5]), "Nothing was written. Fix the cells and call again.")

    def mutate(ws: Any) -> tuple[str, dict[str, Any]]:
        for (row, col), (_, value) in targets.items():
            ws.cell(row=row, column=col).value = value
        formulas = [lbl for lbl, v in targets.values() if isinstance(v, str) and v.startswith("=")]
        refs = [lbl for lbl, _ in targets.values()]
        shown = ", ".join(refs[:12]) + (f" and {len(refs) - 12} more" if len(refs) > 12 else "")
        text = f"Wrote {len(refs)} cell{'s' * (len(refs) != 1)} ({shown})"
        if formulas:
            text += f"; {len(formulas)} formula{'s' * (len(formulas) != 1)} ({', '.join(formulas[:6])})"
        return text, {"cells": refs, "formulas": formulas}

    return _modify(path, sheet, mutate)


# --------------------------------------------------------------- excel_append_rows
def _last_used_row(ws: Any) -> int:
    # ws.max_row also counts formatted-but-empty rows, so look for real content.
    for r in range(ws.max_row, 0, -1):
        if any(c.value is not None for c in ws[r]):
            return r
    return 0


@tool(group=GROUP, risk=_write_risk, describe=_append_describe, timeout=60.0)
def excel_append_rows(
    path: str, sheet: str, rows: list[list[str | int | float | bool | None]]
) -> ToolResult:
    """Add rows below the last used row of a sheet, starting at column A. Strings starting
    with "=" are formulas. Creates the sheet (and file) if missing. Keeps formatting and
    other sheets, saves the previous file as .bak, needs user approval when the file exists.
    Formulas are recalculated with Excel when installed. Fails if open in Excel.

    Args:
        path: Workbook to modify (.xlsx or .xlsm).
        sheet: Sheet name; created if missing.
        rows: Rows to append; each row is a list of cell values in column order.
    """
    if not rows:
        return ToolResult.fail("rows is empty", 'Give at least one row, e.g. [["Rent", 1200]].')
    if len(rows) > MAX_WRITE_ROWS:
        return ToolResult.fail(f"{len(rows)} rows in one call is too many (max {MAX_WRITE_ROWS})", "Split it into several calls.")
    problems = []
    for i, row in enumerate(rows, 1):
        if len(row) > MAX_COLS:
            problems.append(f"row {i} has {len(row)} columns (max {MAX_COLS})")
            continue
        for j, value in enumerate(row, 1):
            if bad := _check_value(f"row {i} column {col_letter(j)}", value):
                problems.append(bad)
    if problems:
        return ToolResult.fail("; ".join(problems[:5]), "Nothing was written. Fix the rows and call again.")

    def mutate(ws: Any) -> tuple[str, dict[str, Any]]:
        start = _last_used_row(ws) + 1
        formulas: list[str] = []
        for i, row in enumerate(rows):
            for j, value in enumerate(row, 1):
                if value is not None:
                    ws.cell(row=start + i, column=j).value = value
                    if isinstance(value, str) and value.startswith("="):
                        formulas.append(f"{col_letter(j)}{start + i}")
        width = max(1, max(len(r) for r in rows))
        span = f"A{start}:{col_letter(width)}{start + len(rows) - 1}"
        return f"Appended {len(rows)} row{'s' * (len(rows) != 1)} at {span}", {"range": span, "formulas": formulas}

    return _modify(path, sheet, mutate)


# ------------------------------------------------------------------- excel_create
def _display_width(value: Any) -> int:
    text = _cell_text(value).split("\n")[0]
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _fit_columns(ws: Any, rows: list[list[CellValue]]) -> None:
    for j in range(max(len(r) for r in rows)):
        widest = max((_display_width(r[j]) for r in rows[:200] if j < len(r) and r[j] is not None), default=0)
        ws.column_dimensions[col_letter(j + 1)].width = max(8, min(widest + 2, 60))


def _check_create_input(sheets: dict[str, list[list[CellValue]]]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for name, rows in sheets.items():
        if bad := valid_sheet_name(name):
            problems.append(bad)
        elif name.casefold() in seen:
            problems.append(f"sheet name '{name}' is used twice (names are not case-sensitive)")
        seen.add(name.casefold())
        if len(rows) > MAX_CREATE_ROWS:
            problems.append(f"sheet '{name}' has {len(rows)} rows (max {MAX_CREATE_ROWS})")
            continue
        for i, row in enumerate(rows, 1):
            if len(row) > MAX_COLS:
                problems.append(f"sheet '{name}' row {i} has {len(row)} columns (max {MAX_COLS})")
                break
            problems.extend(
                bad for j, v in enumerate(row, 1) if (bad := _check_value(f"{name}!{col_letter(j)}{i}", v))
            )
    return problems


@tool(group=GROUP, timeout=60.0)
def excel_create(
    path: str,
    sheets: dict[str, list[list[str | int | float | bool | None]]],
    header_style: bool = True,
) -> ToolResult:
    """Create a new .xlsx workbook from rows of values, one list of rows per sheet name. With
    header_style the first row of each sheet is bold and frozen, and columns get sensible widths.
    Strings starting with "=" become formulas (calculated with Excel when installed). Never
    overwrites an existing file. A bare file name (no folder) is saved in the Documents folder.

    Args:
        path: New file path ending in .xlsx, or just a file name like "budget.xlsx".
        sheets: Map of sheet name to its rows, e.g. {"Budget": [["Item", "Cost"], ["Rent", 1200]]}.
        header_style: Make the first row bold and frozen.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils.exceptions import IllegalCharacterError

    if not sheets:
        return ToolResult.fail("sheets is empty", 'Give at least one sheet, e.g. {"Sheet1": [["A", "B"], [1, 2]]}.')
    raw = (path or "").strip().strip('"')
    if not raw:
        return ToolResult.fail("empty path", "Give a file name like budget.xlsx.")
    try:
        target = known_folder("Documents") / raw if Path(raw).name == raw else resolve_path(raw)
    except ValueError as exc:
        return ToolResult.fail(str(exc), "Give a valid file name.")
    if not target.suffix:
        target = target.with_suffix(".xlsx")
    if target.suffix.lower() != ".xlsx":
        return ToolResult.fail(
            f"new workbooks must be .xlsx, not {target.suffix}",
            "Use a name ending in .xlsx (macros cannot be created; legacy .xls is not supported).",
        )
    if target.exists():
        return ToolResult.fail(
            f"{target} already exists; excel_create never overwrites",
            "Choose a different file name, or modify it with excel_write / excel_append_rows.",
        )
    if problems := _check_create_input(sheets):
        return ToolResult.fail("; ".join(problems[:5]), "Nothing was created. Fix the input and call again.")

    wb = Workbook()
    wb.remove(wb.active)
    total_rows = 0
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(list(row))
        total_rows += len(rows)
        if rows:
            _fit_columns(ws, rows)
            if header_style:
                for cell in ws[1]:
                    cell.font = Font(bold=True)
                ws.freeze_panes = "A2"
    tmp = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        os.close(fd)
        wb.save(tmp)
        if target.exists():  # appeared while the workbook was being built
            raise FileExistsError(str(target))
        os.replace(tmp, target)
        tmp = None
    except FileExistsError:
        return ToolResult.fail(f"{target} already exists; excel_create never overwrites", "Choose a different file name.")
    except PermissionError:
        return ToolResult.fail(f"cannot write {target}: access denied", "Pick a folder you can write to.")
    except IllegalCharacterError as exc:
        return ToolResult.fail(f"a value contains a control character Excel cannot store: {exc}", "Remove it and retry.")
    except OSError as exc:
        return ToolResult.fail(f"cannot write {target}: {exc.strerror or exc}")
    finally:
        wb.close()
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    formula_cells = [
        (name, f"{col_letter(j)}{i}")
        for name, rows in sheets.items()
        for i, row in enumerate(rows, 1)
        for j, v in enumerate(row, 1)
        if _is_formula(v)
    ]
    msg = f"Created {target} with {len(sheets)} sheet(s) ({', '.join(sheets)}) and {total_rows} row(s)."
    if formula_cells:
        recalc = excel_calc.recalculate(target)
        msg += " " + _formula_report(target, formula_cells, recalc, len(formula_cells), qualify=len(sheets) > 1)
    return ToolResult.success(msg, data=str(target))


# -------------------------------------------------------------------- excel_query
def _resolve_column(sheet: _Sheet, header_row: int | None, headers: list[Any], column: str) -> int | str:
    """Column index for a header name or letter, or an error message."""
    wanted = column.strip().casefold()
    norm = [str(h).strip().casefold() if h not in (None, NO_CACHE) else "" for h in headers]
    if wanted in norm:
        return norm.index(wanted) + 1
    if re.fullmatch(r"[A-Za-z]{1,3}", column.strip()) and col_index(column.strip()) <= sheet.max_col:
        return col_index(column.strip())
    hits = [i for i, h in enumerate(norm, 1) if wanted and wanted in h]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        names = ", ".join(f"{col_letter(i)}={headers[i - 1]}" for i in hits)
        return f"'{column}' matches several headers ({names}); use the exact header or the column letter"
    listing = ", ".join(f"{col_letter(i)}={h}" for i, h in enumerate(headers, 1) if h not in (None, ""))
    return f"no column '{column}' in sheet '{sheet.name}'. " + (
        f"Columns: {listing}" if header_row else f"There is no header row; use a letter A-{col_letter(sheet.max_col)}"
    )


def _column_values(sheet: _Sheet, col: int, first_row: int) -> Iterator[tuple[int, Any]]:
    for start in range(first_row, sheet.max_row + 1, 2000):
        end = min(start + 1999, sheet.max_row)
        for i, row in enumerate(sheet.block(start, col, end, col)):
            yield start + i, row[0]


@tool(group=GROUP, timeout=60.0)
def excel_query(
    path: str,
    sheet: str | None = None,
    column: str | None = None,
    operation: Literal["sum", "average", "min", "max", "count", "unique", "find"] = "sum",
    match: str | None = None,
) -> ToolResult:
    """Answer quick questions about a spreadsheet without doing the arithmetic yourself:
    sum, average, min, max of a column; count its filled cells; list its distinct values; or
    find the rows containing some text. Numbers are parsed from text like "₹1,20,000.50",
    "(450)" or "12%" (percentages become fractions, 12% = 0.12). Reports how many cells were
    skipped as non-numeric. Uses calculated values, so uncalculated formulas are reported
    and skipped.

    Args:
        path: Workbook (.xlsx, .xlsm) or .csv file.
        sheet: Sheet name. Default: the active (first) sheet.
        column: Header name (e.g. "Amount") or column letter (e.g. "C"). Required except for find, where it optionally limits the search.
        operation: sum, average, min, max, count (filled cells), unique (distinct values with counts), or find (rows containing `match`).
        match: Text to look for (case-insensitive substring). Required for find.
    """
    located = _locate(path)
    if isinstance(located, ToolResult):
        return located
    if operation == "find" and not (match or "").strip():
        return ToolResult.fail("find needs `match`", "Give the text to look for, e.g. match='rent'.")
    if operation != "find" and not column:
        return ToolResult.fail(f"{operation} needs `column`", "Give a header name or column letter (see excel_overview).")
    try:
        book = _open_book(located)
    except _OpenError as exc:
        return exc.result
    try:
        picked = _pick_sheet(book, sheet)
        if isinstance(picked, ToolResult):
            return picked
        if picked.is_empty:
            return ToolResult.fail(f"sheet '{picked.name}' is empty")
        return _run_query(picked, column, operation, match)
    finally:
        book.close()


def _run_query(sheet: _Sheet, column: str | None, operation: str, match: str | None) -> ToolResult:
    top = sheet.block(1, 1, min(sheet.max_row, 10), sheet.max_col)
    header_row = _guess_header(top)
    headers = top[header_row - 1] if header_row else []
    first = (header_row or 0) + 1
    col = None
    if column:
        resolved = _resolve_column(sheet, header_row, headers, column)
        if isinstance(resolved, str):
            return ToolResult.fail(resolved, "Check the header names with excel_overview.")
        col = resolved
    if operation == "find":
        return _find(sheet, str(match), col, header_row, headers, first)
    assert col is not None
    label = f"'{headers[col - 1]}'" if header_row and headers[col - 1] is not None else f"column {col_letter(col)}"
    span = f"{col_letter(col)}{first}:{col_letter(col)}{sheet.max_row}"
    header_note = f"header row {header_row}" if header_row else "no header row"
    src = f"{label} ({span}, sheet '{sheet.name}', {header_note})"
    if operation == "unique":
        return _unique(sheet, col, first, src)
    return _aggregate(sheet, col, first, operation, src)


def _aggregate(sheet: _Sheet, col: int, first: int, operation: str, src: str) -> ToolResult:
    numbers: list[float] = []
    texts: list[tuple[int, Any]] = []
    empty = uncalc = errors = percents = 0
    for row, value in _column_values(sheet, col, first):
        if value is None or (isinstance(value, str) and not value.strip()):
            empty += 1
        elif value is NO_CACHE:
            uncalc += 1
        elif isinstance(value, str) and value.strip().upper() in ERROR_VALUES:
            errors += 1
        elif (n := parse_number(value)) is not None:
            numbers.append(n)
            percents += _is_percent(value)
        else:
            texts.append((row, value))
    skipped = _skipped_note(texts, col, uncalc, errors, empty, percents)
    calculated = sum(1 for r, c in _marks(sheet) if c == col and r >= first)
    if calculated:
        skipped += f" {calculated} formula value(s) in this column were calculated by Jarvis (the file stores no results)."
    if uncalc and _unsupported_names(sheet):
        skipped += f" Jarvis cannot calculate: {_unsupported_names(sheet)}."
    if operation == "count":
        filled = len(numbers) + len(texts) + errors + uncalc
        return ToolResult.success(f"Count of filled cells in {src}: {filled}.{skipped}", data={"count": filled})
    if not numbers:
        return ToolResult.fail(
            f"no numbers found in {src}.{skipped}",
            "Check the column choice with excel_overview; the column may hold text.",
        )
    result = {
        "sum": math.fsum(numbers),
        "average": math.fsum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }[operation]
    return ToolResult.success(
        f"{operation.capitalize()} of {src}: {_num(result)}\nBased on {len(numbers)} numeric cell(s).{skipped}",
        data={"value": result, "numeric_cells": len(numbers), "skipped_non_numeric": len(texts)},
    )


def _skipped_note(
    texts: list[tuple[int, Any]], col: int, uncalc: int, errors: int, empty: int, percents: int
) -> str:
    notes = []
    if texts:
        sample = ", ".join(f"{col_letter(col)}{r}={_cell_text(v)!r}" for r, v in texts[:3])
        notes.append(f"Skipped {len(texts)} non-numeric cell(s) (e.g. {sample})")
    else:
        notes.append("Skipped 0 non-numeric cells")
    if errors:
        notes.append(f"{errors} Excel error cell(s) skipped")
    if uncalc:
        notes.append(f"{uncalc} formula cell(s) skipped because they have no cached value (open and save in Excel)")
    if empty:
        notes.append(f"{empty} empty")
    if percents:
        notes.append("percent values counted as fractions (12% = 0.12)")
    return " " + "; ".join(notes) + "."


def _unique(sheet: _Sheet, col: int, first: int, src: str) -> ToolResult:
    counts: dict[str, int] = {}
    empty = 0
    for _, value in _column_values(sheet, col, first):
        text = _cell_text(value).strip() if value is not None else ""
        if text:
            counts[text] = counts.get(text, 0) + 1
        else:
            empty += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [f"{label}: {n}" for label, n in ranked[:50]]
    more = f"\n... and {len(ranked) - 50} more distinct values." if len(ranked) > 50 else ""
    return ToolResult.success(
        f"{len(ranked)} distinct value(s) in {src} ({empty} empty); value: count\n" + "\n".join(lines) + more,
        data={"distinct": len(ranked)},
    )


def _find(
    sheet: _Sheet, match: str, col: int | None, header_row: int | None, headers: list[Any], first: int
) -> ToolResult:
    needle = match.strip().casefold()
    width = min(sheet.max_col, MAX_SHOWN_COLS)
    lo, hi = (col - 1, col) if col else (0, width)
    hits: list[tuple[int, list[Any]]] = []
    total = 0
    for start in range(first, sheet.max_row + 1, 2000):
        end = min(start + 1999, sheet.max_row)
        for i, row in enumerate(sheet.block(start, 1, end, sheet.max_col)):
            if any(v is not None and v is not NO_CACHE and needle in _cell_text(v).casefold() for v in row[lo:hi]):
                total += 1
                if len(hits) < 50:
                    hits.append((start + i, row[:width]))
    where = f"column {col_letter(col)}" if col else f"sheet '{sheet.name}'"
    if not hits:
        return ToolResult.success(f"No rows contain {match!r} (case-insensitive) in {where}.", data={"rows": []})
    rows = hits
    if header_row:
        head = (headers + [None] * width)[:width]
        rows = [(header_row, head), *hits]
    text, _ = _render(rows, 1)
    note = f"\n... {total - len(hits)} more matching rows not shown; use a more specific match." if total > len(hits) else ""
    return ToolResult.success(
        f"{total} row(s) contain {match!r} (case-insensitive) in {where}:\n{text}{note}",
        data={"rows": [r for r, _ in hits], "total": total},
    )
