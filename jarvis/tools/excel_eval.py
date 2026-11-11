"""A small, safe Excel formula evaluator (no ``eval``).

openpyxl writes formulas but cannot calculate them, so a workbook saved by a script has no
values until Excel opens it. This module is the fallback for that case: numbers, strings,
booleans, cell references and ranges (``A1``, ``Sheet!A1``, ``'My sheet'!A1:B3``), arithmetic,
comparison, ``&``, percent, unary minus, and SUM AVERAGE MIN MAX COUNT COUNTA ROUND ABS IF AND
OR NOT SUMIF COUNTIF AVERAGEIF. It is not a spreadsheet engine: anything else evaluates to
``#NAME?`` with ``.detail`` naming the unsupported function, so callers can say what was missed.
"""

from __future__ import annotations

import fnmatch
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable, Iterable, Protocol

MAX_RANGE_CELLS = 200_000


class XlError(str):
    """An Excel error value such as ``#DIV/0!``; ``detail`` says why, for the human."""

    detail: str = ""

    def __new__(cls, code: str, detail: str = "") -> "XlError":
        obj = super().__new__(cls, code)
        obj.detail = detail
        return obj


ERROR_CODES = frozenset({"#DIV/0!", "#N/A", "#NAME?", "#NULL!", "#NUM!", "#REF!", "#VALUE!"})


class Source(Protocol):
    def dims(self, sheet: str) -> tuple[int, int] | None:
        """(max_row, max_col) of a sheet, or None if there is no such sheet."""

    def cell(self, sheet: str, row: int, col: int) -> tuple[Any, Any]:
        """(content, cached value): content is a value or a '=formula' string."""


FUNCTIONS = frozenset(
    "SUM AVERAGE MIN MAX COUNT COUNTA ROUND ABS IF AND OR NOT SUMIF COUNTIF AVERAGEIF".split()
)


class _Bad(Exception):
    def __init__(self, error: XlError) -> None:
        super().__init__(str(error))
        self.error = error


# ------------------------------------------------------------------- tokenizer
_TOKEN = re.compile(
    r"""\s*(?:
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
   |(?P<str>"(?:[^"]|"")*")
   |(?P<err>\#(?:DIV/0!|N/A|NAME\?|NULL!|NUM!|REF!|VALUE!))
   |(?P<ref>(?:(?:'(?:[^']|'')+'|[A-Za-z_][\w.]*)!)?\$?[A-Za-z]{1,3}\$?\d+(?::\$?[A-Za-z]{1,3}\$?\d+)?(?![\w(]))
   |(?P<func>[A-Za-z_][\w.]*)(?=\()
   |(?P<bool>TRUE|FALSE)(?![\w(])
   |(?P<op><>|<=|>=|[-+*/^&=<>%(),])
   )""",
    re.VERBOSE | re.IGNORECASE,
)
_CELL = re.compile(r"\$?([A-Za-z]{1,3})\$?(\d+)")


def _tokenize(text: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    while text[pos:].strip():
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            out.append(("bad", text[pos:].strip()[:20]))  # reported only if the parser reaches it
            break
        kind = m.lastgroup or ""
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + ord(ch) - 64
    return n


# ---------------------------------------------------------------------- parser
class _Parser:
    """Recursive descent: comparison < & < +,- < *,/ < ^ < unary minus < percent."""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.t, self.i = tokens, 0

    def peek(self) -> tuple[str, str]:
        return self.t[self.i] if self.i < len(self.t) else ("end", "")

    def take(self) -> tuple[str, str]:
        tok = self.peek()
        self.i += 1
        return tok

    def accept(self, *ops: str) -> str | None:
        kind, val = self.peek()
        if kind == "op" and val in ops:
            self.i += 1
            return val
        return None

    def parse(self) -> tuple:
        node = self.compare()
        if self.peek()[0] != "end":
            raise _Bad(XlError("#VALUE!", "unexpected " + self.peek()[1]))
        return node

    def _binary(self, sub: Callable[[], tuple], ops: tuple[str, ...]) -> tuple:
        node = sub()
        while (op := self.accept(*ops)) is not None:
            node = ("bin", op, node, sub())
        return node

    def compare(self) -> tuple:
        return self._binary(self.concat, ("=", "<>", "<=", ">=", "<", ">"))

    def concat(self) -> tuple:
        return self._binary(self.add, ("&",))

    def add(self) -> tuple:
        return self._binary(self.mul, ("+", "-"))

    def mul(self) -> tuple:
        return self._binary(self.power, ("*", "/"))

    def power(self) -> tuple:
        return self._binary(self.unary, ("^",))

    def unary(self) -> tuple:
        if (op := self.accept("-", "+")) is not None:
            inner = self.unary()
            return ("neg", inner) if op == "-" else inner
        node = self.primary()
        while self.accept("%"):
            node = ("pct", node)
        return node

    def primary(self) -> tuple:
        kind, val = self.take()
        if kind == "num":
            return ("val", float(val))
        if kind == "str":
            return ("val", val[1:-1].replace('""', '"'))
        if kind == "bool":
            return ("val", val.upper() == "TRUE")
        if kind == "err":
            return ("val", XlError(val.upper()))
        if kind == "ref":
            return _ref_node(val)
        if kind == "func":
            return self.call(val.upper())
        if kind == "op" and val == "(":
            node = self.compare()
            if not self.accept(")"):
                raise _Bad(XlError("#VALUE!", "missing )"))
            return node
        if kind == "bad":
            raise _Bad(XlError("#NAME?", val))
        raise _Bad(XlError("#VALUE!", f"unexpected {val or 'end of formula'}"))

    def call(self, name: str) -> tuple:
        if name not in FUNCTIONS:  # fail before parsing arguments we could not read anyway
            raise _Bad(XlError("#NAME?", name))
        self.take()  # the "("
        args: list[tuple] = []
        if not self.accept(")"):
            while True:
                args.append(self.compare())
                if self.accept(")"):
                    break
                if not self.accept(","):
                    raise _Bad(XlError("#VALUE!", f"bad arguments for {name}"))
        return ("call", name, args)


def _ref_node(text: str) -> tuple:
    sheet = None
    if "!" in text:
        sheet, text = text.rsplit("!", 1)
        sheet = sheet[1:-1].replace("''", "'") if sheet.startswith("'") else sheet
    parts = [_CELL.fullmatch(p) for p in text.split(":")]
    (r1, c1), (r2, c2) = [(int(m.group(2)), _col_index(m.group(1))) for m in (parts[0], parts[-1])]  # type: ignore[union-attr]
    return ("ref", sheet, min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2), len(parts) == 2)


# ------------------------------------------------------------------- semantics
def _to_num(v: Any) -> float:
    if isinstance(v, XlError):
        raise _Bad(v)
    if v is None:
        return 0.0
    if isinstance(v, (bool, int, float)):
        return float(v)
    if isinstance(v, str):
        text = v.strip()
        try:
            return float(text[:-1]) / 100 if text.endswith("%") else float(text)
        except ValueError:
            pass
    raise _Bad(XlError("#VALUE!", f"{v!r} is not a number"))


def _to_text(v: Any) -> str:
    if isinstance(v, XlError):
        raise _Bad(v)
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() and abs(v) < 1e15 else format(v, ".15g")
    return str(v)


def _to_bool(v: Any) -> bool:
    if isinstance(v, str) and not isinstance(v, XlError):
        if v.upper() in ("TRUE", "FALSE"):
            return v.upper() == "TRUE"
        raise _Bad(XlError("#VALUE!", f"{v!r} is not TRUE or FALSE"))
    return bool(_to_num(v))


def _rank(v: Any) -> tuple[int, Any]:
    """Excel's ordering: numbers < text < booleans; text compares case-insensitively."""
    if v is None:
        return (0, 0.0)
    if isinstance(v, bool):
        return (3, v)
    if isinstance(v, (int, float)):
        return (0, float(v))
    return (2, str(v).casefold())


def _compare(op: str, a: Any, b: Any) -> bool:
    for v in (a, b):
        if isinstance(v, XlError):
            raise _Bad(v)
    ra, rb = _rank(a), _rank(b)
    if a is None and isinstance(b, str):
        ra = (2, "")
    if b is None and isinstance(a, str):
        rb = (2, "")
    return {"=": ra == rb, "<>": ra != rb, "<": ra < rb, ">": ra > rb, "<=": ra <= rb, ">=": ra >= rb}[op]


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _criterion(crit: Any) -> Callable[[Any], bool]:
    """Predicate for SUMIF-style criteria: ">100", "<>0", "food", "a*", or a plain value."""
    if isinstance(crit, XlError):
        raise _Bad(crit)
    if not isinstance(crit, str):
        want = _to_num(crit)
        return lambda v: _is_number(v) and float(v) == want
    m = re.match(r"(<>|<=|>=|=|<|>)?(.*)", crit, re.DOTALL)
    op, rest = (m.group(1) or "="), m.group(2)  # type: ignore[union-attr]
    try:
        number: float | None = float(rest)
    except ValueError:
        number = None

    def test(v: Any) -> bool:
        if isinstance(v, XlError):
            return False
        if number is not None:
            return _compare(op, float(v), number) if _is_number(v) else op == "<>"
        if op in ("=", "<>"):
            blank = v is None or v == ""
            hit = blank if not rest else (not blank and fnmatch.fnmatchcase(str(v).casefold(), rest.casefold()))
            return hit if op == "=" else not hit
        return isinstance(v, str) and _compare(op, v, rest)

    return test


def _round_half_up(x: float, digits: int) -> float:
    if digits >= 15:
        return x
    quant = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(abs(x))).quantize(quant, rounding=ROUND_HALF_UP)) * (-1 if x < 0 else 1)


def _wrap(v: Any) -> Any:
    return XlError(v) if isinstance(v, str) and v in ERROR_CODES else v


class Evaluator:
    """Evaluates the formula cells of one workbook through a ``Source``; results are memoised."""

    def __init__(self, source: Source) -> None:
        self.src = source
        self._memo: dict[tuple[str, int, int], Any] = {}
        self._active: set[tuple[str, int, int]] = set()

    # ---- public
    def value(self, sheet: str, row: int, col: int) -> Any:
        """The value of one cell: stored, or calculated here if it is an uncached formula."""
        content, cached = self.src.cell(sheet, row, col)
        if not (isinstance(content, str) and content.startswith("=")):
            return _wrap(content)
        if cached is not None:
            return _wrap(cached)
        key = (sheet, row, col)
        if key in self._memo:
            return self._memo[key]
        if key in self._active:
            return XlError("#REF!", "circular reference")
        self._active.add(key)
        try:
            result = self.formula(content, sheet)
        finally:
            self._active.discard(key)
        self._memo[key] = result
        return result

    def formula(self, text: str, sheet: str) -> Any:
        """Evaluate formula text (with or without the leading '=') in the context of a sheet."""
        try:
            tree = _Parser(_tokenize(text[1:] if text.startswith("=") else text)).parse()
            out = self._scalar(tree, sheet)
        except _Bad as bad:
            return bad.error
        except RecursionError:
            return XlError("#VALUE!", "formula too deeply nested")
        except OverflowError:
            return XlError("#NUM!")
        if out is None:  # a reference to an empty cell reads as 0, like Excel
            return 0
        return int(out) if isinstance(out, float) and out.is_integer() and abs(out) < 1e15 else out

    # ---- evaluation
    def _eval(self, node: tuple, sheet: str) -> Any:
        kind = node[0]
        if kind == "val":
            return node[1]
        if kind == "ref":
            return self._ref(node, sheet)
        if kind == "neg":
            return -_to_num(self._scalar(node[1], sheet))
        if kind == "pct":
            return _to_num(self._scalar(node[1], sheet)) / 100
        if kind == "bin":
            return self._binary(node[1], self._scalar(node[2], sheet), self._scalar(node[3], sheet))
        return self._call(node[1], node[2], sheet)

    def _scalar(self, node: tuple, sheet: str) -> Any:
        v = self._eval(node, sheet)
        if isinstance(v, list):
            raise _Bad(XlError("#VALUE!", "a range was used where one value is needed"))
        return v

    def _binary(self, op: str, a: Any, b: Any) -> Any:
        if op in ("=", "<>", "<", ">", "<=", ">="):
            return _compare(op, a, b)
        if op == "&":
            return _to_text(a) + _to_text(b)
        x, y = _to_num(a), _to_num(b)
        if op == "+":
            return x + y
        if op == "-":
            return x - y
        if op == "*":
            return x * y
        if op == "/":
            if y == 0:
                raise _Bad(XlError("#DIV/0!"))
            return x / y
        if x == 0 and y < 0:
            raise _Bad(XlError("#DIV/0!"))
        result = x**y
        if isinstance(result, complex) or not math.isfinite(result):
            raise _Bad(XlError("#NUM!"))
        return result

    def _ref(self, node: tuple, sheet: str) -> Any:
        _, name, r1, c1, r2, c2, is_range = node
        target = name or sheet
        dims = self.src.dims(target)
        if dims is None:
            raise _Bad(XlError("#REF!", f"no sheet named {target!r}"))
        if not is_range:
            return self.value(target, r1, c1)
        r2, c2 = min(r2, dims[0]), min(c2, dims[1])  # beyond the used area every cell is empty
        if (r2 - r1 + 1) * (c2 - c1 + 1) > MAX_RANGE_CELLS:
            raise _Bad(XlError("#VALUE!", "range too large"))
        return [self.value(target, r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)]

    def _flat(self, args: list[tuple], sheet: str) -> Iterable[tuple[Any, bool]]:
        """(value, from_range) for every argument value, ranges expanded."""
        for a in args:
            v = self._eval(a, sheet)
            if isinstance(v, list):
                yield from ((x, True) for x in v)
            else:
                yield v, False

    def _numbers(self, args: list[tuple], sheet: str) -> list[float]:
        out = []
        for v, from_range in self._flat(args, sheet):
            if isinstance(v, XlError):
                raise _Bad(v)
            if from_range:  # text, blanks and booleans inside a range are ignored
                if _is_number(v):
                    out.append(float(v))
            elif v is not None:
                out.append(_to_num(v))
        return out

    def _call(self, name: str, args: list[tuple], sheet: str) -> Any:
        if name == "IF":
            if not 2 <= len(args) <= 3:
                raise _Bad(XlError("#VALUE!", "IF needs 2 or 3 arguments"))
            if _to_bool(self._scalar(args[0], sheet)):
                return self._scalar(args[1], sheet)
            return self._scalar(args[2], sheet) if len(args) == 3 else False
        if name in ("SUM", "AVERAGE", "MIN", "MAX"):
            nums = self._numbers(args, sheet)
            if name == "SUM":
                return math.fsum(nums)
            if name == "AVERAGE":
                if not nums:
                    raise _Bad(XlError("#DIV/0!"))
                return math.fsum(nums) / len(nums)
            return (min if name == "MIN" else max)(nums) if nums else 0.0
        if name in ("COUNT", "COUNTA"):
            vals = [v for v, _ in self._flat(args, sheet)]
            if name == "COUNTA":
                return float(sum(v is not None for v in vals))
            return float(sum(_is_number(v) for v in vals))
        if name in ("AND", "OR"):
            flags = [_to_bool(v) for v, _ in self._flat(args, sheet) if v is not None]
            if not flags:
                raise _Bad(XlError("#VALUE!", f"{name} has no logical values"))
            return all(flags) if name == "AND" else any(flags)
        if name in ("NOT", "ABS", "ROUND"):
            return self._simple_fn(name, args, sheet)
        if name in ("SUMIF", "COUNTIF", "AVERAGEIF"):
            return self._conditional(name, args, sheet)
        raise _Bad(XlError("#NAME?", name))

    def _simple_fn(self, name: str, args: list[tuple], sheet: str) -> Any:
        if name == "ROUND" and len(args) == 1:
            args = [*args, ("val", 0.0)]
        if len(args) != (2 if name == "ROUND" else 1):
            raise _Bad(XlError("#VALUE!", f"wrong number of arguments for {name}"))
        first = self._scalar(args[0], sheet)
        if name == "NOT":
            return not _to_bool(first)
        x = _to_num(first)
        if name == "ABS":
            return abs(x)
        return _round_half_up(x, int(_to_num(self._scalar(args[1], sheet))))

    def _conditional(self, name: str, args: list[tuple], sheet: str) -> Any:
        if not 2 <= len(args) <= 3 or (name == "COUNTIF" and len(args) != 2):
            raise _Bad(XlError("#VALUE!", f"wrong number of arguments for {name}"))
        crit_range = self._eval(args[0], sheet)
        crit_range = crit_range if isinstance(crit_range, list) else [crit_range]
        test = _criterion(self._scalar(args[1], sheet))
        summed = crit_range
        if len(args) == 3:
            other = self._eval(args[2], sheet)
            summed = other if isinstance(other, list) else [other]
        hits = [s for c, s in zip(crit_range, summed) if test(c)]
        if name == "COUNTIF":
            return float(len(hits))
        nums = [float(s) for s in hits if _is_number(s)]
        if name == "SUMIF":
            return math.fsum(nums)
        if not nums:
            raise _Bad(XlError("#DIV/0!"))
        return math.fsum(nums) / len(nums)
