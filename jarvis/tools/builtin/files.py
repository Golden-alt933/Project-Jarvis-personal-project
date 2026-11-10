"""File tools: find, read, search, write, edit, move, copy, delete, open.

Ported from Friday's file_skill / code_skill: the fuzzy name scoring (tuned for
spoken names), exact-once edit with a backup, capped reads. The Skill/Task
wrappers and every piece of sentence parsing are gone; the model supplies typed
arguments.

Safety model: deletes go to the Recycle Bin, replaced or edited files keep a
``.bak`` beside them, and move/copy never overwrite.
"""

from __future__ import annotations

import difflib
import fnmatch
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Literal

from send2trash import send2trash

from jarvis.tools.base import Risk, ToolResult, tool

log = logging.getLogger(__name__)

GROUP = "files"

# Never descended into when searching: huge, generated, or never asked for by name.
EXCLUDED_DIRS = frozenset(
    d.lower()
    for d in (
        "AppData", "Application Data", "node_modules", "__pycache__", ".git", ".venv",
        "venv", "env", ".idea", ".vscode", "$RECYCLE.BIN", "System Volume Information",
        "site-packages", ".cache", ".next", "build", "target", ".mypy_cache",
        ".pytest_cache", ".tox", ".gradle", ".npm", ".nuget", ".cargo", ".rustup",
    )
)

# Opening these runs code (or edits the registry), so open_path asks first.
EXECUTABLE_SUFFIXES = frozenset(
    {
        ".exe", ".bat", ".cmd", ".com", ".scr", ".pif", ".msi", ".msp", ".ps1", ".psm1",
        ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".hta", ".cpl", ".jar", ".reg",
        ".inf", ".lnk", ".url", ".py", ".pyw",
    }
)

# Windows device names: os.path.exists("NUL") is True and nobody means that.
RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

DOCUMENT_HINTS = {
    ".pdf": "a PDF; use the PDF/document tools to read it",
    ".docx": "a Word document; use the document tools to read it",
    ".doc": "a Word document; use the document tools to read it",
    ".xlsx": "an Excel workbook; use the spreadsheet/document tools to read it",
    ".xls": "an Excel workbook; use the spreadsheet/document tools to read it",
    ".pptx": "a PowerPoint file; use the document tools to read it",
}
BINARY_KINDS = {
    ".png": "an image", ".jpg": "an image", ".jpeg": "an image", ".gif": "an image",
    ".bmp": "an image", ".webp": "an image", ".ico": "an image", ".mp3": "audio",
    ".wav": "audio", ".flac": "audio", ".mp4": "a video", ".mkv": "a video",
    ".avi": "a video", ".mov": "a video", ".zip": "a zip archive", ".7z": "an archive",
    ".rar": "an archive", ".gz": "an archive", ".tar": "an archive", ".exe": "an executable",
    ".dll": "a library", ".pyc": "compiled Python", ".sqlite": "a database", ".db": "a database",
}

MAX_READ_BYTES = 20 * 1024 * 1024
MAX_LINE_CHARS = 2000
MAX_GREP_BYTES = 2 * 1024 * 1024
FIND_TIME_BUDGET = 8.0
GREP_TIME_BUDGET = 15.0
FIND_MAX_DEPTH = 5
MAX_ENTRIES = 200_000

# Fuzzy-score bands do not overlap: exact > prefix > substring > token match > fuzzy.
MIN_SUBSTRING_LEN = 3
FUZZY_MIN = 0.75
FUZZY_FLOOR, FUZZY_CEILING = 0.55, 0.70
MIN_SCORE = 0.55


# --------------------------------------------------------------------- paths
def _home() -> Path:
    return Path(os.path.expanduser("~"))


def resolve_path(raw: str) -> Path:
    """Absolute path from ``~``, %VARS%/$VARS, either slash style, or a name relative to home."""
    text = (raw or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError("empty path")
    expanded = os.path.expanduser(os.path.expandvars(text))
    path = Path(expanded)
    # Checked before abspath(), which rewrites device names into odd device paths.
    if path.name.split(".")[0].upper() in RESERVED_NAMES:
        raise ValueError(f"'{raw}' is a reserved Windows device name, not a file")
    if not path.is_absolute() and not path.drive:
        path = _home() / path
    return Path(os.path.abspath(path))


def _norm(path: Path | str) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def _protected_reason(path: Path) -> str | None:
    """Why this path must never be moved or deleted, or None."""
    real = _norm(path)
    if Path(real).parent == Path(real):
        return "it is a drive root"
    home = _norm(_home())
    if real == home:
        return "it is your home folder"
    if real == _norm(_home().parent):
        return "it is the Users folder"
    for var in ("SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData"):
        base = os.environ.get(var)
        if not base:
            continue
        b = _norm(base)
        if real == b or real.startswith(b + os.sep):
            return f"it is inside the system folder {base}"
    return None


def _similar_names(path: Path) -> tuple[Path, list[str]]:
    """Nearest existing ancestor and up to 3 entries there that resemble the missing name."""
    missing = path.name
    parent = path.parent
    while not parent.exists() and parent.parent != parent:
        missing = parent.name
        parent = parent.parent
    try:
        names = os.listdir(parent)
    except OSError:
        return parent, []
    close = difflib.get_close_matches(missing, names, n=3, cutoff=0.4)
    if len(close) < 3:  # substring hits catch spoken partial names
        low = missing.lower()
        for n in names:
            if n not in close and low and low in n.lower():
                close.append(n)
            if len(close) == 3:
                break
    return parent, close


def _missing(path: Path, what: str = "path") -> ToolResult:
    parent, close = _similar_names(path)
    if close:
        hint = f"In {parent} there is: {', '.join(close)}. Use find_files if unsure."
    else:
        hint = f"The nearest existing folder is {parent}. Use find_files or list_directory to look around."
    return ToolResult.fail(f"{what} does not exist: {path}", hint)


def _bad_path(exc: ValueError) -> ToolResult:
    return ToolResult.fail(str(exc), "Give a non-empty path.")


def _fmt_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _stat_line(path: Path) -> str:
    try:
        st = path.stat()
    except OSError:
        return f"{path}"
    if path.is_dir():
        return f"{path}  <folder>  modified {_fmt_time(st.st_mtime)}"
    return f"{path}  {_fmt_size(st.st_size)}  modified {_fmt_time(st.st_mtime)}"


# ------------------------------------------------------------------ encoding
def _decode(raw: bytes) -> tuple[str, str]:
    """Decode text bytes: UTF-16/UTF-8 BOM, then utf-8, then cp1252."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "cp1252"


def _is_binary(raw: bytes) -> bool:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16 text is full of NULs
        return False
    return b"\x00" in raw[:4096]


def _encode(text: str, encoding: str) -> bytes:
    if encoding == "utf-8-sig":
        return b"\xef\xbb\xbf" + text.encode("utf-8")
    if encoding == "utf-16":
        return b"\xff\xfe" + text.encode("utf-16-le")
    return text.encode(encoding, errors="replace")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _backup(path: Path) -> Path:
    bak = path.with_name(path.name + ".bak")
    shutil.copy2(path, bak)
    return bak


# -------------------------------------------------------------------- search
def _score(needle: str, filename: str) -> float:
    """How well a file name matches a (possibly spoken) name; 0 means no match."""
    needle = (needle or "").strip().lower()
    if not needle:
        return 0.0
    full = filename.lower()
    stem = os.path.splitext(full)[0]
    if needle == full:
        return 1.0
    if needle == stem:
        return 0.98
    best = 0.0
    for cand in (full, stem):
        if not cand:
            continue
        if len(needle) >= MIN_SUBSTRING_LEN:
            if cand.startswith(needle):
                best = max(best, 0.89 + 0.08 * len(needle) / len(cand))
            elif needle in cand:
                best = max(best, 0.76 + 0.11 * len(needle) / len(cand))
        # The needle may hold more words than the name ("budget spreadsheet" vs budget.xlsx);
        # only whole-word occurrences count so "sum" does not match "resume".
        if len(cand) >= MIN_SUBSTRING_LEN and re.search(rf"\b{re.escape(cand)}\b", needle):
            best = max(best, 0.76 + 0.11 * len(cand) / len(needle))
        tokens = [t for t in re.split(r"[\s_\-.]+", needle) if len(t) >= 2]
        if len(tokens) > 1 and all(t in cand for t in tokens):
            best = max(best, 0.72)
        ratio = difflib.SequenceMatcher(None, needle, cand).ratio()
        if ratio >= FUZZY_MIN:
            frac = (ratio - FUZZY_MIN) / (1.0 - FUZZY_MIN)
            best = max(best, FUZZY_FLOOR + (FUZZY_CEILING - FUZZY_FLOOR) * frac)
    return best


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self.end = time.monotonic() + seconds
        self.expired = False

    def hit(self) -> bool:
        if time.monotonic() > self.end:
            self.expired = True
        return self.expired


def _walk(
    root: Path,
    max_depth: int,
    deadline: _Deadline,
    budget: list[int],
    skip: frozenset[str] = frozenset(),
) -> Iterator[tuple[Path, bool]]:
    """Yield (path, is_dir) under root, pruned by noise dirs, depth, time and entry budget."""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if deadline.hit():
            return
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        subdirs: list[Path] = []
        for entry in entries:
            budget[0] -= 1
            if budget[0] <= 0:
                deadline.expired = True
                return
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            path = Path(entry.path)
            yield path, is_dir
            if is_dir and depth + 1 < max_depth:
                low = entry.name.lower()
                if low in EXCLUDED_DIRS or entry.name.startswith(".") or _norm(path) in skip:
                    continue
                subdirs.append(path)
        stack.extend((d, depth + 1) for d in reversed(subdirs))


def _default_roots() -> list[tuple[Path, int]]:
    home = _home()
    names = ["Desktop", "OneDrive/Desktop", "Documents", "OneDrive/Documents", "Downloads",
             "Pictures", "Videos", "Music"]
    roots = [(home.joinpath(*n.split("/")), FIND_MAX_DEPTH) for n in names]
    roots = [(p, d) for p, d in roots if p.is_dir()]
    roots.append((home, 2))  # loose files and top-level project folders
    return roots


@tool(group=GROUP, timeout=30.0)
def find_files(
    name: str,
    folder: str | None = None,
    kind: Literal["any", "file", "folder"] = "any",
    limit: int = 10,
) -> ToolResult:
    """Find files or folders by (approximate) name. Use this when the user names a file but
    you do not know its location. Tolerates partial names, spelling slips and extra words.
    Returns ranked absolute paths with size and modified time.

    Args:
        name: The file or folder name, or part of it, e.g. "budget" or "thesis draft".
        folder: Optional folder to search under. Default: Desktop, Documents, Downloads, Pictures, Videos, Music and the home folder.
        kind: Restrict results to files, folders, or either.
        limit: Maximum results to return.
    """
    if not name.strip():
        return ToolResult.fail("name is empty", "Give part of the file or folder name.")
    limit = max(1, min(limit, 50))
    if folder:
        try:
            base = resolve_path(folder)
        except ValueError as exc:
            return _bad_path(exc)
        if not base.is_dir():
            return _missing(base, "folder")
        roots = [(base, FIND_MAX_DEPTH + 1)]
    else:
        roots = _default_roots()

    deadline, budget = _Deadline(FIND_TIME_BUDGET), [MAX_ENTRIES]
    seen: set[str] = set()
    scored: list[tuple[float, Path, bool]] = []
    searched: set[str] = set()
    for root, depth in roots:
        if deadline.expired:
            break
        # Folders searched earlier (e.g. Desktop) are skipped when scanning the home folder.
        skip = frozenset(searched)
        for path, is_dir in _walk(root, depth, deadline, budget, skip):
            if kind == "file" and is_dir or kind == "folder" and not is_dir:
                continue
            score = _score(name, path.name)
            if score >= MIN_SCORE:
                key = _norm(path)
                if key not in seen:
                    seen.add(key)
                    scored.append((score, path, is_dir))
        searched.add(_norm(root))
    scored.sort(key=lambda t: (-t[0], len(str(t[1]))))
    partial = " Search was partial (time or entry limit reached); narrow it with `folder`." if deadline.expired else ""
    if not scored:
        where = str(roots[0][0]) if folder else "Desktop, Documents, Downloads, Pictures, Videos, Music, home"
        return ToolResult.fail(
            f"nothing matching '{name}' found in {where}.{partial}",
            "Try a shorter or different part of the name, or pass `folder`.",
        )
    top = scored[:limit]
    lines = [_stat_line(p) for _, p, _ in top]
    extra = f"\n({len(scored) - len(top)} more matches not shown.)" if len(scored) > len(top) else ""
    return ToolResult.success(
        "\n".join(lines) + extra + partial, data=[str(p) for _, p, _ in top]
    )


@tool(group=GROUP, timeout=40.0)
def search_file_contents(query: str, folder: str, glob: str = "*", limit: int = 20) -> ToolResult:
    """Search for text inside files under a folder (case-insensitive). Use this to find which
    file mentions something. Skips binary files and files over 2 MB. Returns path:line: snippet.

    Args:
        query: Text to look for.
        folder: Folder to search under (recursive).
        glob: File-name pattern to restrict the search, e.g. "*.py".
        limit: Maximum matching lines to return.
    """
    if not query:
        return ToolResult.fail("query is empty", "Give the text to search for.")
    try:
        base = resolve_path(folder)
    except ValueError as exc:
        return _bad_path(exc)
    if not base.is_dir():
        return _missing(base, "folder")
    limit = max(1, min(limit, 200))
    needle = query.lower()
    pattern = (glob or "*").lower()
    deadline, budget = _Deadline(GREP_TIME_BUDGET), [MAX_ENTRIES]
    hits: list[str] = []
    files_checked = 0
    for path, is_dir in _walk(base, 12, deadline, budget):
        if is_dir or not fnmatch.fnmatch(path.name.lower(), pattern):
            continue
        try:
            if path.stat().st_size > MAX_GREP_BYTES:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        if _is_binary(raw):
            continue
        files_checked += 1
        text, _ = _decode(raw)
        for lineno, line in enumerate(text.splitlines(), 1):
            if needle in line.lower():
                snippet = line.strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                hits.append(f"{path}:{lineno}: {snippet}")
                if len(hits) >= limit:
                    break
        if len(hits) >= limit:
            break
    notes = []
    if len(hits) >= limit:
        notes.append(f"Stopped at {limit} matches; narrow the query or glob.")
    if deadline.expired:
        notes.append("Search was partial (time limit reached).")
    if not hits:
        return ToolResult.fail(
            f"no text matching '{query}' in {files_checked} files under {base}. {' '.join(notes)}".strip(),
            "Try a different spelling, a broader glob, or a higher folder.",
        )
    return ToolResult.success("\n".join(hits + notes), data=hits)


@tool(group=GROUP)
def list_directory(path: str, limit: int = 100) -> ToolResult:
    """List the contents of a folder: sub-folders first, then files with sizes. Use this to
    look around before reading or editing.

    Args:
        path: Folder to list.
        limit: Maximum entries to show.
    """
    try:
        target = resolve_path(path)
    except ValueError as exc:
        return _bad_path(exc)
    if not target.exists():
        return _missing(target, "folder")
    if not target.is_dir():
        return ToolResult.fail(f"{target} is a file, not a folder", "Use read_file to read it.")
    try:
        entries = list(os.scandir(target))
    except OSError as exc:
        return ToolResult.fail(f"cannot list {target}: {exc.strerror or exc}")
    entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
    limit = max(1, limit)
    lines = []
    for e in entries[:limit]:
        try:
            if e.is_dir():
                lines.append(f"{e.name}/")
            else:
                lines.append(f"{e.name}  {_fmt_size(e.stat().st_size)}")
        except OSError:
            lines.append(e.name)
    header = f"{target}: {len(entries)} entries"
    if len(entries) > limit:
        header += f" (showing first {limit})"
    return ToolResult.success(header + "\n" + "\n".join(lines) if lines else header + " (empty)")


@tool(group=GROUP)
def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> ToolResult:
    """Read a text file (code, notes, config, logs) with line numbers. Returns up to max_lines
    lines from start_line and says when more remains. Not for PDF/Word/Excel or other binary files.

    Args:
        path: File to read.
        start_line: First line to return (1-based).
        max_lines: Maximum lines to return.
    """
    try:
        target = resolve_path(path)
    except ValueError as exc:
        return _bad_path(exc)
    if not target.exists():
        return _missing(target, "file")
    if target.is_dir():
        return ToolResult.fail(f"{target} is a folder", "Use list_directory to see its contents.")
    suffix = target.suffix.lower()
    if suffix in DOCUMENT_HINTS:
        return ToolResult.fail(f"{target.name} is {DOCUMENT_HINTS[suffix]}", "read_file only handles plain text.")
    try:
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            return ToolResult.fail(
                f"{target.name} is {_fmt_size(size)}, too large to read whole",
                "Use search_file_contents to find the part you need, or run_command with Select-Object -First/-Last.",
            )
        raw = target.read_bytes()
    except OSError as exc:
        return ToolResult.fail(f"cannot read {target}: {exc.strerror or exc}")
    if suffix in BINARY_KINDS or _is_binary(raw):
        kind = BINARY_KINDS.get(suffix, "a binary file")
        return ToolResult.fail(f"{target.name} is {kind}, not text", "It cannot be shown as text.")
    text, _ = _decode(raw)
    lines = text.splitlines()
    total = len(lines)
    start = max(1, start_line)
    count = max(1, max_lines)
    chunk = lines[start - 1 : start - 1 + count]
    if not chunk:
        return ToolResult.fail(f"{target.name} has {total} lines; start_line {start} is past the end")
    width = len(str(start + len(chunk)))
    body = "\n".join(
        f"{i:>{width}}\t{l if len(l) <= MAX_LINE_CHARS else l[:MAX_LINE_CHARS] + ' [line truncated]'}"
        for i, l in enumerate(chunk, start)
    )
    end = start + len(chunk) - 1
    note = f"\n[Showing lines {start}-{end} of {total}.]"
    if end < total:
        note = f"\n[Showing lines {start}-{end} of {total}. {total - end} more; call again with start_line={end + 1}.]"
    return ToolResult.success(f"{target}\n{body}{note}")


# ------------------------------------------------------------------- writing
def _write_risk(args: dict[str, Any]) -> Risk:
    if not args.get("overwrite"):
        return Risk.SAFE
    try:
        return Risk.CONFIRM if resolve_path(args["path"]).exists() else Risk.SAFE
    except (ValueError, KeyError, OSError):
        return Risk.SAFE


def _write_describe(args: dict[str, Any]) -> str:
    try:
        p = resolve_path(args["path"])
    except (ValueError, KeyError):
        p = args.get("path")
    return f"Overwrite {p} with {len(args.get('content', ''))} characters (the old version is kept as .bak)"


@tool(group=GROUP, risk=_write_risk, describe=_write_describe)
def write_file(path: str, content: str, overwrite: bool = False) -> ToolResult:
    """Create a text file with the given content (parent folders are created). Fails if the file
    exists unless overwrite is true; overwriting needs user approval and keeps a .bak copy.
    For small changes to an existing file prefer edit_file.

    Args:
        path: Where to write the file.
        content: Full text content.
        overwrite: Set true to replace an existing file.
    """
    try:
        target = resolve_path(path)
    except ValueError as exc:
        return _bad_path(exc)
    if target.is_dir():
        return ToolResult.fail(f"{target} is a folder", "Give a file path including the file name.")
    existed = target.exists()
    if existed and not overwrite:
        return ToolResult.fail(
            f"{target} already exists",
            "Pass overwrite=true to replace it (a .bak copy is kept), or use edit_file for a small change.",
        )
    try:
        bak = _backup(target) if existed else None
        _atomic_write(target, content.encode("utf-8"))
    except OSError as exc:
        return ToolResult.fail(f"cannot write {target}: {exc.strerror or exc}")
    msg = f"Wrote {len(content)} characters to {target}."
    if bak:
        msg += f" Previous version kept at {bak}."
    return ToolResult.success(msg, data=str(target))


@tool(group=GROUP)
def edit_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> ToolResult:
    """Replace exact text in an existing file. old_text must match exactly (including
    whitespace) and be unique unless replace_all is true. Keeps a .bak of the previous version
    and returns a short diff. Read the file first so old_text is accurate.

    Args:
        path: File to edit.
        old_text: Exact text to find.
        new_text: Text to put in its place.
        replace_all: Replace every occurrence instead of requiring exactly one.
    """
    try:
        target = resolve_path(path)
    except ValueError as exc:
        return _bad_path(exc)
    if not target.is_file():
        return _missing(target, "file") if not target.exists() else ToolResult.fail(f"{target} is a folder")
    if not old_text:
        return ToolResult.fail("old_text is empty", "Use write_file to create or replace a whole file.")
    try:
        raw = target.read_bytes()
    except OSError as exc:
        return ToolResult.fail(f"cannot read {target}: {exc.strerror or exc}")
    if _is_binary(raw):
        return ToolResult.fail(f"{target.name} is binary and cannot be edited as text")
    text, encoding = _decode(raw)
    crlf = "\r\n" in text
    work = text.replace("\r\n", "\n") if crlf else text  # models send \n; match either style
    old = old_text.replace("\r\n", "\n")
    new = new_text.replace("\r\n", "\n")
    count = work.count(old)
    if count == 0:
        return ToolResult.fail(
            f"old_text not found in {target.name}",
            "Use read_file to copy the exact text, including indentation and line breaks.",
        )
    if count > 1 and not replace_all:
        return ToolResult.fail(
            f"old_text occurs {count} times in {target.name}",
            "Include more surrounding lines to make it unique, or set replace_all=true.",
        )
    updated = work.replace(old, new) if replace_all else work.replace(old, new, 1)
    out = updated.replace("\n", "\r\n") if crlf else updated
    try:
        bak = _backup(target)
        _atomic_write(target, _encode(out, encoding))
    except OSError as exc:
        return ToolResult.fail(f"cannot write {target}: {exc.strerror or exc}")
    diff = list(
        difflib.unified_diff(
            work.splitlines(), updated.splitlines(), "before", "after", lineterm="", n=2
        )
    )
    shown = diff[:60] + ([f"... ({len(diff) - 60} more diff lines)"] if len(diff) > 60 else [])
    return ToolResult.success(
        f"Edited {target} ({count if replace_all else 1} replacement(s)). Backup of the previous "
        f"version: {bak}\n" + "\n".join(shown),
        data=str(target),
    )


# --------------------------------------------------------- move/copy/delete
def _arg_path(args: dict[str, Any], key: str) -> str:
    try:
        return str(resolve_path(args[key]))
    except (ValueError, KeyError):
        return str(args.get(key))


@tool(
    group=GROUP,
    risk=Risk.CONFIRM,
    describe=lambda a: f"Move {_arg_path(a, 'source')} to {_arg_path(a, 'destination')}",
)
def move_path(source: str, destination: str) -> ToolResult:
    """Move or rename a file or folder. Never overwrites: fails if the destination exists.
    Give the full destination path including the new name (not just the target folder).

    Args:
        source: Existing file or folder.
        destination: New full path (parent folders are created).
    """
    try:
        src, dst = resolve_path(source), resolve_path(destination)
    except ValueError as exc:
        return _bad_path(exc)
    if not src.exists():
        return _missing(src)
    if reason := _protected_reason(src):
        return ToolResult.fail(f"refusing to move {src}: {reason}")
    problem = _dest_problem(src, dst)
    if problem:
        return problem
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as exc:
        return ToolResult.fail(f"cannot move {src}: {exc.strerror or exc}")
    return ToolResult.success(f"Moved {src} to {dst}.", data=str(dst))


def _dest_problem(src: Path, dst: Path) -> ToolResult | None:
    if dst.exists():
        return ToolResult.fail(
            f"destination already exists: {dst}",
            "Nothing is overwritten. Pick a new name, or to put it inside a folder give the full path including the file name.",
        )
    if _norm(dst).startswith(_norm(src) + os.sep) and src.is_dir():
        return ToolResult.fail("cannot put a folder inside itself")
    return None


@tool(group=GROUP)
def copy_path(source: str, destination: str) -> ToolResult:
    """Copy a file or folder (recursively). Never overwrites: fails if the destination exists.
    Give the full destination path including the new name.

    Args:
        source: Existing file or folder.
        destination: New full path (parent folders are created).
    """
    try:
        src, dst = resolve_path(source), resolve_path(destination)
    except ValueError as exc:
        return _bad_path(exc)
    if not src.exists():
        return _missing(src)
    problem = _dest_problem(src, dst)
    if problem:
        return problem
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    except OSError as exc:
        return ToolResult.fail(f"cannot copy {src}: {exc.strerror or exc}")
    return ToolResult.success(f"Copied {src} to {dst}.", data=str(dst))


@tool(
    group=GROUP,
    risk=Risk.CONFIRM,
    describe=lambda a: f"Move {_arg_path(a, 'path')} to the Recycle Bin",
)
def delete_path(path: str) -> ToolResult:
    """Delete a file or folder by sending it to the Recycle Bin (recoverable). Needs user
    approval. Refuses drive roots, the home folder and Windows/Program Files folders.

    Args:
        path: File or folder to delete.
    """
    try:
        target = resolve_path(path)
    except ValueError as exc:
        return _bad_path(exc)
    if not target.exists():
        return _missing(target)
    if reason := _protected_reason(target):
        return ToolResult.fail(f"refusing to delete {target}: {reason}", "Pick a specific file or folder inside it.")
    try:
        send2trash(str(target))
    except Exception as exc:  # send2trash raises OSError and platform-specific errors
        return ToolResult.fail(f"cannot delete {target}: {exc}")
    return ToolResult.success(f"Sent {target} to the Recycle Bin (restorable from there).")


@tool(group=GROUP)
def create_folder(path: str) -> ToolResult:
    """Create a folder, including any missing parent folders. Succeeds if it already exists.

    Args:
        path: Folder to create.
    """
    try:
        target = resolve_path(path)
    except ValueError as exc:
        return _bad_path(exc)
    if target.is_file():
        return ToolResult.fail(f"a file already exists at {target}", "Choose a different folder name.")
    existed = target.is_dir()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ToolResult.fail(f"cannot create {target}: {exc.strerror or exc}")
    return ToolResult.success(f"{'Folder already exists' if existed else 'Created folder'}: {target}")


def _open_risk(args: dict[str, Any]) -> Risk:
    try:
        return Risk.CONFIRM if resolve_path(args["path"]).suffix.lower() in EXECUTABLE_SUFFIXES else Risk.SAFE
    except (ValueError, KeyError):
        return Risk.CONFIRM


@tool(
    group=GROUP,
    risk=_open_risk,
    describe=lambda a: f"Open {_arg_path(a, 'path')} (this type of file runs code)",
)
def open_path(path: str) -> ToolResult:
    """Open a file or folder with its default application (like double-clicking it). Use for
    "open my report" once you know the path. Programs and scripts need user approval.

    Args:
        path: File or folder to open.
    """
    try:
        target = resolve_path(path)
    except ValueError as exc:
        return _bad_path(exc)
    if not target.exists():
        return _missing(target)
    try:
        os.startfile(str(target))  # type: ignore[attr-defined]  # Windows only
    except OSError as exc:
        return ToolResult.fail(f"cannot open {target}: {exc.strerror or exc}")
    return ToolResult.success(f"Opened {target}.")
