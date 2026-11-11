"""Path, encoding and formatting helpers shared by the file, document, spreadsheet and
screen tools, so each tool reports the same hints and handles text the same way.
"""

from __future__ import annotations

import difflib
import logging
import os
import shutil
import stat
import sys
from pathlib import Path

from jarvis.tools.base import ToolResult

log = logging.getLogger(__name__)

# Windows device names: os.path.exists("NUL") is True and nobody means that.
RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

KNOWN_FOLDERS = {
    "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "Downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "Pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "Music": "4BD8D571-6D19-48D3-BE97-422220080E43",
    "Videos": "18989B1D-99B5-42AC-8B2D-CC9CF2A3C0D2",
}


def known_folder(name: str) -> Path:
    """Resolve Desktop, Documents, Downloads, Pictures, Music or Videos.

    On Windows these are often redirected (OneDrive), so the shell is asked instead of
    assuming ``~/Desktop``. Falls back to ``~/<name>`` when the lookup is unavailable.
    """
    fallback = Path.home() / name
    guid_text = KNOWN_FOLDERS.get(name)
    if guid_text is None or sys.platform != "win32":
        return fallback
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        raw = uuid.UUID(guid_text)
        guid = _GUID(
            raw.time_low,
            raw.time_mid,
            raw.time_hi_version,
            (ctypes.c_ubyte * 8)(*raw.bytes[8:]),
        )
        out = ctypes.c_wchar_p()
        rc = ctypes.windll.shell32.SHGetKnownFolderPath(  # type: ignore[attr-defined]
            ctypes.byref(guid), 0, None, ctypes.byref(out)
        )
        if rc == 0 and out.value:
            path = Path(out.value)
            ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]
            return path
    except Exception:  # best effort; the fallback is right for most machines
        log.debug("known folder lookup failed for %s", name, exc_info=True)
    return fallback


def home() -> Path:
    return Path(os.path.expanduser("~"))


def resolve_path(raw: str) -> Path:
    """Absolute path from ``~``, %VARS%/$VARS, either slash style, or a name relative to home.

    Raises ValueError for an empty path or a reserved Windows device name.
    """
    text = (raw or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError("empty path")
    expanded = os.path.expanduser(os.path.expandvars(text))
    path = Path(expanded)
    # Checked before abspath(), which rewrites device names into odd device paths.
    if path.name.split(".")[0].upper() in RESERVED_NAMES:
        raise ValueError(f"'{raw}' is a reserved Windows device name, not a file")
    if not path.is_absolute() and not path.drive:
        path = home() / path
    return Path(os.path.abspath(path))


def similar_names(path: Path) -> tuple[Path, list[str]]:
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


def not_found(path: Path, what: str = "path") -> ToolResult:
    """Failure for a missing path, hinting at similarly named entries in the nearest folder."""
    parent, close = similar_names(path)
    if close:
        hint = f"In {parent} there is: {', '.join(close)}. Use find_files if unsure."
    else:
        hint = f"The nearest existing folder is {parent}. Use find_files or list_directory to look around."
    return ToolResult.fail(f"{what} does not exist: {path}", hint)


def bad_path(exc: ValueError) -> ToolResult:
    """Failure for a path that resolve_path rejected."""
    return ToolResult.fail(str(exc), "Give a non-empty path.")


def decode_text(raw: bytes) -> tuple[str, str]:
    """Decode text bytes (UTF-16/UTF-8 BOM, then utf-8, then cp1252); returns (text, encoding)."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "cp1252"


def is_binary(raw: bytes) -> bool:
    """True if the first 4 KB contain NUL bytes (UTF-16 text is exempt)."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return False
    return b"\x00" in raw[:4096]


def format_size(n: int) -> str:
    """Human size such as '512 B' or '1.5 MB'."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def make_backup(path: Path) -> Path:
    """Copy ``path`` to ``path + '.bak'`` and return that path.

    An existing .bak is replaced even when it is read-only (Windows refuses to overwrite
    such a file, which would otherwise block every later edit of the original).
    """
    bak = path.with_name(path.name + ".bak")
    if bak.exists():
        os.chmod(bak, stat.S_IWRITE | stat.S_IREAD)
    shutil.copy2(path, bak)
    return bak
