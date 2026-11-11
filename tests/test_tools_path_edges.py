"""Edge cases of the file tools: non-ASCII names, files held by another process,
atomic writes beside the destination, and read-only backups."""

from __future__ import annotations

import asyncio
import io
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from jarvis.config import Settings
from jarvis.llm.types import ToolCall
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import excel, files
from jarvis.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_module(files)
    return reg


@pytest.fixture(autouse=True)
def fake_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path_factory.mktemp("users") / "home"
    home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    return home


def call(reg: ToolRegistry, tool_name: str, /, **args: Any):
    async def confirm(_: str) -> bool:
        return True

    ctx = ToolContext(settings=Settings(), confirm=confirm)
    return asyncio.run(reg.execute(ToolCall(id="t", name=tool_name, arguments=args), ctx))


# ------------------------------------------------------------ non-ASCII names
NAMES = ["résumé_final.txt", "Übersicht Größe.txt", "日本語のメモ.txt", "تقرير.txt", "naïve café 😀.txt"]


@pytest.mark.parametrize("name", NAMES)
def test_find_files_with_non_ascii_names(registry: ToolRegistry, tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("x", encoding="utf-8")
    r = call(registry, "find_files", name=name, folder=str(tmp_path))
    assert r.ok, r.content
    assert str(tmp_path / name) in r.content
    r = call(registry, "find_files", name=Path(name).stem[:3], folder=str(tmp_path))
    assert r.ok and str(tmp_path / name) in r.content


def test_find_files_matches_without_accents(registry: ToolRegistry, tmp_path: Path) -> None:
    """The model (or speech-to-text) says 'resume'; the file is called résumé."""
    (tmp_path / "résumé_final.txt").write_text("x", encoding="utf-8")
    r = call(registry, "find_files", name="resume final", folder=str(tmp_path))
    assert r.ok and "résumé_final.txt" in r.content


def test_find_files_matches_decomposed_unicode(registry: ToolRegistry, tmp_path: Path) -> None:
    """Names can be stored with combining accents (e + U+0301) while the query is precomposed."""
    (tmp_path / "résumé_final.txt").write_text("x", encoding="utf-8")
    r = call(registry, "find_files", name="résumé_final", folder=str(tmp_path))
    assert r.ok and "final.txt" in r.content


def test_search_contents_in_non_ascii_files(registry: ToolRegistry, tmp_path: Path) -> None:
    folder = tmp_path / "Ünïcode dir"
    folder.mkdir()
    (folder / "日本語.txt").write_text("première ligne\nl'École normale\n", encoding="utf-8")
    (folder / "legacy.txt").write_bytes("café au lait\n".encode("cp1252"))
    r = call(registry, "search_file_contents", query="école", folder=str(folder))
    assert r.ok and "日本語.txt:2: l'École normale" in r.content
    r = call(registry, "search_file_contents", query="CAFÉ", folder=str(folder))
    assert r.ok and "legacy.txt:1: café au lait" in r.content
    r = call(registry, "search_file_contents", query="ligne", folder=str(folder), glob="日本*")
    assert r.ok and "日本語.txt:1" in r.content


# ------------------------------------------------- file held by another process
@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing modes")
def test_read_file_locked_by_another_process(registry: ToolRegistry, tmp_path: Path) -> None:
    import win32file

    target = tmp_path / "held.log"
    target.write_text("data", encoding="utf-8")
    handle = win32file.CreateFile(
        str(target), win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, win32file.OPEN_EXISTING, 0, None
    )  # share mode 0: nobody else may open it
    try:
        r = call(registry, "read_file", path=str(target))
    finally:
        handle.Close()
    assert not r.ok
    assert "cannot read" in r.content and "Traceback" not in r.content
    assert "locks it" in r.content  # says why, and what to do


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing modes")
def test_read_file_being_appended_by_a_writer(registry: ToolRegistry, tmp_path: Path) -> None:
    target = tmp_path / "growing.log"
    with open(target, "w", encoding="utf-8") as writer:
        writer.write("line1\nline2\n")
        writer.flush()
        r = call(registry, "read_file", path=str(target))
    assert r.ok and "line2" in r.content


# --------------------------------------------------------- atomic destination
def test_write_file_creates_its_temp_file_beside_the_destination(
    registry: ToolRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.replace cannot cross drives, so the temp file must live in the destination folder."""
    dirs: list[Any] = []
    real = tempfile.mkstemp

    def spy(*args: Any, **kwargs: Any):
        dirs.append(kwargs.get("dir"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", spy)
    dest = tmp_path / "sub" / "out.txt"
    assert call(registry, "write_file", path=str(dest), content="hello").ok
    assert call(registry, "write_file", path=str(dest), content="again", overwrite=True).ok
    assert dirs and all(Path(d) == dest.parent for d in dirs)
    assert dest.read_text() == "again"
    assert [p.name for p in dest.parent.iterdir()] == ["out.txt", "out.txt.bak"]  # no stray temp files


def test_excel_saves_beside_the_destination_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dirs: list[Any] = []
    real = tempfile.mkstemp

    def spy(*args: Any, **kwargs: Any):
        dirs.append(kwargs.get("dir"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", spy)
    reg = ToolRegistry()
    reg.register_module(excel)
    from jarvis.tools import excel_calc

    monkeypatch.setattr(excel_calc, "excel_installed", lambda: False)
    assert call(reg, "excel_write", path=str(tmp_path / "x.xlsx"), sheet="S", cells={"A1": 1}).ok
    assert dirs and all(Path(d) == tmp_path for d in dirs)


# ------------------------------------------------------------- read-only .bak
def test_overwrite_when_bak_is_read_only(registry: ToolRegistry, tmp_path: Path) -> None:
    target = tmp_path / "doc.txt"
    target.write_text("v1", encoding="utf-8")
    bak = tmp_path / "doc.txt.bak"
    bak.write_text("ancient", encoding="utf-8")
    os.chmod(bak, stat.S_IREAD)
    r = call(registry, "write_file", path=str(target), content="v2", overwrite=True)
    assert r.ok, r.content
    assert target.read_text() == "v2"
    assert bak.read_text() == "v1"  # the backup is the previous version, not stale


def test_edit_when_bak_is_read_only(registry: ToolRegistry, tmp_path: Path) -> None:
    target = tmp_path / "doc.txt"
    target.write_text("alpha beta", encoding="utf-8")
    bak = tmp_path / "doc.txt.bak"
    bak.write_text("ancient", encoding="utf-8")
    os.chmod(bak, stat.S_IREAD)
    r = call(registry, "edit_file", path=str(target), old_text="beta", new_text="gamma")
    assert r.ok, r.content
    assert target.read_text() == "alpha gamma" and bak.read_text() == "alpha beta"


def test_excel_write_when_bak_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from openpyxl import Workbook, load_workbook

    from jarvis.tools import excel_calc

    monkeypatch.setattr(excel_calc, "excel_installed", lambda: False)
    path = tmp_path / "b.xlsx"
    wb = Workbook()
    wb.active["A1"] = "old"
    wb.save(path)
    bak = tmp_path / "b.xlsx.bak"
    bak.write_bytes(b"stale")
    os.chmod(bak, stat.S_IREAD)
    reg = ToolRegistry()
    reg.register_module(excel)
    r = call(reg, "excel_write", path=str(path), sheet="Sheet", cells={"A1": "new"})
    assert r.ok, r.content
    assert load_workbook(io.BytesIO(bak.read_bytes())).active["A1"].value == "old"
