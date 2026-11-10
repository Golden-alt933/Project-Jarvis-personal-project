"""Tests for the files and shell tool groups, run through a real ToolRegistry."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from jarvis.config import Settings
from jarvis.llm.types import ToolCall
from jarvis.tools.base import Risk, ToolContext
from jarvis.tools.builtin import files, shell
from jarvis.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_module(files)
    reg.register_module(shell)
    return reg


class Asker:
    """Confirm callback that records prompts and answers with a fixed value."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, text: str) -> bool:
        self.prompts.append(text)
        return self.answer


def make_ctx(answer: bool = True) -> tuple[ToolContext, Asker]:
    asker = Asker(answer)
    return ToolContext(settings=Settings(), confirm=asker), asker


async def call(reg: ToolRegistry, tool_name: str, ctx: ToolContext | None = None, /, **args: Any):
    ctx = ctx or make_ctx()[0]
    return await reg.execute(ToolCall(id="t", name=tool_name, arguments=args), ctx)


@pytest.fixture(autouse=True)
def fake_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path_factory.mktemp("users") / "home"
    home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    return home


# ----------------------------------------------------------------- registry
def test_all_tools_registered(registry: ToolRegistry):
    names = {s.name for s in registry.specs()}
    assert {"find_files", "search_file_contents", "list_directory", "read_file", "write_file",
            "edit_file", "move_path", "copy_path", "delete_path", "create_folder", "open_path",
            "run_command", "run_python"} <= names
    assert "ctx" not in registry.get("run_command").parameters["properties"]
    assert registry.get("run_command").timeout > 600


# -------------------------------------------------------------------- paths
def test_resolve_path_forms(fake_home: Path, monkeypatch):
    assert files.resolve_path("notes.txt") == fake_home / "notes.txt"
    assert files.resolve_path("~/a/b.txt") == fake_home / "a" / "b.txt"
    assert files.resolve_path("~/a/b.txt") == files.resolve_path(r"~\a\b.txt")
    monkeypatch.setenv("MYDIR", str(fake_home / "x"))
    assert files.resolve_path("%MYDIR%/f.txt") == fake_home / "x" / "f.txt"
    with pytest.raises(ValueError):
        files.resolve_path("NUL")


async def test_not_found_hint_lists_similar(registry, tmp_path):
    (tmp_path / "budget_2024.xlsx").write_text("x")
    (tmp_path / "unrelated.txt").write_text("x")
    r = await call(registry, "read_file", path=str(tmp_path / "budget_2023.xlsx"))
    assert not r.ok and "does not exist" in r.content
    assert "budget_2024.xlsx" in r.content


async def test_not_found_nearest_parent(registry, tmp_path):
    (tmp_path / "Projects").mkdir()
    r = await call(registry, "list_directory", path=str(tmp_path / "Projcts" / "deep" / "x"))
    assert not r.ok and "Projects" in r.content


# --------------------------------------------------------------- find_files
async def test_find_files_ranks_exact_over_fuzzy(registry, tmp_path):
    (tmp_path / "budget.xlsx").write_text("a")
    (tmp_path / "budget notes from the old laptop.docx").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "thesis draft.docx").write_text("a")
    r = await call(registry, "find_files", name="budget", folder=str(tmp_path))
    assert r.ok
    assert r.data[0].endswith("budget.xlsx")
    assert len(r.data) == 2
    assert "KB" in r.content or " B" in r.content and "modified" in r.content


async def test_find_files_kind_and_noise_dirs(registry, tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports.txt").write_text("a")
    nm = tmp_path / "node_modules" / "reports"
    nm.mkdir(parents=True)
    (nm / "reports.txt").write_text("a")
    folders = await call(registry, "find_files", name="reports", folder=str(tmp_path), kind="folder")
    assert [Path(p).name for p in folders.data] == ["reports"]
    only_files = await call(registry, "find_files", name="reports", folder=str(tmp_path), kind="file")
    assert len(only_files.data) == 1  # node_modules was not entered


async def test_find_files_spoken_name_and_no_match(registry, tmp_path):
    (tmp_path / "resume.pdf").write_text("a")
    r = await call(registry, "find_files", name="my resume pdf file", folder=str(tmp_path))
    assert r.ok and r.data[0].endswith("resume.pdf")
    none = await call(registry, "find_files", name="zzqqxx", folder=str(tmp_path))
    assert not none.ok and "nothing matching" in none.content


async def test_find_files_default_roots_and_partial(registry, fake_home, monkeypatch):
    docs = fake_home / "Documents"
    docs.mkdir()
    (docs / "taxes.txt").write_text("a")
    r = await call(registry, "find_files", name="taxes")
    assert r.ok and r.data[0].endswith("taxes.txt")
    monkeypatch.setattr(files, "FIND_TIME_BUDGET", -1.0)
    partial = await call(registry, "find_files", name="taxes", folder=str(docs))
    assert "partial" in partial.content


# ------------------------------------------------------- search contents
async def test_search_file_contents(registry, tmp_path):
    (tmp_path / "a.py").write_text("import os\nTODO fix this\n")
    (tmp_path / "b.txt").write_text("todo in text\n")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01TODO\x00")
    (tmp_path / "big.txt").write_text("TODO " + "x" * (2 * 1024 * 1024 + 10))
    r = await call(registry, "search_file_contents", query="todo", folder=str(tmp_path))
    assert r.ok
    assert f"a.py:2: TODO fix this" in r.content
    assert "b.txt:1" in r.content
    assert "c.bin" not in r.content and "big.txt" not in r.content
    py = await call(registry, "search_file_contents", query="todo", folder=str(tmp_path), glob="*.py")
    assert "b.txt" not in py.content
    none = await call(registry, "search_file_contents", query="nope", folder=str(tmp_path))
    assert not none.ok
    lim = await call(registry, "search_file_contents", query="todo", folder=str(tmp_path), limit=1)
    assert len(lim.data) == 1
    bad = await call(registry, "search_file_contents", query="x", folder=str(tmp_path / "missing"))
    assert not bad.ok


# ----------------------------------------------------------- list_directory
async def test_list_directory(registry, tmp_path):
    (tmp_path / "zdir").mkdir()
    (tmp_path / "b.txt").write_text("hello")
    (tmp_path / "a.txt").write_text("hi")
    r = await call(registry, "list_directory", path=str(tmp_path))
    assert r.ok
    lines = r.content.splitlines()
    assert lines[1] == "zdir/"  # folders first
    assert lines[2].startswith("a.txt")
    limited = await call(registry, "list_directory", path=str(tmp_path), limit=1)
    assert "showing first 1" in limited.content
    notdir = await call(registry, "list_directory", path=str(tmp_path / "a.txt"))
    assert not notdir.ok and "read_file" in notdir.content


# ---------------------------------------------------------------- read_file
async def test_read_file_numbers_and_paging(registry, tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)))
    r = await call(registry, "read_file", path=str(f), max_lines=3)
    assert "line1" in r.content and "line3" in r.content and "line4" not in r.content
    assert "start_line=4" in r.content
    r2 = await call(registry, "read_file", path=str(f), start_line=9)
    assert "line10" in r2.content and "more" not in r2.content
    past = await call(registry, "read_file", path=str(f), start_line=99)
    assert not past.ok


async def test_read_file_encodings(registry, tmp_path):
    (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbfcaf\xc3\xa9")
    (tmp_path / "cp.txt").write_bytes("café".encode("cp1252"))
    (tmp_path / "u16.txt").write_bytes("café".encode("utf-16"))
    for name in ("bom.txt", "cp.txt", "u16.txt"):
        r = await call(registry, "read_file", path=str(tmp_path / name))
        assert r.ok and "café" in r.content, name
        assert "\ufeff" not in r.content


async def test_read_file_refuses_binary(registry, tmp_path):
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    (tmp_path / "mystery.dat").write_bytes(b"ab\x00cd")
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4")
    png = await call(registry, "read_file", path=str(tmp_path / "pic.png"))
    assert not png.ok and "image" in png.content
    dat = await call(registry, "read_file", path=str(tmp_path / "mystery.dat"))
    assert not dat.ok and "binary" in dat.content
    pdf = await call(registry, "read_file", path=str(tmp_path / "x.pdf"))
    assert not pdf.ok and "PDF" in pdf.content
    folder = await call(registry, "read_file", path=str(tmp_path))
    assert not folder.ok and "list_directory" in folder.content


# --------------------------------------------------------------- write_file
async def test_write_file_create_and_parent_dirs(registry, tmp_path):
    ctx, asker = make_ctx()
    target = tmp_path / "new" / "deep" / "f.txt"
    r = await call(registry, "write_file", ctx, path=str(target), content="héllo")
    assert r.ok and target.read_text(encoding="utf-8") == "héllo"
    assert asker.prompts == []  # creating is SAFE
    assert not list(target.parent.glob("*.tmp"))


async def test_write_file_exists_without_overwrite(registry, tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("old")
    ctx, asker = make_ctx()
    r = await call(registry, "write_file", ctx, path=str(f), content="new")
    assert not r.ok and "overwrite=true" in r.content
    assert f.read_text() == "old" and asker.prompts == []


async def test_write_file_overwrite_confirm_and_backup(registry, tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("old")
    ctx, asker = make_ctx(True)
    r = await call(registry, "write_file", ctx, path=str(f), content="new", overwrite=True)
    assert r.ok and f.read_text() == "new"
    assert (tmp_path / "f.txt.bak").read_text() == "old"
    assert len(asker.prompts) == 1 and "Overwrite" in asker.prompts[0]


async def test_write_file_overwrite_denied(registry, tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("old")
    ctx, asker = make_ctx(False)
    r = await call(registry, "write_file", ctx, path=str(f), content="new", overwrite=True)
    assert not r.ok and "declined" in r.content
    assert f.read_text() == "old" and not (tmp_path / "f.txt.bak").exists()


async def test_write_file_overwrite_of_missing_file_needs_no_confirm(registry, tmp_path):
    ctx, asker = make_ctx(False)
    r = await call(registry, "write_file", ctx, path=str(tmp_path / "n.txt"), content="x", overwrite=True)
    assert r.ok and asker.prompts == []


async def test_write_file_onto_folder(registry, tmp_path):
    r = await call(registry, "write_file", path=str(tmp_path), content="x")
    assert not r.ok and "folder" in r.content


# ---------------------------------------------------------------- edit_file
async def test_edit_file_success_diff_and_backup(registry, tmp_path):
    f = tmp_path / "c.py"
    f.write_text("a = 1\nb = 2\nc = 3\n")
    r = await call(registry, "edit_file", path=str(f), old_text="b = 2", new_text="b = 20")
    assert r.ok and "-b = 2" in r.content and "+b = 20" in r.content
    assert "c.py.bak" in r.content
    assert f.read_text() == "a = 1\nb = 20\nc = 3\n"
    assert (tmp_path / "c.py.bak").read_text() == "a = 1\nb = 2\nc = 3\n"


async def test_edit_file_ambiguity_and_replace_all(registry, tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("x\nx\ny\n")
    r = await call(registry, "edit_file", path=str(f), old_text="x", new_text="z")
    assert not r.ok and "2 times" in r.content
    assert f.read_text() == "x\nx\ny\n" and not (tmp_path / "c.txt.bak").exists()
    r2 = await call(registry, "edit_file", path=str(f), old_text="x", new_text="z", replace_all=True)
    assert r2.ok and f.read_text() == "z\nz\ny\n"


async def test_edit_file_missing_text_and_missing_file(registry, tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("hello")
    r = await call(registry, "edit_file", path=str(f), old_text="nope", new_text="x")
    assert not r.ok and "not found" in r.content
    r = await call(registry, "edit_file", path=str(tmp_path / "c.tx"), old_text="a", new_text="b")
    assert not r.ok and "c.txt" in r.content
    r = await call(registry, "edit_file", path=str(f), old_text="", new_text="b")
    assert not r.ok


async def test_edit_file_preserves_crlf_and_bom(registry, tmp_path):
    f = tmp_path / "w.txt"
    f.write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\nthree\r\n")
    r = await call(registry, "edit_file", path=str(f), old_text="one\ntwo", new_text="1\n2")
    assert r.ok
    assert f.read_bytes() == b"\xef\xbb\xbf1\r\n2\r\nthree\r\n"


# --------------------------------------------------------- move/copy/folder
async def test_move_confirm_deny_and_approve(registry, tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("a")
    dst = tmp_path / "out" / "b.txt"
    ctx, asker = make_ctx(False)
    r = await call(registry, "move_path", ctx, source=str(src), destination=str(dst))
    assert not r.ok and src.exists() and not dst.exists()
    assert asker.prompts == [f"Move {src} to {dst}"]
    ctx, _ = make_ctx(True)
    r = await call(registry, "move_path", ctx, source=str(src), destination=str(dst))
    assert r.ok and dst.read_text() == "a" and not src.exists()


async def test_move_refuses_overwrite_and_missing(registry, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    r = await call(registry, "move_path", source=str(tmp_path / "a.txt"), destination=str(tmp_path / "b.txt"))
    assert not r.ok and "already exists" in r.content
    assert (tmp_path / "b.txt").read_text() == "b"
    r = await call(registry, "move_path", source=str(tmp_path / "zz.txt"), destination=str(tmp_path / "q.txt"))
    assert not r.ok and "does not exist" in r.content


async def test_copy_file_and_folder_no_confirm(registry, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    d = tmp_path / "d"
    d.mkdir()
    (d / "inner.txt").write_text("i")
    ctx, asker = make_ctx(False)
    r = await call(registry, "copy_path", ctx, source=str(tmp_path / "a.txt"), destination=str(tmp_path / "a2.txt"))
    assert r.ok and (tmp_path / "a2.txt").read_text() == "a" and asker.prompts == []
    r = await call(registry, "copy_path", ctx, source=str(d), destination=str(tmp_path / "d2"))
    assert r.ok and (tmp_path / "d2" / "inner.txt").read_text() == "i"
    r = await call(registry, "copy_path", ctx, source=str(d), destination=str(d / "inside"))
    assert not r.ok and "inside itself" in r.content
    r = await call(registry, "copy_path", ctx, source=str(tmp_path / "a.txt"), destination=str(tmp_path / "a2.txt"))
    assert not r.ok and "already exists" in r.content


async def test_create_folder(registry, tmp_path):
    r = await call(registry, "create_folder", path=str(tmp_path / "x" / "y"))
    assert r.ok and (tmp_path / "x" / "y").is_dir()
    again = await call(registry, "create_folder", path=str(tmp_path / "x" / "y"))
    assert again.ok and "already exists" in again.content
    (tmp_path / "f").write_text("x")
    clash = await call(registry, "create_folder", path=str(tmp_path / "f"))
    assert not clash.ok


# ------------------------------------------------------------------- delete
@pytest.fixture
def trash(monkeypatch) -> list[str]:
    sent: list[str] = []
    monkeypatch.setattr(files, "send2trash", lambda p: sent.append(p))
    return sent


async def test_delete_path_confirm_flow(registry, tmp_path, trash):
    f = tmp_path / "a.txt"
    f.write_text("a")
    ctx, asker = make_ctx(False)
    r = await call(registry, "delete_path", ctx, path=str(f))
    assert not r.ok and trash == []
    assert asker.prompts == [f"Move {f} to the Recycle Bin"]
    ctx, _ = make_ctx(True)
    r = await call(registry, "delete_path", ctx, path=str(f))
    assert r.ok and trash == [str(f)] and "Recycle Bin" in r.content


async def test_delete_path_missing(registry, tmp_path, trash):
    r = await call(registry, "delete_path", path=str(tmp_path / "ghost.txt"))
    assert not r.ok and trash == []


async def test_delete_refusals(registry, fake_home, trash, monkeypatch):
    windir = fake_home.parent / "FakeWindows"
    (windir / "System32").mkdir(parents=True)
    monkeypatch.setenv("SystemRoot", str(windir))
    monkeypatch.setenv("windir", str(windir))
    drive_root = Path(os.path.abspath(fake_home)).anchor
    targets = [drive_root, str(fake_home), str(fake_home.parent), str(windir), str(windir / "System32")]
    for t in targets:
        r = await call(registry, "delete_path", path=t)
        assert not r.ok and "refusing" in r.content, t
    assert trash == []


async def test_move_refuses_protected_source(registry, fake_home):
    r = await call(registry, "move_path", source=str(fake_home), destination=str(fake_home.parent / "elsewhere"))
    assert not r.ok and "refusing" in r.content


# --------------------------------------------------------------------- open
async def test_open_path(registry, tmp_path, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(os, "startfile", lambda p: opened.append(p), raising=False)
    f = tmp_path / "doc.txt"
    f.write_text("x")
    ctx, asker = make_ctx(False)
    r = await call(registry, "open_path", ctx, path=str(f))
    assert r.ok and opened == [str(f)] and asker.prompts == []
    exe = tmp_path / "setup.bat"
    exe.write_text("echo")
    r = await call(registry, "open_path", ctx, path=str(exe))
    assert not r.ok and "declined" in r.content and opened == [str(f)]
    r = await call(registry, "open_path", ctx, path=str(tmp_path / "nothing.txt"))
    assert not r.ok


# ------------------------------------------------------- risk classifier
SAFE_COMMANDS = [
    "Get-ChildItem",
    "dir",
    "ls -Force",
    "Get-ChildItem -Recurse -Filter *.py | Select-Object -First 5",
    "Get-Content notes.txt",
    "type notes.txt",
    "cat notes.txt | Select-String todo",
    "Get-Process",
    "Get-Process | Sort-Object CPU | Select-Object -First 5",
    "Get-Date",
    "whoami",
    "hostname",
    "ipconfig",
    "ipconfig /all",
    "systeminfo",
    "where.exe python",
    "Get-Command python",
    "git status",
    "git log --oneline -n 5",
    "git diff",
    "git branch",
    "git branch -a",
    "python --version",
    "pip list",
    "python -m pip list",
    "echo hello",
]
UNSAFE_COMMANDS = [
    "Get-Content x | Set-Content y",
    "dir; rm -r .",
    "echo hi > f",
    "echo hi >> f",
    "git status && del x",
    "git status & del x",
    "powershell -enc SQBFAFgA",
    "Get-Process | Stop-Process",
    "Get-ChildItem | Remove-Item",
    "Remove-Item x",
    "rm -r .",
    "del file.txt",
    "Invoke-Expression 'dir'",
    "iex dir",
    "dir `;rm x",
    "echo $(Get-Date)",
    "Get-Process | ForEach-Object { $_.Kill() }",
    "Start-Process notepad",
    "git branch -D main",
    "git push",
    "git log --output=x.txt",
    "git checkout main",
    "git status | Out-File x.txt",
    "ipconfig /release",
    "python script.py",
    "python -c \"print(1)\"",
    "pip install requests",
    "Get-Content x | Invoke-Expression",
    "Set-Location C:\\ ",
    "curl http://example.com",
    "",
    "   ",
    "dir |",
    "echo 'unterminated",
    "shutdown /s",
    "Get-Date -Format ( rm x )",
    "Get-Content (Read-Host)",
    "cat x -EncodedCommand abc",
]


@pytest.mark.parametrize("cmd", SAFE_COMMANDS)
def test_classifier_safe(cmd):
    assert shell.classify_command(cmd) is Risk.SAFE, cmd


@pytest.mark.parametrize("cmd", UNSAFE_COMMANDS)
def test_classifier_unsafe(cmd):
    assert shell.classify_command(cmd) is Risk.CONFIRM, cmd


def test_classifier_table_size():
    assert len(SAFE_COMMANDS) + len(UNSAFE_COMMANDS) >= 30


# ------------------------------------------------------------ execution
@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell tools target Windows")
async def test_run_command_real_safe_no_confirm(registry, tmp_path):
    ctx, asker = make_ctx(False)
    (tmp_path / "marker.txt").write_text("x")
    r = await call(registry, "run_command", ctx, command="Get-ChildItem -Name", cwd=str(tmp_path))
    assert r.ok and "marker.txt" in r.content and "Exit code 0" in r.content
    assert asker.prompts == []


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell tools target Windows")
async def test_run_command_confirm_and_exit_code(registry, tmp_path):
    ctx, asker = make_ctx(True)
    r = await call(registry, "run_command", ctx, command="Write-Output boom; exit 3", cwd=str(tmp_path))
    assert not r.ok and "Exit code 3" in r.content and "boom" in r.content
    assert len(asker.prompts) == 1 and "Write-Output boom" in asker.prompts[0]
    denied, asker2 = make_ctx(False)
    r = await call(registry, "run_command", denied, command="New-Item x.txt", cwd=str(tmp_path))
    assert not r.ok and "declined" in r.content and not (tmp_path / "x.txt").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell tools target Windows")
async def test_run_command_stderr_merged(registry):
    r = await call(registry, "run_command", command="Write-Error 'bad thing'; Write-Output ok")
    assert "bad thing" in r.content and "ok" in r.content


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell tools target Windows")
async def test_run_command_bad_cwd(registry, tmp_path):
    r = await call(registry, "run_command", command="Get-Date", cwd=str(tmp_path / "nope"))
    assert not r.ok and "does not exist" in r.content


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell tools target Windows")
async def test_run_command_timeout_kills_process_tree(registry, tmp_path):
    import psutil

    pidfile = tmp_path / "child.pid"
    # The child python is a grandchild of the shell; the tree kill must reach it.
    cmd = (
        f"$p = Start-Process -PassThru -NoNewWindow '{sys.executable}' "
        f"-ArgumentList '-c','import time; time.sleep(120)'; "
        f"$p.Id | Out-File '{pidfile}'; Start-Sleep 120"
    )
    t0 = time.monotonic()
    r = await call(registry, "run_command", command=cmd, timeout_seconds=4)
    assert time.monotonic() - t0 < 30
    assert not r.ok and "timed out" in r.content
    pid = int(pidfile.read_text(encoding="utf-16").strip().lstrip("\ufeff")) if pidfile.exists() else None
    if pid is not None:
        await asyncio.sleep(0.5)
        assert not psutil.pid_exists(pid)


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell tools target Windows")
async def test_run_command_cancel(registry):
    ctx, _ = make_ctx(True)
    asyncio.get_running_loop().call_later(1.0, ctx.cancelled.set)
    t0 = time.monotonic()
    r = await call(registry, "run_command", ctx, command="Start-Sleep 60", timeout_seconds=60)
    assert time.monotonic() - t0 < 20
    assert not r.ok and "cancelled" in r.content


async def test_timeout_is_clamped():
    assert shell._clamp(0) == 1
    assert shell._clamp(99999) == 600
    assert shell._clamp(30) == 30


async def test_run_command_output_tail_truncated(registry):
    if sys.platform != "win32":
        pytest.skip("Windows only")
    r = await call(registry, "run_command", command="1..3000 | ForEach-Object { 'line number ' + $_ }")
    assert r.ok and "earlier characters omitted" in r.content and "line number 3000" in r.content


# -------------------------------------------------------------- run_python
async def test_run_python_confirm_and_output(registry):
    ctx, asker = make_ctx(True)
    r = await call(registry, "run_python", ctx, code="print(6*7)\nimport sys; sys.exit(0)")
    assert r.ok and "42" in r.content
    assert len(asker.prompts) == 1 and "print(6*7)" in asker.prompts[0]


async def test_run_python_denied_and_failure(registry):
    ctx, _ = make_ctx(False)
    r = await call(registry, "run_python", ctx, code="print('no')")
    assert not r.ok and "declined" in r.content
    ctx, _ = make_ctx(True)
    r = await call(registry, "run_python", ctx, code="raise ValueError('oops')")
    assert not r.ok and "Exit code 1" in r.content and "ValueError: oops" in r.content


async def test_run_python_timeout(registry):
    ctx, _ = make_ctx(True)
    t0 = time.monotonic()
    r = await call(registry, "run_python", ctx, code="import time; time.sleep(60)", timeout_seconds=2)
    assert time.monotonic() - t0 < 20
    assert not r.ok and "timed out" in r.content


def test_run_python_describe_truncates():
    spec = shell.run_python.__tool_spec__
    text = spec.confirmation_text({"code": "\n".join(f"x{i}=1" for i in range(20)), "timeout_seconds": 5})
    assert "x0=1" in text and "12 more lines" in text
