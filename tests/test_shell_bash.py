"""The bash side of the shell tool: risk classifier (table-tested, runs on any OS), tool
description, and real bash execution (POSIX only; skipped where there is no bash)."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from typing import Any

import pytest

from jarvis import platform as plat
from jarvis.config import Settings
from jarvis.llm.types import ToolCall
from jarvis.platform.linux import LinuxPlatform
from jarvis.platform.windows import WindowsPlatform
from jarvis.tools.base import Risk, ToolContext
from jarvis.tools.builtin import files, shell
from jarvis.tools.registry import ToolRegistry

SAFE = [
    "ls",
    "ls -la",
    "ls -la /tmp",
    "ls 'my folder'",
    "ls *.py",
    "cat notes.txt",
    "cat a.txt b.txt",
    "head -n 20 file.log",
    "tail -n 50 file.log",
    "wc -l file.txt",
    "grep -n TODO main.py",
    "grep -rn needle src",
    "rg needle",
    "rg -n 'foo bar' src",
    "find . -name '*.py'",
    "find /var/log -type f -mtime -1",
    "pwd",
    "whoami",
    "hostname",
    "hostname -f",
    "date",
    "date +%Y-%m-%d",
    "date -u",
    "uname -a",
    "df -h",
    "du -sh .",
    "free -m",
    "ps aux",
    "uptime",
    "which python3",
    "env",
    "printenv",
    "printenv HOME",
    "git status",
    "git log --oneline -n 5",
    "git diff",
    "git diff HEAD~1",
    "git show HEAD",
    "git branch",
    "git branch -a",
    "python --version",
    "python3 --version",
    "pip list",
    "pip3 freeze",
    "python3 -m pip list",
    "echo hello",
    "echo 'hello world'",
    "ps aux | grep python",
    "cat file.txt | head -n 5",
    "ls -la | wc -l",
    "git log --oneline | head -n 3",
]

CONFIRM = [
    "",
    "   ",
    "rm -rf .",
    "rm file.txt",
    "mv a b",
    "cp a b",
    "mkdir x",
    "touch x",
    "chmod +x run.sh",
    "ls; rm -rf x",
    "ls && rm x",
    "ls || rm x",
    "ls & rm x",
    "ls &",
    "echo hi > f",
    "echo hi >> f",
    "cat < file",
    "cat file > /dev/null",
    "echo `whoami`",
    "echo $(whoami)",
    "echo $HOME",
    "echo ${HOME}",
    "ls $(pwd)",
    "sudo ls",
    "sudo apt install x",
    "su -c ls",
    "eval ls",
    "eval 'rm x'",
    "xargs rm",
    "ls | xargs rm",
    "sh -c 'ls'",
    "bash -c 'ls'",
    "bash script.sh",
    "ls | sh",
    "cat <<EOF\nhi\nEOF",
    "cat <<EOF",
    "ls\nrm x",
    "find . -delete",
    "find . -name x -exec rm {} +",
    "find . -exec ls {} ;",
    "find . -ok rm {} ;",
    "find . -execdir ls ;",
    "find . -fprint out.txt",
    "rg --pre ./evil needle",
    "rg --hostname-bin evil x",
    "tail -f log",
    "tail -F log",
    "hostname newname",
    "date -s '2020-01-01'",
    "date --set=2020-01-01",
    "date 010100002020",
    "env FOO=1 ls",
    "env rm x",
    "git push",
    "git commit -m x",
    "git checkout main",
    "git reset --hard",
    "git branch -D main",
    "git branch newbranch",
    "git log --output=x.txt",
    "git diff --output=x",
    "git status; git push",
    "python script.py",
    "python3 -c 'print(1)'",
    "python -m http.server",
    "pip install requests",
    "pip uninstall x",
    "curl http://example.com",
    "wget http://example.com",
    "ls | tee out.txt",
    "ls | grep x | sort",
    "cat x | python3",
    "./script.sh",
    "/bin/ls",
    "FOO=bar ls",
    "nohup ls",
    "timeout 5 ls",
    "ls 'unterminated",
    "ls |",
    "| ls",
    "ls ||",
    "echo 'a;b'",
    "ls {a,b}",
    "ls (x)",
    "ls \\;",
    "systemctl stop x",
    "kill -9 1",
    "shutdown now",
    "x" * 500,
]


@pytest.mark.parametrize("cmd", SAFE)
def test_bash_safe(cmd):
    assert shell.classify_bash(cmd) is Risk.SAFE, cmd


@pytest.mark.parametrize("cmd", CONFIRM)
def test_bash_confirm(cmd):
    assert shell.classify_bash(cmd) is Risk.CONFIRM, cmd


def test_bash_table_size():
    assert len(SAFE) + len(CONFIRM) >= 40


def test_classify_command_follows_the_platform_shell(monkeypatch):
    plat.set_backend(LinuxPlatform())
    assert shell.classify_command("ls -la") is Risk.SAFE
    assert shell.classify_command("Get-ChildItem") is Risk.CONFIRM  # not a bash command
    plat.set_backend(WindowsPlatform())
    assert shell.classify_command("Get-ChildItem") is Risk.SAFE
    assert shell.classify_command("ls -la") is Risk.SAFE  # PowerShell alias
    assert shell.classify_command("ls -la", shell="bash") is Risk.SAFE
    assert shell.classify_command("rm -rf x", shell="bash") is Risk.CONFIRM


def test_run_command_describes_the_shell_in_use():
    spec = shell.run_command.__tool_spec__
    if sys.platform == "win32":
        assert "PowerShell" in spec.description and "Windows" in spec.description
    else:
        assert "bash" in spec.description and "Linux" in spec.description
        assert "PowerShell" not in spec.description
    assert spec.parameters["properties"]["command"]["description"].startswith(plat.current().shell_name)


def test_confirmation_text_names_the_shell(fake_platform):
    spec = shell.run_command.__tool_spec__
    assert spec.confirmation_text({"command": "rm x"}).startswith("Run bash command")
    assert spec.confirmation_text({"command": "rm x", "cwd": "/tmp"}) == "Run bash command in /tmp: rm x"


def test_risk_is_decided_per_call_with_the_bash_classifier(fake_platform):
    spec = shell.run_command.__tool_spec__
    assert spec.risk_for({"command": "ls -la"}) is Risk.SAFE
    assert spec.risk_for({"command": "rm -rf ~"}) is Risk.CONFIRM


def test_linux_shell_argv_is_a_login_bash():
    assert LinuxPlatform().shell_argv("echo hi") == ["bash", "-lc", "echo hi"]
    argv = WindowsPlatform().shell_argv("Get-Date")
    assert argv[0] == "powershell.exe" and argv[-1].endswith("Get-Date")


# ------------------------------------------------------------ real bash (POSIX)
posix_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs a POSIX system with bash"
)


class Asker:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, text: str) -> bool:
        self.prompts.append(text)
        return self.answer


def make_ctx(answer: bool = True) -> tuple[ToolContext, Asker]:
    asker = Asker(answer)
    return ToolContext(settings=Settings(), confirm=asker), asker


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_module(files)
    reg.register_module(shell)
    return reg


async def call(reg: ToolRegistry, name: str, ctx: ToolContext | None = None, /, **args: Any):
    return await reg.execute(ToolCall(id="t", name=name, arguments=args), ctx or make_ctx()[0])


@posix_bash
async def test_run_command_safe_needs_no_confirmation(registry, tmp_path):
    ctx, asker = make_ctx(False)
    (tmp_path / "marker.txt").write_text("x")
    r = await call(registry, "run_command", ctx, command="ls", cwd=str(tmp_path))
    assert r.ok and "marker.txt" in r.content and "Exit code 0" in r.content
    assert asker.prompts == []


@posix_bash
async def test_run_command_confirm_and_exit_code(registry, tmp_path):
    ctx, asker = make_ctx(True)
    r = await call(registry, "run_command", ctx, command="echo boom; exit 3", cwd=str(tmp_path))
    assert not r.ok and "Exit code 3" in r.content and "boom" in r.content
    assert len(asker.prompts) == 1 and "echo boom" in asker.prompts[0]
    denied, _ = make_ctx(False)
    r = await call(registry, "run_command", denied, command="touch x.txt", cwd=str(tmp_path))
    assert not r.ok and "declined" in r.content and not (tmp_path / "x.txt").exists()


@posix_bash
async def test_run_command_stderr_merged(registry):
    r = await call(registry, "run_command", command="echo bad >&2; echo ok")
    assert "bad" in r.content and "ok" in r.content


@posix_bash
async def test_run_command_bad_cwd(registry, tmp_path):
    r = await call(registry, "run_command", command="pwd", cwd=str(tmp_path / "nope"))
    assert not r.ok and "does not exist" in r.content


@posix_bash
async def test_run_command_output_tail_truncated(registry):
    r = await call(registry, "run_command", command="for i in $(seq 1 3000); do echo line number $i; done")
    assert r.ok and "earlier characters omitted" in r.content and "line number 3000" in r.content


@posix_bash
async def test_run_command_timeout_kills_the_whole_process_group(registry, tmp_path):
    import psutil

    pidfile = tmp_path / "child.pid"
    # A background grandchild: the group kill must reach it, not just the shell.
    cmd = f"sleep 120 & echo $! > '{pidfile}'; wait"
    t0 = time.monotonic()
    r = await call(registry, "run_command", command=cmd, timeout_seconds=3)
    assert time.monotonic() - t0 < 30
    assert not r.ok and "timed out" in r.content
    pid = int(pidfile.read_text().strip())
    await asyncio.sleep(0.5)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@posix_bash
async def test_run_command_cancel_kills_the_group(registry):
    ctx, _ = make_ctx(True)
    asyncio.get_running_loop().call_later(1.0, ctx.cancelled.set)
    t0 = time.monotonic()
    r = await call(registry, "run_command", ctx, command="sleep 60", timeout_seconds=60)
    assert time.monotonic() - t0 < 20
    assert not r.ok and "cancelled" in r.content


@posix_bash
async def test_login_shell_is_used():
    # `bash -lc` makes $0 start with a dash only for login shells launched via `bash -l`
    ctx, _ = make_ctx(True)
    r = await shell.run_command("shopt -q login_shell && echo login || echo notlogin", ctx)
    assert "login" in r.content and "notlogin" not in r.content
    assert os.path.isdir(os.path.expanduser("~"))
