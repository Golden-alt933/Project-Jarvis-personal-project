"""Shell tools: run a PowerShell command or a Python snippet.

``run_command`` is SAFE only for a single read-only command on a conservative
allowlist (see ``classify_command``); anything else asks the user first. When
unsure the classifier says CONFIRM. The process tree is killed on timeout or
cancellation. Python snippets always need approval.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import psutil

from jarvis.tools.base import Risk, ToolContext, ToolResult, tool
from jarvis.tools.builtin.files import resolve_path

log = logging.getLogger(__name__)

GROUP = "shell"
MAX_OUTPUT_CHARS = 8000
MAX_TIMEOUT = 600
SPEC_TIMEOUT = 630.0  # must exceed the largest timeout_seconds so the tool enforces its own limit

# ----------------------------------------------------------- risk classifier
# Characters that chain commands, redirect, expand, or open script blocks. Rejected
# anywhere in the string (even inside quotes): being conservative costs only a prompt.
_FORBIDDEN_CHARS = set(";&><`${}()\n\r")
_FORBIDDEN_WORDS = {
    "invoke-expression", "iex", "start-process", "saps", "start", "-encodedcommand", "-enc",
    "-ec", "invoke-command", "icm", "invoke-webrequest", "iwr", "curl", "wget", "invoke-restmethod",
    "irm", "powershell", "powershell.exe", "pwsh", "cmd", "cmd.exe", "-command", "-file",
}
_SIMPLE_READERS = {
    "get-childitem", "gci", "dir", "ls", "get-content", "gc", "type", "cat", "get-process", "gps",
    "ps", "get-date", "whoami", "hostname", "systeminfo", "get-command", "gcm", "where.exe",
    "echo", "write-output", "get-location", "pwd", "get-service", "test-path", "get-item", "gi",
    "tasklist", "select-string", "sls", "findstr", "get-computerinfo", "get-volume",
}
_PIPE_FILTERS = {
    "select-object", "select", "sort-object", "sort", "measure-object", "measure", "format-table",
    "ft", "format-list", "fl", "out-string", "group-object", "group", "where-object", "select-string",
    "sls", "findstr", "get-unique", "unique",
}
_GIT_READ = {"status", "log", "diff", "show"}
_GIT_BRANCH_FLAGS = {"-a", "-r", "-v", "-vv", "--list", "--show-current", "--all", "--remotes", "--verbose"}
_GIT_BAD_ARG_PARTS = ("output", "ext-diff", "textconv", "exec", "upload-pack")
_PIP_READ = {"list", "freeze", "show", "--version", "-v"}


def _tokens(segment: str) -> list[str] | None:
    try:
        return shlex.split(segment, posix=False)
    except ValueError:
        return None


def _simple_command_ok(tokens: list[str]) -> bool:
    head = tokens[0].lower()
    args = [t.lower() for t in tokens[1:]]
    if head == "git":
        if not args:
            return False
        sub = args[0]
        if any(part in a for a in args for part in _GIT_BAD_ARG_PARTS):
            return False
        if sub in _GIT_READ:
            return True
        return sub == "branch" and all(a in _GIT_BRANCH_FLAGS for a in args[1:])
    if head in ("python", "python3", "py"):
        if args in (["--version"], ["-v"]):
            return True
        return len(args) >= 3 and args[:3] == ["-m", "pip", args[2]] and args[2] in _PIP_READ
    if head in ("pip", "pip3"):
        return bool(args) and args[0] in _PIP_READ
    if head == "ipconfig":
        return args in ([], ["/all"])
    return head in _SIMPLE_READERS


def classify_command(command: str) -> Risk:
    """SAFE only for a single simple read-only command (optionally piped through read-only
    filters); CONFIRM for everything else, including anything the parser is unsure about."""
    text = (command or "").strip()
    if not text or len(text) > 400 or any(c in _FORBIDDEN_CHARS for c in text):
        return Risk.CONFIRM
    segments = [s.strip() for s in text.split("|")]
    for i, seg in enumerate(segments):
        toks = _tokens(seg) if seg else None
        if not toks:
            return Risk.CONFIRM
        if any(t.strip("'\"").lower() in _FORBIDDEN_WORDS for t in toks):
            return Risk.CONFIRM
        if i == 0:
            if not _simple_command_ok(toks):
                return Risk.CONFIRM
        elif toks[0].lower() not in _PIPE_FILTERS:
            return Risk.CONFIRM
    return Risk.SAFE


def _command_risk(args: dict[str, Any]) -> Risk:
    return classify_command(str(args.get("command", "")))


def _command_describe(args: dict[str, Any]) -> str:
    cmd = str(args.get("command", ""))
    shown = cmd if len(cmd) <= 300 else cmd[:300] + "..."
    where = f" in {args['cwd']}" if args.get("cwd") else ""
    return f"Run PowerShell command{where}: {shown}"


# ----------------------------------------------------------------- execution
def _kill_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=3)


def _tail(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"[... {omitted} earlier characters omitted; showing the end ...]\n{text[-MAX_OUTPUT_CHARS:]}"


async def _run_process(
    argv: list[str], cwd: str | None, timeout: int, ctx: ToolContext, env: dict[str, str] | None = None
) -> ToolResult:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merged so interleaving matches what a terminal shows
            stdin=asyncio.subprocess.DEVNULL,
            cwd=cwd,
            env=env,
            creationflags=flags,
        )
    except OSError as exc:
        return ToolResult.fail(f"could not start {argv[0]}: {exc}")

    comm = asyncio.ensure_future(proc.communicate())
    cancel = asyncio.ensure_future(ctx.cancelled.wait())
    reason = None
    try:
        done, _ = await asyncio.wait({comm, cancel}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if comm not in done:
            reason = "cancelled by the user" if cancel in done else f"timed out after {timeout}s"
            _kill_tree(proc.pid)
    except asyncio.CancelledError:
        _kill_tree(proc.pid)
        comm.cancel()
        raise
    finally:
        cancel.cancel()

    try:
        out, _ = await asyncio.wait_for(comm, timeout=5 if reason else None)
    except asyncio.TimeoutError:
        out = b""
    text = _tail((out or b"").decode("utf-8", errors="replace").strip())
    if reason:
        return ToolResult.fail(
            f"command {reason}; the process tree was killed" + (f"\nOutput so far:\n{text}" if text else ""),
            "Run something shorter, or raise timeout_seconds (max 600).",
        )
    code = proc.returncode
    body = f"Exit code {code}" + (f"\n{text}" if text else " (no output)")
    return ToolResult(ok=code == 0, content=body, data={"exit_code": code})


def _clamp(seconds: int) -> int:
    return max(1, min(int(seconds), MAX_TIMEOUT))


@tool(group=GROUP, risk=_command_risk, describe=_command_describe, timeout=SPEC_TIMEOUT)
async def run_command(
    command: str, ctx: ToolContext, cwd: str | None = None, timeout_seconds: int = 60
) -> ToolResult:
    """Run a PowerShell command on this Windows PC and return its output and exit code. Use for
    anything the dedicated tools do not cover (git, pip, system info, scripts). Simple read-only
    commands run immediately; everything else asks the user first. A non-zero exit code is
    reported as a failure with the output.

    Args:
        command: PowerShell command line to run.
        cwd: Working folder (default: the home folder).
        timeout_seconds: Kill the command after this many seconds (1-600).
    """
    if not command.strip():
        return ToolResult.fail("command is empty")
    try:
        directory = resolve_path(cwd) if cwd else Path(os.path.expanduser("~"))
    except ValueError as exc:
        return ToolResult.fail(str(exc))
    if not directory.is_dir():
        return ToolResult.fail(f"working folder does not exist: {directory}", "Check the path with list_directory.")
    # UTF-8 output and no progress bars (they pollute captured output).
    script = "$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command
    argv = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]
    return await _run_process(argv, str(directory), _clamp(timeout_seconds), ctx)


def _python_describe(args: dict[str, Any]) -> str:
    lines = str(args.get("code", "")).splitlines()
    head = "\n".join(lines[:8])
    more = f"\n... ({len(lines) - 8} more lines)" if len(lines) > 8 else ""
    return f"Run this Python code:\n{head}{more}"


@tool(group=GROUP, risk=Risk.CONFIRM, describe=_python_describe, timeout=SPEC_TIMEOUT)
async def run_python(code: str, ctx: ToolContext, timeout_seconds: int = 60) -> ToolResult:
    """Run a Python snippet in a fresh subprocess and return its printed output and exit code.
    Use for calculations, data wrangling or quick scripts; print what you want to see. Always
    needs user approval. State does not persist between calls.

    Args:
        code: Python source code to execute.
        timeout_seconds: Kill the script after this many seconds (1-600).
    """
    if not code.strip():
        return ToolResult.fail("code is empty")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(prefix="jarvis-py-") as tmp:
        script = Path(tmp) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        argv = [sys.executable, "-u", str(script)]
        return await _run_process(argv, os.path.expanduser("~"), _clamp(timeout_seconds), ctx, env)
