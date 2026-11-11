"""Shell tools: run a shell command (PowerShell on Windows, bash on Linux) or a Python snippet.

``run_command`` is SAFE only for a single read-only command on a conservative
allowlist (``classify_powershell`` / ``classify_bash``, chosen by the platform's shell);
anything else asks the user first. When unsure the classifier says CONFIRM. The process
tree (on Linux, the whole process group) is killed on timeout or cancellation. Python
snippets always need approval.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

from jarvis import platform as plat
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


def classify_powershell(command: str) -> Risk:
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


# ---- bash. Single simple commands from an allowlist, optionally piped through other
# allowlisted commands. Anything that chains, redirects, expands, substitutes, elevates or
# runs another program is CONFIRM. The character check looks at the raw text, so a banned
# character inside quotes also asks: being conservative costs only a prompt.
_BASH_FORBIDDEN_CHARS = set(";&<>`$(){}\\\n\r")
_BASH_FORBIDDEN_WORDS = {
    "sudo", "su", "doas", "pkexec", "eval", "exec", "xargs", "sh", "bash", "zsh", "dash", "fish",
    "source", "nohup", "setsid", "env", "command", "builtin", "time", "watch", "timeout", "nice",
}
_BASH_READERS = {
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "pwd", "whoami", "hostname", "date",
    "uname", "df", "du", "free", "ps", "uptime", "which", "env", "printenv", "echo",
}
_FIND_BAD = {
    "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls",
}
_RG_BAD_PREFIXES = ("--pre", "--hostname-bin")
_TAIL_FOLLOW = {"-f", "-F", "--follow"}


def _bash_simple_ok(tokens: list[str]) -> bool:
    head, args = tokens[0], tokens[1:]
    if head in _BASH_FORBIDDEN_WORDS and head != "env":
        return False
    if head == "git":
        if not args:
            return False
        sub = args[0]
        if any(part in a.lower() for a in args for part in _GIT_BAD_ARG_PARTS):
            return False
        if sub in _GIT_READ:
            return True
        return sub == "branch" and all(a in _GIT_BRANCH_FLAGS for a in args[1:])
    if head in ("python", "python3"):
        if args in (["--version"], ["-V"]):
            return True
        return len(args) >= 3 and args[:2] == ["-m", "pip"] and args[2] in _PIP_READ
    if head in ("pip", "pip3"):
        return bool(args) and args[0] in _PIP_READ
    if head not in _BASH_READERS:
        return False
    if head == "env":
        return not args  # `env cmd` would run cmd
    if head == "find":
        return not any(a in _FIND_BAD for a in args)
    if head == "rg":
        return not any(a.startswith(_RG_BAD_PREFIXES) for a in args)
    if head == "tail":
        return not any(a in _TAIL_FOLLOW or (a.startswith("-") and not a.startswith("--") and "f" in a.lower() and a[1:].isalpha()) for a in args)
    if head == "hostname":
        return all(a.startswith("-") for a in args)  # a bare argument would set the name
    if head == "date":
        return all((a.startswith("+") or a.startswith("-")) and not a.startswith(("-s", "--set")) for a in args)
    return True


def classify_bash(command: str) -> Risk:
    """SAFE only for a single simple read-only command (optionally piped into other
    allowlisted commands); CONFIRM for everything else, including anything unclear."""
    text = (command or "").strip()
    if not text or len(text) > 400 or any(c in _BASH_FORBIDDEN_CHARS for c in text):
        return Risk.CONFIRM
    for segment in text.split("|"):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return Risk.CONFIRM
        if not tokens or not _bash_simple_ok(tokens):
            return Risk.CONFIRM
    return Risk.SAFE


def classify_command(command: str, shell: str | None = None) -> Risk:
    """Risk of a command for ``shell`` ('powershell' or 'bash'; default: this machine's)."""
    kind = shell or plat.current().shell_kind
    return classify_bash(command) if kind == "bash" else classify_powershell(command)


def _command_risk(args: dict[str, Any]) -> Risk:
    return classify_command(str(args.get("command", "")))


def _command_describe(args: dict[str, Any]) -> str:
    cmd = str(args.get("command", ""))
    shown = cmd if len(cmd) <= 300 else cmd[:300] + "..."
    where = f" in {args['cwd']}" if args.get("cwd") else ""
    return f"Run {plat.current().shell_name} command{where}: {shown}"


# ----------------------------------------------------------------- execution
def _kill_tree(pid: int) -> None:
    plat.current().kill_tree(pid)


def _tail(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"[... {omitted} earlier characters omitted; showing the end ...]\n{text[-MAX_OUTPUT_CHARS:]}"


async def _run_process(
    argv: list[str], cwd: str | None, timeout: int, ctx: ToolContext, env: dict[str, str] | None = None
) -> ToolResult:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merged so interleaving matches what a terminal shows
            stdin=asyncio.subprocess.DEVNULL,
            cwd=cwd,
            env=env,
            **plat.current().subprocess_kwargs(),
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


_RUN_COMMAND_DOC = """Run a {shell} command on this {os} machine and return its output and exit code. Use
    for anything the dedicated tools do not cover (git, pip, system info, scripts). Simple
    read-only commands run immediately; everything else asks the user first. A non-zero exit
    code is reported as a failure with the output.

    Args:
        command: {shell} command line to run.
        cwd: Working folder (default: the home folder).
        timeout_seconds: Kill the command after this many seconds (1-600).
    """


async def run_command(
    command: str, ctx: ToolContext, cwd: str | None = None, timeout_seconds: int = 60
) -> ToolResult:
    if not command.strip():
        return ToolResult.fail("command is empty")
    try:
        directory = resolve_path(cwd) if cwd else Path(os.path.expanduser("~"))
    except ValueError as exc:
        return ToolResult.fail(str(exc))
    if not directory.is_dir():
        return ToolResult.fail(f"working folder does not exist: {directory}", "Check the path with list_directory.")
    argv = plat.current().shell_argv(command)
    return await _run_process(argv, str(directory), _clamp(timeout_seconds), ctx)


# The model reads the docstring, so it names the shell this machine really uses; the
# platform is known at import time, before the decorator builds the tool spec.
run_command.__doc__ = _RUN_COMMAND_DOC.format(
    shell=plat.current().shell_name, os={"windows": "Windows", "linux": "Linux"}.get(plat.current().name, "this")
)
run_command = tool(group=GROUP, risk=_command_risk, describe=_command_describe, timeout=SPEC_TIMEOUT)(run_command)


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
