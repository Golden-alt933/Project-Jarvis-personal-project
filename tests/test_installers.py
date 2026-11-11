"""The one-command installers. Static checks run everywhere; install.sh is also driven for
real on Linux with a fake `uv` (nothing is downloaded or installed)."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SH = ROOT / "install.sh"
PS1 = ROOT / "install.ps1"
REPO = "https://github.com/Golden-alt933/Project-Jarvis-personal-project"

pytestmark = pytest.mark.skipif(not SH.is_file(), reason="installers are not part of this checkout")


def project_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


# ----------------------------------------------------------------- static checks
def test_scripts_are_strict_and_pin_the_release_in_one_clearly_marked_line():
    sh, ps1 = SH.read_text(encoding="utf-8"), PS1.read_text(encoding="utf-8")
    assert "set -euo pipefail" in sh
    assert "$ErrorActionPreference = 'Stop'" in ps1
    refs = re.findall(r'^JARVIS_DEFAULT_REF="(v[^"]+)"$', sh, re.M)
    assert refs == [f"v{project_version()}"]
    assert re.findall(r"^\$DefaultRef = '(v[^']+)'$", ps1, re.M) == refs
    assert sh.count("bump this one line") == 1 and ps1.count("bump this one line") == 1
    assert sh.startswith("#!/usr/bin/env bash")


def test_the_install_command_is_the_documented_one():
    sh, ps1 = SH.read_text(encoding="utf-8"), PS1.read_text(encoding="utf-8")
    assert f'REPO_URL="{REPO}"' in sh and f"$RepoUrl = '{REPO}'" in ps1
    assert 'SPEC="jarvis[${EXTRAS}] @ git+${REPO_URL}@${JARVIS_REF}"' in sh
    assert "uv tool install --python 3.13" in sh and "uv tool install --python 3.13" in ps1
    assert "@ git+${RepoUrl}@${Ref}" in ps1


def test_scripts_use_the_official_uv_installers_and_say_so():
    sh, ps1 = SH.read_text(encoding="utf-8"), PS1.read_text(encoding="utf-8")
    assert "https://astral.sh/uv/install.sh" in sh and "https://astral.sh/uv/install.ps1" in ps1
    assert "Installing it with Astral's official installer" in sh and "Installing it with Astral's official installer" in ps1


def test_options_exist_in_both_scripts():
    sh, ps1 = SH.read_text(encoding="utf-8"), PS1.read_text(encoding="utf-8")
    for option in ("--minimal", "--gpu", "--ref"):
        assert option in sh
    for option in ("minimal", "gpu", "ref"):
        assert f"(--|-){option}" in ps1
    assert "--system-deps" in sh
    for env in ("JARVIS_REF", "JARVIS_NO_SETUP"):
        assert env in sh and env in ps1


def test_sudo_only_runs_behind_the_system_deps_flag():
    lines = SH.read_text(encoding="utf-8").splitlines()
    runs = [i for i, line in enumerate(lines) if re.match(r'\s*sudo "', line)]
    assert len(runs) == 1
    gate = next(i for i, line in enumerate(lines) if line.strip() == 'if [ "$SYSTEM_DEPS" = 1 ]; then')
    otherwise = next(i for i, line in enumerate(lines) if "Not run: pass --system-deps" in line)
    assert gate < runs[0] < otherwise  # inside the flag's branch, before its else


def test_no_secrets_in_the_scripts():
    for path in (SH, PS1):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"nvapi-|sk-[A-Za-z0-9]{10}|ghp_|api[_-]?key\s*=\s*\S{12,}", text, re.I), path


def test_powershell_script_parses():
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("no PowerShell on this machine")
    cmd = f"$e=$null;[void][System.Management.Automation.Language.Parser]::ParseFile('{PS1}',[ref]$null,[ref]$e);if($e){{$e|%{{$_.Message}};exit 1}}"
    done = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs a POSIX bash (on Windows `bash` may be the WSL stub)")
def test_bash_syntax():
    done = subprocess.run(["bash", "-n", str(SH)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# ----------------------------------------------------------------- driving install.sh (Linux)
linux_only = pytest.mark.skipif(sys.platform != "linux" or shutil.which("bash") is None, reason="install.sh targets Linux")


@pytest.fixture
def sandbox(tmp_path):
    """A PATH with a fake uv (records its arguments), a fake jarvis and a fake sudo."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv.log"
    sudo_log = tmp_path / "sudo.log"
    tool_bin = tmp_path / "toolbin"
    tool_bin.mkdir()

    def script(path: Path, body: str) -> None:
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    script(
        bin_dir / "uv",
        f'''case "$1 $2" in
  "tool install") printf '%s\\n' "$*" >> "{log}" ;;
  "tool dir") echo "{tool_bin}" ;;
  *) [ "$1" = "--version" ] && echo "uv 0.0.0-test" ;;
esac
exit 0
''',
    )
    script(tool_bin / "jarvis", 'echo "jarvis 0.0.0-test"; echo "ARGS:$*"\n')
    script(bin_dir / "sudo", f'printf \'%s\\n\' "$*" >> "{sudo_log}"\nexit 0\n')
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "JARVIS_NO_SETUP": "1",
        "TERM": "dumb",
    }
    return types_ns(bin_dir=bin_dir, log=log, sudo_log=sudo_log, env=env)


def types_ns(**kw):
    import types

    return types.SimpleNamespace(**kw)


def run_install(sandbox, *args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(sandbox.env, **(env_extra or {}))
    return subprocess.run(["bash", str(SH), *args], capture_output=True, text=True, env=env, timeout=60, stdin=subprocess.DEVNULL)


def uv_calls(sandbox) -> list[str]:
    return sandbox.log.read_text().splitlines() if sandbox.log.exists() else []


@linux_only
def test_default_install_pins_the_release_and_uses_all_extras(sandbox):
    done = run_install(sandbox)
    assert done.returncode == 0, done.stdout + done.stderr
    (call,) = uv_calls(sandbox)
    assert call.startswith("tool install --python 3.13")
    assert f"jarvis[all] @ git+{REPO}@v{project_version()}" in call
    assert "jarvis 0.0.0-test" in done.stdout and "Next steps" in done.stdout
    assert "uv tool uninstall jarvis" in done.stdout


@linux_only
def test_minimal_gpu_and_ref_options(sandbox):
    done = run_install(sandbox, "--minimal", "--ref", "my-branch")
    assert done.returncode == 0, done.stderr
    assert f"jarvis @ git+{REPO}@my-branch" in uv_calls(sandbox)[-1]
    run_install(sandbox, "--gpu")
    assert f"jarvis[all,gpu] @ git+{REPO}@v{project_version()}" in uv_calls(sandbox)[-1]
    run_install(sandbox, "--minimal", "--gpu", "--ref=abc123")
    assert f"jarvis[gpu] @ git+{REPO}@abc123" in uv_calls(sandbox)[-1]


@linux_only
def test_ref_from_the_environment_and_rerunning_is_idempotent(sandbox):
    assert run_install(sandbox, env_extra={"JARVIS_REF": "v9.9.9"}).returncode == 0
    assert run_install(sandbox, env_extra={"JARVIS_REF": "v9.9.9"}).returncode == 0
    calls = uv_calls(sandbox)
    assert len(calls) == 2 and calls[0] == calls[1] and "@v9.9.9" in calls[0]
    assert "--force" in calls[0] and "--reinstall-package jarvis" in calls[0]  # re-running upgrades


@linux_only
def test_flag_overrides_environment_ref(sandbox):
    run_install(sandbox, "--ref", "flag-wins", env_extra={"JARVIS_REF": "env-loses"})
    assert "@flag-wins" in uv_calls(sandbox)[-1]


@linux_only
def test_bad_options_fail_fast_without_installing(sandbox):
    for args in (("--bogus",), ("--ref",)):
        done = run_install(sandbox, *args)
        assert done.returncode != 0 and uv_calls(sandbox) == []
    assert "Usage" in run_install(sandbox, "--help").stdout


@linux_only
def test_missing_system_packages_are_printed_not_run(sandbox):
    done = run_install(sandbox)
    assert not sandbox.sudo_log.exists(), "sudo must never run without --system-deps"
    if "missing" in done.stdout:
        assert re.search(r"sudo (apt-get install -y|dnf install -y|pacman -S|zypper install -y) ", done.stdout) or "Unknown distribution" in done.stdout
        assert "Not run: pass --system-deps" in done.stdout or "Unknown distribution" in done.stdout


@linux_only
@pytest.mark.skipif(os.geteuid() == 0 if hasattr(os, "geteuid") else True, reason="as root the script would call the real package manager")
def test_system_deps_runs_only_the_printed_command_through_sudo(sandbox):
    done = run_install(sandbox, "--system-deps", "--minimal")
    assert done.returncode == 0, done.stdout + done.stderr
    if "missing" not in done.stdout:
        pytest.skip("this machine already has every helper program")
    if "Unknown distribution" in done.stdout:
        pytest.skip("distribution not recognised: the script only prints advice")
    assert "Running with sudo" in done.stdout
    ran = sandbox.sudo_log.read_text().strip()
    assert re.match(r"(apt-get install -y|dnf install -y|pacman -S --needed --noconfirm|zypper install -y) ", ran)
    printed = re.search(r"^\s+sudo (.+)$", done.stdout, re.M).group(1)
    assert ran == printed  # exactly what was shown to the user


@linux_only
def test_refuses_to_run_elsewhere_than_linux(sandbox, tmp_path):
    fake_uname = sandbox.bin_dir / "uname"
    fake_uname.write_text("#!/bin/sh\necho Darwin\n")
    fake_uname.chmod(fake_uname.stat().st_mode | stat.S_IEXEC)
    done = run_install(sandbox)
    assert done.returncode != 0 and "for Linux" in done.stderr and uv_calls(sandbox) == []
