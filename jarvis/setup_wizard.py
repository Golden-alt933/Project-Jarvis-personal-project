"""``jarvis --setup``: a short first-run in the terminal.

Choose a provider, paste the API key (input hidden), optionally say what to call you, and
the answers are written to the per-user config file with user-only permissions where the
OS supports them. The key is never echoed, printed or logged. Then the doctor runs.

When stdin is not a terminal nothing is asked: the text to put in the file is printed.
"""

from __future__ import annotations

import getpass
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from jarvis.config import Settings, get_settings, user_config_file

Prompt = Callable[[str], str]


@dataclass(frozen=True)
class ProviderChoice:
    id: str
    label: str
    key_var: str | None  # None: no key (local)
    key_help: str = ""


PROVIDER_CHOICES = [
    ProviderChoice("nvidia", "NVIDIA (free tier, recommended)", "NVIDIA_API_KEY", "Get a free key at https://build.nvidia.com"),
    ProviderChoice("groq", "Groq", "GROQ_API_KEY", "Get a key at https://console.groq.com/keys"),
    ProviderChoice("openrouter", "OpenRouter", "OPENROUTER_API_KEY", "Get a key at https://openrouter.ai/keys"),
    ProviderChoice("gemini", "Google Gemini", "GEMINI_API_KEY", "Get a key at https://aistudio.google.com/apikey"),
    ProviderChoice("custom", "Any OpenAI-compatible endpoint", "CUSTOM_API_KEY", "You need the endpoint's base URL and a key"),
    ProviderChoice("ollama", "Ollama only (fully local, no key)", None),
]

_SAFE_VALUE = re.compile(r"^[^\r\n\"'\\$`]*$")


def _quote(value: str) -> str:
    return f'"{value}"' if re.search(r"\s|#", value) else value


def read_env_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def env_value(lines: list[str], name: str) -> str:
    for line in lines:
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            return value.strip().strip('"').strip("'")
    return ""


def update_env_lines(lines: list[str], updates: dict[str, str]) -> list[str]:
    """Set each NAME=value, replacing an existing assignment in place and appending the rest.
    Comments and unrelated lines are kept."""
    out = list(lines)
    pending = dict(updates)
    for i, line in enumerate(out):
        key, sep, _ = line.partition("=")
        name = key.strip()
        if sep and name in pending and not line.lstrip().startswith("#"):
            out[i] = f"{name}={_quote(pending.pop(name))}"
    for name, value in pending.items():
        out.append(f"{name}={_quote(value)}")
    return out


def write_private_file(path: Path, text: str) -> None:
    """Write ``text`` atomically; the file (and a newly made folder) is readable by this
    user only where the OS supports permissions (a no-op on Windows, where the profile
    folder is already private)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def non_interactive_instructions(path: Path) -> str:
    return (
        "Not a terminal, so nothing was asked. Put these lines in:\n"
        f"    {path}\n\n"
        "    JARVIS_PROVIDER=nvidia\n"
        "    NVIDIA_API_KEY=<your key from https://build.nvidia.com>\n"
        "    JARVIS_USER_NAME=<what Jarvis should call you>   (optional)\n\n"
        "Then run: jarvis --doctor\n"
        "(For a fully local setup use JARVIS_PROVIDER=ollama and no key.)"
    )


def _choose_provider(console: Any, ask: Prompt) -> ProviderChoice:
    console.print("Which brain should Jarvis use?")
    for number, choice in enumerate(PROVIDER_CHOICES, 1):
        console.print(f"  {number}. {choice.label}")
    while True:
        answer = ask("Choose 1-%d [1]: " % len(PROVIDER_CHOICES)).strip()
        if not answer:
            return PROVIDER_CHOICES[0]
        if answer.isdigit() and 1 <= int(answer) <= len(PROVIDER_CHOICES):
            return PROVIDER_CHOICES[int(answer) - 1]
        for choice in PROVIDER_CHOICES:
            if answer.lower() == choice.id:
                return choice
        console.print("Please enter a number from the list.")


def run_setup(
    console: Any,
    *,
    ask: Prompt | None = None,
    ask_secret: Prompt | None = None,
    interactive: bool | None = None,
    config_path: Path | None = None,
    doctor: Callable[[Settings], int] | None = None,
) -> int:
    """Run the first-run questions. ``ask``/``ask_secret`` default to input() and getpass;
    ``doctor`` is called with fresh settings at the end (default: the real doctor, online)."""
    path = config_path or user_config_file()
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        console.print(non_interactive_instructions(path))
        return 0
    ask = ask or input
    ask_secret = ask_secret or getpass.getpass
    lines = read_env_lines(path)
    updates: dict[str, str] = {}

    console.print("Jarvis setup. Your answers go to a file only you can read:")
    console.print(f"  {path}\n")
    choice = _choose_provider(console, ask)
    updates["JARVIS_PROVIDER"] = choice.id

    if choice.id == "custom":
        base = ask("Base URL of the endpoint (for example https://host/v1): ").strip()
        if not base or not _SAFE_VALUE.match(base):
            console.print("That is not a usable URL; nothing was changed.")
            return 2
        updates["CUSTOM_BASE_URL"] = base
    if choice.key_var:
        if choice.key_help:
            console.print(choice.key_help)
        existing = env_value(lines, choice.key_var)
        prompt = (
            "A key is already saved. Press Enter to keep it, or paste a new one (hidden): "
            if existing
            else "Paste your API key (hidden, nothing is shown as you type): "
        )
        key = ask_secret(prompt).strip()
        if key:
            if not _SAFE_VALUE.match(key):
                console.print("That key contains characters that cannot be stored; nothing was changed.")
                return 2
            updates[choice.key_var] = key
        elif not existing and choice.id != "custom":
            console.print("No key entered; nothing was changed.")
            return 2

    name = ask("What should Jarvis call you? (optional, Enter to skip): ").strip()
    if name:
        if not _SAFE_VALUE.match(name):
            console.print("Skipped the name: it contains characters that cannot be stored.")
        else:
            updates["JARVIS_USER_NAME"] = name

    write_private_file(path, "\n".join(update_env_lines(lines, updates)) + "\n")
    console.print(f"\nSaved to {path}")
    console.print("Checking the setup...\n")

    get_settings.cache_clear()
    if doctor is None:
        from jarvis.doctor import run_doctor

        def doctor(settings: Settings) -> int:  # noqa: F811
            return run_doctor(settings, console)

    return doctor(Settings())


__all__ = ["run_setup", "PROVIDER_CHOICES", "update_env_lines", "write_private_file", "non_interactive_instructions"]
