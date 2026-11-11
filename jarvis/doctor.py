"""``jarvis --doctor``: a readable checklist of what works, what is optional and missing, and
what is broken, each with a one-line fix.

Exit code 0 unless something needed for text mode fails (Python version, a usable brain).
The API key is never printed. The one network probe (the hosted model) is skipped with
``--offline``. Every outside fact is read through a small function so tests can replace it.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from jarvis import platform as plat
from jarvis.config import Settings, config_files, user_config_file

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
HOSTED_TIMEOUT = 15.0
MIN_PYTHON = (3, 13)

# optional extras and the module that proves each is installed
EXTRAS: dict[str, tuple[str, ...]] = {
    "office": ("openpyxl", "pypdf", "docx"),
    "screen": ("mss", "PIL", "rapidocr_onnxruntime"),
    "browser": ("websockets",),
    "voice": ("numpy", "sounddevice", "faster_whisper", "piper", "onnxruntime", "av", "edge_tts", "riva"),
    "hud": ("PyQt6", "qasync"),
}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""
    required: bool = False  # a FAIL here makes the exit code non-zero


def _key_attr(settings: Settings) -> tuple[str, str] | None:
    from jarvis.llm.providers import _KEY_ENV, PROFILES

    profile = PROFILES.get(settings.provider.lower())
    if profile is None:
        return None
    return profile.key_setting_attr, _KEY_ENV[profile.key_setting_attr]


# ------------------------------------------------------------------ probes (replaceable)
def probe_hosted(settings: Settings) -> tuple[str, str]:
    """One tiny chat request to the configured provider: (status, detail)."""
    from jarvis.llm.providers import PROFILES

    provider = settings.provider.lower()
    profile = PROFILES[provider]
    base = settings.custom_base_url if provider == "custom" else profile.base_url
    key = getattr(settings, profile.key_setting_attr)
    model = settings.model or profile.default_model
    if not model:
        return WARN, "no model configured (set JARVIS_MODEL)"
    try:
        response = httpx.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            timeout=HOSTED_TIMEOUT,
        )
    except httpx.TimeoutException:
        return FAIL, f"no answer within {HOSTED_TIMEOUT:.0f} s"
    except httpx.HTTPError as exc:
        return FAIL, f"could not connect ({type(exc).__name__})"
    code = response.status_code
    if code == 200:
        return PASS, f"{provider} answered ({model})"
    if code in (401, 403):
        return FAIL, f"the key was rejected (HTTP {code})"
    if code == 404:
        return FAIL, f"model '{model}' was not found (HTTP 404)"
    if code == 429:
        return WARN, "reachable, but rate limited right now (HTTP 429)"
    return FAIL, f"HTTP {code}"


def probe_ollama(settings: Settings) -> tuple[bool, list[str]]:
    """(server answers, names of pulled models)."""
    try:
        response = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2.0)
        if response.status_code != 200:
            return False, []
        return True, [m.get("name", "") for m in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        return False, []


def find_ollama() -> str | None:
    from jarvis.llm.ollama import find_ollama_exe

    return find_ollama_exe()


def module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def probe_audio() -> tuple[int, int] | str:
    """(input devices, output devices), or an error text."""
    from jarvis.voice import audio

    sd = audio.sd
    if sd is None:  # not installed, PortAudio missing, or no sound server: audio.py caught it
        return audio.sounddevice_problem()
    try:
        devices = list(sd.query_devices())
    except Exception as exc:
        return f"could not query devices ({exc})"
    inputs = sum(1 for d in devices if d["max_input_channels"] > 0)
    outputs = sum(1 for d in devices if d["max_output_channels"] > 0)
    return inputs, outputs


def whisper_models_cached() -> list[str]:
    hub = os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    base = Path(hub)
    if not base.is_dir():
        return []
    return sorted(p.name.rsplit("--", 1)[-1] for p in base.glob("models--*faster-whisper-*"))


def cuda_devices() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def nvidia_smi_present() -> bool:
    return shutil.which("nvidia-smi") is not None


# ------------------------------------------------------------------ the checks
def check_python() -> Check:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] >= MIN_PYTHON:
        return Check("Python", PASS, version, required=True)
    return Check(
        "Python", FAIL, f"{version}; Jarvis needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
        "install Python 3.13 (uv python install 3.13)", required=True,
    )


def check_platform() -> Check:
    return Check("Platform", PASS, plat.current().describe())


def check_config() -> Check:
    files = config_files()
    if files:
        return Check("Config file", PASS, ", ".join(str(f) for f in files))
    return Check("Config file", WARN, f"none yet (would be {user_config_file()})", "run: jarvis --setup")


def check_key(settings: Settings) -> Check:
    provider = settings.provider.lower()
    if provider == "ollama":
        return Check("API key", SKIP, "provider is ollama: no key needed")
    attr = _key_attr(settings)
    if attr is None:
        return Check("API key", FAIL, f"unknown provider '{settings.provider}'", "run: jarvis --setup", required=True)
    field, env = attr
    if getattr(settings, field):
        return Check("API key", PASS, f"{env} is set for {provider}", required=True)
    if provider == "custom":
        if settings.custom_base_url:
            return Check("API key", PASS, "custom endpoint without a key", required=True)
        return Check("API key", FAIL, "CUSTOM_BASE_URL is not set", "run: jarvis --setup", required=True)
    return Check("API key", FAIL, f"{env} is not set", "run: jarvis --setup (free key: https://build.nvidia.com)", required=True)


def check_hosted(settings: Settings, offline: bool, key_ok: bool, local_ok: bool) -> Check:
    provider = settings.provider.lower()
    if provider == "ollama":
        return Check("Hosted model", SKIP, "provider is ollama")
    if offline:
        return Check("Hosted model", SKIP, "skipped (--offline)")
    if not key_ok:
        return Check("Hosted model", SKIP, "no key to try")
    status, detail = probe_hosted(settings)
    required = not local_ok  # with a working local model, an unreachable hosted one is not fatal
    if status == FAIL and local_ok:
        status = WARN
    fix = "" if status == PASS else "check the key and your connection, or use: jarvis --local"
    return Check("Hosted model", status, detail, fix, required=required and status == FAIL)


def check_ollama(settings: Settings) -> tuple[Check, bool]:
    needed = settings.provider.lower() == "ollama"
    exe = find_ollama()
    if exe is None:
        return Check(
            "Ollama", FAIL if needed else WARN, "not installed",
            "install from https://ollama.com/download (local fallback and --local mode)", required=needed,
        ), False
    running, models = probe_ollama(settings)
    if not running:
        return Check(
            "Ollama", FAIL if needed else WARN, f"installed ({exe}) but not running",
            "start it: ollama serve", required=needed,
        ), False
    wanted = settings.ollama_model
    have = any(m == wanted or m.split(":")[0] == wanted.split(":")[0] and ":" not in wanted for m in models)
    if not have:
        return Check(
            "Ollama", FAIL if needed else WARN, f"running, but model {wanted} is not pulled",
            f"run: ollama pull {wanted}", required=needed,
        ), False
    return Check("Ollama", PASS, f"running, {wanted} available"), True


def check_extras() -> list[Check]:
    checks = []
    for extra, modules in EXTRAS.items():
        missing = [m for m in modules if not module_present(m)]
        if not missing:
            checks.append(Check(f"Extra: {extra}", PASS, "installed"))
        else:
            checks.append(Check(
                f"Extra: {extra}", WARN, f"missing {', '.join(missing)}",
                f"reinstall with the extra: uv tool install --reinstall 'jarvis[{extra}]' (from a checkout: uv sync --extra {extra})",
            ))
    return checks


def check_audio(voice_installed: bool) -> Check:
    if not voice_installed:
        return Check("Audio devices", SKIP, "voice extra not installed")
    result = probe_audio()
    if isinstance(result, str):
        fix = "install the system library for PortAudio"
        for helper, present, hint in plat.current().helper_report():
            if helper == "libportaudio" and not present:
                fix = hint
        return Check("Audio devices", WARN, result, fix)
    inputs, outputs = result
    if inputs == 0 or outputs == 0:
        missing = "microphone" if inputs == 0 else "speaker"
        return Check("Audio devices", WARN, f"{inputs} input(s), {outputs} output(s): no {missing} found", "connect one, then see: jarvis --devices")
    return Check("Audio devices", PASS, f"{inputs} input(s), {outputs} output(s)")


def check_speech_models(settings: Settings, voice_installed: bool) -> Check:
    if not voice_installed:
        return Check("Speech models", SKIP, "voice extra not installed")
    piper_file = settings.voices_dir / f"{settings.tts_voice}.onnx"
    have_piper = piper_file.is_file()
    cached = whisper_models_cached()
    if have_piper and cached:
        return Check("Speech models", PASS, f"voice {settings.tts_voice}; recognition {', '.join(cached)}")
    missing = []
    if not have_piper:
        missing.append(f"voice {settings.tts_voice}")
    if not cached:
        missing.append("recognition model")
    return Check(
        "Speech models", WARN, f"not downloaded yet: {', '.join(missing)}",
        "they download on first use of: jarvis --voice (needs internet once)",
    )


SPEECH_PROBE_TIMEOUT = 8.0


def probe_online_tts(settings: Settings) -> tuple[str, str, str]:
    """(status, detail, engine names): synthesize one word with each online engine."""
    import asyncio

    from jarvis.voice.tts import FallbackSynthesizer, build_synthesizer

    synth = build_synthesizer(settings)
    if not isinstance(synth, FallbackSynthesizer):
        return SKIP, "", synth.name
    online = synth.engines[:-1]
    names = " > ".join(e.name for e in synth.engines)
    results = []
    for engine in online:
        try:
            audio = asyncio.run(asyncio.wait_for(engine.synthesize("Ready."), SPEECH_PROBE_TIMEOUT))
            results.append((engine.name, len(audio) > 0, ""))
        except Exception as exc:
            results.append((engine.name, False, type(exc).__name__))
    ok = [n for n, good, _ in results if good]
    if len(ok) == len(results):
        return PASS, f"{', '.join(ok)} answered", names
    bad = ", ".join(f"{n} ({why})" for n, good, why in results if not good)
    return WARN, f"not answering: {bad}", names


def probe_online_stt(settings: Settings) -> tuple[str, str]:
    """(status, detail) for the online recogniser, with one short silent request."""
    import asyncio

    import numpy as np

    from jarvis.voice.stt import build_transcriber
    from jarvis.voice.stt_online import GatedTranscriber

    transcriber = build_transcriber(settings)
    if not isinstance(transcriber, GatedTranscriber):
        return SKIP, transcriber.name
    try:
        asyncio.run(asyncio.wait_for(transcriber.online.transcribe(np.zeros(8000, dtype=np.float32)), SPEECH_PROBE_TIMEOUT))
    except Exception as exc:
        return WARN, f"{transcriber.online.name} did not answer ({type(exc).__name__})"
    return PASS, f"{transcriber.online.name} answered"


def check_speech_engines(settings: Settings, voice_installed: bool, offline: bool) -> list[Check]:
    """Which engines speech will use, and whether the online ones answer."""
    if not voice_installed:
        return [Check("Speech synthesis", SKIP, "voice extra not installed"),
                Check("Speech recognition", SKIP, "voice extra not installed")]
    from jarvis.voice.stt import resolve_stt_provider
    from jarvis.voice.tts import build_synthesizer, FallbackSynthesizer

    checks = []
    synth = build_synthesizer(settings)
    if isinstance(synth, FallbackSynthesizer):
        chain = " > ".join(e.name for e in synth.engines)
        if offline:
            checks.append(Check("Speech synthesis", SKIP, f"online: {chain} (not probed: --offline)"))
        else:
            status, detail, _ = probe_online_tts(settings)
            fix = "" if status == PASS else "speech falls back to the local Piper voice; check your connection"
            checks.append(Check("Speech synthesis", status, f"online, falls back to local: {chain}; {detail}", fix))
    else:
        checks.append(Check("Speech synthesis", PASS, f"local: {synth.name}"))
    stt = resolve_stt_provider(settings)
    if stt == "local":
        checks.append(Check("Speech recognition", PASS, "local: faster-whisper on this machine; no audio leaves it"))
    elif offline:
        checks.append(Check("Speech recognition", SKIP, f"online ({stt}), wake word gated locally (not probed: --offline)"))
    else:
        status, detail = probe_online_stt(settings)
        fix = "" if status == PASS else "recognition falls back to the local model; check the key and your connection"
        checks.append(Check("Speech recognition", status, f"online, only for speech addressed to Jarvis: {detail}", fix))
    return checks


def check_gpu(voice_installed: bool) -> Check:
    if not voice_installed:
        return Check("Speech GPU", SKIP, "voice extra not installed")
    if cuda_devices() > 0:
        return Check("Speech GPU", PASS, "CUDA device available to speech recognition")
    if nvidia_smi_present():
        return Check("Speech GPU", WARN, "an NVIDIA GPU exists but speech recognition cannot use it", "install the gpu extra: uv tool install --reinstall 'jarvis[all,gpu]'")
    return Check("Speech GPU", PASS, "no NVIDIA GPU: speech recognition runs on the CPU")


def check_extension(settings: Settings) -> Check:
    from jarvis.browser.bridge import extension_dir, token_path

    folder = extension_dir()
    if token_path(settings).is_file():
        return Check("Chrome extension", PASS, f"pairing token exists; extension folder {folder}")
    return Check(
        "Chrome extension", WARN, "not paired yet (no token file)",
        f"run: jarvis --browser-token, then load {folder} at chrome://extensions",
    )


def check_helpers() -> list[Check]:
    checks = []
    for name, present, fix in plat.current().helper_report():
        checks.append(Check(f"Helper: {name}", PASS if present else WARN, "found" if present else "missing", "" if present else fix))
    return checks


def run_checks(settings: Settings, *, offline: bool = False) -> list[Check]:
    checks = [check_python(), check_platform(), check_config()]
    key = check_key(settings)
    checks.append(key)
    ollama, local_ok = check_ollama(settings)
    checks.append(check_hosted(settings, offline, key.status == PASS, local_ok))
    checks.append(ollama)
    extras = check_extras()
    checks.extend(extras)
    voice_installed = next(c for c in extras if c.name == "Extra: voice").status == PASS
    checks.append(check_audio(voice_installed))
    checks.append(check_speech_models(settings, voice_installed))
    checks.extend(check_speech_engines(settings, voice_installed, offline))
    checks.append(check_gpu(voice_installed))
    checks.append(check_extension(settings))
    checks.extend(check_helpers())
    return checks


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.required and c.status == FAIL for c in checks) else 0


_LABELS = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}
_STYLES = {PASS: "green", WARN: "yellow", FAIL: "bold red", SKIP: "dim"}


def render(checks: list[Check], console: Any) -> None:
    from rich.text import Text

    width = max(len(c.name) for c in checks)
    for check in checks:
        line = Text()
        line.append(f"{_LABELS[check.status]}  ", style=_STYLES[check.status])
        line.append(f"{check.name:<{width}}  ")
        line.append(check.detail, style="dim")
        console.print(line, soft_wrap=True)
        if check.fix and check.status in (WARN, FAIL):
            console.print(Text(f"{'':6}{'':<{width}}  fix: {check.fix}", style="cyan"), soft_wrap=True)
    failed = sum(c.status == FAIL for c in checks)
    warned = sum(c.status == WARN for c in checks)
    summary = "Everything needed for text mode works." if exit_code(checks) == 0 else "Something needed for text mode is broken: fix the FAIL lines."
    console.print(Text(f"\n{summary} ({failed} failed, {warned} warnings)", style="bold"))


def run_doctor(settings: Settings, console: Any, *, offline: bool = False) -> int:
    checks = run_checks(settings, offline=offline)
    render(checks, console)
    return exit_code(checks)


__all__ = ["Check", "run_checks", "run_doctor", "exit_code", "render", "PASS", "WARN", "FAIL", "SKIP"]
