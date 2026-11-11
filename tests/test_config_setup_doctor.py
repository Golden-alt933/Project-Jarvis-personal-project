"""Config lookup, --config-path, --version, --setup and --doctor. Everything outside the
process (network, devices, installed programs) is mocked."""

from __future__ import annotations

import os
import re
import stat
import sys
import types
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from jarvis import config, doctor, setup_wizard
from jarvis import platform as plat
from jarvis.config import Settings
from jarvis.interfaces import cli

SECRET = "nvapi-SECRET-KEY-1234567890"


def console() -> Console:
    return Console(record=True, width=200, color_system=None)


# ===================================================================== config lookup
@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A fake repo checkout with its own .env, plus a separate per-user config file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    user = tmp_path / "cfg" / ".env"
    user.parent.mkdir()
    monkeypatch.setattr(config, "REPO_ROOT", repo)
    monkeypatch.setenv("JARVIS_CONFIG_FILE", str(user))
    for var in ("NVIDIA_API_KEY", "JARVIS_USER_NAME", "JARVIS_PROVIDER", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return repo, user


def load() -> Settings:
    return Settings(_env_file=config._env_file_chain())


def test_lookup_order_env_then_repo_then_user(checkout, monkeypatch):
    repo, user = checkout
    user.write_text("NVIDIA_API_KEY=from-user\nJARVIS_USER_NAME=UserName\nGROQ_API_KEY=user-groq\n")
    (repo / ".env").write_text("NVIDIA_API_KEY=from-repo\nJARVIS_USER_NAME=RepoName\n")
    s = load()
    assert s.nvidia_api_key == "from-repo" and s.user_name == "RepoName"  # repo .env beats the user file
    assert s.groq_api_key == "user-groq"  # a key only the user file has still applies
    monkeypatch.setenv("NVIDIA_API_KEY", "from-env")
    assert load().nvidia_api_key == "from-env"  # the environment beats both


def test_user_file_alone_is_used_when_installed(checkout, monkeypatch):
    repo, user = checkout
    (repo / "pyproject.toml").unlink()  # not a checkout any more: an installed wheel
    (repo / ".env").write_text("NVIDIA_API_KEY=stray-file-in-site-packages\n")
    user.write_text("NVIDIA_API_KEY=from-user\n")
    assert not config.is_checkout() and config.repo_env_file() is None
    assert config._env_file_chain() == (user,)
    assert load().nvidia_api_key == "from-user"
    assert config.config_files() == [user]


def test_config_files_priority_and_existence(checkout):
    repo, user = checkout
    assert config.config_files() == []
    user.write_text("A=1\n")
    assert config.config_files() == [user]
    (repo / ".env").write_text("A=2\n")
    assert config.config_files() == [repo / ".env", user]


def test_default_user_config_location(monkeypatch, tmp_path):
    monkeypatch.delenv("JARVIS_CONFIG_FILE")
    path = config.user_config_file()
    assert path.name == ".env"
    if sys.platform == "win32":
        assert path.parts[-2:] == ("Jarvis", ".env")
        assert Path(os.environ["APPDATA"]) in path.parents  # %APPDATA%\Jarvis\.env
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert config.user_config_file() == tmp_path / "jarvis" / ".env"
        monkeypatch.delenv("XDG_CONFIG_HOME")
        assert config.user_config_file() == Path.home() / ".config" / "jarvis" / ".env"


def test_app_dir_name_follows_the_platform():
    plat.set_backend(plat.create("win32"))
    assert config.app_dir_name() == "Jarvis"
    plat.set_backend(plat.create("linux"))
    assert config.app_dir_name() == "jarvis"


def test_data_dir_follows_override_and_platform_name(tmp_path):
    assert Settings(data_dir_override=str(tmp_path / "d")).data_dir == tmp_path / "d"
    assert (tmp_path / "d").is_dir()
    plat.set_backend(plat.create("linux"))
    assert Settings().data_dir.name in ("jarvis", "Jarvis")  # platformdirs decides the root; the leaf is ours


# ===================================================================== version, config-path
def test_version_flag(capsys):
    assert cli.main(["--version"]) == 0
    assert re.fullmatch(r"jarvis \d+\.\d+\.\d+\S*", capsys.readouterr().out.strip())


def test_version_without_metadata(monkeypatch, capsys):
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    assert cli.main(["--version"]) == 0
    assert "unknown" in capsys.readouterr().out


def test_config_path_prints_the_file_in_use(checkout, capsys):
    repo, user = checkout
    assert cli.main(["--config-path"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == str(user) and "no config file yet" in out.err
    user.write_text("A=1\n")
    assert cli.main(["--config-path"]) == 0
    assert capsys.readouterr().out.strip() == str(user)
    (repo / ".env").write_text("A=2\n")
    assert cli.main(["--config-path"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == str(repo / ".env") and str(user) in out.err


def test_missing_key_message_points_to_setup():
    text = cli.config_error_message("No API key for provider 'nvidia'")
    assert "jarvis --setup" in text and "jarvis --local" in text and str(config.user_config_file()) in text


def test_new_flags_are_parsed():
    args = cli.build_parser().parse_args(["--doctor", "--offline"])
    assert args.doctor and args.offline and not args.setup
    assert cli.build_parser().parse_args(["--setup"]).setup


# ===================================================================== setup wizard
class Script:
    """Scripted answers; records every prompt text it was shown."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)


def run(tmp_path, ask, secret, **kw):
    out = console()
    seen: list[Settings] = []
    path = tmp_path / "cfg" / "jarvis" / ".env"
    code = setup_wizard.run_setup(
        out, ask=ask, ask_secret=secret, interactive=True, config_path=path,
        doctor=lambda s: seen.append(s) or 0, **kw,
    )
    return code, path, out.export_text(), seen


def test_setup_writes_key_name_and_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    ask, secret = Script("", "Tony"), Script(SECRET)
    code, path, shown, seen = run(tmp_path, ask, secret)
    assert code == 0 and len(seen) == 1
    text = path.read_text(encoding="utf-8")
    assert f"NVIDIA_API_KEY={SECRET}" in text and "JARVIS_PROVIDER=nvidia" in text and "JARVIS_USER_NAME=Tony" in text
    assert "build.nvidia.com" in shown  # tells the user where a free key comes from
    assert SECRET not in shown  # never echoed
    assert all(SECRET not in p for p in ask.prompts + secret.prompts)
    assert "hidden" in secret.prompts[0]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits; Windows relies on the private profile folder")
def test_setup_file_is_user_only(tmp_path):
    ask, secret = Script("", ""), Script(SECRET)
    _, path, _, _ = run(tmp_path, ask, secret)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_setup_keeps_other_lines_and_replaces_in_place(tmp_path):
    path = tmp_path / "cfg" / "jarvis" / ".env"
    path.parent.mkdir(parents=True)
    path.write_text("# my notes\nOLLAMA_MODEL=qwen3:4b\nNVIDIA_API_KEY=old\nJARVIS_PROVIDER=groq\n")
    ask, secret = Script("", "Pepper Potts"), Script("new-key")
    setup_wizard.run_setup(console(), ask=ask, ask_secret=secret, interactive=True, config_path=path, doctor=lambda s: 0)
    lines = path.read_text().splitlines()
    assert lines[:2] == ["# my notes", "OLLAMA_MODEL=qwen3:4b"]
    assert "NVIDIA_API_KEY=new-key" in lines and "NVIDIA_API_KEY=old" not in lines
    assert "JARVIS_PROVIDER=nvidia" in lines and lines.count("JARVIS_PROVIDER=nvidia") == 1
    assert 'JARVIS_USER_NAME="Pepper Potts"' in lines


def test_setup_enter_keeps_an_existing_key(tmp_path):
    path = tmp_path / "cfg" / "jarvis" / ".env"
    path.parent.mkdir(parents=True)
    path.write_text("NVIDIA_API_KEY=keepme\n")
    secret = Script("")
    code = setup_wizard.run_setup(
        console(), ask=Script("", ""), ask_secret=secret, interactive=True, config_path=path, doctor=lambda s: 0
    )
    assert code == 0 and "NVIDIA_API_KEY=keepme" in path.read_text()
    assert "already saved" in secret.prompts[0]


def test_setup_without_a_key_changes_nothing(tmp_path):
    code, path, shown, seen = run(tmp_path, Script(""), Script(""))
    assert code == 2 and not path.exists() and seen == [] and "nothing was changed" in shown


def test_setup_ollama_needs_no_key(tmp_path):
    secret = Script()
    code, path, _, seen = run(tmp_path, Script("6", ""), secret)
    assert code == 0 and "JARVIS_PROVIDER=ollama" in path.read_text() and secret.prompts == []


def test_setup_custom_endpoint_needs_a_url(tmp_path):
    code, path, _, _ = run(tmp_path, Script("5", "https://llm.example.com/v1", ""), Script("k-1"))
    text = path.read_text()
    assert code == 0 and "JARVIS_PROVIDER=custom" in text and "CUSTOM_BASE_URL=https://llm.example.com/v1" in text
    assert "CUSTOM_API_KEY=k-1" in text
    code, path2, shown, _ = run(tmp_path / "other", Script("5", ""), Script("k"))
    assert code == 2 and not path2.exists()


def test_setup_reprompts_on_a_bad_choice_and_accepts_names(tmp_path):
    ask = Script("9", "banana", "groq", "")
    code, path, shown, _ = run(tmp_path, ask, Script("gk"))
    assert code == 0 and "GROQ_API_KEY=gk" in path.read_text() and shown.count("Please enter a number") == 2


def test_setup_rejects_keys_that_cannot_be_stored(tmp_path):
    code, path, shown, _ = run(tmp_path, Script("", ""), Script('bad"key'))
    assert code == 2 and not path.exists() and 'bad"key' not in shown


def test_setup_skips_an_unstorable_name(tmp_path):
    code, path, shown, _ = run(tmp_path, Script("", 'Tony "Iron" Stark'), Script("k"))
    assert code == 0 and "JARVIS_USER_NAME" not in path.read_text() and "Skipped the name" in shown


def test_setup_not_a_terminal_prints_instructions(tmp_path):
    out = console()
    path = tmp_path / "x" / ".env"
    code = setup_wizard.run_setup(out, interactive=False, config_path=path, ask=lambda p: pytest.fail("asked"))
    text = out.export_text()
    assert code == 0 and not path.exists()
    assert str(path) in text and "NVIDIA_API_KEY=" in text and "build.nvidia.com" in text


def test_setup_through_the_cli_when_stdin_is_not_a_tty(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("JARVIS_CONFIG_FILE", str(tmp_path / "c" / ".env"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli.main(["--setup"]) == 0
    assert "NVIDIA_API_KEY" in capsys.readouterr().out
    assert not (tmp_path / "c").exists()


def test_setup_runs_the_doctor_with_fresh_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_CONFIG_FILE", str(tmp_path / "c" / ".env"))
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr(doctor, "run_doctor", lambda settings, con, offline=False: calls.append((settings, offline)) or 0)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", Script("", ""))
    monkeypatch.setattr(setup_wizard.getpass, "getpass", Script(SECRET))
    assert cli.main(["--setup", "--offline"]) == 0
    assert calls and calls[0][1] is True


def test_update_env_lines_edge_cases():
    assert setup_wizard.update_env_lines([], {"A": "1"}) == ["A=1"]
    assert setup_wizard.update_env_lines(["# A=old", "A=old"], {"A": "new"}) == ["# A=old", "A=new"]
    assert setup_wizard.update_env_lines(["A = spaced"], {"A": "x y"}) == ['A="x y"']
    assert setup_wizard.env_value(["A=\"quoted\""], "A") == "quoted"
    assert setup_wizard.env_value([], "A") == ""


# ===================================================================== doctor
class World:
    """Everything the doctor looks at, defaulting to a healthy machine."""

    def __init__(self) -> None:
        self.hosted = (doctor.PASS, "nvidia answered")
        self.hosted_calls = 0
        self.ollama_exe: str | None = "/usr/bin/ollama"
        self.ollama = (True, ["qwen3:4b-instruct"])
        self.modules: set[str] | None = None  # None = everything present
        self.audio: tuple[int, int] | str = (1, 2)
        self.whisper = ["faster-whisper-base.en"]
        self.cuda = 0
        self.smi = False
        self.tts = (doctor.PASS, "edge en-GB-RyanNeural answered", "")
        self.stt = (doctor.PASS, "nvidia parakeet-ctc-1.1b answered")


@pytest.fixture
def world(monkeypatch, tmp_path, fake_platform):
    w = World()

    def hosted(settings):
        w.hosted_calls += 1
        return w.hosted

    monkeypatch.setattr(doctor, "probe_hosted", hosted)
    monkeypatch.setattr(doctor, "find_ollama", lambda: w.ollama_exe)
    monkeypatch.setattr(doctor, "probe_ollama", lambda s: w.ollama)
    monkeypatch.setattr(doctor, "probe_online_tts", lambda s: w.tts)
    monkeypatch.setattr(doctor, "probe_online_stt", lambda s: w.stt)
    monkeypatch.setattr(doctor, "module_present", lambda m: w.modules is None or m in w.modules)
    monkeypatch.setattr(doctor, "probe_audio", lambda: w.audio)
    monkeypatch.setattr(doctor, "whisper_models_cached", lambda: w.whisper)
    monkeypatch.setattr(doctor, "cuda_devices", lambda: w.cuda)
    monkeypatch.setattr(doctor, "nvidia_smi_present", lambda: w.smi)
    monkeypatch.setenv("JARVIS_CONFIG_FILE", str(tmp_path / "cfg" / ".env"))
    fake_platform.helper_rows = [("xdg-open", True, ""), ("wmctrl", False, "sudo apt install wmctrl")]
    fake_platform.helper_report = lambda: fake_platform.helper_rows
    return w


def settings(tmp_path, **kw) -> Settings:
    base = dict(provider="nvidia", nvidia_api_key=SECRET, data_dir_override=str(tmp_path), ollama_model="qwen3:4b-instruct")
    base.update(kw)
    return Settings(_env_file=None, **base)


def by_name(checks):
    return {c.name: c for c in checks}


def test_doctor_reports_which_speech_engines_are_active(world, tmp_path):
    c = by_name(doctor.run_checks(settings(tmp_path, tts_provider="edge", stt_provider="nvidia")))
    assert c["Speech synthesis"].status == doctor.PASS and "edge en-GB-RyanNeural" in c["Speech synthesis"].detail
    assert "piper" in c["Speech synthesis"].detail  # the local fallback is shown
    assert c["Speech recognition"].status == doctor.PASS and "only for speech addressed to Jarvis" in c["Speech recognition"].detail
    world.tts = (doctor.WARN, "not answering: edge (OSError)", "")
    world.stt = (doctor.WARN, "nvidia parakeet-ctc-1.1b did not answer (TimeoutError)")
    c = by_name(doctor.run_checks(settings(tmp_path, tts_provider="edge", stt_provider="nvidia")))
    assert c["Speech synthesis"].status == doctor.WARN and "Piper" in c["Speech synthesis"].fix
    assert c["Speech recognition"].status == doctor.WARN and "local" in c["Speech recognition"].fix


def test_doctor_speech_probes_are_skipped_offline_and_local_mode_is_local(world, tmp_path):
    world.tts = world.stt = None  # would blow up if probed
    c = by_name(doctor.run_checks(settings(tmp_path, tts_provider="edge", stt_provider="nvidia"), offline=True))
    assert c["Speech synthesis"].status == doctor.SKIP and "--offline" in c["Speech synthesis"].detail
    assert c["Speech recognition"].status == doctor.SKIP and "--offline" in c["Speech recognition"].detail
    c = by_name(doctor.run_checks(settings(tmp_path, provider="ollama")))
    assert c["Speech synthesis"].detail.startswith("local: piper")
    assert c["Speech recognition"].detail.startswith("local:")


def test_healthy_machine_exits_zero_and_never_prints_the_key(world, tmp_path):
    checks = doctor.run_checks(settings(tmp_path))
    out = console()
    doctor.render(checks, out)
    text = out.export_text()
    assert doctor.exit_code(checks) == 0
    assert SECRET not in text and "NVIDIA_API_KEY is set" in text
    c = by_name(checks)
    assert c["Hosted model"].status == doctor.PASS and c["Ollama"].status == doctor.PASS
    assert c["Helper: xdg-open"].status == doctor.PASS
    assert c["Helper: wmctrl"].status == doctor.WARN and c["Helper: wmctrl"].fix == "sudo apt install wmctrl"
    assert "fix: sudo apt install wmctrl" in text
    assert "Extra: voice" in c and c["Extra: hud"].status == doctor.PASS


def test_offline_skips_the_hosted_probe(world, tmp_path):
    checks = doctor.run_checks(settings(tmp_path), offline=True)
    assert world.hosted_calls == 0
    assert by_name(checks)["Hosted model"].status == doctor.SKIP and "--offline" in by_name(checks)["Hosted model"].detail
    assert doctor.exit_code(checks) == 0


def test_missing_key_fails_with_the_setup_fix(world, tmp_path):
    checks = doctor.run_checks(settings(tmp_path, nvidia_api_key=""))
    key = by_name(checks)["API key"]
    assert key.status == doctor.FAIL and "jarvis --setup" in key.fix and "build.nvidia.com" in key.fix
    assert doctor.exit_code(checks) == 1
    assert world.hosted_calls == 0  # nothing to try without a key


def test_ollama_provider_needs_no_key_but_needs_ollama(world, tmp_path):
    cfg = settings(tmp_path, provider="ollama", nvidia_api_key="")
    checks = by_name(doctor.run_checks(cfg))
    assert checks["API key"].status == doctor.SKIP and checks["Hosted model"].status == doctor.SKIP
    assert doctor.exit_code(list(checks.values())) == 0
    world.ollama_exe = None
    checks = doctor.run_checks(cfg)
    assert by_name(checks)["Ollama"].status == doctor.FAIL and doctor.exit_code(checks) == 1
    world.ollama_exe, world.ollama = "/usr/bin/ollama", (False, [])
    assert "ollama serve" in by_name(doctor.run_checks(cfg))["Ollama"].fix
    world.ollama = (True, ["other:7b"])
    c = by_name(doctor.run_checks(cfg))["Ollama"]
    assert c.status == doctor.FAIL and c.fix == "run: ollama pull qwen3:4b-instruct"


def test_ollama_is_optional_with_a_cloud_provider(world, tmp_path):
    world.ollama_exe = None
    checks = doctor.run_checks(settings(tmp_path))
    assert by_name(checks)["Ollama"].status == doctor.WARN and doctor.exit_code(checks) == 0


@pytest.mark.parametrize(
    "status, expected_exit", [(doctor.FAIL, 1), (doctor.WARN, 0)]
)
def test_unreachable_hosted_model_without_a_local_fallback(world, tmp_path, status, expected_exit):
    world.ollama = (False, [])  # no local model to fall back on
    world.hosted = (status, "could not connect")
    checks = doctor.run_checks(settings(tmp_path))
    hosted = by_name(checks)["Hosted model"]
    assert hosted.status == status and hosted.fix and "--local" in hosted.fix
    assert doctor.exit_code(checks) == expected_exit  # only a hard failure is fatal; a 429 is not


def test_hosted_failure_is_a_warning_when_ollama_works(world, tmp_path):
    world.hosted = (doctor.FAIL, "no answer within 15 s")
    world.ollama = (True, ["qwen3:4b-instruct"])
    checks = doctor.run_checks(settings(tmp_path))
    assert by_name(checks)["Hosted model"].status == doctor.WARN and doctor.exit_code(checks) == 0
    world.ollama = (False, [])
    checks = doctor.run_checks(settings(tmp_path))
    assert by_name(checks)["Hosted model"].status == doctor.FAIL and doctor.exit_code(checks) == 1


def test_old_python_is_fatal(world, tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "MIN_PYTHON", (3, 99))
    checks = doctor.run_checks(settings(tmp_path))
    py = by_name(checks)["Python"]
    assert py.status == doctor.FAIL and "3.13" in py.fix and doctor.exit_code(checks) == 1


def test_config_file_check(world, tmp_path, monkeypatch):
    # A developer checkout usually has a real .env; this test is about the per-user file.
    monkeypatch.setattr("jarvis.config.repo_env_file", lambda: None)
    c = by_name(doctor.run_checks(settings(tmp_path)))["Config file"]
    assert c.status == doctor.WARN and "jarvis --setup" in c.fix
    cfg = Path(os.environ["JARVIS_CONFIG_FILE"])
    cfg.parent.mkdir(parents=True)
    cfg.write_text("A=1\n")
    c = by_name(doctor.run_checks(settings(tmp_path)))["Config file"]
    assert c.status == doctor.PASS and str(cfg) in c.detail


def test_missing_extras_are_warnings_with_a_reinstall_line(world, tmp_path):
    world.modules = {"openpyxl", "pypdf", "docx", "websockets"}
    checks = by_name(doctor.run_checks(settings(tmp_path)))
    assert checks["Extra: office"].status == doctor.PASS
    for extra in ("screen", "voice", "hud"):
        c = checks[f"Extra: {extra}"]
        assert c.status == doctor.WARN and f"jarvis[{extra}]" in c.fix
    # without the voice extra the voice checks are skipped, not failed
    for name in ("Audio devices", "Speech models", "Speech GPU"):
        assert checks[name].status == doctor.SKIP
    assert doctor.exit_code(list(checks.values())) == 0


def test_audio_devices(world, tmp_path):
    assert by_name(doctor.run_checks(settings(tmp_path)))["Audio devices"].status == doctor.PASS
    world.audio = (0, 2)
    c = by_name(doctor.run_checks(settings(tmp_path)))["Audio devices"]
    assert c.status == doctor.WARN and "no microphone" in c.detail
    world.audio = (1, 0)
    assert "no speaker" in by_name(doctor.run_checks(settings(tmp_path)))["Audio devices"].detail
    world.audio = "PortAudio library not found (libportaudio.so.2)"
    c = by_name(doctor.run_checks(settings(tmp_path)))["Audio devices"]
    assert c.status == doctor.WARN and "PortAudio" in c.detail and c.fix


def test_audio_missing_library_uses_the_platform_hint(world, tmp_path, fake_platform):
    world.audio = "PortAudio library not found"
    fake_platform.helper_report = lambda: [("libportaudio", False, "sudo apt install libportaudio2")]
    assert by_name(doctor.run_checks(settings(tmp_path)))["Audio devices"].fix == "sudo apt install libportaudio2"


def test_speech_models(world, tmp_path):
    cfg = settings(tmp_path)
    c = by_name(doctor.run_checks(cfg))["Speech models"]
    assert c.status == doctor.WARN and "voice en_GB-alan-medium" in c.detail and "recognition" not in c.detail
    (cfg.voices_dir / f"{cfg.tts_voice}.onnx").write_text("x")
    assert by_name(doctor.run_checks(cfg))["Speech models"].status == doctor.PASS
    world.whisper = []
    assert "recognition model" in by_name(doctor.run_checks(cfg))["Speech models"].detail


def test_speech_gpu(world, tmp_path):
    assert "CPU" in by_name(doctor.run_checks(settings(tmp_path)))["Speech GPU"].detail
    world.smi = True
    c = by_name(doctor.run_checks(settings(tmp_path)))["Speech GPU"]
    assert c.status == doctor.WARN and "gpu" in c.fix
    world.cuda = 1
    assert by_name(doctor.run_checks(settings(tmp_path)))["Speech GPU"].status == doctor.PASS


def test_chrome_extension_pairing(world, tmp_path):
    from jarvis.browser.bridge import token_path

    cfg = settings(tmp_path)
    c = by_name(doctor.run_checks(cfg))["Chrome extension"]
    assert c.status == doctor.WARN and "--browser-token" in c.fix and "chrome://extensions" in c.fix
    token_path(cfg).write_text("t")
    assert by_name(doctor.run_checks(cfg))["Chrome extension"].status == doctor.PASS


def test_run_doctor_returns_the_exit_code_and_prints(world, tmp_path):
    out = console()
    assert doctor.run_doctor(settings(tmp_path, nvidia_api_key=""), out) == 1
    shown = out.export_text()
    assert "FAIL" in shown and "Something needed for text mode is broken" in shown
    assert doctor.run_doctor(settings(tmp_path), out, offline=True) == 0


def test_doctor_through_the_cli(world, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_settings", lambda: settings(tmp_path))
    assert cli.main(["--doctor", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and SECRET not in out
    monkeypatch.setattr(cli, "get_settings", lambda: settings(tmp_path, nvidia_api_key=""))
    assert cli.main(["--doctor", "--offline"]) == 1


# -- the real probes, with the network / devices mocked at the lowest level
def response(code: int) -> httpx.Response:
    return httpx.Response(code, request=httpx.Request("POST", "https://x"))


@pytest.mark.parametrize(
    "code, status, text",
    [(200, doctor.PASS, "answered"), (401, doctor.FAIL, "rejected"), (403, doctor.FAIL, "rejected"),
     (404, doctor.FAIL, "not found"), (429, doctor.WARN, "rate limited"), (500, doctor.FAIL, "HTTP 500")],
)
def test_probe_hosted_status_mapping(monkeypatch, tmp_path, code, status, text):
    seen = {}

    def post(url, **kw):
        seen.update(url=url, **kw)
        return response(code)

    monkeypatch.setattr(doctor.httpx, "post", post)
    got_status, detail = doctor.probe_hosted(settings(tmp_path))
    assert got_status == status and text in detail and SECRET not in detail
    assert seen["timeout"] == 15.0 and seen["json"]["max_tokens"] == 1
    assert seen["url"].endswith("/chat/completions") and seen["headers"]["Authorization"] == f"Bearer {SECRET}"


def test_probe_hosted_network_errors(monkeypatch, tmp_path):
    def slow(url, **kw):
        raise httpx.ReadTimeout("t")

    def down(url, **kw):
        raise httpx.ConnectError("c")

    monkeypatch.setattr(doctor.httpx, "post", slow)
    assert doctor.probe_hosted(settings(tmp_path))[0] == doctor.FAIL and "15" in doctor.probe_hosted(settings(tmp_path))[1]
    monkeypatch.setattr(doctor.httpx, "post", down)
    assert "could not connect" in doctor.probe_hosted(settings(tmp_path))[1]
    assert doctor.probe_hosted(settings(tmp_path, provider="groq", groq_api_key="g"))[0] == doctor.WARN  # no default model


def test_probe_ollama(monkeypatch, tmp_path):
    body = httpx.Response(200, json={"models": [{"name": "a:1"}, {"name": "b:2"}]}, request=httpx.Request("GET", "http://x"))
    monkeypatch.setattr(doctor.httpx, "get", lambda url, timeout: body)
    assert doctor.probe_ollama(settings(tmp_path)) == (True, ["a:1", "b:2"])
    monkeypatch.setattr(doctor.httpx, "get", lambda url, timeout: response(500))
    assert doctor.probe_ollama(settings(tmp_path)) == (False, [])

    def refuse(url, timeout):
        raise httpx.ConnectError("no")

    monkeypatch.setattr(doctor.httpx, "get", refuse)
    assert doctor.probe_ollama(settings(tmp_path)) == (False, [])


def test_probe_audio_counts_devices(monkeypatch):
    from jarvis.voice import audio

    fake = types.SimpleNamespace(query_devices=lambda: [
        {"max_input_channels": 2, "max_output_channels": 0},
        {"max_input_channels": 0, "max_output_channels": 2},
        {"max_input_channels": 1, "max_output_channels": 1},
    ])
    monkeypatch.setattr(audio, "sd", fake)
    assert doctor.probe_audio() == (2, 2)

    def boom():
        raise RuntimeError("no host api")

    fake.query_devices = boom
    assert "could not query" in doctor.probe_audio()


@pytest.mark.parametrize(
    "error, expected",
    [
        (ImportError("No module named 'sounddevice'"), "not installed"),
        (OSError("PortAudio library not found"), "PortAudio library not found"),
        (RuntimeError("PulseAudio_Initialize: Can't connect to server"), "Can't connect to server"),
    ],
)
def test_probe_audio_reports_why_audio_is_unavailable(monkeypatch, error, expected):
    """A missing library or sound server must be a doctor line, never a traceback."""
    from jarvis.voice import audio

    monkeypatch.setattr(audio, "sd", None)
    monkeypatch.setattr(audio, "_SD_ERROR", error)
    text = doctor.probe_audio()
    assert isinstance(text, str) and expected in text
    if not isinstance(error, ImportError):
        assert "libportaudio2" in text and "sound server" in text
    assert "not installed" in audio.list_devices() or "could not start" in audio.list_devices()


def test_whisper_models_cached(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    (hub / "models--Systran--faster-whisper-tiny.en").mkdir(parents=True)
    (hub / "models--Systran--faster-whisper-small").mkdir()
    (hub / "models--other--thing").mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    assert doctor.whisper_models_cached() == ["faster-whisper-small", "faster-whisper-tiny.en"]
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "nope"))
    assert doctor.whisper_models_cached() == []


def test_module_present():
    assert doctor.module_present("json") and not doctor.module_present("definitely_not_a_module_xyz")
