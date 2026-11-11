"""Places where other modules meet the platform layer: prompt, extension folder, known
folders, capture helpers, GPU libraries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from jarvis import platform as plat
from jarvis.agent import prompt
from jarvis.browser import bridge
from jarvis.config import Settings


@pytest.mark.parametrize(
    "name, os_word, shell",
    [("win32", "Windows", "PowerShell"), ("linux", "Linux", "bash")],
)
def test_prompt_names_the_real_os_and_shell(name, os_word, shell):
    plat.set_backend(plat.create(name))
    text = prompt.build_system_prompt(
        settings=Settings(_env_file=None), facts=[], summary=None, output_mode="text", is_local=False,
        now=datetime(2026, 11, 11, 12, 0),
    )
    assert f"running on the user's {os_word} computer" in text
    assert f"Commands run in {shell}." in text
    other = "Linux" if os_word == "Windows" else "Windows"
    assert f"user's {other}" not in text


def test_prompt_for_an_unsupported_os_does_not_claim_windows():
    plat.set_backend(plat.create("darwin"))
    text = prompt.build_system_prompt(
        settings=Settings(_env_file=None), facts=[], summary=None, output_mode="text", is_local=False,
    )
    assert "Windows" not in text.split("The machine:")[0]
    assert "running on the user's desktop computer" in text


def test_extension_dir_prefers_the_checkout_then_the_packaged_copy(tmp_path, monkeypatch):
    checkout, packaged = tmp_path / "checkout", tmp_path / "packaged"
    monkeypatch.setattr(bridge, "CHECKOUT_EXTENSION_DIR", checkout)
    monkeypatch.setattr(bridge, "PACKAGED_EXTENSION_DIR", packaged)
    assert bridge.extension_dir() == checkout  # neither exists: the documented default
    packaged.mkdir()
    (packaged / "manifest.json").write_text("{}")
    assert bridge.extension_dir() == packaged  # an installed wheel
    checkout.mkdir()
    (checkout / "manifest.json").write_text("{}")
    assert bridge.extension_dir() == checkout  # a checkout wins


def test_the_real_extension_folder_is_found_in_a_checkout():
    folder = bridge.extension_dir()
    assert (folder / "manifest.json").is_file() and (folder / "service-worker.js").is_file()


def test_known_folder_goes_through_the_platform(fake_platform, monkeypatch, tmp_path):
    from jarvis.tools import paths

    monkeypatch.setattr(fake_platform, "known_folder", lambda name: tmp_path / "X" / name)
    assert paths.known_folder("Desktop") == tmp_path / "X" / "Desktop"

    def broken(name):
        raise RuntimeError("shell unavailable")

    monkeypatch.setattr(fake_platform, "known_folder", broken)
    assert paths.known_folder("Documents") == Path.home() / "Documents"


def test_reserved_device_names_depend_on_the_platform(fake_platform):
    from jarvis.tools import paths

    paths.resolve_path("NUL")  # no error: this backend has no device names
    fake_platform.reserved_device_names = True
    with pytest.raises(ValueError, match="reserved"):
        paths.resolve_path("NUL")


def test_protected_paths_come_from_the_platform(fake_platform, tmp_path):
    from jarvis.tools.builtin import files

    target = tmp_path / "precious"
    target.mkdir()
    assert files._protected_reason(target) is None
    import os

    fake_platform.protected[os.path.normcase(os.path.realpath(target))] = "it is inside the system folder X"
    assert files._protected_reason(target) == "it is inside the system folder X"


def test_capture_helpers_use_the_platform(fake_platform, monkeypatch):
    from jarvis.vision import capture

    monkeypatch.setattr(fake_platform, "foreground_window_title", lambda: "Terminal")
    monkeypatch.setattr(fake_platform, "cursor_position", lambda: (5, 6))
    assert capture.capture_window_title() == "Terminal"
    assert capture._cursor_position() == (5, 6)

    def boom():
        raise RuntimeError("x")

    monkeypatch.setattr(fake_platform, "foreground_window_title", boom)
    assert capture.capture_window_title() == ""


def test_capture_screen_prefers_an_os_tool_image(fake_platform, monkeypatch):
    pytest.importorskip("mss")
    PIL = pytest.importorskip("PIL.Image")
    from jarvis.vision import capture

    monkeypatch.setattr(fake_platform, "screen_grab", lambda: PIL.new("RGB", (40, 30), (9, 9, 9)))
    assert capture.capture_screen().size == (40, 30)


def test_gpu_library_preparation_never_raises(monkeypatch, fake_platform):
    from jarvis.voice import stt

    def boom():
        raise OSError("cannot load")

    monkeypatch.setattr(fake_platform, "prepare_gpu_libs", boom)
    stt._add_nvidia_dll_dirs()  # logged, not raised: the CPU path still works


def test_ollama_failure_message_names_where_it_looked(fake_platform, monkeypatch):
    import asyncio

    from jarvis.llm import ollama as ollama_mod

    monkeypatch.setattr(ollama_mod, "_autostart", ollama_mod._AutostartState())
    monkeypatch.setattr(ollama_mod, "find_ollama_exe", lambda: None)
    assert asyncio.run(ollama_mod.ensure_ollama_running("http://localhost:11434")) is False
    assert ollama_mod._autostart.failure == "ollama was not found on PATH"
    plat.set_backend(plat.create("linux"))
    monkeypatch.setattr(ollama_mod, "_autostart", ollama_mod._AutostartState())
    assert asyncio.run(ollama_mod.ensure_ollama_running("http://localhost:11434")) is False
    assert str(Path("/usr/local/bin")) in ollama_mod._autostart.failure


@pytest.mark.skipif(__import__("shutil").which("uv") is None, reason="uv is needed to build the wheel")
def test_built_wheel_ships_the_extension_and_finds_it(tmp_path):
    """Build the wheel, unpack it somewhere that is not a checkout, and ask where the extension is."""
    import subprocess
    import sys
    import zipfile

    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "dist"
    built = subprocess.run(["uv", "build", "--wheel", "-o", str(out)], cwd=root, capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next(out.glob("jarvis-*.whl"))
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as z:
        z.extractall(site)
        names = z.namelist()
    for needed in ("jarvis/browser/extension/manifest.json", "jarvis/browser/extension/service-worker.js", "jarvis/platform/linux.py"):
        assert needed in names, needed
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from jarvis.browser import bridge;"
        "from jarvis import config;"
        "print(bridge.extension_dir()); print(config.is_checkout())"
    )
    done = subprocess.run([sys.executable, "-c", probe, str(site)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    folder, checkout = done.stdout.split()
    assert Path(folder) == site / "jarvis" / "browser" / "extension" and checkout == "False"
