"""Suite-wide test settings."""

import os
import tempfile
from pathlib import Path

import pytest

from jarvis.platform.base import BasePlatform

# Auto-starting Ollama is a real side effect (a server process that outlives the run).
# Tests that exercise the auto-start logic pass it explicitly with the launch mocked.
os.environ["JARVIS_OLLAMA_AUTOSTART"] = "false"

# Settings() must not read the developer's real config: neither the checkout's .env (set
# before jarvis.config is imported, because the file list is fixed at import time) nor the
# per-user file. Either may hold a real key.
os.environ["JARVIS_IGNORE_REPO_ENV"] = "1"
os.environ["JARVIS_CONFIG_FILE"] = str(Path(tempfile.gettempdir()) / "jarvis-tests-no-such-config" / ".env")


@pytest.fixture(autouse=True)
def _fresh_platform_backend():
    """Each test starts with the real backend for this OS (tests may install fakes)."""
    from jarvis import platform as plat

    plat.set_backend(None)
    yield
    plat.set_backend(None)


@pytest.fixture(autouse=True)
def _no_libreoffice(monkeypatch):
    """Recalculation tests must not depend on whether the machine has LibreOffice."""
    from jarvis.tools import excel_calc

    monkeypatch.setattr(excel_calc, "find_soffice", lambda: None)


class FakePlatform(BasePlatform):
    """An in-memory OS backend: records what tools ask of it. Not tied to any real OS.
    Inherits the defaults for operations a test does not care about (camera, desktop control)."""

    name = "fake"
    app_dir_name = "jarvis"
    shell_kind = "bash"
    shell_name = "bash"
    reserved_device_names = False
    trash_name = "Trash"
    executable_suffixes = frozenset()

    def __init__(self) -> None:
        self.apps: list = []
        self.launched: list = []
        self.window_rows: list = []
        self.windows_error: Exception | None = None
        self.closed: list = []
        self.volume = (50, False)
        self.volume_error: Exception | None = None
        self.keys: list = []
        self.key_error: Exception | None = None
        self.locked = 0
        self.lock_error: Exception | None = None
        self.clipboard_text: str | None = None
        self.urls: list = []
        self.paths: list = []
        self.open_error: Exception | None = None
        self.protected: dict = {}

    def describe(self) -> str:
        return "Fake OS 1.0"

    def shell_argv(self, command):
        return ["bash", "-lc", command]

    def subprocess_kwargs(self):
        return {"start_new_session": True}

    def kill_tree(self, pid):
        pass

    def open_path(self, path):
        if self.open_error:
            raise self.open_error
        self.paths.append(path)

    def open_url(self, url):
        if self.open_error:
            raise self.open_error
        self.urls.append(url)
        return True

    def known_folder(self, name):
        return Path.home() / name

    def scan_apps(self):
        return list(self.apps)

    def launch_app(self, entry):
        self.launched.append(entry)

    def windows(self):
        if self.windows_error:
            raise self.windows_error
        return list(self.window_rows)

    def close_window(self, handle):
        self.closed.append(handle)

    def never_close_titles(self):
        return {"program manager"}

    def foreground_window_title(self):
        return ""

    def _volume(self):
        if self.volume_error:
            raise self.volume_error
        return self.volume

    def get_volume(self):
        return self._volume()

    def set_volume(self, percent):
        self._volume()
        self.volume = (percent, self.volume[1])
        return self.volume

    def set_mute(self, muted):
        self._volume()
        self.volume = (self.volume[0], muted)
        return self.volume

    def media_key(self, action):
        if self.key_error:
            raise self.key_error
        self.keys.append(action)

    def lock_screen(self):
        if self.lock_error:
            raise self.lock_error
        self.locked += 1

    def clipboard_problem(self):
        return self.clipboard_text

    def protected_path_reason(self, real_path):
        return self.protected.get(real_path)

    def ollama_extra_paths(self):
        return []

    def detached_popen_kwargs(self):
        return {"start_new_session": True}

    def screen_grab(self):
        return None

    def cursor_position(self):
        return None

    def prepare_capture(self):
        pass

    def has_excel_com(self):
        return False

    def prepare_gpu_libs(self):
        pass

    def helper_report(self):
        return []


@pytest.fixture
def fake_platform():
    """Install a FakePlatform for the test (the autouse fixture restores the real one)."""
    from jarvis import platform as plat

    backend = FakePlatform()
    plat.set_backend(backend)
    return backend
