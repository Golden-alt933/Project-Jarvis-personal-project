"""Windows backend. Everything here used to live in the tool modules.

The module imports on any OS (ctypes.windll and friends are only touched inside methods),
so its logic can be unit-tested anywhere with the system calls mocked.

Volume note: master volume is driven by a small C# COM interop snippet compiled by
PowerShell (``Add-Type``) against the Core Audio ``IAudioEndpointVolume`` interface. It
needs no extra Python packages, reads and sets an exact percent, and takes roughly 1-2
seconds per call because of the compile step.
"""

from __future__ import annotations

import base64
import ctypes
import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

import psutil

from jarvis.platform.base import (
    JUNK_NAME,
    AppEntry,
    BasePlatform,
    PlatformUnsupported,
    UiElement,
    UiSnapshot,
    WindowInfo,
    WindowRow,
    kill_with_psutil,
)

log = logging.getLogger(__name__)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

KNOWN_FOLDERS = {
    "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "Downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "Pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "Music": "4BD8D571-6D19-48D3-BE97-422220080E43",
    "Videos": "18989B1D-99B5-42AC-8B2D-CC9CF2A3C0D2",
}

# Executables that exist on a stock Windows install, reachable by name.
COMMON_EXES: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "command prompt": "cmd.exe",
    "task manager": "taskmgr.exe",
    "file explorer": "explorer.exe",
    "snipping tool": "snippingtool.exe",
    "control panel": "control.exe",
    "registry editor": "regedit.exe",
    "wordpad": "write.exe",
}

# Browsers install site shortcuts ("Calculator", "YouTube") that launch through these.
_BROWSER_LAUNCHERS = ("chrome_proxy.exe", "msedge_proxy.exe", "brave_proxy.exe")
_BROWSER_APP_ID = re.compile(r"(chrome|msedge|brave|chromium)\._crx_", re.IGNORECASE)

_WM_CLOSE = 0x0010
_MEDIA_VK = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}

_AUDIO_CS = r"""
using System;
using System.Runtime.InteropServices;
[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {
  int a(); int b(); int c(); int d();
  int SetMasterVolumeLevelScalar(float level, Guid ctx);
  int e();
  int GetMasterVolumeLevelScalar(out float level);
  int f(); int g(); int h(); int i();
  int SetMute([MarshalAs(UnmanagedType.Bool)] bool mute, Guid ctx);
  int GetMute(out bool mute);
}
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice { int Activate(ref Guid id, int clsCtx, IntPtr p, out IAudioEndpointVolume v); }
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator { int f(); int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice d); }
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumeratorComObject { }
public class JarvisAudio {
  static IAudioEndpointVolume Ep() {
    var en = (IMMDeviceEnumerator)(new MMDeviceEnumeratorComObject());
    IMMDevice dev; en.GetDefaultAudioEndpoint(0, 1, out dev);
    Guid iid = typeof(IAudioEndpointVolume).GUID; IAudioEndpointVolume v;
    dev.Activate(ref iid, 23, IntPtr.Zero, out v); return v;
  }
  public static string Get() { float l; bool m; var v = Ep(); v.GetMasterVolumeLevelScalar(out l); v.GetMute(out m);
    return ((int)Math.Round(l * 100)).ToString() + "," + (m ? "1" : "0"); }
  public static void SetLevel(int pct) { Ep().SetMasterVolumeLevelScalar(pct / 100f, Guid.Empty); }
  public static void SetMute(bool m) { Ep().SetMute(m, Guid.Empty); }
}
"""


def _powershell(script: str, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    """Run a PowerShell script (passed encoded, so no quoting problems)."""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_NO_WINDOW,
    )


def _start_menu_dirs() -> list[Path]:
    roots = [os.environ.get("PROGRAMDATA"), os.environ.get("APPDATA")]
    return [Path(r) / "Microsoft" / "Windows" / "Start Menu" / "Programs" for r in roots if r]


def _is_webapp_shortcut(path: Path) -> bool:
    """True if the .lnk opens a website in a browser window rather than a real app.
    The target path is stored in the file as ANSI and/or UTF-16, so a byte search is enough."""
    if any(part.lower().endswith(" apps") for part in path.parts[-3:-1]):
        return True  # e.g. Start Menu\Programs\Chrome Apps
    try:
        raw = path.read_bytes()[:8192].lower()
    except OSError:
        return False
    return any(n.encode() in raw or n.encode("utf-16-le") in raw for n in _BROWSER_LAUNCHERS)


def scan_shortcuts(dirs: list[Path]) -> list[AppEntry]:
    found: dict[str, AppEntry] = {}
    for base in dirs:
        if not base.is_dir():
            continue
        for path in base.rglob("*.lnk"):
            name = path.stem
            if JUNK_NAME.search(name):
                continue
            kind = "webapp" if _is_webapp_shortcut(path) else "lnk"
            key = name.lower()
            if key not in found or (found[key].kind == "webapp" and kind == "lnk"):
                found[key] = AppEntry(name, kind, str(path))
    return list(found.values())


def scan_uwp() -> list[AppEntry]:
    try:
        out = _powershell("Get-StartApps | ConvertTo-Json -Compress", timeout=25)
        data = json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        log.warning("Get-StartApps failed", exc_info=True)
        return []
    if isinstance(data, dict):
        data = [data]
    entries = []
    for item in data:
        name, app_id = item.get("Name"), item.get("AppID")
        if name and app_id and not JUNK_NAME.search(name):
            # Get-StartApps also lists browser site shortcuts ("Chrome._crx_..."), often
            # with the same name as a real app and ahead of it.
            kind = "webapp" if _BROWSER_APP_ID.match(app_id) else "uwp"
            entries.append(AppEntry(name, kind, app_id))
    return entries


def common_exes() -> list[AppEntry]:
    return [AppEntry(n.title(), "exe", exe) for n, exe in COMMON_EXES.items()]


class WindowsPlatform(BasePlatform):
    name = "windows"
    app_dir_name = "Jarvis"
    shell_kind = "powershell"
    shell_name = "PowerShell"
    reserved_device_names = True
    trash_name = "Recycle Bin"

    def describe(self) -> str:
        import platform

        return f"Windows {platform.release()}"

    # -- shell ------------------------------------------------------------
    def shell_argv(self, command: str) -> list[str]:
        # UTF-8 output and no progress bars (they pollute captured output).
        script = (
            "$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.Encoding]::UTF8; "
            + command
        )
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]

    def subprocess_kwargs(self) -> dict[str, Any]:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags}

    def kill_tree(self, pid: int) -> None:
        kill_with_psutil(pid)

    # -- opening things ---------------------------------------------------
    def open_path(self, path: str) -> None:
        os.startfile(path)  # type: ignore[attr-defined]  # Windows only

    def known_folder(self, name: str) -> Path:
        """Desktop, Documents, ...: these are often redirected (OneDrive), so the shell is
        asked instead of assuming ``~/Desktop``. Falls back to ``~/<name>``."""
        fallback = Path.home() / name
        guid_text = KNOWN_FOLDERS.get(name)
        if guid_text is None or os.name != "nt":
            return fallback
        try:
            from ctypes import wintypes

            class _GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8),
                ]

            raw = uuid.UUID(guid_text)
            guid = _GUID(raw.time_low, raw.time_mid, raw.time_hi_version, (ctypes.c_ubyte * 8)(*raw.bytes[8:]))
            out = ctypes.c_wchar_p()
            rc = ctypes.windll.shell32.SHGetKnownFolderPath(  # type: ignore[attr-defined]
                ctypes.byref(guid), 0, None, ctypes.byref(out)
            )
            if rc == 0 and out.value:
                path = Path(out.value)
                ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]
                return path
        except Exception:  # best effort; the fallback is right for most machines
            log.debug("known folder lookup failed for %s", name, exc_info=True)
        return fallback

    # -- apps and windows -------------------------------------------------
    def scan_apps(self) -> list[AppEntry]:
        return scan_shortcuts(_start_menu_dirs()) + scan_uwp() + common_exes()

    def launch_app(self, entry: AppEntry) -> None:
        if not entry.target.lower().endswith((".lnk", ".exe")):  # a Start-menu AppID
            subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{entry.target}"], creationflags=_NO_WINDOW)
        else:
            os.startfile(entry.target)  # type: ignore[attr-defined]  # Windows only

    def windows(self) -> list[WindowRow]:
        if os.name != "nt":
            raise self.unsupported("listing windows")
        user32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi  # type: ignore[attr-defined]
        results: list[WindowRow] = []
        enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)  # type: ignore[attr-defined]

        def callback(hwnd: int, _: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            cloaked = ctypes.c_int(0)  # DWMWA_CLOAKED: hidden UWP shells and other virtual-desktop windows
            dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
            length = user32.GetWindowTextLengthW(hwnd)
            if cloaked.value or length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                proc = psutil.Process(pid.value).name()
            except psutil.Error:
                return True
            results.append((proc, buf.value, pid.value, int(hwnd or 0)))
            return True

        user32.EnumWindows(enum_proc(callback), 0)
        return results

    def close_window(self, handle: int) -> None:
        ctypes.windll.user32.PostMessageW(handle, _WM_CLOSE, 0, 0)  # type: ignore[attr-defined]

    def never_close_titles(self) -> set[str]:
        return {"program manager"}  # the desktop shell itself

    def foreground_window_title(self) -> str:
        if os.name != "nt":
            return ""
        try:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return ""
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            return buffer.value.strip()
        except Exception:
            log.debug("window title unavailable", exc_info=True)
            return ""

    # -- audio, media, session -------------------------------------------
    def _audio(self, action: str) -> tuple[int, bool]:
        """Run an audio action ('Get()', 'SetLevel(40)', 'SetMute($true)'); return (percent, muted)."""
        script = (
            "$ProgressPreference = 'SilentlyContinue'\n"  # else stderr fills with CLIXML noise
            f"$src = @'\n{_AUDIO_CS}\n'@\n"
            "Add-Type -TypeDefinition $src\n"
            f"[JarvisAudio]::{action}\n"
            "[JarvisAudio]::Get()"
        )
        out = _powershell(script, timeout=25)
        last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if out.returncode != 0 or not re.fullmatch(r"\d+,[01]", last):
            raise OSError((out.stderr or out.stdout or "no audio endpoint").strip()[:200])
        pct, muted = last.split(",")
        return int(pct), muted == "1"

    def get_volume(self) -> tuple[int, bool]:
        return self._audio("Get()")

    def set_volume(self, percent: int) -> tuple[int, bool]:
        return self._audio(f"SetLevel({int(percent)})")

    def set_mute(self, muted: bool) -> tuple[int, bool]:
        return self._audio(f"SetMute(${'true' if muted else 'false'})")

    def media_key(self, action: str) -> None:
        vk = _MEDIA_VK[action]
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP

    def lock_screen(self) -> None:
        if not ctypes.windll.user32.LockWorkStation():  # type: ignore[attr-defined]
            raise OSError("LockWorkStation failed")

    # -- files and processes ---------------------------------------------
    def protected_path_reason(self, real_path: str) -> str | None:
        for var in ("SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData"):
            base = os.environ.get(var)
            if not base:
                continue
            b = os.path.normcase(os.path.realpath(base))
            if real_path == b or real_path.startswith(b + os.sep):
                return f"it is inside the system folder {base}"
        return None

    def ollama_extra_paths(self) -> list[Path]:
        local = os.environ.get("LOCALAPPDATA", "")
        return [Path(local) / "Programs" / "Ollama" / "ollama.exe"] if local else []

    def detached_popen_kwargs(self) -> dict[str, Any]:
        return {
            "creationflags": subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        }

    # -- screen -----------------------------------------------------------
    _dpi_set = False

    def prepare_capture(self) -> None:
        """Without this, cursor and monitor coordinates disagree at non-100% scaling."""
        if self._dpi_set or os.name != "nt":
            return
        WindowsPlatform._dpi_set = True
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]  # per-monitor
        except Exception:  # already set (e.g. by mss) or unavailable
            log.debug("could not set DPI awareness", exc_info=True)

    def cursor_position(self) -> tuple[int, int] | None:
        if os.name != "nt":
            return None
        try:
            from ctypes import wintypes

            point = wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):  # type: ignore[attr-defined]
                return int(point.x), int(point.y)
        except Exception:
            log.debug("GetCursorPos failed", exc_info=True)
        return None

    # -- camera -----------------------------------------------------------
    def camera_backends(self) -> list[str]:
        # DirectShow opens in about a second here; Media Foundation took five. MSMF is the fallback.
        return ["dshow", "msmf"]

    def camera_access_problem(self) -> str | None:
        """Windows has a master camera switch and one for desktop apps; either set to Deny blocks us."""
        try:
            import winreg
        except ImportError:
            return None
        base = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
        checks = [
            (winreg.HKEY_LOCAL_MACHINE, base, "Windows 'Camera access' is off for this device"),
            (winreg.HKEY_CURRENT_USER, base, "Windows 'Camera access' is off for this account"),
            (winreg.HKEY_CURRENT_USER, base + r"\NonPackaged", "'Let desktop apps access your camera' is off"),
        ]
        for hive, path, meaning in checks:
            try:
                with winreg.OpenKey(hive, path) as key:
                    if str(winreg.QueryValueEx(key, "Value")[0]).lower() == "deny":
                        return f"{meaning}. Turn it on in Settings > Privacy & security > Camera."
            except OSError:
                continue
        return None

    def camera_devices(self) -> list[str] | None:
        try:
            out = _powershell(
                "Get-PnpDevice -Class Camera -PresentOnly -ErrorAction SilentlyContinue | "
                "Select-Object -ExpandProperty FriendlyName | ConvertTo-Json -Compress",
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        text = out.stdout.strip()
        if out.returncode != 0:
            return None
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return [str(n) for n in (data if isinstance(data, list) else [data]) if n]

    # -- desktop control ----------------------------------------------------
    _uia_session: Any = None

    def _uia(self) -> Any:
        if WindowsPlatform._uia_session is None:
            from jarvis.platform.windows_uia import UiaSession

            WindowsPlatform._uia_session = UiaSession()
        return WindowsPlatform._uia_session

    def desktop_problem(self) -> str | None:
        if os.name != "nt":
            return "desktop control needs Windows"
        try:
            import comtypes  # noqa: F401
        except ImportError:
            return "the 'desktop' extra is not installed (needs comtypes): uv sync --extra desktop"
        return None

    def can_inspect_elements(self) -> bool:
        return True

    def self_elevated(self) -> bool:
        from jarvis.platform import windows_input as wi

        return wi.self_elevated()

    def input_desktop_available(self) -> bool:
        from jarvis.platform import windows_input as wi

        return wi.input_desktop_available()

    def window_info(self, handle: int) -> WindowInfo | None:
        from jarvis.platform import windows_input as wi

        if os.name != "nt" or not handle:
            return None
        title, class_name, pid, minimized, rect = wi.window_details(handle)
        try:
            proc = psutil.Process(pid).name()
        except psutil.Error:
            proc = "?"
        return WindowInfo(
            title=title.strip(), process=proc, class_name=class_name, pid=pid, handle=handle,
            elevated=wi.process_elevated(pid), minimized=minimized, rect=rect,
        )

    def window_at(self, x: int, y: int) -> WindowInfo | None:
        from jarvis.platform import windows_input as wi

        return self.window_info(wi.root_window_at(int(x), int(y)))

    def active_window(self) -> WindowInfo | None:
        from jarvis.platform import windows_input as wi

        return self.window_info(wi.foreground_handle())

    def focus_window(self, handle: int) -> bool:
        from jarvis.platform import windows_input as wi

        return wi.bring_to_front(handle)

    def ui_snapshot(self, handle: int) -> UiSnapshot:
        self.prepare_capture()
        info = self.window_info(handle)
        if info is None:
            raise OSError("that window no longer exists")
        session = self._uia()
        elements, total = session.run(session.snapshot, handle)
        return UiSnapshot(window=info, elements=elements, total=total)

    def ui_read(self, handle: int, ref: int) -> UiElement:
        session = self._uia()
        return session.run(session.read, handle, ref)

    def ui_act(self, handle: int, ref: int, action: str) -> str:
        from jarvis.platform import windows_input as wi

        self.prepare_capture()
        session = self._uia()
        return session.run(session.act, handle, ref, action, wi.click, timeout=30)

    def ui_clear(self, handle: int, ref: int) -> bool:
        session = self._uia()
        return session.run(session.clear, handle, ref)

    def ui_focus_element(self, handle: int, ref: int) -> str:
        from jarvis.platform import windows_input as wi

        self.prepare_capture()
        session = self._uia()
        return session.run(session.focus, handle, ref, wi.click)

    def ui_focused(self) -> UiElement | None:
        session = self._uia()
        return session.run(session.focused)

    def input_click(self, x: int, y: int, double: bool = False) -> None:
        from jarvis.platform import windows_input as wi

        self.prepare_capture()
        wi.click(x, y, double)

    def input_text(self, text: str) -> None:
        from jarvis.platform import windows_input as wi

        wi.type_text(text)

    def input_keys(self, keys: list[str]) -> None:
        from jarvis.platform import windows_input as wi

        wi.press_chord(keys)

    def input_scroll(self, x: int, y: int, direction: str, amount: int) -> None:
        from jarvis.platform import windows_input as wi

        self.prepare_capture()
        wi.scroll(x, y, direction, amount)

    # -- office / voice ---------------------------------------------------
    def has_excel_com(self) -> bool:
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Excel.Application\CLSID"):
                return True
        except (OSError, ImportError):
            return False

    def prepare_gpu_libs(self) -> None:
        """Make the pip-installed cuBLAS/cuDNN (nvidia-*-cu12) DLLs findable by ctranslate2."""
        import sys

        for base in sys.path:
            root = Path(base) / "nvidia"
            if not root.is_dir():
                continue
            for bin_dir in root.glob("*/bin"):
                try:
                    os.add_dll_directory(str(bin_dir))  # type: ignore[attr-defined]
                    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                except (OSError, AttributeError):
                    pass

    def helper_report(self) -> list[tuple[str, bool, str]]:
        import shutil

        return [("powershell", bool(shutil.which("powershell")), "PowerShell ships with Windows; check PATH")]


__all__ = ["WindowsPlatform", "PlatformUnsupported"]
