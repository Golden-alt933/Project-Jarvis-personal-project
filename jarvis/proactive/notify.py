"""A desktop notification through an OS command, for results that arrive while no
window is open (``jarvis --daemon``). Fails quietly: a missing helper is not an error."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)

# The title and text travel in environment variables and are escaped by PowerShell
# itself, so nothing the user or a website wrote can be interpreted as script.
_TOAST = r"""
$ErrorActionPreference = 'Stop'
[void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
[void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]
$title = [System.Security.SecurityElement]::Escape($env:JARVIS_NOTIFY_TITLE)
$text = [System.Security.SecurityElement]::Escape($env:JARVIS_NOTIFY_TEXT)
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml("<toast><visual><binding template='ToastGeneric'><text>$title</text><text>$text</text></binding></visual></toast>")
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($xml))
"""

MAX_TEXT = 240


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def command_for(title: str, text: str, platform: str | None = None) -> tuple[list[str], dict[str, str]] | None:
    """The (argv, extra environment) that shows the notification, or None if this system has no way."""
    platform = platform or sys.platform
    title, text = _clip(title, 80), _clip(text, MAX_TEXT)
    if platform == "win32":
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            return None
        argv = [shell, "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", _TOAST]
        return argv, {"JARVIS_NOTIFY_TITLE": title, "JARVIS_NOTIFY_TEXT": text}
    if platform.startswith("linux"):
        tool = shutil.which("notify-send")
        if tool is None:
            return None
        return [tool, "--app-name=Jarvis", "--", title, text], {}
    return None


def notify(title: str, text: str) -> bool:
    """Show a desktop notification. Returns True if the OS command ran without error."""
    try:
        plan = command_for(title, text)
        if plan is None:
            return False
        argv, extra = plan
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(argv, env={**os.environ, **extra}, capture_output=True, timeout=15, check=False, **kwargs)
        return result.returncode == 0
    except Exception:
        log.debug("notification failed", exc_info=True)
        return False
