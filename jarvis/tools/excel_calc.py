"""Recalculate a saved workbook with a real spreadsheet engine.

openpyxl writes formulas but never their values, so after Jarvis saves a workbook every
formula has no cached result until a spreadsheet program opens and saves it. Two engines:

- Excel (Windows, COM), in a private hidden instance, described below;
- LibreOffice (any OS where ``soffice`` is installed; the Linux path): a headless
  convert-to-xlsx of a temp copy in a private profile (LibreOffice calculates formulas
  that have no cached value on load), swapped in atomically under the same rules. A
  LibreOffice re-save can change styling or drop features openpyxl never touched, so
  Excel is preferred when both exist.

The Excel path, safely:

- a NEW Excel process is started (``DispatchEx``); the user's own Excel is never attached to,
  changed or closed;
- the work is done on a temp copy beside the target and swapped in with ``os.replace`` only
  if Excel saved a valid file, so a failure leaves the original untouched;
- the helper's PID is captured at creation. On timeout that process, and only that one, is
  killed; after a normal quit it is killed too if it lingers.
"""

from __future__ import annotations

import functools
import gc
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis import platform as plat

log = logging.getLogger(__name__)

RECALC_TIMEOUT = 40.0
EXIT_GRACE = 5.0  # how long a quit Excel gets to disappear before it is killed
MS_AUTOMATION_SECURITY_FORCE_DISABLE = 3


@dataclass(frozen=True)
class Recalc:
    """Outcome of a recalculation. ``status`` is ok, unavailable, locked, timeout or failed."""

    status: str
    detail: str = ""
    engine: str = "Excel"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@functools.lru_cache(maxsize=1)
def excel_installed() -> bool:
    """True if the Excel.Application COM class is registered (detected once)."""
    return plat.current().has_excel_com()


def find_soffice() -> str | None:
    """Path of LibreOffice's command-line launcher, if installed."""
    return shutil.which("soffice") or shutil.which("libreoffice")


def _com_modules() -> tuple[Any, Any, Any]:
    """(pythoncom, DispatchEx, win32process); a seam so tests can substitute fakes."""
    import pythoncom
    import win32process
    from win32com.client import DispatchEx

    return pythoncom, DispatchEx, win32process


def is_locked(path: Path) -> bool:
    """True if another program (normally Excel) holds the file open for writing."""
    try:
        with open(path, "r+b"):
            return False
    except PermissionError:
        return True
    except OSError:
        return False


# ---------------------------------------------------------------- the COM job
class _Job:
    """State shared between the caller and the worker thread."""

    def __init__(self) -> None:
        self.pid: int | None = None
        self.error: str | None = None
        self.done = threading.Event()


def _app_pid(app: Any, win32process: Any) -> int | None:
    try:
        return int(win32process.GetWindowThreadProcessId(app.Hwnd)[1])
    except Exception:
        log.debug("could not read the helper Excel PID", exc_info=True)
        return None


def _configure(app: Any) -> None:
    app.Visible = False
    app.DisplayAlerts = False
    app.ScreenUpdating = False
    app.AskToUpdateLinks = False
    app.EnableEvents = False
    app.AutomationSecurity = MS_AUTOMATION_SECURITY_FORCE_DISABLE  # no macros run on open


def _run_com(copy: Path, job: _Job) -> None:
    """Worker thread: open the copy in a private Excel, calculate, save, quit."""
    pythoncom, dispatch_ex, win32process = _com_modules()
    pythoncom.CoInitialize()
    app = book = None
    try:
        app = dispatch_ex("Excel.Application")
        job.pid = _app_pid(app, win32process)
        _configure(app)
        book = app.Workbooks.Open(str(copy), 0, False)
        app.CalculateFull()
        book.Save()
        book.Close(False)
        book = None
        gc.collect()  # release the workbook proxy while Excel is still alive
        app.Quit()
    except BaseException as exc:  # includes COM errors, which are not always Exceptions
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if book is not None:
                book.Close(False)
            if app is not None and job.error:
                app.Quit()
        except BaseException:
            log.debug("cleanup of helper Excel failed", exc_info=True)
        book = app = None
        gc.collect()  # drop the COM wrappers before the apartment goes away
        pythoncom.CoUninitialize()
        job.done.set()


def kill_process(pid: int) -> None:
    """Terminate a helper Excel (and its children) if the PID is still an EXCEL.EXE."""
    import psutil

    try:
        proc = psutil.Process(pid)
        if proc.name().lower() != "excel.exe":  # the PID may have been reused
            return
        for child in proc.children(recursive=True):
            child.kill()
        proc.kill()
    except psutil.NoSuchProcess:
        pass
    except psutil.Error:
        log.warning("could not terminate helper Excel pid %s", pid, exc_info=True)


def _wait_gone(pid: int | None, grace: float = EXIT_GRACE) -> None:
    """After Quit, make sure the helper process is really gone."""
    if pid is None:
        return
    import psutil

    deadline = time.monotonic() + grace
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if psutil.pid_exists(pid):
        log.warning("helper Excel pid %s did not exit after Quit; terminating it", pid)
        kill_process(pid)


def _discard(path: Path) -> None:
    for _ in range(10):  # Excel may need a moment to release the file
        try:
            path.unlink(missing_ok=True)
            return
        except OSError:
            time.sleep(0.2)
    log.warning("could not remove temp file %s", path)


def _valid_xlsx(path: Path) -> bool:
    try:
        return path.stat().st_size > 0 and zipfile.is_zipfile(path)
    except OSError:
        return False


# ------------------------------------------------------------------ LibreOffice
def _run_soffice(argv: list[str], timeout: float) -> tuple[int, str]:
    """Run LibreOffice in its own process group; (exit code, output). A seam for tests."""
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        **plat.current().subprocess_kwargs(),
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        plat.current().kill_tree(proc.pid)
        proc.communicate()
        raise
    return proc.returncode, out or ""


def _recalculate_libreoffice(path: Path, soffice: str, timeout: float) -> Recalc:
    engine = "LibreOffice"
    if (path.parent / f".~lock.{path.name}#").exists():
        return Recalc("locked", f"{path.name} is open in LibreOffice", engine)
    if is_locked(path):
        return Recalc("locked", f"{path.name} is open in another program", engine)
    staged: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-lo-") as tmp:
            work = Path(tmp)
            (work / "in").mkdir()
            (work / "out").mkdir()
            source = work / "in" / path.name
            shutil.copyfile(path, source)
            profile = (work / "profile").as_uri()
            argv = [
                soffice, f"-env:UserInstallation={profile}", "--headless", "--norestore", "--nolockcheck",
                "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", str(work / "out"), str(source),
            ]
            try:
                code, output = _run_soffice(argv, timeout)
            except subprocess.TimeoutExpired:
                return Recalc("timeout", f"LibreOffice did not finish within {timeout:.0f}s and was stopped", engine)
            produced = work / "out" / path.name
            if code != 0 or not _valid_xlsx(produced):
                return Recalc("failed", f"LibreOffice did not produce a valid file ({output.strip()[:120] or code})", engine)
            fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.calc.", suffix=path.suffix)
            os.close(fd)
            staged = Path(name)
            shutil.copyfile(produced, staged)
        try:
            os.replace(staged, path)
        except PermissionError:
            return Recalc("locked", f"{path.name} was opened elsewhere during recalculation", engine)
        staged = None
        return Recalc("ok", "", engine)
    except OSError as exc:
        return Recalc("failed", f"{type(exc).__name__}: {exc}", engine)
    finally:
        if staged is not None:
            _discard(staged)


# ------------------------------------------------------------------ public API
def recalculate(path: Path, timeout: float = RECALC_TIMEOUT) -> Recalc:
    """Recalculate ``path`` in place and store the results: with a private hidden Excel
    when installed, else with headless LibreOffice when installed.

    Never raises. Returns status ``unavailable`` (no engine), ``locked`` (the user has the
    file open), ``timeout`` or ``failed`` without having touched the original file.
    """
    if excel_installed():
        return _recalculate_excel(path, timeout)
    soffice = find_soffice()
    if soffice:
        return _recalculate_libreoffice(path, soffice, timeout)
    return Recalc("unavailable", "Excel is not installed and neither is LibreOffice (soffice)")


def _recalculate_excel(path: Path, timeout: float) -> Recalc:
    try:
        _com_modules()  # confirms pywin32 is usable before any copying
    except ImportError:
        return Recalc("unavailable", "pywin32 is not installed")
    if is_locked(path):
        return Recalc("locked", f"{path.name} is open in Excel")

    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.calc.", suffix=path.suffix)
    os.close(fd)
    copy = Path(name)
    job = _Job()
    try:
        with open(path, "rb") as src, open(copy, "wb") as dst:
            dst.write(src.read())
        worker = threading.Thread(target=_run_com, args=(copy, job), name="excel-recalc", daemon=True)
        worker.start()
        if not job.done.wait(timeout):
            if job.pid is not None:
                kill_process(job.pid)
            else:
                log.warning("Excel recalculation timed out before the helper PID was known")
            worker.join(5)
            return Recalc("timeout", f"Excel did not finish within {timeout:.0f}s and was stopped")
        _wait_gone(job.pid)
        if job.error:
            return Recalc("failed", job.error)
        if not _valid_xlsx(copy):
            return Recalc("failed", "Excel did not produce a valid file")
        try:
            os.replace(copy, path)
        except PermissionError:
            return Recalc("locked", f"{path.name} was opened in Excel during recalculation")
        return Recalc("ok")
    except OSError as exc:
        return Recalc("failed", f"{type(exc).__name__}: {exc}")
    finally:
        _discard(copy)
