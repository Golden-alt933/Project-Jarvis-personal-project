"""System stats sampled on a background thread; results are delivered by a Qt signal."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger(__name__)
INTERVAL = 3.0
GPU_EVERY = 3


@dataclass(frozen=True)
class Sample:
    cpu: float | None = None  # 0..1
    memory: float | None = None
    gpu: float | None = None
    battery: float | None = None
    charging: bool = False


def read_gpu(exe: str | None) -> float | None:
    if not exe:
        return None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        out = subprocess.run([exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2.5, creationflags=flags)
        return float(out.stdout.strip().splitlines()[0]) / 100.0
    except Exception:
        return None


def take_sample(psutil_mod, gpu_exe: str | None) -> Sample:
    cpu = mem = bat = None
    charging = False
    try:
        cpu = psutil_mod.cpu_percent(interval=None) / 100.0
        mem = psutil_mod.virtual_memory().percent / 100.0
        battery = psutil_mod.sensors_battery()
        if battery is not None:
            bat, charging = battery.percent / 100.0, bool(battery.power_plugged)
    except Exception:
        log.debug("sampling failed", exc_info=True)
    return Sample(cpu, mem, read_gpu(gpu_exe), bat, charging)


class SysMonitor(QObject):
    sampled = pyqtSignal(object)

    def __init__(self, interval: float = INTERVAL) -> None:
        super().__init__()
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive() and not self._stop.is_set():
            return
        self._stop = threading.Event()
        try:
            import psutil
        except ImportError:
            return
        gpu_exe = shutil.which("nvidia-smi")
        psutil.cpu_percent(interval=None)

        stop = self._stop

        def loop() -> None:
            n = 0
            gpu = None
            while not stop.is_set():
                # nvidia-smi is a process spawn: ask it every third sample, reuse the reading between
                sample = take_sample(psutil, gpu_exe if n % GPU_EVERY == 0 else None)
                if n % GPU_EVERY == 0:
                    gpu = sample.gpu
                elif gpu is not None:
                    sample = Sample(sample.cpu, sample.memory, gpu, sample.battery, sample.charging)
                n += 1
                if stop.is_set():
                    break
                try:
                    self.sampled.emit(sample)
                except RuntimeError:  # the Qt object is already gone
                    break
                stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, name="hud-sysmon", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
