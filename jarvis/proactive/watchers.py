"""Proactive alerts: Jarvis speaks first, sparingly.

Plain checks over cheap sensors, no model call. Each watcher fires once per *episode*
(it re-arms only after the condition has clearly ended, with hysteresis) and has a
cool-down so a value hovering around a threshold cannot repeat itself. Quiet hours
hold everything back; an alert that is still true afterwards is then delivered.

``ProactiveRunner`` delivers through the same path as reminders and never while the
user is mid-turn or being spoken to (the long-turn notice is the one exception, by
design, and only through its own ``deliver_busy`` path).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

if TYPE_CHECKING:
    from jarvis.config import Settings

log = logging.getLogger(__name__)

GIB = 1024**3
POLL_SECONDS = 30.0


# -- sensors ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Battery:
    percent: float
    plugged: bool


@dataclass(frozen=True)
class Disk:
    free_bytes: int
    total_bytes: int

    @property
    def free_percent(self) -> float:
        return 100.0 * self.free_bytes / self.total_bytes if self.total_bytes else 100.0


class Sensors(Protocol):
    def battery(self) -> Battery | None: ...
    def disk(self) -> Disk | None: ...


class SystemSensors:
    """The real readings (psutil for the battery, shutil for the disk). None = unavailable."""

    @staticmethod
    def system_drive() -> str:
        return (os.environ.get("SystemDrive", "C:") + "\\") if os.name == "nt" else "/"

    def battery(self) -> Battery | None:
        try:
            import psutil

            reading = psutil.sensors_battery()
        except Exception:
            return None
        if reading is None or reading.percent is None:
            return None
        return Battery(float(reading.percent), bool(reading.power_plugged))

    def disk(self) -> Disk | None:
        try:
            usage = shutil.disk_usage(self.system_drive())
        except OSError:
            return None
        return Disk(usage.free, usage.total)


# -- quiet hours ---------------------------------------------------------------------------------
def parse_quiet_hours(text: str) -> tuple[time, time] | None:
    """'22:00-07:00' -> (22:00, 07:00); empty or malformed -> None (never quiet)."""
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        start, end = raw.replace(" ", "").split("-")
        a = time.fromisoformat(start.zfill(5))
        b = time.fromisoformat(end.zfill(5))
    except ValueError:
        log.warning("JARVIS_QUIET_HOURS=%r is not HH:MM-HH:MM; ignoring it", text)
        return None
    return None if a == b else (a, b)


def is_quiet(window: tuple[time, time] | None, now: datetime) -> bool:
    if window is None:
        return False
    start, end = window
    t = now.time().replace(second=0, microsecond=0)
    return start <= t < end if start < end else (t >= start or t < end)  # the window may cross midnight


# -- watchers ----------------------------------------------------------------------------------------
class Watcher:
    """Base: episode bookkeeping shared by the checks.

    A key that has fired stays fired until the subclass calls ``_rearm`` because the
    condition ended; ``cooldown`` additionally spaces two firings of the same key.
    """

    name = "watcher"
    cooldown = timedelta(minutes=30)

    def __init__(self) -> None:
        self._fired: set[str] = set()
        self._last: dict[str, datetime] = {}

    def _fire(self, key: str, now: datetime, text: str) -> str | None:
        if key in self._fired:
            return None
        last = self._last.get(key)
        if last is not None and now - last < self.cooldown:
            return None
        self._fired.add(key)
        self._last[key] = now
        return text

    def _rearm(self) -> None:
        self._fired.clear()

    def check(self, sensors: Sensors, now: datetime) -> str | None:  # pragma: no cover - interface
        raise NotImplementedError


class BatteryLow(Watcher):
    name = "battery"
    HYSTERESIS = 5  # percentage points above the threshold before the next episode may start

    def __init__(self, low: int = 20, critical: int = 10) -> None:
        super().__init__()
        self.low, self.critical = low, critical

    def check(self, sensors: Sensors, now: datetime) -> str | None:
        reading = sensors.battery()
        if reading is None:
            return None
        if reading.plugged or reading.percent >= self.low + self.HYSTERESIS:
            self._rearm()
            return None
        pct = int(reading.percent)
        if reading.percent <= self.critical:
            self._fired.add("low")  # the earlier warning is moot now
            return self._fire("critical", now, f"The battery is critically low at {pct} percent. Please plug in.")
        if reading.percent <= self.low:
            return self._fire("low", now, f"The battery is at {pct} percent and not charging.")
        return None


class BatteryFull(Watcher):
    name = "battery-full"
    FULL = 99
    REARM_BELOW = 90

    def check(self, sensors: Sensors, now: datetime) -> str | None:
        reading = sensors.battery()
        if reading is None:
            return None
        if not reading.plugged or reading.percent <= self.REARM_BELOW:
            self._rearm()
            return None
        if reading.percent >= self.FULL:
            return self._fire("full", now, "The battery is fully charged. You can unplug it.")
        return None


class DiskLow(Watcher):
    name = "disk"
    cooldown = timedelta(hours=6)
    HYSTERESIS = 1.25  # free space must exceed the threshold by this factor to end the episode

    def __init__(self, min_gb: float = 10.0, min_percent: float = 5.0) -> None:
        super().__init__()
        self.min_bytes, self.min_percent = int(min_gb * GIB), min_percent

    def check(self, sensors: Sensors, now: datetime) -> str | None:
        disk = sensors.disk()
        if disk is None:
            return None
        low = disk.free_bytes < self.min_bytes or disk.free_percent < self.min_percent
        recovered = (
            disk.free_bytes > self.min_bytes * self.HYSTERESIS
            and disk.free_percent > self.min_percent * self.HYSTERESIS
        )
        if recovered:
            self._rearm()
            return None
        if low:
            return self._fire("low", now, f"The system drive is nearly full: only {disk.free_bytes / GIB:.1f} gigabytes free.")
        return None


class LongTurn(Watcher):
    """Not a sensor: notices a turn that has been running a long time and says so once."""

    name = "long-turn"
    cooldown = timedelta(0)

    def __init__(self, seconds: int = 90, turn_active: Callable[[], bool] = lambda: False) -> None:
        super().__init__()
        self.seconds = seconds
        self.turn_active = turn_active
        self._since: datetime | None = None

    def check(self, sensors: Sensors, now: datetime) -> str | None:
        if not self.turn_active():
            self._since = None
            self._rearm()
            return None
        if self._since is None:
            self._since = now
            return None
        if (now - self._since).total_seconds() >= self.seconds:
            return self._fire("long", now, "Still working on that. It is taking longer than usual.")
        return None


def build_watchers(settings: "Settings", turn_active: Callable[[], bool] = lambda: False) -> list[Watcher]:
    """The watchers the settings ask for. JARVIS_PROACTIVE=off gives none."""
    if not proactive_enabled(settings):
        return []
    watchers: list[Watcher] = []
    if settings.watch_battery:
        watchers.append(BatteryLow(settings.battery_low_percent, settings.battery_critical_percent))
    if settings.watch_battery_full:
        watchers.append(BatteryFull())
    if settings.watch_disk:
        watchers.append(DiskLow(settings.disk_low_gb, settings.disk_low_percent))
    if settings.long_turn_seconds > 0:
        watchers.append(LongTurn(settings.long_turn_seconds, turn_active))
    return watchers


def proactive_enabled(settings: "Settings") -> bool:
    return settings.proactive.strip().lower() not in ("off", "false", "0", "no")


# -- delivery ---------------------------------------------------------------------------------------
class ProactiveRunner:
    """Polls the watchers and delivers what fires, one short sentence at a time."""

    def __init__(
        self,
        watchers: list[Watcher],
        sensors: Sensors,
        deliver: Callable[[str], Awaitable[None]],
        is_idle: Callable[[], bool] = lambda: True,
        *,
        deliver_busy: Callable[[str], Awaitable[None]] | None = None,
        quiet: tuple[time, time] | None = None,
        clock: Callable[[], datetime] = datetime.now,
        interval: float = POLL_SECONDS,
    ) -> None:
        self.watchers = watchers
        self.sensors = sensors
        self.deliver = deliver
        self.deliver_busy = deliver_busy  # the long-turn notice is spoken mid-turn by design
        self.is_idle = is_idle
        self.quiet = quiet
        self.clock = clock
        self.interval = interval
        self._task: asyncio.Task[None] | None = None

    async def tick(self) -> list[str]:
        now = self.clock()
        quiet = is_quiet(self.quiet, now)
        idle = self.is_idle()
        sent: list[str] = []
        for watcher in self.watchers:
            busy_kind = isinstance(watcher, LongTurn)
            if busy_kind:
                if quiet or self.deliver_busy is None:
                    continue
            elif quiet or not idle:
                continue  # state is untouched, so a still-true alert fires after quiet hours / the turn
            try:
                text = watcher.check(self.sensors, now)
            except Exception:
                log.exception("watcher %s failed", watcher.name)
                continue
            if not text:
                continue
            send = self.deliver_busy if busy_kind else self.deliver
            try:
                await send(text)  # type: ignore[misc]
            except Exception:
                log.exception("could not deliver alert from %s", watcher.name)
            sent.append(text)
            if not busy_kind:
                break  # one spoken alert per pass: sparingly
        return sent

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("proactive pass failed")
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self.watchers and (self._task is None or self._task.done()):
            self._task = asyncio.ensure_future(self.run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
