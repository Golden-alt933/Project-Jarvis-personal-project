"""Proactive watchers: episodes, hysteresis, cool-downs, quiet hours and delivery,
all with a fake clock and fake sensors."""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.proactive.watchers import (
    GIB,
    Battery,
    BatteryFull,
    BatteryLow,
    Disk,
    DiskLow,
    LongTurn,
    ProactiveRunner,
    SystemSensors,
    build_watchers,
    is_quiet,
    parse_quiet_hours,
)

T0 = datetime(2026, 6, 10, 12, 0)


class Sensors:
    def __init__(self) -> None:
        self.bat: Battery | None = None
        self.dsk: Disk | None = None

    def battery(self) -> Battery | None:
        return self.bat

    def disk(self) -> Disk | None:
        return self.dsk


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def gb(free: float, total: float = 500.0) -> Disk:
    return Disk(int(free * GIB), int(total * GIB))


# -- battery low -------------------------------------------------------------------------------------
def run_battery(steps, watcher=None):
    """steps: (minute, percent, plugged) -> list of (minute, message)."""
    watcher, sensors, fired = watcher or BatteryLow(), Sensors(), []
    for minute, pct, plugged in steps:
        sensors.bat = Battery(pct, plugged)
        text = watcher.check(sensors, at(minute))
        if text:
            fired.append((minute, text))
    return fired


def test_battery_fires_once_at_twenty_and_again_at_ten():
    fired = run_battery([(0, 30, False), (1, 21, False), (2, 20, False), (3, 19, False), (4, 15, False),
                         (5, 11, False), (6, 10, False), (7, 9, False), (8, 5, False)])
    assert [m for m, _ in fired] == [2, 6]
    assert "20 percent" in fired[0][1] and "critically" in fired[1][1] and "10 percent" in fired[1][1]


def test_battery_does_not_fire_while_plugged_in():
    assert run_battery([(0, 5, True), (1, 3, True)]) == []


def test_battery_no_battery_means_no_alert():
    w, s = BatteryLow(), Sensors()
    assert w.check(s, T0) is None


def test_battery_hysteresis_a_wobble_around_the_threshold_is_one_episode():
    steps = [(0, 20, False), (1, 22, False), (2, 24, False), (3, 19, False), (4, 24, False), (5, 20, False)]
    assert [m for m, _ in run_battery(steps)] == [0]  # never reached 25, so never re-armed


def test_battery_new_episode_after_charging_clearly_past_the_threshold_respects_the_cooldown():
    steps = [(0, 20, False), (10, 40, False), (20, 20, False)]  # re-armed at 40, but only 20 minutes after the first alert
    assert [m for m, _ in run_battery(steps)] == [0]
    steps = [(0, 20, False), (10, 40, False), (31, 20, False)]
    assert [m for m, _ in run_battery(steps)] == [0, 31]


def test_battery_plugging_in_ends_the_episode():
    steps = [(0, 15, False), (5, 15, True), (50, 15, False)]
    assert [m for m, _ in run_battery(steps)] == [0, 50]


def test_battery_that_starts_critical_gets_only_the_critical_message():
    fired = run_battery([(0, 8, False), (1, 7, False)])
    assert len(fired) == 1 and "critically" in fired[0][1]


def test_battery_custom_thresholds():
    fired = run_battery([(0, 40, False), (1, 30, False), (2, 14, False)], BatteryLow(low=35, critical=15))
    assert [m for m, _ in fired] == [1, 2]


# -- battery full ---------------------------------------------------------------------------------------
def test_battery_full_fires_once_while_plugged_and_rearms_after_unplugging():
    w = BatteryFull()
    steps = [(0, 95, True), (1, 100, True), (2, 100, True), (3, 100, False), (4, 80, False), (45, 100, True)]
    assert [m for m, _ in run_battery(steps, w)] == [1, 45]


def test_battery_full_stays_quiet_when_not_plugged():
    assert run_battery([(0, 100, False)], BatteryFull()) == []


# -- disk ---------------------------------------------------------------------------------------------------
def run_disk(steps, watcher=None):
    watcher, sensors, fired = watcher or DiskLow(), Sensors(), []
    for minute, disk in steps:
        sensors.dsk = disk
        text = watcher.check(sensors, at(minute))
        if text:
            fired.append((minute, text))
    return fired


def test_disk_fires_below_ten_gigabytes_once():
    fired = run_disk([(0, gb(50)), (1, gb(9.9)), (2, gb(8)), (3, gb(5))])
    assert len(fired) == 1 and "9.9 gigabytes" in fired[0][1]


def test_disk_fires_below_five_percent_even_with_plenty_of_gigabytes():
    assert len(run_disk([(0, gb(40, total=1000))])) == 1  # 4 percent of a 1 TB drive


def test_disk_hysteresis_and_cooldown():
    steps = [(0, gb(8)), (10, gb(11)), (20, gb(9)), (30, gb(40)), (40, gb(8)), (6 * 60 + 1, gb(40)), (6 * 60 + 2, gb(8))]
    # 11 GB is not recovered (needs > 12.5), so 9 GB is the same episode; 40 GB (8 percent) re-arms; 8 GB again within 6 h is held back
    assert [m for m, _ in run_disk(steps)] == [0, 6 * 60 + 2]


def test_disk_unreadable_means_no_alert():
    assert DiskLow().check(Sensors(), T0) is None


# -- long turn ----------------------------------------------------------------------------------------------
def test_long_turn_says_so_once_per_turn():
    active = {"on": False}
    w, s, fired = LongTurn(90, lambda: active["on"]), Sensors(), []

    def step(sec: float, on: bool) -> None:
        active["on"] = on
        text = w.check(s, T0 + timedelta(seconds=sec))
        if text:
            fired.append(sec)

    step(0, False)
    step(10, True)  # the turn starts being noticed
    step(99, True)  # 89 s in
    step(100, True)  # 90 s: fires
    step(150, True)
    step(160, False)
    step(200, True)
    step(250, True)
    step(291, True)  # 91 s into the second turn
    assert fired == [100, 291]


# -- quiet hours ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [("22:00-07:00", (time(22), time(7))), ("9:00-17:30", (time(9), time(17, 30))), (" 22:00 - 07:00 ", (time(22), time(7))),
     ("", None), ("nonsense", None), ("10:00-10:00", None), ("25:00-26:00", None)],
)
def test_parse_quiet_hours(text, expected):
    assert parse_quiet_hours(text) == expected


@pytest.mark.parametrize(
    ("window", "hour", "minute", "quiet"),
    [("22:00-07:00", 23, 0, True), ("22:00-07:00", 3, 0, True), ("22:00-07:00", 7, 0, False), ("22:00-07:00", 6, 59, True),
     ("22:00-07:00", 12, 0, False), ("22:00-07:00", 22, 0, True), ("13:00-14:00", 13, 30, True), ("13:00-14:00", 14, 0, False),
     ("", 3, 0, False)],
)
def test_is_quiet(window, hour, minute, quiet):
    assert is_quiet(parse_quiet_hours(window), datetime(2026, 6, 10, hour, minute)) is quiet


# -- delivery -------------------------------------------------------------------------------------------------------
class Rig:
    def __init__(self, watchers, *, quiet=None, busy=False, with_busy_channel=True) -> None:
        self.sensors = Sensors()
        self.now = T0
        self.said: list[str] = []
        self.said_busy: list[str] = []
        self.busy = busy

        async def say(text):
            self.said.append(text)

        async def say_busy(text):
            self.said_busy.append(text)

        self.runner = ProactiveRunner(
            watchers, self.sensors, say, lambda: not self.busy,
            deliver_busy=say_busy if with_busy_channel else None, quiet=quiet, clock=lambda: self.now, interval=0.01,
        )


async def test_runner_delivers_one_short_sentence():
    rig = Rig([BatteryLow(), DiskLow()])
    rig.sensors.bat, rig.sensors.dsk = Battery(15, False), gb(5)
    assert len(await rig.runner.tick()) == 1  # sparingly: one alert per pass
    assert len(await rig.runner.tick()) == 1  # the other one on the next pass
    assert len(await rig.runner.tick()) == 0
    assert len(rig.said) == 2 and all(len(t) < 100 for t in rig.said)


async def test_runner_never_speaks_while_the_user_is_busy_and_does_not_lose_the_alert():
    rig = Rig([BatteryLow()], busy=True)
    rig.sensors.bat = Battery(15, False)
    assert await rig.runner.tick() == []
    rig.busy = False
    assert len(await rig.runner.tick()) == 1


async def test_runner_holds_alerts_during_quiet_hours_then_delivers_if_still_true():
    rig = Rig([BatteryLow()], quiet=parse_quiet_hours("12:00-13:00"))
    rig.sensors.bat = Battery(15, False)
    assert await rig.runner.tick() == []
    rig.now = datetime(2026, 6, 10, 13, 0)
    assert len(await rig.runner.tick()) == 1
    # and an alert whose cause has gone is never delivered late
    rig2 = Rig([BatteryLow()], quiet=parse_quiet_hours("12:00-13:00"))
    rig2.sensors.bat = Battery(15, False)
    await rig2.runner.tick()
    rig2.sensors.bat = Battery(15, True)
    rig2.now = datetime(2026, 6, 10, 13, 5)
    assert await rig2.runner.tick() == []


async def test_long_turn_notice_uses_its_own_channel_even_mid_turn():
    active = {"on": True}
    rig = Rig([LongTurn(90, lambda: active["on"])], busy=True)
    await rig.runner.tick()
    rig.now = T0 + timedelta(seconds=95)
    await rig.runner.tick()
    assert rig.said == [] and rig.said_busy == ["Still working on that. It is taking longer than usual."]


async def test_long_turn_notice_is_silent_without_a_busy_channel_and_in_quiet_hours():
    rig = Rig([LongTurn(1, lambda: True)], with_busy_channel=False)
    rig.now = T0
    await rig.runner.tick()
    rig.now = T0 + timedelta(seconds=5)
    assert await rig.runner.tick() == []
    quiet = Rig([LongTurn(1, lambda: True)], quiet=parse_quiet_hours("11:00-13:00"))
    await quiet.runner.tick()
    quiet.now = T0 + timedelta(seconds=5)
    assert await quiet.runner.tick() == []


async def test_a_failing_watcher_or_delivery_does_not_stop_the_others():
    class Broken(BatteryLow):
        def check(self, sensors, now):
            raise RuntimeError("sensor crashed")

    rig = Rig([Broken(), DiskLow()])
    rig.sensors.dsk = gb(5)
    assert len(await rig.runner.tick()) == 1

    async def failing_say(text):
        raise OSError("no audio")

    runner = ProactiveRunner([DiskLow()], rig.sensors, failing_say, clock=lambda: T0)
    assert len(await runner.tick()) == 1  # reported as sent; the episode still counts


async def test_runner_start_and_stop():
    rig = Rig([DiskLow()])
    rig.sensors.dsk = gb(5)
    rig.runner.start()
    for _ in range(100):
        if rig.said:
            break
        await asyncio.sleep(0.01)
    await rig.runner.stop()
    assert len(rig.said) == 1
    ProactiveRunner([], rig.sensors, rig.runner.deliver).start()  # no watchers: nothing starts


# -- settings ------------------------------------------------------------------------------------------------------------
def names(settings: Settings) -> list[str]:
    return [w.name for w in build_watchers(settings)]


def test_default_watchers(tmp_path: Path):
    assert names(Settings(data_dir_override=str(tmp_path))) == ["battery", "disk", "long-turn"]


@pytest.mark.parametrize("off", ["off", "OFF", "false", "0", "no"])
def test_proactive_off_disables_everything(tmp_path: Path, off):
    assert names(Settings(data_dir_override=str(tmp_path), proactive=off)) == []


def test_each_watcher_is_individually_switchable(tmp_path: Path):
    base = {"data_dir_override": str(tmp_path)}
    assert names(Settings(**base, watch_battery=False)) == ["disk", "long-turn"]
    assert names(Settings(**base, watch_disk=False)) == ["battery", "long-turn"]
    assert names(Settings(**base, long_turn_seconds=0)) == ["battery", "disk"]
    assert names(Settings(**base, watch_battery_full=True)) == ["battery", "battery-full", "disk", "long-turn"]


def test_thresholds_come_from_settings(tmp_path: Path):
    watchers = build_watchers(Settings(data_dir_override=str(tmp_path), battery_low_percent=30, battery_critical_percent=12,
                                       disk_low_gb=20.0, disk_low_percent=8.0, long_turn_seconds=45))
    by_name = {w.name: w for w in watchers}
    assert (by_name["battery"].low, by_name["battery"].critical) == (30, 12)
    assert by_name["disk"].min_bytes == 20 * GIB and by_name["disk"].min_percent == 8.0
    assert by_name["long-turn"].seconds == 45


def test_system_sensors_degrade_to_none(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "sensors_battery", lambda: None)
    assert SystemSensors().battery() is None
    monkeypatch.setattr(psutil, "sensors_battery", lambda: (_ for _ in ()).throw(OSError("no ACPI")))
    assert SystemSensors().battery() is None
    monkeypatch.setattr("jarvis.proactive.watchers.shutil.disk_usage", lambda p: (_ for _ in ()).throw(OSError("gone")))
    assert SystemSensors().disk() is None
    assert SystemSensors.system_drive()
