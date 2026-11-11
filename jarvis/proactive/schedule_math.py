"""Schedule arithmetic: when does a repeating task run next?

Pure functions, no I/O, no clock of their own. Everything takes the zone explicitly
(``tz=None`` means the machine's local zone, resolved by the OS, which knows DST).

Model
-----
A task has an *anchor* (the naive local wall-clock time of its first run) and a *rule*.
Rules fall in two families:

* absolute intervals (``hourly``, ``every N minutes/hours``): a grid of fixed-length
  steps in real time, starting at the anchor. A DST change does not move the grid.
* wall-clock rules (``daily``, ``weekdays``, ``weekly``, ``monthly``, ``every N days``):
  the same local time of day on the matching dates, so "9 am" stays 9 am across DST.
  A time that does not exist (spring-forward gap) runs just after the gap; a time that
  happens twice (fall-back) runs the first time only.

``next_after(rule, anchor, after, tz)`` is the first run strictly later than ``after``,
so "catch up" can never produce a burst: the scheduler asks once, from "now".
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Iterator

MIN_INTERVAL_SECONDS = 5 * 60
UTC = timezone.utc

NAMED = ("once", "hourly", "daily", "weekdays", "weekly", "monthly")
_EVERY = re.compile(r"^every\s+(\d+)\s*(minutes?|mins?|hours?|hrs?|days?)$", re.I)
_UNIT = {"min": 60, "minute": 60, "hr": 3600, "hour": 3600, "day": 86400}


class ScheduleError(ValueError):
    """The repeat rule or time cannot be used."""


@dataclass(frozen=True)
class Rule:
    kind: str  # once | interval | days | weekdays | monthly
    seconds: int = 0  # interval length (kind == "interval")
    step_days: int = 1  # kind == "days": every N days (weekly is 7)
    text: str = "once"  # canonical spelling, stored and shown

    @property
    def repeats(self) -> bool:
        return self.kind != "once"

    @property
    def period_seconds(self) -> int:
        """Longest legitimate gap between two consecutive runs (no slack)."""
        if self.kind == "interval":
            return self.seconds
        if self.kind == "days":
            return self.step_days * 86400
        if self.kind == "weekdays":
            return 4 * 86400  # Friday to Monday plus a DST hour
        if self.kind == "monthly":
            return 31 * 86400
        return 0


def parse_repeat(text: str | None) -> Rule:
    """Turn 'daily', 'every 30 minutes', ... into a Rule. Raises ScheduleError."""
    raw = " ".join((text or "once").lower().split())
    if raw in ("", "once", "one-off", "one off", "never", "no"):
        return Rule("once", text="once")
    if raw in ("hourly", "every hour"):
        return Rule("interval", seconds=3600, text="hourly")
    if raw in ("daily", "every day", "everyday"):
        return Rule("days", step_days=1, text="daily")
    if raw in ("weekdays", "every weekday", "weekday"):
        return Rule("weekdays", text="weekdays")
    if raw in ("weekly", "every week"):
        return Rule("days", step_days=7, text="weekly")
    if raw in ("monthly", "every month"):
        return Rule("monthly", text="monthly")
    m = _EVERY.match(raw)
    if not m:
        raise ScheduleError(
            f"'{text}' is not a repeat rule. Use once, hourly, daily, weekdays, weekly, monthly, "
            "or 'every N minutes/hours/days' (at least 5 minutes)"
        )
    count, unit = int(m.group(1)), m.group(2).lower().rstrip("s")
    if count < 1:
        raise ScheduleError("the repeat count must be at least 1")
    if unit == "day":
        return Rule("days", step_days=count, text="daily" if count == 1 else f"every {count} days")
    seconds = count * _UNIT[unit]
    if seconds < MIN_INTERVAL_SECONDS:
        raise ScheduleError("the shortest repeat interval is 5 minutes")
    if seconds == 3600:
        return Rule("interval", seconds=3600, text="hourly")
    label = f"every {count} {'minute' if unit.startswith('min') else 'hour'}{'' if count == 1 else 's'}"
    return Rule("interval", seconds=seconds, text=label)


# -- zones -----------------------------------------------------------------------------
def to_local(moment: datetime, tz: tzinfo | None) -> datetime:
    """An aware instant expressed in the zone (the OS local zone when tz is None)."""
    return moment.astimezone(tz)


def localize(naive: datetime, tz: tzinfo | None, fold: int = 0) -> datetime:
    """The aware instant for a naive wall-clock time in the zone.

    A wall time inside a spring-forward gap does not exist: it is moved forward by the
    size of the gap, so it runs just after the clocks jump. An ambiguous time (fall-back)
    resolves to its first occurrence.
    """
    naive = naive.replace(tzinfo=None)
    if tz is None:
        aware = naive.replace(fold=fold).astimezone()
        return aware
    aware = naive.replace(tzinfo=tz, fold=fold)
    round_trip = aware.astimezone(UTC).astimezone(tz)
    if round_trip.replace(tzinfo=None) != naive:  # nonexistent wall time
        return round_trip
    return aware


def _u(moment: datetime) -> datetime:
    """The instant in UTC. Always compare and subtract through this: two aware datetimes that
    share a tzinfo are compared by wall clock, which is wrong inside a repeated DST hour."""
    return moment.astimezone(UTC)


def utc_iso(moment: datetime) -> str:
    return moment.astimezone(UTC).replace(microsecond=0).isoformat()


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


# -- the arithmetic -------------------------------------------------------------------
def _month_days(anchor_day: int, year: int, month: int) -> int:
    return min(anchor_day, calendar.monthrange(year, month)[1])


def _dates(rule: Rule, anchor: date, start: date) -> Iterator[date]:
    """Matching dates, ascending, beginning no later than `start`."""
    if rule.kind == "days":
        step = rule.step_days
        k = max(0, (start - anchor).days // step - 1)
        while True:
            yield anchor + timedelta(days=k * step)
            k += 1
    elif rule.kind == "weekdays":
        d = max(anchor, start - timedelta(days=1))
        while True:
            if d.weekday() < 5:
                yield d
            d += timedelta(days=1)
    elif rule.kind == "monthly":
        year, month = anchor.year, anchor.month
        if (start.year, start.month) > (year, month):
            year, month = start.year, start.month
            month -= 1  # one month of slack before `start`
            if month == 0:
                year, month = year - 1, 12
        while True:
            yield date(year, month, _month_days(anchor.day, year, month))
            month += 1
            if month == 13:
                year, month = year + 1, 1
    else:  # pragma: no cover - guarded by the callers
        raise ScheduleError(f"no calendar dates for rule {rule.kind}")


def next_after(rule: Rule, anchor: datetime, after: datetime, tz: tzinfo | None = None) -> datetime | None:
    """First run strictly later than `after` (an aware instant), or None for a finished one-off.

    `anchor` is the naive local wall-clock time of the first run.
    """
    anchor = anchor.replace(tzinfo=None, microsecond=0)
    first = localize(anchor, tz)
    after_u, first_u = _u(after), _u(first)
    if rule.kind == "once":
        return first if first_u > after_u else None
    if rule.kind == "interval":
        if first_u > after_u:
            return first
        steps = int((after_u - first_u).total_seconds() // rule.seconds) + 1
        return to_local(first_u + timedelta(seconds=steps * rule.seconds), tz)
    start = to_local(after, tz).date()
    clock = anchor.time()
    for day in _dates(rule, anchor.date(), start):
        candidate = localize(datetime.combine(day, clock), tz)
        if _u(candidate) > after_u and _u(candidate) >= first_u:
            return candidate
    return None  # pragma: no cover - the generators never end


def first_run(rule: Rule, anchor: datetime, tz: tzinfo | None = None) -> datetime:
    """The first run: the anchor itself, or the next matching date for calendar rules
    (a weekday rule anchored on a Saturday starts on Monday)."""
    anchor_instant = localize(anchor, tz)
    result = next_after(rule, anchor, _u(anchor_instant) - timedelta(microseconds=1), tz)
    return result if result is not None else anchor_instant


def clock_looks_wrong(rule: Rule, next_run: datetime, now: datetime, slack: timedelta = timedelta(hours=2)) -> bool:
    """True when a stored next run is further away than the rule could ever put it:
    the system clock must have been set backwards since it was computed."""
    if not rule.repeats:
        return False
    return _u(next_run) - _u(now) > timedelta(seconds=rule.period_seconds) + slack
