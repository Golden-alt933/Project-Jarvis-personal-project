"""Schedule arithmetic: a table of next-run cases, including DST in several zones,
month ends, leap days and the rule parser. No clock, no database."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from jarvis.proactive.schedule_math import (
    ScheduleError,
    clock_looks_wrong,
    first_run,
    localize,
    next_after,
    parse_repeat,
)

UTC = timezone.utc
NY = ZoneInfo("America/New_York")  # spring forward 2026-03-08 02:00, fall back 2026-11-01 02:00
LON = ZoneInfo("Europe/London")  # spring forward 2026-03-29 01:00
SYD = ZoneInfo("Australia/Sydney")  # fall back 2026-04-05 03:00 (southern hemisphere)
KOL = ZoneInfo("Asia/Kolkata")  # no DST, half-hour offset


def u(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def local(tz, *parts: int) -> datetime:
    return datetime(*parts, tzinfo=tz)


def naive(*parts: int) -> datetime:
    return datetime(*parts)


# (rule, anchor wall time, after, zone, expected instant)
CASES = [
    # --- daily across the NY spring-forward gap -------------------------------------------
    ("daily", naive(2026, 3, 1, 9), local(NY, 2026, 3, 7, 12), NY, local(NY, 2026, 3, 8, 9)),
    ("daily", naive(2026, 3, 5, 2, 30), local(NY, 2026, 3, 7, 3), NY, u(2026, 3, 8, 7, 30)),  # 02:30 does not exist: 03:30 EDT
    ("daily", naive(2026, 3, 5, 2, 30), u(2026, 3, 8, 7, 30), NY, local(NY, 2026, 3, 9, 2, 30)),  # next day it exists again
    ("daily", naive(2026, 3, 1, 9), u(2026, 3, 8, 13), NY, local(NY, 2026, 3, 9, 9)),  # strictly after
    # --- daily across the NY fall-back repeat -----------------------------------------------
    ("daily", naive(2026, 10, 30, 1, 30), local(NY, 2026, 10, 31, 2), NY, u(2026, 11, 1, 5, 30)),  # first 01:30 (EDT)
    ("daily", naive(2026, 10, 30, 1, 30), u(2026, 11, 1, 5, 30), NY, u(2026, 11, 2, 6, 30)),  # not the second 01:30 (EST)
    ("daily", naive(2026, 10, 30, 9), local(NY, 2026, 10, 31, 9), NY, local(NY, 2026, 11, 1, 9)),  # 9 am stays 9 am
    # --- other zones -------------------------------------------------------------------------
    ("daily", naive(2026, 3, 20, 1, 30), local(LON, 2026, 3, 28, 2), LON, u(2026, 3, 29, 1, 30)),  # London gap: 02:30 BST
    ("daily", naive(2026, 3, 20, 9), local(LON, 2026, 3, 28, 12), LON, local(LON, 2026, 3, 29, 9)),
    ("daily", naive(2026, 4, 1, 2, 30), local(SYD, 2026, 4, 4, 10), SYD, u(2026, 4, 4, 15, 30)),  # Sydney repeat: first 02:30 (+11)
    ("daily", naive(2026, 4, 1, 2, 30), u(2026, 4, 4, 15, 30), SYD, u(2026, 4, 5, 16, 30)),  # then the next day (+10)
    ("daily", naive(2026, 6, 1, 7, 15), local(KOL, 2026, 6, 1, 8), KOL, local(KOL, 2026, 6, 2, 7, 15)),
    # --- absolute intervals do not follow the wall clock ---------------------------------------
    ("hourly", naive(2026, 3, 8, 0), local(NY, 2026, 3, 8, 1, 30), NY, u(2026, 3, 8, 7)),  # 06:30Z -> 07:00Z = 03:00 EDT
    ("hourly", naive(2026, 11, 1, 0), u(2026, 11, 1, 5, 30), NY, u(2026, 11, 1, 6)),  # 01:00 EST, an hour later in real time
    ("every 90 minutes", naive(2026, 1, 1, 8), local(NY, 2026, 1, 1, 9), NY, local(NY, 2026, 1, 1, 9, 30)),
    ("every 2 hours", naive(2026, 6, 1, 8), local(NY, 2026, 6, 1, 10), NY, local(NY, 2026, 6, 1, 12)),  # on the grid: the next one
    ("every 5 minutes", naive(2026, 6, 1, 10), local(NY, 2026, 6, 1, 10, 3), NY, local(NY, 2026, 6, 1, 10, 5)),
    ("every 15 minutes", naive(2026, 6, 1, 10), local(NY, 2026, 6, 3, 10, 7), NY, local(NY, 2026, 6, 3, 10, 15)),
    ("every 6 hours", naive(2026, 6, 1, 6), local(NY, 2026, 5, 1), NY, local(NY, 2026, 6, 1, 6)),  # before the anchor: the anchor
    # --- weekdays ------------------------------------------------------------------------------------
    ("weekdays", naive(2026, 6, 1, 18), local(NY, 2026, 6, 5, 18), NY, local(NY, 2026, 6, 8, 18)),  # Friday -> Monday
    ("weekdays", naive(2026, 6, 1, 18), local(NY, 2026, 6, 5, 17), NY, local(NY, 2026, 6, 5, 18)),
    ("weekdays", naive(2026, 6, 1, 18), local(NY, 2026, 6, 6, 9), NY, local(NY, 2026, 6, 8, 18)),  # Saturday
    ("weekdays", naive(2026, 6, 1, 18), local(NY, 2026, 6, 3, 19), NY, local(NY, 2026, 6, 4, 18)),
    # --- weekly --------------------------------------------------------------------------------------
    ("weekly", naive(2026, 6, 2, 10), local(NY, 2026, 6, 2, 10), NY, local(NY, 2026, 6, 9, 10)),
    ("weekly", naive(2026, 3, 1, 10), local(NY, 2026, 3, 8, 10), NY, local(NY, 2026, 3, 15, 10)),  # across the DST change
    ("weekly", naive(2026, 6, 2, 10), local(NY, 2026, 6, 20, 0), NY, local(NY, 2026, 6, 23, 10)),
    # --- monthly: month ends and leap days -----------------------------------------------------------
    ("monthly", naive(2026, 1, 31, 9), local(NY, 2026, 1, 31, 9), NY, local(NY, 2026, 2, 28, 9)),
    ("monthly", naive(2026, 1, 31, 9), local(NY, 2026, 2, 28, 9), NY, local(NY, 2026, 3, 31, 9)),  # the 31st comes back
    ("monthly", naive(2028, 1, 31, 9), local(NY, 2028, 1, 31, 9), NY, local(NY, 2028, 2, 29, 9)),  # leap year
    ("monthly", naive(2026, 4, 30, 9), local(NY, 2026, 4, 30, 9), NY, local(NY, 2026, 5, 30, 9)),
    ("monthly", naive(2026, 12, 15, 10), local(NY, 2026, 12, 15, 10), NY, local(NY, 2027, 1, 15, 10)),  # year rollover
    ("monthly", naive(2026, 3, 8, 2, 30), local(NY, 2026, 3, 8, 12), NY, local(NY, 2026, 4, 8, 2, 30)),
    ("monthly", naive(2026, 1, 15, 9), local(NY, 2030, 6, 1), NY, local(NY, 2030, 6, 15, 9)),  # far from the anchor
    # --- daily at month, year and leap boundaries ------------------------------------------------------
    ("daily", naive(2028, 2, 28, 9), local(NY, 2028, 2, 28, 10), NY, local(NY, 2028, 2, 29, 9)),  # leap day exists
    ("daily", naive(2028, 2, 28, 9), local(NY, 2028, 2, 29, 10), NY, local(NY, 2028, 3, 1, 9)),
    ("daily", naive(2026, 2, 27, 9), local(NY, 2026, 2, 28, 10), NY, local(NY, 2026, 3, 1, 9)),  # no leap day in 2026
    ("daily", naive(2026, 1, 1, 9), local(NY, 2026, 1, 31, 10), NY, local(NY, 2026, 2, 1, 9)),
    ("daily", naive(2026, 1, 1, 23, 30), local(NY, 2026, 12, 31, 23, 45), NY, local(NY, 2027, 1, 1, 23, 30)),
    ("daily", naive(2026, 6, 1, 9), local(NY, 2026, 5, 1), NY, local(NY, 2026, 6, 1, 9)),  # before the anchor
    # --- every N days --------------------------------------------------------------------------------------
    ("every 3 days", naive(2026, 6, 1, 9), local(NY, 2026, 6, 1, 9), NY, local(NY, 2026, 6, 4, 9)),
    ("every 3 days", naive(2026, 6, 1, 9), local(NY, 2026, 6, 5), NY, local(NY, 2026, 6, 7, 9)),
    ("every 3 days", naive(2026, 1, 1, 9), local(NY, 2026, 6, 15, 10), NY, local(NY, 2026, 6, 18, 9)),
    ("every 2 days", naive(2026, 3, 6, 9), local(NY, 2026, 3, 6, 10), NY, local(NY, 2026, 3, 8, 9)),  # lands on the DST day
    # --- one-offs ----------------------------------------------------------------------------------------------
    ("once", naive(2026, 6, 1, 9), local(NY, 2026, 5, 31), NY, local(NY, 2026, 6, 1, 9)),
    ("once", naive(2026, 6, 1, 9), local(NY, 2026, 6, 1, 9), NY, None),
    ("once", naive(2026, 6, 1, 9), local(NY, 2026, 6, 2), NY, None),
    ("once", naive(2026, 3, 8, 2, 30), local(NY, 2026, 3, 1), NY, u(2026, 3, 8, 7, 30)),  # a one-off inside the gap
]


@pytest.mark.parametrize(("rule", "anchor", "after", "tz", "expected"), CASES)
def test_next_after_table(rule, anchor, after, tz, expected):
    result = next_after(parse_repeat(rule), anchor, after, tz)
    if expected is None:
        assert result is None
    else:
        # compare as instants: aware values sharing a zone compare by wall clock inside a repeated hour
        assert result.astimezone(UTC) == expected.astimezone(UTC), f"{result.astimezone(UTC)} != {expected.astimezone(UTC)}"
        assert result.astimezone(UTC) > after.astimezone(UTC)


def test_the_table_is_large_enough():
    assert len(CASES) >= 30


def test_repeated_calls_from_now_never_burst():
    """Catch-up safety: asking from 'now' after any downtime yields one future slot."""
    rule, anchor = parse_repeat("every 15 minutes"), naive(2026, 6, 1, 10)
    after = local(NY, 2026, 6, 20, 12, 7)  # nineteen days of downtime
    nxt = next_after(rule, anchor, after, NY)
    assert nxt == local(NY, 2026, 6, 20, 12, 15)
    assert nxt.astimezone(UTC) - after.astimezone(UTC) < timedelta(minutes=15)


def test_runs_stay_ordered_and_distinct_over_a_year_of_dst():
    rule, anchor = parse_repeat("daily"), naive(2026, 1, 1, 2, 30)  # 02:30 meets both DST oddities
    cursor, seen = local(NY, 2025, 12, 31), []
    for _ in range(366):
        cursor = next_after(rule, anchor, cursor, NY)
        seen.append(cursor)
    instants = [m.astimezone(UTC) for m in seen]
    assert instants == sorted(set(instants))
    assert len(instants) == 366


@pytest.mark.parametrize(
    ("rule", "anchor", "expected"),
    [
        ("weekdays", naive(2026, 6, 6, 9), local(NY, 2026, 6, 8, 9)),  # a Saturday anchor starts on Monday
        ("weekdays", naive(2026, 6, 7, 9), local(NY, 2026, 6, 8, 9)),  # a Sunday anchor too
        ("weekdays", naive(2026, 6, 3, 9), local(NY, 2026, 6, 3, 9)),
        ("daily", naive(2026, 6, 6, 9), local(NY, 2026, 6, 6, 9)),
        ("once", naive(2026, 6, 6, 9), local(NY, 2026, 6, 6, 9)),
        ("hourly", naive(2026, 6, 6, 9), local(NY, 2026, 6, 6, 9)),
    ],
)
def test_first_run(rule, anchor, expected):
    assert first_run(parse_repeat(rule), anchor, NY).astimezone(UTC) == expected.astimezone(UTC)


def test_localize_system_zone_keeps_the_wall_time():
    moment = localize(datetime(2026, 6, 1, 9, 0), None)
    assert moment.tzinfo is not None
    assert moment.replace(tzinfo=None) == datetime(2026, 6, 1, 9, 0)


def test_localize_gap_and_fold():
    assert localize(naive(2026, 3, 8, 2, 30), NY).astimezone(UTC) == u(2026, 3, 8, 7, 30)
    first = localize(naive(2026, 11, 1, 1, 30), NY, fold=0)
    second = localize(naive(2026, 11, 1, 1, 30), NY, fold=1)
    assert second.astimezone(UTC) - first.astimezone(UTC) == timedelta(hours=1)


# -- repeat parsing ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "kind", "text_out"),
    [
        ("once", "once", "once"),
        (None, "once", "once"),
        ("Daily", "days", "daily"),
        ("  every   day ", "days", "daily"),
        ("weekdays", "weekdays", "weekdays"),
        ("weekly", "days", "weekly"),
        ("monthly", "monthly", "monthly"),
        ("hourly", "interval", "hourly"),
        ("every 1 hour", "interval", "hourly"),
        ("every 30 minutes", "interval", "every 30 minutes"),
        ("every 5 min", "interval", "every 5 minutes"),
        ("every 2 hours", "interval", "every 2 hours"),
        ("every 3 days", "days", "every 3 days"),
        ("EVERY 2 HRS", "interval", "every 2 hours"),
    ],
)
def test_parse_repeat_accepts(text, kind, text_out):
    rule = parse_repeat(text)
    assert (rule.kind, rule.text) == (kind, text_out)


@pytest.mark.parametrize("text", ["every 4 minutes", "every 1 minute", "every 0 hours", "sometimes", "every minute", "fortnightly", "every -2 days"])
def test_parse_repeat_rejects(text):
    with pytest.raises(ScheduleError):
        parse_repeat(text)


def test_minimum_interval_is_five_minutes():
    assert parse_repeat("every 5 minutes").seconds == 300
    with pytest.raises(ScheduleError, match="5 minutes"):
        parse_repeat("every 3 minutes")


# -- clock jumps ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("rule", "next_run", "now", "wrong"),
    [
        ("daily", u(2026, 6, 2, 9), u(2026, 6, 1, 10), False),  # normal
        ("daily", u(2026, 6, 2, 9), u(2026, 5, 20, 10), True),  # the clock went back by days
        ("hourly", u(2026, 6, 1, 11), u(2026, 6, 1, 10, 30), False),
        ("hourly", u(2026, 6, 1, 11), u(2026, 6, 1, 4), True),  # set back by six hours
        ("weekdays", u(2026, 6, 8, 18), u(2026, 6, 5, 19), False),  # Friday evening -> Monday
        ("monthly", u(2026, 3, 31, 9), u(2026, 2, 28, 10), False),
        ("once", u(2027, 1, 1), u(2026, 1, 1), False),  # a far one-off is simply far
    ],
)
def test_clock_looks_wrong(rule, next_run, now, wrong):
    assert clock_looks_wrong(parse_repeat(rule), next_run, now) is wrong
