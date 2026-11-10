"""Notes and reminders, stored in the shared SQLite memory database.

Times are local, naive ISO strings ("2026-11-10T18:30:00"), which sort correctly
as text. ``parse_when`` is a deliberately strict grammar (ported from Friday's
utils/when.py): anything it does not recognise is refused rather than guessed,
because a reminder at the wrong time is worse than none.

``due_reminders`` and ``mark_reminder_fired`` are plain functions for the future
scheduler, not tools.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any

from jarvis.tools.base import Risk, ToolContext, ToolResult, tool

log = logging.getLogger(__name__)

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL DEFAULT '',
        body TEXT NOT NULL,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        due_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        fired_at TEXT,
        cancelled INTEGER NOT NULL DEFAULT 0)""",
    "CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_at)",
)

_NO_MEMORY = ToolResult.fail("the memory database is not available", "Tell the user notes and reminders cannot be used right now.")


def _ready(store: Any) -> None:
    for ddl in _SCHEMA:
        store.ensure_schema(ddl)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0, tzinfo=None).isoformat()


def _pretty(iso: str) -> str:
    try:
        return f"{datetime.fromisoformat(iso):%a %d %b %Y %H:%M}"
    except ValueError:
        return iso


# --------------------------------------------------------------------------
# time parsing
# --------------------------------------------------------------------------
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty five": 45, "fortyfive": 45,
    "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90,
}
_UNIT_SECONDS = {
    "second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600,
    "day": 86400, "week": 604800,
}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_HALF_RE = re.compile(r"\bin\s+half\s+an?\s+(hour|minute)\b", re.I)
_OFFSET_RE = re.compile(
    r"\bin\s+(\d+|[a-z]+(?:\s+[a-z]+)?)\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)\b", re.I
)
# "9am", "6:45 pm", "18:30", or "at 9" -- a bare number only counts after "at".
_CLOCK_RE = re.compile(
    r"(?:\bat\s+)?\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?(?![\d:])", re.I
)
_TOMORROW_RE = re.compile(r"\btomorrow\b", re.I)
_TODAY_RE = re.compile(r"\btoday\b", re.I)
_TONIGHT_RE = re.compile(r"\b(tonight|this evening)\b", re.I)
_NOON_RE = re.compile(r"\b(noon|midday|midnight)\b", re.I)
_WEEKDAY_RE = re.compile(r"\b(?:next\s+|on\s+)?(" + "|".join(_WEEKDAYS) + r")\b", re.I)

MAX_OFFSET = timedelta(days=365)
DEFAULT_MORNING_HOUR = 9
DEFAULT_EVENING_HOUR = 20


class WhenError(ValueError):
    """The text could not be turned into a usable future time."""


def _number(token: str) -> int | None:
    token = token.strip().lower()
    return int(token) if token.isdigit() else _WORD_NUMBERS.get(token)


def _find_clock(text: str) -> tuple[int, int] | None:
    """The (hour, minute) named in text, or None. Raises WhenError for impossible times."""
    for m in _CLOCK_RE.finditer(text):
        has_at = m.group(0).lower().startswith("at")
        hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "").lower()
        # A bare integer ("in 20 minutes", "May 5") is not a clock time.
        if not (has_at or mer or m.group(2)):
            continue
        if minute > 59 or (mer and not 1 <= hour <= 12) or hour > 23:
            raise WhenError(f"'{m.group(0).strip()}' is not a valid clock time")
        if mer == "pm" and hour < 12:
            hour += 12
        elif mer == "am" and hour == 12:
            hour = 0
        return hour, minute
    return None


def parse_when(text: str, now: datetime) -> datetime:
    """Resolve an ISO-8601 local datetime or a relative expression to a future datetime.

    Raises WhenError (never guesses) when the text is unrecognised or in the past.
    A bare clock time that already passed today ("at 9am" at 14:30) means tomorrow;
    an explicit day ("today 9am", an ISO timestamp) in the past is refused.
    """
    raw = (text or "").strip()
    if not raw:
        raise WhenError("no time given")
    now = now.replace(tzinfo=None, microsecond=0)

    try:  # ISO-8601, optionally with an offset (converted to local time)
        iso: datetime | None = datetime.fromisoformat(raw)
    except ValueError:
        iso = None
    if iso is not None:
        if iso.tzinfo is not None:
            iso = iso.astimezone().replace(tzinfo=None)
        if iso <= now:
            raise WhenError(f"{_iso(iso)} is in the past")
        return iso

    if m := _HALF_RE.search(raw):
        return now + timedelta(minutes=30 if m.group(1).lower() == "hour" else 0.5)

    if m := _OFFSET_RE.search(raw):
        count, unit = _number(m.group(1)), _UNIT_SECONDS.get(m.group(2).lower().rstrip("s"))
        if not count or not unit:
            raise WhenError(f"could not understand the duration '{m.group(0)}'")
        delta = timedelta(seconds=count * unit)
        if delta > MAX_OFFSET:
            raise WhenError("that is more than a year away")
        return now + delta

    clock = _find_clock(raw)
    if clock is None and (n := _NOON_RE.search(raw)):
        clock = (0, 0) if n.group(1).lower() == "midnight" else (12, 0)
    day_named = False
    base = now
    if _TOMORROW_RE.search(raw):
        base, day_named = now + timedelta(days=1), True
    elif _TODAY_RE.search(raw):
        day_named = True
    elif w := _WEEKDAY_RE.search(raw):
        ahead = (_WEEKDAYS.index(w.group(1).lower()) - now.weekday()) % 7 or 7
        base, day_named = now + timedelta(days=ahead), True
    elif _TONIGHT_RE.search(raw):
        clock = clock or (DEFAULT_EVENING_HOUR, 0)
    if clock is None:
        if not day_named or _TODAY_RE.search(raw):
            raise WhenError(f"could not understand the time '{raw}'")
        clock = (DEFAULT_MORNING_HOUR, 0)  # a named day with no time

    target = base.replace(hour=clock[0], minute=clock[1], second=0)
    if target <= now:
        if day_named:
            raise WhenError(f"{_iso(target)} is in the past")
        target += timedelta(days=1)
    return target


# --------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------
@tool(group="core")
def add_note(text: str, ctx: ToolContext, title: str | None = None) -> ToolResult:
    """Save a note for the user (ideas, lists, things to remember that are not
    time-based). Notes persist across sessions. For something that must happen at
    a time, use set_reminder instead.

    Args:
        text: The note content.
        title: Optional short title.
    """
    if ctx.memory is None:
        return _NO_MEMORY
    if not text.strip():
        return ToolResult.fail("the note is empty")
    _ready(ctx.memory)
    rows = ctx.memory.execute(
        "INSERT INTO notes (title, body, created_at) VALUES (?, ?, ?) RETURNING id",
        (title.strip() if title else "", text.strip(), _iso(datetime.now())),
    )
    note_id = rows[0]["id"] if rows else None
    return ToolResult.success(f"Saved note #{note_id}.", data={"id": note_id})


def _note_line(row: Any) -> str:
    title = f"{row['title']}: " if row["title"] else ""
    body = row["body"] if len(row["body"]) <= 200 else row["body"][:199] + "…"
    return f"#{row['id']} [{_pretty(row['created_at'])}] {title}{body}"


@tool(group="core")
def list_notes(ctx: ToolContext, limit: int = 20) -> ToolResult:
    """List the user's saved notes, newest first.

    Args:
        limit: Maximum notes to return (1-100).
    """
    if ctx.memory is None:
        return _NO_MEMORY
    _ready(ctx.memory)
    rows = ctx.memory.execute("SELECT * FROM notes ORDER BY id DESC LIMIT ?", (max(1, min(limit, 100)),))
    if not rows:
        return ToolResult.success("No notes saved.", data=[])
    return ToolResult.success("\n".join(_note_line(r) for r in rows), data=[dict(r) for r in rows])


@tool(group="core")
def search_notes(query: str, ctx: ToolContext) -> ToolResult:
    """Find saved notes whose title or text contains the query words.

    Args:
        query: Words to look for; all must appear (any order, case-insensitive).
    """
    if ctx.memory is None:
        return _NO_MEMORY
    words = query.split()
    if not words:
        return ToolResult.fail("empty query")
    _ready(ctx.memory)
    clause = " AND ".join("(LOWER(title) LIKE ? OR LOWER(body) LIKE ?)" for _ in words)
    params = [p for w in words for p in (f"%{w.lower()}%",) * 2]
    rows = ctx.memory.execute(f"SELECT * FROM notes WHERE {clause} ORDER BY id DESC LIMIT 20", params)
    if not rows:
        return ToolResult.success(f"No notes match '{query}'.", data=[])
    return ToolResult.success("\n".join(_note_line(r) for r in rows), data=[dict(r) for r in rows])


@tool(group="core", risk=Risk.CONFIRM, describe=lambda a: f"Delete note #{a.get('note_id')}")
def delete_note(note_id: int, ctx: ToolContext) -> ToolResult:
    """Permanently delete a saved note by its number (shown by list_notes or search_notes).

    Args:
        note_id: The note's number.
    """
    if ctx.memory is None:
        return _NO_MEMORY
    _ready(ctx.memory)
    if not ctx.memory.execute("SELECT id FROM notes WHERE id = ?", (note_id,)):
        return ToolResult.fail(f"no note #{note_id}", "Use list_notes to see valid numbers.")
    ctx.memory.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    return ToolResult.success(f"Deleted note #{note_id}.")


# --------------------------------------------------------------------------
# reminders
# --------------------------------------------------------------------------
_WHEN_HELP = (
    'Use an ISO local datetime like "2026-11-10T18:30", or "in 20 minutes", "in 2 hours", '
    '"tomorrow 9am", "tomorrow", "at 18:30", "tonight", "friday 3pm"'
)


@tool(group="core")
def set_reminder(message: str, when: str, ctx: ToolContext) -> ToolResult:
    """Schedule a reminder that Jarvis will deliver at a future time.

    Use for "remind me to ... in/at/tomorrow ...". If the user gave no usable
    time, ask them rather than guessing. Returns the reminder number and the
    exact time it was set for; repeat that time back to the user.

    Args:
        message: What to remind the user about, phrased as the thing to do ("call mum").
        when: ISO-8601 local datetime, or a relative expression such as "in 20 minutes", "tomorrow 9am", "at 18:30".
    """
    if ctx.memory is None:
        return _NO_MEMORY
    if not message.strip():
        return ToolResult.fail("the reminder message is empty")
    now = datetime.now()
    try:
        due = parse_when(when, now)
    except WhenError as exc:
        return ToolResult.fail(
            f"could not set the reminder: {exc}",
            f"{_WHEN_HELP}. The current local time is {now:%Y-%m-%dT%H:%M} ({now:%A}).",
        )
    _ready(ctx.memory)
    rows = ctx.memory.execute(
        "INSERT INTO reminders (message, due_at, created_at) VALUES (?, ?, ?) RETURNING id",
        (message.strip(), _iso(due), _iso(now)),
    )
    rid = rows[0]["id"] if rows else None
    return ToolResult.success(
        f"Reminder #{rid} set for {_pretty(_iso(due))}: {message.strip()}",
        data={"id": rid, "due_at": _iso(due)},
    )


@tool(group="core")
def list_reminders(ctx: ToolContext, include_done: bool = False) -> ToolResult:
    """List the user's reminders, soonest first. By default only pending ones.

    Args:
        include_done: Also show reminders that already fired or were cancelled.
    """
    if ctx.memory is None:
        return _NO_MEMORY
    _ready(ctx.memory)
    where = "" if include_done else "WHERE fired_at IS NULL AND cancelled = 0"
    rows = ctx.memory.execute(f"SELECT * FROM reminders {where} ORDER BY due_at LIMIT 100")
    if not rows:
        return ToolResult.success("No pending reminders." if not include_done else "No reminders.", data=[])

    def line(r: Any) -> str:
        state = "cancelled" if r["cancelled"] else "done" if r["fired_at"] else "pending"
        return f"#{r['id']} {_pretty(r['due_at'])} [{state}] {r['message']}"

    return ToolResult.success("\n".join(line(r) for r in rows), data=[dict(r) for r in rows])


@tool(group="core")
def cancel_reminder(reminder_id: int, ctx: ToolContext) -> ToolResult:
    """Cancel a pending reminder by its number (from list_reminders).

    Args:
        reminder_id: The reminder's number.
    """
    if ctx.memory is None:
        return _NO_MEMORY
    _ready(ctx.memory)
    rows = ctx.memory.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
    if not rows:
        return ToolResult.fail(f"no reminder #{reminder_id}", "Use list_reminders to see valid numbers.")
    if rows[0]["cancelled"] or rows[0]["fired_at"]:
        return ToolResult.fail(f"reminder #{reminder_id} is already {'cancelled' if rows[0]['cancelled'] else 'done'}")
    ctx.memory.execute("UPDATE reminders SET cancelled = 1 WHERE id = ?", (reminder_id,))
    return ToolResult.success(f"Cancelled reminder #{reminder_id}: {rows[0]['message']}")


# -- scheduler API (not tools) ----------------------------------------------
def due_reminders(store: Any, now: datetime) -> list[dict[str, Any]]:
    """Pending reminders whose time has come, oldest first."""
    _ready(store)
    rows = store.execute(
        "SELECT * FROM reminders WHERE fired_at IS NULL AND cancelled = 0 AND due_at <= ? ORDER BY due_at",
        (_iso(now),),
    )
    return [dict(r) for r in rows]


def mark_reminder_fired(store: Any, reminder_id: int) -> None:
    """Record that a reminder was delivered so it is not delivered again."""
    _ready(store)
    store.execute("UPDATE reminders SET fired_at = ? WHERE id = ?", (_iso(datetime.now()), reminder_id))
