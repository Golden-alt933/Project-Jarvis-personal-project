"""Scheduled tasks in SQLite: plain functions over the shared memory database.

Times in the database are UTC ISO strings (``2026-11-10T04:30:00+00:00``), so they sort
as text and survive a change of time zone. ``anchor`` is the first run as the naive
local wall-clock time the user meant, which is what calendar rules repeat.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

KEEP_RUNS = 10

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS scheduled_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        instruction TEXT NOT NULL,
        repeat_text TEXT NOT NULL DEFAULT 'once',
        anchor TEXT NOT NULL,
        next_run TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        force_run INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        last_run_at TEXT,
        last_status TEXT,
        run_count INTEGER NOT NULL DEFAULT 0)""",
    "CREATE INDEX IF NOT EXISTS idx_tasks_next ON scheduled_tasks(status, next_run)",
    """CREATE TABLE IF NOT EXISTS task_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL,
        scheduled_for TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        conversation_id TEXT,
        steps INTEGER NOT NULL DEFAULT 0)""",
    "CREATE INDEX IF NOT EXISTS idx_runs_task ON task_runs(task_id, id)",
)

ACTIVE, DONE, CANCELLED, MISSED = "active", "done", "cancelled", "missed"


def ready(store: Any) -> None:
    for ddl in _SCHEMA:
        store.ensure_schema(ddl)


def add_task(
    store: Any, *, title: str, instruction: str, repeat_text: str, anchor: datetime, next_run: str, created: str
) -> int:
    ready(store)
    rows = store.execute(
        "INSERT INTO scheduled_tasks (title, instruction, repeat_text, anchor, next_run, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
        (title, instruction, repeat_text, anchor.replace(microsecond=0, tzinfo=None).isoformat(), next_run, created),
    )
    return int(rows[0]["id"])


def get_task(store: Any, task_id: int) -> dict[str, Any] | None:
    ready(store)
    rows = store.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,))
    return dict(rows[0]) if rows else None


def list_tasks(store: Any, include_finished: bool = False) -> list[dict[str, Any]]:
    ready(store)
    where = "" if include_finished else "WHERE status = 'active'"
    rows = store.execute(f"SELECT * FROM scheduled_tasks {where} ORDER BY (next_run IS NULL), next_run, id LIMIT 200")
    return [dict(r) for r in rows]


def due_tasks(store: Any, now_utc: str) -> list[dict[str, Any]]:
    """Active tasks whose time has come (or that were asked to run now), oldest first."""
    ready(store)
    rows = store.execute(
        "SELECT * FROM scheduled_tasks WHERE status = 'active' AND (force_run = 1 OR next_run <= ?) "
        "ORDER BY COALESCE(next_run, ''), id",
        (now_utc,),
    )
    return [dict(r) for r in rows]


def cancel_task(store: Any, task_id: int) -> bool:
    ready(store)
    rows = store.execute(
        "UPDATE scheduled_tasks SET status = 'cancelled', next_run = NULL, force_run = 0 "
        "WHERE id = ? AND status = 'active' RETURNING id",
        (task_id,),
    )
    return bool(rows)


def request_run_now(store: Any, task_id: int) -> bool:
    ready(store)
    rows = store.execute(
        "UPDATE scheduled_tasks SET force_run = 1 WHERE id = ? AND status = 'active' RETURNING id", (task_id,)
    )
    return bool(rows)


def claim(
    store: Any, task: dict[str, Any], *, new_next: str | None, new_status: str, now_iso: str
) -> bool:
    """Take a due task for running and move it forward in one atomic step.

    The row only changes if it still has the next_run (and force flag) we saw, so a
    second process (the daemon next to a terminal) cannot run the same slot twice.
    Moving forward *before* the run makes it at-most-once: a crash mid-run never repeats.
    """
    rows = store.execute(
        "UPDATE scheduled_tasks SET next_run = ?, status = ?, force_run = 0, last_run_at = ?, "
        "run_count = run_count + 1 WHERE id = ? AND status = 'active' AND force_run = ? "
        "AND COALESCE(next_run, '') = ? RETURNING id",
        (new_next, new_status, now_iso, task["id"], task["force_run"], task["next_run"] or ""),
    )
    return bool(rows)


def advance_without_run(store: Any, task: dict[str, Any], *, new_next: str | None, new_status: str) -> bool:
    """Skip a slot (missed or deferred too long) without counting it as a run."""
    rows = store.execute(
        "UPDATE scheduled_tasks SET next_run = ?, status = ?, force_run = 0 "
        "WHERE id = ? AND status = 'active' AND COALESCE(next_run, '') = ? RETURNING id",
        (new_next, new_status, task["id"], task["next_run"] or ""),
    )
    return bool(rows)


def reschedule(store: Any, task_id: int, next_run: str) -> None:
    ready(store)
    store.execute("UPDATE scheduled_tasks SET next_run = ? WHERE id = ? AND status = 'active'", (next_run, task_id))


def start_run(store: Any, task_id: int, scheduled_for: str | None, started: str) -> int:
    ready(store)
    rows = store.execute(
        "INSERT INTO task_runs (task_id, scheduled_for, started_at, status) VALUES (?, ?, ?, 'running') RETURNING id",
        (task_id, scheduled_for, started),
    )
    return int(rows[0]["id"])


def finish_run(
    store: Any, run_id: int, *, status: str, summary: str, finished: str, conversation_id: str | None = None, steps: int = 0
) -> None:
    store.execute(
        "UPDATE task_runs SET status = ?, summary = ?, finished_at = ?, conversation_id = ?, steps = ? WHERE id = ?",
        (status, summary, finished, conversation_id, steps, run_id),
    )
    store.execute(
        "UPDATE scheduled_tasks SET last_status = ? WHERE id = (SELECT task_id FROM task_runs WHERE id = ?)",
        (status, run_id),
    )
    store.execute(
        "DELETE FROM task_runs WHERE task_id = (SELECT task_id FROM task_runs WHERE id = ?) AND id NOT IN "
        "(SELECT id FROM task_runs WHERE task_id = (SELECT task_id FROM task_runs WHERE id = ?) "
        f"ORDER BY id DESC LIMIT {KEEP_RUNS})",
        (run_id, run_id),
    )


def record_event(store: Any, task_id: int, *, status: str, summary: str, when: str, scheduled_for: str | None = None) -> None:
    """A run that never started (missed, skipped): one row with the reason."""
    ready(store)
    store.execute(
        "INSERT INTO task_runs (task_id, scheduled_for, started_at, finished_at, status, summary) VALUES (?, ?, ?, ?, ?, ?)",
        (task_id, scheduled_for, when, when, status, summary),
    )
    store.execute("UPDATE scheduled_tasks SET last_status = ? WHERE id = ?", (status, task_id))


def recent_runs(store: Any, task_id: int, limit: int = 3) -> list[dict[str, Any]]:
    ready(store)
    rows = store.execute("SELECT * FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT ?", (task_id, limit))
    return [dict(r) for r in rows]


def interrupted_runs(store: Any, cutoff_iso: str, now_iso: str) -> int:
    """Runs left 'running' since before the cutoff by a crash or a stopped process: mark
    them so they are not reported as in progress forever. Returns how many."""
    ready(store)
    rows = store.execute(
        "UPDATE task_runs SET status = 'error', summary = 'Interrupted: Jarvis stopped before this run finished.', "
        "finished_at = ? WHERE status = 'running' AND started_at < ? RETURNING id",
        (now_iso, cutoff_iso),
    )
    return len(rows)


def counts(store: Any) -> tuple[int, str | None]:
    """(active tasks, earliest next run as UTC ISO or None) for the doctor."""
    ready(store)
    rows = store.execute("SELECT COUNT(*) AS n, MIN(next_run) AS nxt FROM scheduled_tasks WHERE status = 'active'")
    return int(rows[0]["n"]), rows[0]["nxt"]
