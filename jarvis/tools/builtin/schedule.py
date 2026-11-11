"""Scheduled tasks: requests Jarvis carries out by itself at a set time.

A reminder only *says* something at a time (``set_reminder``). A scheduled task is a
request in plain language that Jarvis *does* with its tools when the time comes, once or
on a repeat, while no one is watching. The scheduler that runs them lives in
``jarvis/proactive/tasks.py``; these tools only store and inspect tasks.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from jarvis.proactive import taskstore as ts
from jarvis.proactive import tasks as sched
from jarvis.proactive.schedule_math import ScheduleError, Rule, first_run, parse_repeat, utc_iso
from jarvis.tools.base import Risk, ToolContext, ToolResult, tool
from jarvis.tools.builtin.productivity import _WHEN_HELP, WhenError, parse_when

_NO_MEMORY = ToolResult.fail(
    "the memory database is not available", "Tell the user scheduled tasks cannot be used right now."
)
TITLE_LIMIT = 60


def _rule(repeat: str) -> Rule | None:
    try:
        return parse_repeat(repeat)
    except ScheduleError:
        return None


def _risk(args: dict[str, Any]) -> Risk:
    """Asking before a task that would run more often than hourly is created."""
    rule = _rule(str(args.get("repeat") or "once"))
    if rule is not None and rule.kind == "interval" and rule.seconds < 3600:
        return Risk.CONFIRM
    return Risk.SAFE


def _describe(args: dict[str, Any]) -> str:
    rule = _rule(str(args.get("repeat") or "once"))
    every = rule.text if rule else str(args.get("repeat"))
    return f"Schedule a task to run {every} (more often than hourly): {str(args.get('instruction', ''))[:200]}"


def _default_title(instruction: str) -> str:
    one_line = " ".join(instruction.split())
    return one_line if len(one_line) <= TITLE_LIMIT else one_line[: TITLE_LIMIT - 1] + "…"


@tool(group="core", risk=_risk, describe=_describe)
def schedule_task(
    instruction: str,
    when: str,
    ctx: ToolContext,
    repeat: Literal["once", "hourly", "daily", "weekdays", "weekly", "monthly"] | str = "once",
    title: str | None = None,
) -> ToolResult:
    """Schedule a TASK: something Jarvis will DO by itself later, with its tools, once or on a repeat.

    Use for "every morning at nine, brief me", "in an hour, check whether my download
    finished", "every weekday at 6 pm remind me to log my hours and open the timesheet".
    The instruction is carried out when the time comes with nobody watching, so write it as
    a complete request. Use set_reminder instead when the user only wants to be TOLD
    something at a time (a plain message, no action). Asks the user to confirm a repeat more
    often than hourly. Tell the user the next run time from the result.

    Args:
        instruction: What to do, in plain language, complete enough to act on without asking questions ("check the weather in Pune and tell me whether to take an umbrella").
        when: When it first runs: ISO local datetime, or "in 20 minutes", "tomorrow 9am", "at 18:30", "friday 3pm".
        repeat: once, hourly, daily, weekdays, weekly, monthly, or "every N minutes/hours/days" (at least 5 minutes). Repeats keep the clock time of the first run.
        title: Optional short name; defaults to the start of the instruction.
    """
    if ctx.memory is None:
        return _NO_MEMORY
    if not instruction.strip():
        return ToolResult.fail("the instruction is empty")
    try:
        rule = parse_repeat(repeat)
    except ScheduleError as exc:
        return ToolResult.fail(str(exc), "Use once, hourly, daily, weekdays, weekly, monthly, or 'every N minutes/hours/days'.")
    now = sched.local_now()
    try:
        anchor = parse_when(when, now.replace(tzinfo=None))
    except WhenError as exc:
        return ToolResult.fail(
            f"could not schedule the task: {exc}",
            f"{_WHEN_HELP}. The current local time is {now:%Y-%m-%dT%H:%M} ({now:%A}).",
        )
    first = first_run(rule, anchor)
    task_id = ts.add_task(
        ctx.memory,
        title=(title or "").strip() or _default_title(instruction),
        instruction=instruction.strip(),
        repeat_text=rule.text,
        anchor=anchor,
        next_run=utc_iso(first),
        created=utc_iso(now),
    )
    when_text = sched.format_local(utc_iso(first))
    cadence = "" if not rule.repeats else f", repeating {rule.text}"
    return ToolResult.success(
        f"Scheduled task #{task_id} for {when_text} local time{cadence}: {instruction.strip()[:120]}\n"
        "It runs by itself while Jarvis is open (a window or `jarvis --daemon`).",
        data={"id": task_id, "next_run": utc_iso(first), "repeat": rule.text},
    )


def _task_line(row: dict[str, Any]) -> str:
    state = row["status"]
    nxt = f"next {sched.format_local(row['next_run'])}" if row["next_run"] else state
    rep = "" if row["repeat_text"] == "once" else f", {row['repeat_text']}"
    queued = ", queued to run now" if row["force_run"] else ""
    return f"#{row['id']} \"{row['title']}\" [{state}{rep}{queued}] {nxt}: {row['instruction'][:160]}"


def _run_line(run: dict[str, Any]) -> str:
    text = " ".join(str(run["summary"]).split())
    text = text if len(text) <= 300 else text[:299] + "…"
    return f"    {sched.format_local(run['started_at'])} {run['status']}: {text}"


@tool(group="core")
def list_scheduled_tasks(ctx: ToolContext, include_finished: bool = False) -> ToolResult:
    """List the user's scheduled tasks with their next run time and what the last runs found.

    Use it to answer "what do I have scheduled?" and "what did the 9 am task find?": each
    task shows its three most recent runs with their results.

    Args:
        include_finished: Also show tasks that finished, were cancelled or were missed.
    """
    if ctx.memory is None:
        return _NO_MEMORY
    rows = ts.list_tasks(ctx.memory, include_finished)
    if not rows:
        return ToolResult.success("No scheduled tasks." if not include_finished else "No tasks.", data=[])
    lines: list[str] = []
    for row in rows:
        lines.append(_task_line(row))
        lines.extend(_run_line(r) for r in ts.recent_runs(ctx.memory, row["id"], 3))
    return ToolResult.success("\n".join(lines), data=[{k: r[k] for k in ("id", "title", "next_run", "status")} for r in rows])


@tool(group="core")
def cancel_scheduled_task(task_id: int, ctx: ToolContext) -> ToolResult:
    """Cancel a scheduled task by its number (from list_scheduled_tasks). Its past results stay.

    Args:
        task_id: The task's number.
    """
    if ctx.memory is None:
        return _NO_MEMORY
    row = ts.get_task(ctx.memory, task_id)
    if row is None:
        return ToolResult.fail(f"no scheduled task #{task_id}", "Use list_scheduled_tasks to see valid numbers.")
    if not ts.cancel_task(ctx.memory, task_id):
        return ToolResult.fail(f"task #{task_id} is already {row['status']}")
    return ToolResult.success(f"Cancelled scheduled task #{task_id}: {row['title']}")


@tool(group="core")
def run_scheduled_task_now(task_id: int, ctx: ToolContext) -> ToolResult:
    """Ask for a scheduled task to run as soon as Jarvis is idle, without changing its schedule.

    It does not run inside the current turn: it starts right after this reply, and its
    result is delivered like any scheduled result and kept for list_scheduled_tasks.

    Args:
        task_id: The task's number.
    """
    if ctx.memory is None:
        return _NO_MEMORY
    row = ts.get_task(ctx.memory, task_id)
    if row is None:
        return ToolResult.fail(f"no scheduled task #{task_id}", "Use list_scheduled_tasks to see valid numbers.")
    if not ts.request_run_now(ctx.memory, task_id):
        return ToolResult.fail(f"task #{task_id} is {row['status']} and cannot run")
    return ToolResult.success(
        f"Task #{task_id} \"{row['title']}\" is queued; it starts as soon as nothing else is happening "
        "(within about ten seconds), and the result will be delivered then.",
    )
