"""Scheduled tasks: finding the due ones, running each as an unattended agent turn,
and telling the user the outcome.

The scheduler knows nothing about voice or terminals. It is given a ``deliver``
coroutine (speak it, print it, notify) and an ``is_idle`` predicate, exactly like
``ReminderWatcher``, and a ``runner`` that performs one task.

Guarantees (each has a test with a fake clock):
* after downtime a repeating task runs at most once, a one-off that is more than the
  grace period late is reported as missed and not run;
* one task at a time, and never while the user's own turn is in progress; a task that
  keeps being deferred for ``defer_limit`` is reported and skipped until its next slot;
* a task that fails is recorded with its error and the scheduler carries on;
* a slot is claimed in the database before it runs, so two Jarvis processes (the
  daemon and a terminal) never run the same task twice.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from jarvis.proactive import taskstore as ts
from jarvis.proactive.schedule_math import (
    ScheduleError,
    clock_looks_wrong,
    next_after,
    parse_repeat,
    parse_utc,
    to_local,
    utc_iso,
)

if TYPE_CHECKING:
    from jarvis.agent.loop import Agent
    from jarvis.config import Settings

log = logging.getLogger(__name__)

POLL_SECONDS = 10.0
IDLE_RECHECK_SECONDS = 0.5
SPEECH_LIMIT = 600  # characters of a result read aloud; the full text is always stored

UNATTENDED_PREFIX = (
    "[Scheduled task #{id} \"{title}\", running by itself at its scheduled time ({when}) for the "
    "user. Nobody is watching this turn. Do not ask questions: carry the task out with your tools. "
    "Anything that needs the user's approval will be declined automatically, so skip it and say "
    "what was left undone. Finish with the short result the user should be told.]\n\nTask: {instruction}"
)


def local_now() -> datetime:
    """The current time as an aware datetime in the machine's local zone."""
    return datetime.now(timezone.utc).astimezone()


def format_local(utc_text: str | None, tz: tzinfo | None = None) -> str:
    """'Tue 11 Nov 2026 09:00' in local time, or 'never' for no time."""
    if not utc_text:
        return "never"
    try:
        return f"{to_local(parse_utc(utc_text), tz):%a %d %b %Y %H:%M}"
    except ValueError:
        return utc_text


@dataclass
class RunResult:
    status: str  # ok | error | incomplete
    text: str
    conversation_id: str | None = None
    steps: int = 0


# -- running one task ---------------------------------------------------------------------
class AgentTaskRunner:
    """Runs a task as an unattended agent turn in its own conversation.

    A sibling ``Agent`` is built for each run (same model clients, tools and memory), so
    the user's current conversation is never touched and the run gets a smaller step and
    time budget than an interactive turn.
    """

    def __init__(
        self,
        source: "Agent",
        settings: "Settings",
        *,
        confirm: Callable[[str], Awaitable[bool]] | None = None,
        output_mode: str = "text",
    ) -> None:
        self.source = source
        self.settings = settings
        self.confirm = confirm  # None: nobody is present, every confirmation is declined
        self.output_mode = output_mode

    def _task_settings(self) -> "Settings":
        return self.settings.model_copy(
            update={"max_steps": self.settings.task_max_steps, "turn_time_budget": self.settings.task_time_budget}
        )

    def new_agent(self) -> "Agent":
        from jarvis.agent.loop import Agent

        src = self.source
        return Agent(src.primary, src.registry, src.store, self._task_settings(), fallback=src.fallback, fast=src.fast)

    async def __call__(self, task: dict[str, Any], scheduled_for: datetime | None) -> RunResult:
        from jarvis.agent.events import TurnDone

        when = f"{scheduled_for:%A %d %B %H:%M}" if scheduled_for else "now"
        prompt = UNATTENDED_PREFIX.format(
            id=task["id"], title=task["title"], when=when, instruction=task["instruction"]
        )
        agent = self.new_agent()
        declined: list[str] = []

        async def confirm(description: str) -> bool:
            if self.confirm is not None:
                try:
                    if await self.confirm(f"Scheduled task \"{task['title']}\" wants to: {description}"):
                        return True
                except Exception:
                    log.exception("attended confirmation failed; declining")
            declined.append(description.splitlines()[0][:120])
            return False  # an unattended task never approves its own risky action

        done: TurnDone | None = None
        budget = self.settings.task_time_budget + 30
        try:
            async with asyncio.timeout(budget):
                async for event in agent.run_turn(prompt, confirm=confirm, output_mode=self.output_mode):
                    if isinstance(event, TurnDone):
                        done = event
        except TimeoutError:
            agent.cancel()
            return RunResult("incomplete", self._with_declined("It ran out of time and was stopped.", declined), agent.conversation_id)
        except Exception as exc:
            log.exception("scheduled task %s crashed", task["id"])
            return RunResult("error", f"{type(exc).__name__}: {exc}", agent.conversation_id)
        if done is None:
            return RunResult("error", "The run ended without a result.", agent.conversation_id)
        text = self._with_declined(done.text.strip() or "(no result)", declined)
        status = "ok" if done.stopped == "done" else "error" if done.stopped == "error" else "incomplete"
        return RunResult(status, text, agent.conversation_id, done.steps)

    @staticmethod
    def _with_declined(text: str, declined: list[str]) -> str:
        if not declined:
            return text
        return text + "\nNot done because nobody was there to approve it: " + "; ".join(declined)


Runner = Callable[[dict[str, Any], "datetime | None"], Awaitable[RunResult]]


# -- the scheduler ---------------------------------------------------------------------------
class TaskScheduler:
    def __init__(
        self,
        store: Any,
        runner: Runner,
        deliver: Callable[[str], Awaitable[None]],
        is_idle: Callable[[], bool] = lambda: True,
        *,
        grace: timedelta = timedelta(minutes=30),
        defer_limit: timedelta = timedelta(minutes=10),
        interval: float = POLL_SECONDS,
        idle_recheck: float = IDLE_RECHECK_SECONDS,
        clock: Callable[[], datetime] = local_now,
        tz: tzinfo | None = None,
    ) -> None:
        self.store = store
        self.runner = runner
        self.deliver = deliver
        self.is_idle = is_idle
        self.grace = grace
        self.defer_limit = defer_limit
        self.interval = interval
        self.idle_recheck = idle_recheck
        self.clock = clock
        self.tz = tz
        self._running = False
        self._deferred: dict[int, datetime] = {}
        self._notices: list[str] = []
        self._task: asyncio.Task[None] | None = None

    @classmethod
    def from_settings(cls, store: Any, runner: Runner, deliver: Any, is_idle: Any, settings: "Settings", **kw: Any) -> "TaskScheduler":
        return cls(
            store, runner, deliver, is_idle,
            grace=timedelta(minutes=settings.task_grace_minutes),
            defer_limit=timedelta(minutes=settings.task_defer_minutes),
            **kw,
        )

    # -- helpers ---------------------------------------------------------------------------
    def _now(self) -> datetime:
        """The clock as a UTC instant (aware datetimes sharing a zone compare by wall time)."""
        return self.clock().astimezone(timezone.utc)

    def _next(self, task: dict[str, Any], now: datetime) -> datetime | None:
        rule = parse_repeat(task["repeat_text"])
        return next_after(rule, datetime.fromisoformat(task["anchor"]), now, self.tz)

    def _busy(self) -> bool:
        return self._running or not self.is_idle()

    def _notice(self, text: str) -> None:
        self._notices.append(text)

    async def _flush_notices(self) -> None:
        while self._notices and not self._busy():
            text = self._notices.pop(0)
            await self.deliver(text)

    # -- one pass ----------------------------------------------------------------------------
    async def tick(self) -> int:
        """Handle everything due right now. Returns how many tasks were run."""
        await self._flush_notices()
        now = self._now()
        self._repair_clock(now)
        ran = 0
        for row in ts.due_tasks(self.store, utc_iso(now)):
            now = self._now()
            try:
                ran += await self._handle(row, now)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scheduler could not handle task %s", row.get("id"))
        return ran

    def _repair_clock(self, now: datetime) -> None:
        """A next run further away than its rule allows means the clock went backwards."""
        for row in ts.list_tasks(self.store):
            if not row["next_run"] or row["force_run"]:
                continue
            try:
                rule = parse_repeat(row["repeat_text"])
            except ScheduleError:
                continue
            if clock_looks_wrong(rule, parse_utc(row["next_run"]), now):
                nxt = self._next(row, now)
                if nxt is not None:
                    log.warning("clock moved back: task %s rescheduled to %s", row["id"], nxt)
                    ts.reschedule(self.store, row["id"], utc_iso(nxt))

    async def _handle(self, row: dict[str, Any], now: datetime) -> int:
        task_id = row["id"]
        try:
            rule = parse_repeat(row["repeat_text"])
        except ScheduleError as exc:
            ts.advance_without_run(self.store, row, new_next=None, new_status=ts.CANCELLED)
            ts.record_event(self.store, task_id, status="error", summary=f"Cancelled: {exc}", when=utc_iso(now))
            return 0
        forced = bool(row["force_run"])
        due_at = parse_utc(row["next_run"]) if row["next_run"] else now
        late = now - due_at if not forced else timedelta(0)

        if not rule.repeats and not forced and late > self.grace:
            return self._miss(row, now, due_at, late, new_status=ts.MISSED, why=f"Jarvis was not running (limit {self._mins(self.grace)})")

        if self._busy():
            first = self._deferred.setdefault(task_id, now)
            if now - first >= self.defer_limit:
                self._deferred.pop(task_id, None)
                return self._miss(row, now, due_at, late, new_status=ts.MISSED if not rule.repeats else ts.ACTIVE,
                                  why=f"you were busy for {self._mins(self.defer_limit)}", skipped=True)
            return 0
        self._deferred.pop(task_id, None)

        # claim the slot, moving the task forward before it runs
        if forced and due_at > now:
            new_next, new_status = row["next_run"], ts.ACTIVE
            if not rule.repeats:
                new_next, new_status = None, ts.DONE
        else:
            nxt = self._next(row, now) if rule.repeats else None
            new_next, new_status = (utc_iso(nxt), ts.ACTIVE) if nxt is not None else (None, ts.DONE)
        if not ts.claim(self.store, row, new_next=new_next, new_status=new_status, now_iso=utc_iso(now)):
            return 0  # another process took it
        await self._run_one(row, now, due_at, late, forced)
        return 1

    @staticmethod
    def _mins(delta: timedelta) -> str:
        return f"{int(delta.total_seconds() // 60)} minutes"

    def _miss(self, row: dict[str, Any], now: datetime, due_at: datetime, late: timedelta, *, new_status: str, why: str, skipped: bool = False) -> int:
        if new_status == ts.ACTIVE:
            nxt = self._next(row, now)
            new_next, new_status = (utc_iso(nxt), ts.ACTIVE) if nxt is not None else (None, ts.DONE)
        else:
            new_next = None
        if not ts.advance_without_run(self.store, row, new_next=new_next, new_status=new_status):
            return 0
        at = format_local(row["next_run"], self.tz)
        word = "skipped" if skipped else "missed"
        again = f" Next run: {format_local(new_next, self.tz)}." if new_next else ""
        summary = f"Task {word}: it was due {at} and was not run because {why}.{again}"
        ts.record_event(self.store, row["id"], status=word, summary=summary, when=utc_iso(now), scheduled_for=row["next_run"])
        self._notice(f"Scheduled task \"{row['title']}\" was {word}: it was due {at}, and {why}.{again}")
        return 0

    async def _run_one(self, row: dict[str, Any], now: datetime, due_at: datetime, late: timedelta, forced: bool) -> None:
        self._running = True
        run_id = ts.start_run(self.store, row["id"], row["next_run"], utc_iso(now))
        try:
            result = await self.runner(row, to_local(due_at, self.tz) if not forced else None)
        except asyncio.CancelledError:
            ts.finish_run(self.store, run_id, status="error", summary="Cancelled while running.", finished=utc_iso(self._now()))
            self._running = False
            raise
        except Exception as exc:
            log.exception("scheduled task %s failed", row["id"])
            result = RunResult("error", f"{type(exc).__name__}: {exc}")
        finally:
            self._running = False
        text = result.text
        if late > timedelta(minutes=5):
            text += f"\n(Ran {int(late.total_seconds() // 60)} minutes after its scheduled time.)"
        ts.finish_run(
            self.store, run_id, status=result.status, summary=text,
            finished=utc_iso(self._now()), conversation_id=result.conversation_id, steps=result.steps,
        )
        await self._deliver_result(row, result.status, text)

    async def _deliver_result(self, row: dict[str, Any], status: str, text: str) -> None:
        title = row["title"]
        if status == "ok":
            line = f"Scheduled task {title}: {text}"
        elif status == "incomplete":
            line = f"Scheduled task {title} did not finish. {text}"
        else:
            line = f"Scheduled task {title} failed. {text}"
        while self._busy():
            await asyncio.sleep(self.idle_recheck)
        try:
            await self.deliver(line)
        except Exception:
            log.exception("could not deliver the result of task %s", row["id"])

    # -- lifecycle ---------------------------------------------------------------------------
    async def run(self) -> None:
        cutoff = self._now() - timedelta(hours=1)
        try:
            n = ts.interrupted_runs(self.store, utc_iso(cutoff), utc_iso(self._now()))
            if n:
                log.warning("%d scheduled run(s) were interrupted by a restart", n)
        except Exception:
            log.exception("could not clean up interrupted runs")
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scheduler pass failed")
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self.run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def speakable(text: str, limit: int = SPEECH_LIMIT) -> str:
    """The first part of a result, cut at a sentence end when it is long (for speech)."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: end + 1] if end > limit // 3 else cut.rstrip()) + " The rest is saved; ask me for the full result."
