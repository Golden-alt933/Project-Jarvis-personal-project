"""Scheduled tasks: the scheduler (fake clock, fake runner), persistence, unattended runs
with a real Agent and a scripted model, and the four tools. No network, no browser."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import pytest

from jarvis.agent.loop import Agent
from jarvis.config import Settings
from jarvis.llm.types import LLMResponse, ResponseDone, TextDelta, ToolCall
from jarvis.memory.store import MemoryStore
from jarvis.proactive import taskstore as ts
from jarvis.proactive import tasks as sched
from jarvis.proactive.schedule_math import first_run, parse_repeat, utc_iso
from jarvis.proactive.tasks import AgentTaskRunner, RunResult, TaskScheduler, speakable
from jarvis.tools.base import Risk, ToolContext, ToolResult, tool
from jarvis.tools.builtin import schedule as tools
from jarvis.tools.registry import ToolRegistry

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


# -- fakes ---------------------------------------------------------------------------------------
class Clock:
    def __init__(self, *parts: int) -> None:
        self.now = datetime(*parts, tzinfo=NY)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now = (self.now.astimezone(UTC) + timedelta(**kw)).astimezone(NY)


class Runner:
    def __init__(self, status: str = "ok", text: str = "result", raises: Exception | None = None, gate: asyncio.Event | None = None) -> None:
        self.status, self.text, self.raises, self.gate = status, text, raises, gate
        self.calls: list[tuple[int, datetime | None]] = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, task: dict[str, Any], scheduled_for: datetime | None) -> RunResult:
        self.calls.append((task["id"], scheduled_for))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            if self.gate is not None:
                await self.gate.wait()
            if self.raises is not None:
                raise self.raises
            return RunResult(self.status, f"{self.text} #{task['id']}", "conv-x", 2)
        finally:
            self.active -= 1


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "jarvis.db")
    yield s
    s.close()


def add(store, clock: Clock, repeat: str = "once", at=(2026, 6, 10, 9, 0), title: str = "T", instruction: str = "do the thing") -> int:
    rule = parse_repeat(repeat)
    anchor = datetime(*at)
    return ts.add_task(
        store, title=title, instruction=instruction, repeat_text=rule.text, anchor=anchor,
        next_run=utc_iso(first_run(rule, anchor, NY)), created=utc_iso(clock()),
    )


def make(store, clock, runner=None, idle=lambda: True, **kw):
    delivered: list[str] = []

    async def deliver(text: str) -> None:
        delivered.append(text)

    sch = TaskScheduler(store, runner or Runner(), deliver, idle, clock=clock, tz=NY, interval=0.01, idle_recheck=0.001, **kw)
    return sch, delivered


def row(store, task_id: int) -> dict[str, Any]:
    r = ts.get_task(store, task_id)
    assert r is not None
    return r


# -- running and finishing --------------------------------------------------------------------------
async def test_one_off_runs_at_its_time_then_is_done(store):
    clock = Clock(2026, 6, 10, 8, 59)
    tid = add(store, clock)
    sch, delivered = make(store, clock)
    assert await sch.tick() == 0  # not yet due
    clock.advance(minutes=2)
    assert await sch.tick() == 1
    assert row(store, tid)["status"] == "done"
    assert row(store, tid)["next_run"] is None
    assert delivered == ["Scheduled task T: result #1"]
    assert await sch.tick() == 0  # never again


async def test_daily_task_moves_to_the_next_day_and_keeps_its_clock_time(store):
    clock = Clock(2026, 6, 10, 9, 0, 5)
    tid = add(store, clock, "daily")
    sch, _ = make(store, clock)
    assert await sch.tick() == 1
    assert sched.format_local(row(store, tid)["next_run"], NY) == "Thu 11 Jun 2026 09:00"
    assert row(store, tid)["status"] == "active"


async def test_daily_task_at_the_spring_forward_gap_runs_once_after_it(store):
    clock = Clock(2026, 3, 6, 1, 0)
    tid = add(store, clock, "daily", at=(2026, 3, 6, 2, 30))
    runner = Runner()
    sch, _ = make(store, clock, runner)
    clock.now = datetime(2026, 3, 6, 2, 31, tzinfo=NY)
    assert await sch.tick() == 1
    clock.now = datetime(2026, 3, 7, 2, 31, tzinfo=NY)
    assert await sch.tick() == 1
    # 02:30 on the 8th does not exist: the run lands just after the gap, at 03:30 EDT
    assert sched.format_local(row(store, tid)["next_run"], NY) == "Sun 08 Mar 2026 03:30"
    clock.now = datetime(2026, 3, 8, 3, 29, tzinfo=NY)
    assert await sch.tick() == 0
    clock.now = datetime(2026, 3, 8, 3, 31, tzinfo=NY)
    assert await sch.tick() == 1
    assert sched.format_local(row(store, tid)["next_run"], NY) == "Mon 09 Mar 2026 02:30"
    assert len(runner.calls) == 3


# -- downtime: never a burst ---------------------------------------------------------------------------
async def test_after_downtime_a_repeating_task_runs_exactly_once(store):
    clock = Clock(2026, 6, 10, 8, 0)
    tid = add(store, clock, "every 15 minutes", at=(2026, 6, 10, 9, 0))
    runner = Runner()
    sch, _ = make(store, clock, runner)
    clock.now = datetime(2026, 6, 14, 17, 7, tzinfo=NY)  # four days of downtime: ~400 missed slots
    assert await sch.tick() == 1
    assert await sch.tick() == 0
    assert len(runner.calls) == 1
    assert sched.format_local(row(store, tid)["next_run"], NY) == "Sun 14 Jun 2026 17:15"


async def test_a_daily_task_missed_for_days_runs_once_and_reports_lateness(store):
    clock = Clock(2026, 6, 10, 8, 0)
    add(store, clock, "daily")
    sch, delivered = make(store, clock)
    clock.now = datetime(2026, 6, 13, 16, 0, tzinfo=NY)
    assert await sch.tick() == 1
    assert "minutes after its scheduled time" in delivered[0]
    assert await sch.tick() == 0


async def test_a_one_off_missed_by_more_than_the_grace_is_reported_not_run(store):
    clock = Clock(2026, 6, 10, 8, 0)
    tid = add(store, clock)
    runner = Runner()
    sch, delivered = make(store, clock, runner)
    clock.now = datetime(2026, 6, 10, 9, 31, tzinfo=NY)  # 31 minutes late
    assert await sch.tick() == 0
    assert runner.calls == []
    assert row(store, tid)["status"] == "missed"
    assert sch._notices and "missed" in sch._notices[0]
    await sch.tick()  # idle: the notice is delivered
    assert any("was missed" in d for d in delivered)
    assert ts.recent_runs(store, tid)[0]["status"] == "missed"


async def test_a_one_off_inside_the_grace_still_runs(store):
    clock = Clock(2026, 6, 10, 8, 0)
    add(store, clock)
    runner = Runner()
    sch, _ = make(store, clock, runner)
    clock.now = datetime(2026, 6, 10, 9, 29, tzinfo=NY)
    assert await sch.tick() == 1


async def test_grace_period_is_configurable(store):
    clock = Clock(2026, 6, 10, 8, 0)
    add(store, clock)
    sch, _ = make(store, clock, grace=timedelta(minutes=5))
    clock.now = datetime(2026, 6, 10, 9, 10, tzinfo=NY)
    assert await sch.tick() == 0


# -- clock jumps ---------------------------------------------------------------------------------------------
async def test_clock_set_backwards_reschedules_instead_of_waiting_for_weeks(store):
    clock = Clock(2026, 6, 10, 10, 0)
    tid = add(store, clock, "daily", at=(2026, 1, 1, 9, 0))
    sch, _ = make(store, clock)
    assert await sch.tick() == 1  # Jun 10 09:00 had passed: runs once, next is Jun 11 09:00
    clock.now = datetime(2026, 5, 20, 12, 0, tzinfo=NY)  # the clock is set back three weeks
    await sch.tick()
    assert sched.format_local(row(store, tid)["next_run"], NY) == "Thu 21 May 2026 09:00"


async def test_clock_jumping_forward_is_just_downtime(store):
    clock = Clock(2026, 6, 10, 8, 0)
    add(store, clock, "hourly", at=(2026, 6, 10, 9, 0))
    runner = Runner()
    sch, _ = make(store, clock, runner)
    clock.now = datetime(2026, 6, 10, 20, 30, tzinfo=NY)
    await sch.tick()
    await sch.tick()
    assert len(runner.calls) == 1


# -- idle, deferral, one at a time -------------------------------------------------------------------------------
async def test_waits_for_idle_then_runs(store):
    clock = Clock(2026, 6, 10, 9, 1)
    add(store, clock)
    busy = {"on": True}
    runner = Runner()
    sch, _ = make(store, clock, runner, idle=lambda: not busy["on"])
    assert await sch.tick() == 0
    assert runner.calls == []
    busy["on"] = False
    assert await sch.tick() == 1


async def test_a_task_deferred_for_ten_minutes_is_reported_and_skipped_until_the_next_slot(store):
    clock = Clock(2026, 6, 10, 9, 0, 30)
    tid = add(store, clock, "daily")
    runner = Runner()
    sch, delivered = make(store, clock, runner, idle=lambda: False)
    assert await sch.tick() == 0
    clock.advance(minutes=9)
    assert await sch.tick() == 0
    assert row(store, tid)["next_run"] is not None and sched.format_local(row(store, tid)["next_run"], NY) == "Wed 10 Jun 2026 09:00"
    clock.advance(minutes=2)  # now 11 minutes deferred
    assert await sch.tick() == 0
    assert runner.calls == []
    assert sched.format_local(row(store, tid)["next_run"], NY) == "Thu 11 Jun 2026 09:00"
    assert row(store, tid)["status"] == "active"
    assert ts.recent_runs(store, tid)[0]["status"] == "skipped"
    assert "busy" in sch._notices[0]


async def test_a_deferred_one_off_is_marked_missed(store):
    clock = Clock(2026, 6, 10, 9, 0, 30)
    tid = add(store, clock)
    sch, _ = make(store, clock, idle=lambda: False)
    await sch.tick()
    clock.advance(minutes=11)
    await sch.tick()
    assert row(store, tid)["status"] == "missed"


async def test_never_two_tasks_at_once(store):
    clock = Clock(2026, 6, 10, 9, 1)
    a, b = add(store, clock, title="A"), add(store, clock, title="B")
    gate = asyncio.Event()
    runner = Runner(gate=gate)
    sch, _ = make(store, clock, runner)
    first = asyncio.ensure_future(sch.tick())
    await asyncio.sleep(0.01)
    assert runner.active == 1
    assert await sch.tick() == 0  # a second pass while one runs does not start the other
    gate.set()
    assert await first == 2  # the first pass runs A, then B, one after the other
    assert runner.max_active == 1
    assert sorted(c[0] for c in runner.calls) == [a, b]


async def test_a_running_task_blocks_the_next_even_when_the_user_is_idle(store):
    clock = Clock(2026, 6, 10, 9, 1)
    add(store, clock)
    sch, _ = make(store, clock)
    sch._running = True
    assert await sch.tick() == 0


# -- failures --------------------------------------------------------------------------------------------------------
async def test_a_failing_task_is_recorded_and_does_not_stop_the_scheduler(store):
    clock = Clock(2026, 6, 10, 9, 1)
    bad, good = add(store, clock, title="bad"), add(store, clock, title="good")

    class Mixed(Runner):
        async def __call__(self, task, scheduled_for):
            if task["id"] == bad:
                raise RuntimeError("boom")
            return await super().__call__(task, scheduled_for)

    sch, delivered = make(store, clock, Mixed())
    assert await sch.tick() == 2
    failed = ts.recent_runs(store, bad)[0]
    assert failed["status"] == "error" and "boom" in failed["summary"]
    assert ts.recent_runs(store, good)[0]["status"] == "ok"
    assert any("failed" in d and "boom" in d for d in delivered)


async def test_a_failing_repeating_task_keeps_its_schedule(store):
    clock = Clock(2026, 6, 10, 9, 1)
    tid = add(store, clock, "daily")
    sch, _ = make(store, clock, Runner(raises=ValueError("nope")))
    await sch.tick()
    assert row(store, tid)["status"] == "active"
    assert sched.format_local(row(store, tid)["next_run"], NY) == "Thu 11 Jun 2026 09:00"


async def test_a_broken_delivery_does_not_lose_the_result(store):
    clock = Clock(2026, 6, 10, 9, 1)
    tid = add(store, clock)

    async def deliver(text: str) -> None:
        raise OSError("speaker gone")

    sch = TaskScheduler(store, Runner(), deliver, clock=clock, tz=NY)
    assert await sch.tick() == 1
    assert ts.recent_runs(store, tid)[0]["summary"] == "result #1"


async def test_an_unknown_repeat_rule_cancels_the_task_instead_of_looping(store):
    clock = Clock(2026, 6, 10, 9, 1)
    tid = add(store, clock)
    store.execute("UPDATE scheduled_tasks SET repeat_text = 'gibberish' WHERE id = ?", (tid,))
    sch, _ = make(store, clock)
    assert await sch.tick() == 0
    assert row(store, tid)["status"] == "cancelled"


# -- persistence --------------------------------------------------------------------------------------------------------
async def test_state_survives_a_restart(tmp_path: Path):
    path = tmp_path / "jarvis.db"
    clock = Clock(2026, 6, 10, 8, 0)
    s1 = MemoryStore(path)
    tid = add(s1, clock, "daily")
    once = add(s1, clock, title="once", at=(2026, 6, 10, 12, 0))
    s1.close()  # the process stops

    s2 = MemoryStore(path)  # and starts again
    runner = Runner()
    sch, _ = make(s2, clock, runner)
    clock.now = datetime(2026, 6, 10, 9, 0, 10, tzinfo=NY)
    assert await sch.tick() == 1
    assert [c[0] for c in runner.calls] == [tid]
    s2.close()

    s3 = MemoryStore(path)  # a second restart, after the daily task ran
    runner3 = Runner()
    sch3, _ = make(s3, clock, runner3)
    assert await sch3.tick() == 0  # nothing runs twice
    clock.now = datetime(2026, 6, 10, 12, 0, 5, tzinfo=NY)
    assert await sch3.tick() == 1
    assert [c[0] for c in runner3.calls] == [once]
    assert sched.format_local(row(s3, tid)["next_run"], NY) == "Thu 11 Jun 2026 09:00"
    s3.close()


async def test_a_second_process_cannot_claim_the_same_slot(store):
    clock = Clock(2026, 6, 10, 9, 1)
    tid = add(store, clock)
    snapshot = row(store, tid)
    now = utc_iso(clock())
    assert ts.claim(store, snapshot, new_next=None, new_status="done", now_iso=now) is True
    assert ts.claim(store, snapshot, new_next=None, new_status="done", now_iso=now) is False


async def test_two_schedulers_on_one_database_run_a_task_once(store):
    clock = Clock(2026, 6, 10, 9, 1)
    add(store, clock)
    runner = Runner()
    one, _ = make(store, clock, runner)
    two, _ = make(store, clock, runner)
    await asyncio.gather(one.tick(), two.tick())
    assert len(runner.calls) == 1


async def test_runs_left_running_by_a_crash_are_marked_interrupted(store):
    clock = Clock(2026, 6, 10, 9, 1)
    tid = add(store, clock)
    run_id = ts.start_run(store, tid, None, "2026-06-10T07:00:00+00:00")
    assert ts.interrupted_runs(store, "2026-06-10T12:00:00+00:00", "2026-06-10T13:00:00+00:00") == 1
    assert ts.recent_runs(store, tid)[0]["status"] == "error"
    assert run_id


async def test_only_the_last_ten_runs_are_kept(store):
    clock = Clock(2026, 6, 10, 9, 1)
    tid = add(store, clock, "every 5 minutes", at=(2026, 6, 10, 9, 0))
    sch, _ = make(store, clock)
    for _ in range(14):
        clock.advance(minutes=6)
        await sch.tick()
    assert len(ts.recent_runs(store, tid, 50)) == 10


# -- run now, cancel ------------------------------------------------------------------------------------------------------
async def test_run_now_keeps_the_schedule(store):
    clock = Clock(2026, 6, 10, 8, 0)
    tid = add(store, clock, "daily")
    runner = Runner()
    sch, _ = make(store, clock, runner)
    assert ts.request_run_now(store, tid)
    assert await sch.tick() == 1
    assert sched.format_local(row(store, tid)["next_run"], NY) == "Wed 10 Jun 2026 09:00"  # unchanged
    assert await sch.tick() == 0


async def test_run_now_on_a_one_off_finishes_it(store):
    clock = Clock(2026, 6, 10, 8, 0)
    tid = add(store, clock)
    sch, _ = make(store, clock)
    ts.request_run_now(store, tid)
    assert await sch.tick() == 1
    assert row(store, tid)["status"] == "done"
    clock.now = datetime(2026, 6, 10, 9, 5, tzinfo=NY)
    assert await sch.tick() == 0  # its original slot does not run it again


async def test_a_cancelled_task_never_runs(store):
    clock = Clock(2026, 6, 10, 8, 0)
    tid = add(store, clock)
    assert ts.cancel_task(store, tid)
    runner = Runner()
    sch, _ = make(store, clock, runner)
    clock.now = datetime(2026, 6, 10, 9, 1, tzinfo=NY)
    assert await sch.tick() == 0
    assert not ts.cancel_task(store, tid)


async def test_start_and_stop_run_the_loop(store):
    clock = Clock(2026, 6, 10, 9, 1)
    add(store, clock)
    runner = Runner()
    sch, _ = make(store, clock, runner)
    sch.start()
    for _ in range(100):
        if runner.calls:
            break
        await asyncio.sleep(0.01)
    await sch.stop()
    assert len(runner.calls) == 1


# -- unattended runs with a real Agent --------------------------------------------------------------------------------------------
class Script:
    def __init__(self, *replies: LLMResponse) -> None:
        self.replies = list(replies)
        self.name, self.model, self.context_window, self.is_local = "script", "script-model", 100000, False
        self.seen: list[list[Any]] = []

    async def stream(self, messages, tools=None, *, max_tokens=None, temperature=None) -> AsyncIterator:
        self.seen.append(list(messages))
        reply = self.replies.pop(0) if self.replies else LLMResponse("(script exhausted)", [])
        if reply.content:
            yield TextDelta(reply.content)
        yield ResponseDone(reply)

    async def complete(self, messages, tools=None, *, max_tokens=None, temperature=None) -> LLMResponse:
        async for ev in self.stream(messages, tools):
            if isinstance(ev, ResponseDone):
                return ev.response
        raise AssertionError


def call(name: str, i: int = 0, **args: Any) -> LLMResponse:
    return LLMResponse("", [ToolCall(f"c{i}", name, args)], "tool_calls")


@pytest.fixture
def rig(tmp_path: Path):
    ran: list[str] = []

    @tool(group="core", risk=Risk.CONFIRM, describe=lambda a: f"send the email to {a['who']}")
    def send_it(who: str) -> ToolResult:
        """Send an email."""
        ran.append(who)
        return ToolResult.success("sent")

    @tool(group="core")
    def peek(n: int) -> ToolResult:
        """Look at something."""
        return ToolResult.success(f"seen {n}")

    settings = Settings(data_dir_override=str(tmp_path), task_max_steps=3, task_time_budget=60.0, max_steps=25)
    store = MemoryStore(tmp_path / "jarvis.db")
    registry = ToolRegistry()
    registry.register(send_it.__tool_spec__)
    registry.register(peek.__tool_spec__)

    def build(*replies: LLMResponse):
        llm = Script(*replies)
        user_agent = Agent(llm, registry, store, settings)
        return llm, user_agent

    yield build, store, settings, ran
    store.close()


TASK = {"id": 7, "title": "Morning mail", "instruction": "email Sam the report"}


async def test_unattended_confirmation_is_denied_and_reported(rig):
    build, store, settings, ran = rig
    llm, user_agent = build(call("send_it", who="sam@example.com"), LLMResponse("Could not send it.", []))
    result = await AgentTaskRunner(user_agent, settings, confirm=None)(TASK, datetime(2026, 6, 10, 9, 0, tzinfo=NY))
    assert ran == []  # the risky action never ran
    assert result.status == "ok"
    assert "Not done because nobody was there to approve it" in result.text
    assert "send the email to sam@example.com" in result.text


async def test_an_attended_confirmation_is_asked_and_can_approve(rig):
    build, store, settings, ran = rig
    asked: list[str] = []

    async def confirm(description: str) -> bool:
        asked.append(description)
        return True

    llm, user_agent = build(call("send_it", who="sam@example.com"), LLMResponse("Sent.", []))
    result = await AgentTaskRunner(user_agent, settings, confirm=confirm)(TASK, None)
    assert ran == ["sam@example.com"]
    assert result.text == "Sent."
    assert asked and "Morning mail" in asked[0] and "send the email to sam@example.com" in asked[0]


async def test_an_attended_decline_and_a_crashing_confirm_both_deny(rig):
    build, store, settings, ran = rig

    async def no(_: str) -> bool:
        return False

    async def broken(_: str) -> bool:
        raise RuntimeError("frontend died")

    for confirm in (no, broken):
        llm, user_agent = build(call("send_it", who="a@b.co"), LLMResponse("ok", []))
        result = await AgentTaskRunner(user_agent, settings, confirm=confirm)(TASK, None)
        assert "Not done because nobody was there" in result.text
    assert ran == []


async def test_a_task_run_has_its_own_conversation_and_leaves_the_users_alone(rig):
    build, store, settings, ran = rig
    llm, user_agent = build(LLMResponse("Weather is fine.", []))
    user_conv = user_agent.conversation_id
    store.append_message(user_conv, __import__("jarvis.llm.types", fromlist=["Message"]).Message("user", "hello, my own chat"))
    before = [m.content for _, m in store.load_messages(user_conv)]

    result = await AgentTaskRunner(user_agent, settings)(TASK, datetime(2026, 6, 10, 9, 0, tzinfo=NY))

    assert user_agent.conversation_id == user_conv
    assert [m.content for _, m in store.load_messages(user_conv)] == before
    assert result.conversation_id and result.conversation_id != user_conv
    stored = [m for _, m in store.load_messages(result.conversation_id)]
    assert stored[0].role == "user"
    assert stored[0].content.startswith('[Scheduled task #7 "Morning mail"')
    assert "unattended" in stored[0].content or "Nobody is watching" in stored[0].content
    assert "Wednesday 10 June 09:00" in stored[0].content
    assert stored[0].content.endswith("Task: email Sam the report")
    system = llm.seen[0][0].content
    assert "Scheduled task" not in system  # the framing is in the user message, not the system prompt


async def test_a_task_run_has_a_smaller_step_budget_than_a_chat_turn(rig):
    build, store, settings, ran = rig
    llm, user_agent = build(*[call("peek", i, n=i) for i in range(10)], LLMResponse("wrapped up", []))
    result = await AgentTaskRunner(user_agent, settings)(TASK, None)
    assert result.status == "incomplete"  # task_max_steps=3, the chat limit is 25
    assert len(llm.seen) == 4  # three tool steps and one wrap-up call


async def test_a_crashing_agent_becomes_an_error_result(rig, monkeypatch):
    build, store, settings, ran = rig
    llm, user_agent = build()
    runner = AgentTaskRunner(user_agent, settings)

    class Broken:
        conversation_id = "c"

        def run_turn(self, *a, **k):
            raise RuntimeError("model exploded")

        def cancel(self):
            pass

    monkeypatch.setattr(runner, "new_agent", lambda: Broken())
    result = await runner(TASK, None)
    assert result.status == "error" and "model exploded" in result.text


async def test_the_scheduler_plus_runner_end_to_end_stores_a_retrievable_result(rig):
    build, store, settings, ran = rig
    clock = Clock(2026, 6, 10, 9, 0, 5)
    llm, user_agent = build(LLMResponse("It is sunny, 24 degrees.", []))
    tid = add(store, clock, "daily", title="Weather", instruction="check the weather")
    runner = AgentTaskRunner(user_agent, settings)
    delivered: list[str] = []

    async def deliver(text: str) -> None:
        delivered.append(text)

    sch = TaskScheduler(store, runner, deliver, clock=clock, tz=NY)
    assert await sch.tick() == 1
    assert delivered == ["Scheduled task Weather: It is sunny, 24 degrees."]
    run = ts.recent_runs(store, tid)[0]
    assert run["status"] == "ok" and run["summary"] == "It is sunny, 24 degrees."
    # "what did the task find?" is answerable later from the list tool
    ctx = ToolContext(settings=settings, memory=store)
    listing = tools.list_scheduled_tasks(ctx).content
    assert "Weather" in listing and "It is sunny, 24 degrees." in listing


def test_speakable_shortens_long_results_but_not_short_ones():
    assert speakable("Short and sweet.") == "Short and sweet."
    long = "First sentence here. " + "More words follow. " * 80
    spoken = speakable(long, 100)
    assert len(spoken) < 200 and spoken.startswith("First sentence here.") and "saved" in spoken


# -- the tools ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def tool_ctx(store, tmp_path, monkeypatch):
    fixed = datetime(2026, 11, 10, 14, 0).astimezone()  # Tuesday afternoon, machine zone
    monkeypatch.setattr(sched, "local_now", lambda: fixed)
    return ToolContext(settings=Settings(data_dir_override=str(tmp_path)), memory=store)


def test_schedule_task_creates_and_reports_the_next_run(tool_ctx, store):
    res = tools.schedule_task("brief me on my day", "tomorrow 9am", tool_ctx, repeat="daily", title="Briefing")
    assert res.ok, res.content
    assert "Wed 11 Nov 2026 09:00" in res.content and "daily" in res.content
    row_ = ts.get_task(store, res.data["id"])
    assert row_["title"] == "Briefing" and row_["repeat_text"] == "daily" and row_["status"] == "active"


def test_schedule_task_default_title_and_one_off(tool_ctx, store):
    res = tools.schedule_task("check whether my download finished " * 4, "in 1 hour", tool_ctx)
    assert res.ok
    row_ = ts.get_task(store, res.data["id"])
    assert len(row_["title"]) <= tools.TITLE_LIMIT and row_["repeat_text"] == "once"
    assert "15:00" in res.content


@pytest.mark.parametrize(
    ("when", "repeat", "fragment"),
    [("someday", "once", "could not"), ("tomorrow 9am", "fortnightly", "repeat rule"), ("tomorrow 9am", "every 2 minutes", "5 minutes"),
     ("2020-01-01T09:00", "once", "in the past")],
)
def test_schedule_task_rejects_bad_input_with_a_hint(tool_ctx, when, repeat, fragment):
    res = tools.schedule_task("do it", when, tool_ctx, repeat=repeat)
    assert not res.ok and fragment in res.content


def test_schedule_task_rejects_empty_instruction_and_missing_memory(tool_ctx):
    assert not tools.schedule_task("   ", "tomorrow 9am", tool_ctx).ok
    assert not tools.schedule_task("x", "tomorrow 9am", ToolContext(settings=tool_ctx.settings, memory=None)).ok


def test_weekday_task_started_on_a_weekend_begins_monday(tool_ctx, store, monkeypatch):
    saturday = datetime(2026, 11, 14, 10, 0).astimezone()
    monkeypatch.setattr(sched, "local_now", lambda: saturday)
    res = tools.schedule_task("log my hours", "at 18:00", tool_ctx, repeat="weekdays")
    assert "Mon 16 Nov 2026 18:00" in res.content


@pytest.mark.parametrize(
    ("repeat", "risk"),
    [("once", Risk.SAFE), ("hourly", Risk.SAFE), ("daily", Risk.SAFE), ("every 2 hours", Risk.SAFE),
     ("every 30 minutes", Risk.CONFIRM), ("every 5 minutes", Risk.CONFIRM), ("every 59 minutes", Risk.CONFIRM),
     ("every 60 minutes", Risk.SAFE), ("nonsense", Risk.SAFE)],
)
def test_more_often_than_hourly_asks_to_confirm(repeat, risk):
    spec = tools.schedule_task.__tool_spec__
    assert spec.risk_for({"instruction": "x", "when": "in 1 hour", "repeat": repeat}) is risk


async def test_the_confirmation_is_asked_through_the_registry_and_a_refusal_creates_nothing(store, tmp_path):
    from jarvis.llm.types import ToolCall as TC

    registry = ToolRegistry()
    registry.register_module(tools)
    questions: list[str] = []

    async def refuse(text: str) -> bool:
        questions.append(text)
        return False

    ctx = ToolContext(settings=Settings(data_dir_override=str(tmp_path)), memory=store, confirm=refuse)
    res = await registry.execute(TC("1", "schedule_task", {"instruction": "ping the server", "when": "in 1 hour", "repeat": "every 10 minutes"}), ctx)
    assert not res.ok and "declined" in res.content
    assert questions and "every 10 minutes" in questions[0] and "ping the server" in questions[0]
    assert ts.list_tasks(store, True) == []


def test_list_cancel_and_run_now_tools(tool_ctx, store):
    a = tools.schedule_task("task a", "tomorrow 9am", tool_ctx, repeat="daily").data["id"]
    b = tools.schedule_task("task b", "friday 3pm", tool_ctx).data["id"]
    listing = tools.list_scheduled_tasks(tool_ctx).content
    assert f"#{a}" in listing and f"#{b}" in listing
    assert listing.index(f"#{a}") < listing.index(f"#{b}")  # soonest first
    assert "Wed 11 Nov 2026 09:00" in listing

    assert tools.run_scheduled_task_now(a, tool_ctx).ok
    assert "queued to run now" in tools.list_scheduled_tasks(tool_ctx).content

    assert tools.cancel_scheduled_task(b, tool_ctx).ok
    assert f"#{b}" not in tools.list_scheduled_tasks(tool_ctx).content
    assert f"#{b}" in tools.list_scheduled_tasks(tool_ctx, include_finished=True).content
    assert not tools.cancel_scheduled_task(b, tool_ctx).ok  # already cancelled
    assert not tools.run_scheduled_task_now(b, tool_ctx).ok
    for fn in (tools.cancel_scheduled_task, tools.run_scheduled_task_now):
        res = fn(999, tool_ctx)
        assert not res.ok and "list_scheduled_tasks" in res.content


def test_listing_with_nothing_scheduled(tool_ctx):
    assert tools.list_scheduled_tasks(tool_ctx).content == "No scheduled tasks."


def test_reminder_and_task_docstrings_tell_the_model_which_to_use():
    from jarvis.tools.builtin import productivity

    assert "schedule_task" in productivity.set_reminder.__tool_spec__.description
    desc = tools.schedule_task.__tool_spec__.description
    assert "set_reminder" in desc and "TOLD" in desc


def test_the_schedule_tools_are_registered_in_the_core_group():
    registry = ToolRegistry().load_builtin()
    for name in ("schedule_task", "list_scheduled_tasks", "cancel_scheduled_task", "run_scheduled_task_now"):
        assert registry.get(name).group == "core"
    schema = registry.get("schedule_task").parameters
    assert set(schema["required"]) == {"instruction", "when"}
