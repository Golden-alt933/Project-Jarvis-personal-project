"""Agent loop, context manager and prompt tests. No network: models are scripted."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Callable

import pytest

from jarvis.agent.context import ContextManager, estimate_tokens, repair_history
from jarvis.agent.events import Notice, TextChunk, ToolFinished, ToolStarted, TurnDone
from jarvis.agent.loop import Agent
from jarvis.agent.prompt import build_system_prompt
from jarvis.config import Settings
from jarvis.llm.types import (
    LLMBadRequest,
    LLMResponse,
    LLMUnavailable,
    Message,
    ResponseDone,
    TextDelta,
    ToolCall,
)
from jarvis.tools.base import Risk, ToolContext, ToolResult, tool
from jarvis.tools.registry import ToolRegistry


# -- fakes ------------------------------------------------------------------
@dataclass
class FakeFact:
    text: str
    category: str = "general"
    id: int = 0


class FakeStore:
    """Minimal in-memory implementation of the documented MemoryStore interface."""

    def __init__(self) -> None:
        self.messages: dict[str, list[tuple[int, Message]]] = {}
        self.summaries: dict[str, tuple[str, int]] = {}
        self.facts: list[FakeFact] = []
        self._next = 0
        self._conv = 0

    def start_conversation(self) -> str:
        self._conv += 1
        cid = f"c{self._conv}"
        self.messages[cid] = []
        return cid

    def latest_conversation(self, max_age_hours: float | None = None) -> str | None:
        return f"c{self._conv}" if self._conv else None

    def append_message(self, conversation_id: str, message: Message) -> int:
        self._next += 1
        self.messages[conversation_id].append((self._next, message))
        return self._next

    def load_messages(self, conversation_id: str, after_id: int = 0) -> list[tuple[int, Message]]:
        return [(i, m) for i, m in self.messages[conversation_id] if i > after_id]

    def get_summary(self, conversation_id: str) -> tuple[str, int] | None:
        return self.summaries.get(conversation_id)

    def set_summary(self, conversation_id: str, summary: str, upto_message_id: int) -> None:
        self.summaries[conversation_id] = (summary, upto_message_id)

    def recall(self, query: str, limit: int = 8) -> list[FakeFact]:
        return self.facts[:limit]


class ScriptedLLM:
    """Each script item is an LLMResponse, an exception to raise, a string (plain reply),
    or a callable(messages, tools) returning any of those (or an awaitable that hangs)."""

    def __init__(self, script: list[Any], *, name: str = "fake", is_local: bool = False) -> None:
        self.script = list(script)
        self.name = name
        self.model = f"{name}-model"
        self.context_window = 100000
        self.is_local = is_local
        self.calls: list[tuple[list[Message], list[dict[str, Any]] | None]] = []

    async def stream(self, messages, tools=None, *, max_tokens=None, temperature=None) -> AsyncIterator:
        self.calls.append((list(messages), tools))
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of responses")
        item = self.script.pop(0)
        if callable(item):
            item = item(messages, tools)
        if asyncio.iscoroutine(item):
            item = await item
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            item = LLMResponse(item, [])
        if item.content:
            half = len(item.content) // 2
            yield TextDelta(item.content[:half])
            yield TextDelta(item.content[half:])
        yield ResponseDone(item)

    async def complete(self, messages, tools=None, *, max_tokens=None, temperature=None) -> LLMResponse:
        async for ev in self.stream(messages, tools, max_tokens=max_tokens, temperature=temperature):
            if isinstance(ev, ResponseDone):
                return ev.response
        raise AssertionError("no response")

    @property
    def tool_names(self) -> list[list[str]]:
        return [[t["function"]["name"] for t in (tools or [])] for _, tools in self.calls]


def calls(*specs: tuple[str, dict[str, Any]], text: str = "") -> LLMResponse:
    return LLMResponse(
        text, [ToolCall(f"id{i}", name, args) for i, (name, args) in enumerate(specs)], "tool_calls"
    )


async def hang(messages=None, tools=None):
    await asyncio.sleep(60)


# -- tools -------------------------------------------------------------------
EXECUTED: list[str] = []


@tool(group="core")
async def echo(text: str) -> ToolResult:
    """Echo the text back.

    Args:
        text: What to echo.
    """
    EXECUTED.append(f"echo:{text}")
    return ToolResult.success(f"echo {text}")


@tool(group="core")
async def nap(ms: int, label: str) -> ToolResult:
    """Sleep a bit, then report the label.

    Args:
        ms: Milliseconds.
        label: Label to return.
    """
    await asyncio.sleep(ms / 1000)
    return ToolResult.success(label)


@tool(group="core", risk=Risk.CONFIRM)
async def danger(target: str) -> ToolResult:
    """Do something destructive.

    Args:
        target: What to destroy.
    """
    EXECUTED.append(f"danger:{target}")
    return ToolResult.success(f"destroyed {target}")


@tool(group="shell")
async def shelly(cmd: str) -> ToolResult:
    """Pretend shell.

    Args:
        cmd: Command.
    """
    return ToolResult.success("ok")


@tool(group="core")
async def flaky(attempt: int) -> ToolResult:
    """Fails unless attempt is 2.

    Args:
        attempt: Attempt number.
    """
    if attempt != 2:
        return ToolResult.fail("not yet", "use attempt=2")
    return ToolResult.success("worked")


@tool(group="core", timeout=60)
async def stuck() -> ToolResult:
    """Never finishes."""
    await asyncio.sleep(60)
    return ToolResult.success("never")


@pytest.fixture(autouse=True)
def _clear() -> None:
    EXECUTED.clear()


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for fn in (echo, nap, danger, shelly, flaky, stuck):
        reg.register(fn.__tool_spec__)
    return reg


def make_settings(**kw: Any) -> Settings:
    return Settings(_env_file=None, **kw)


@pytest.fixture
def store() -> Any:
    return FakeStore()


def make_agent(store: Any, primary: ScriptedLLM, **kw: Any) -> Agent:
    settings = kw.pop("settings", None) or make_settings()
    return Agent(primary, make_registry(), store, settings, **kw)


async def run(agent: Agent, text: str = "hi", **kw: Any) -> list[Any]:
    return [ev async for ev in agent.run_turn(text, **kw)]


def done(events: list[Any]) -> TurnDone:
    assert isinstance(events[-1], TurnDone)
    return events[-1]


def stored(store: Any, agent: Agent) -> list[Message]:
    return [m for _, m in store.load_messages(agent.conversation_id)]


def assert_valid(messages: list[Message]) -> None:
    """Every assistant tool call has exactly one following tool result, nothing orphaned."""
    i = 0
    while i < len(messages):
        m = messages[i]
        assert m.role != "tool", f"orphan tool message at {i}"
        i += 1
        if m.role == "assistant" and m.tool_calls:
            ids = [c.id for c in m.tool_calls]
            results = []
            while i < len(messages) and messages[i].role == "tool":
                results.append(messages[i].tool_call_id)
                i += 1
            assert results == ids


# -- agent loop ----------------------------------------------------------------
async def test_plain_chat(store):
    llm = ScriptedLLM(["Good evening."])
    agent = make_agent(store, llm)
    events = await run(agent, "hello")
    assert "".join(e.text for e in events if isinstance(e, TextChunk)) == "Good evening."
    d = done(events)
    assert d.text == "Good evening." and d.steps == 1 and d.stopped == "done"
    assert [m.role for m in stored(store, agent)] == ["user", "assistant"]
    assert llm.calls[0][0][0].role == "system"


async def test_single_tool_round_trip(store):
    llm = ScriptedLLM([calls(("echo", {"text": "a"})), "Done."])
    agent = make_agent(store, llm)
    events = await run(agent, "go")
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("ToolStarted") < kinds.index("ToolFinished")
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result.content == "echo a"
    assert [m.role for m in stored(store, agent)] == ["user", "assistant", "tool", "assistant"]
    assert done(events).tool_calls == 1 and done(events).steps == 2


async def test_multi_step_chain_uses_earlier_result(store):
    def second(messages, tools):
        last_tool = [m for m in messages if m.role == "tool"][-1]
        return calls(("echo", {"text": last_tool.content}))

    llm = ScriptedLLM([calls(("echo", {"text": "one"})), second, "All done"])
    agent = make_agent(store, llm)
    await run(agent)
    assert EXECUTED == ["echo:one", "echo:echo one"]
    assert_valid(stored(store, agent))


async def test_parallel_safe_calls_preserve_order(store):
    llm = ScriptedLLM([calls(("nap", {"ms": 80, "label": "slow"}), ("nap", {"ms": 5, "label": "fast"})), "ok"])
    agent = make_agent(store, llm)
    events = await run(agent)
    names = [type(e).__name__ for e in events]
    assert names[:4] == ["ToolStarted", "ToolStarted", "ToolFinished", "ToolFinished"]
    tool_msgs = [m for m in stored(store, agent) if m.role == "tool"]
    assert [m.content for m in tool_msgs] == ["slow", "fast"]
    assert [m.tool_call_id for m in tool_msgs] == ["id0", "id1"]
    # Parallel: total time is closer to the slowest than to the sum.
    assert next(e for e in events if isinstance(e, ToolFinished)).seconds >= 0.07


async def test_mixed_batch_runs_sequentially(store):
    llm = ScriptedLLM([calls(("echo", {"text": "x"}), ("danger", {"target": "t"})), "ok"])
    agent = make_agent(store, llm)

    async def yes(_: str) -> bool:
        return True

    events = await run(agent, confirm=yes)
    names = [type(e).__name__ for e in events[:4]]
    assert names == ["ToolStarted", "ToolFinished", "ToolStarted", "ToolFinished"]


async def test_confirm_denied_and_approved(store):
    seen: list[str] = []

    async def no(text: str) -> bool:
        seen.append(text)
        return False

    llm = ScriptedLLM([calls(("danger", {"target": "x"})), "Not done."])
    agent = make_agent(store, llm)
    events = await run(agent, confirm=no)
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert not finished.result.ok and "declined" in finished.result.content
    assert EXECUTED == [] and "danger" in seen[0]

    async def yes(_: str) -> bool:
        return True

    llm2 = ScriptedLLM([calls(("danger", {"target": "y"})), "Destroyed."])
    agent2 = make_agent(FakeStore(), llm2)
    events2 = await run(agent2, confirm=yes)
    assert next(e for e in events2 if isinstance(e, ToolFinished)).result.ok
    assert EXECUTED == ["danger:y"]


async def test_confirm_defaults_to_deny(store):
    llm = ScriptedLLM([calls(("danger", {"target": "x"})), "ok"])
    agent = make_agent(store, llm)
    await run(agent)
    assert EXECUTED == []


async def test_tool_failure_is_fed_back_and_model_recovers(store):
    llm = ScriptedLLM([calls(("flaky", {"attempt": 1})), calls(("flaky", {"attempt": 2})), "Got it."])
    agent = make_agent(store, llm)
    events = await run(agent)
    results = [e.result for e in events if isinstance(e, ToolFinished)]
    assert [r.ok for r in results] == [False, True]
    second_call_msgs = llm.calls[1][0]
    assert "not yet" in second_call_msgs[-1].content
    assert done(events).text == "Got it."


async def test_unknown_tool_name(store):
    llm = ScriptedLLM([calls(("ecko", {"text": "a"})), "Sorry."])
    agent = make_agent(store, llm)
    events = await run(agent)
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert not finished.result.ok and "unknown tool" in finished.result.content
    assert "echo" in finished.result.content  # close-match hint


async def test_repeated_identical_call_guard(store):
    same = ("echo", {"text": "again"})
    llm = ScriptedLLM([calls(same), calls(same), calls(same), "Stopping."])
    agent = make_agent(store, llm)
    events = await run(agent)
    assert EXECUTED == ["echo:again", "echo:again"]
    third = [e for e in events if isinstance(e, ToolFinished)][2]
    assert not third.result.ok and "will not change" in third.result.content


async def test_max_steps_wrap_up(store):
    counter = iter(range(100))

    def step(messages, tools):
        return calls(("echo", {"text": str(next(counter))}))

    def wrap(messages, tools):
        assert tools is None
        assert "limit" in messages[-1].content
        return "Did two things; the rest is not done."

    llm = ScriptedLLM([step, step, wrap])
    agent = make_agent(store, llm, settings=make_settings(max_steps=2))
    events = await run(agent)
    d = done(events)
    assert d.stopped == "max_steps" and "not done" in d.text
    assert llm.calls[-1][1] is None
    assert_valid(stored(store, agent))
    assert stored(store, agent)[-1].role == "assistant"


async def test_fallback_on_unavailable_restricts_tools(store):
    primary = ScriptedLLM([LLMUnavailable("down")], name="cloud")
    fallback = ScriptedLLM(["Local here."], name="local", is_local=True)
    agent = make_agent(store, primary, fallback=fallback)
    events = await run(agent)
    assert any(isinstance(e, Notice) for e in events)
    assert done(events).text == "Local here." and done(events).model == "local-model"
    assert "shelly" in primary.tool_names[0]
    local_names = fallback.tool_names[0]
    assert "echo" in local_names and "shelly" not in local_names
    assert "small local model" in fallback.calls[0][0][0].content


async def test_unavailable_without_fallback_ends_with_error(store):
    agent = make_agent(store, ScriptedLLM([LLMUnavailable("down")]))
    events = await run(agent)
    d = done(events)
    assert d.stopped == "error" and d.text


async def test_bad_request_compacts_and_retries(store):
    fast = ScriptedLLM(["SUMMARY OF EARLIER"])
    primary = ScriptedLLM([LLMBadRequest("too long"), "Fine."])
    agent = make_agent(store, primary, fast=fast)
    cid = agent.conversation_id
    for i in range(5):
        store.append_message(cid, Message("user", f"q{i}"))
        store.append_message(cid, Message("assistant", f"a{i}"))
    events = await run(agent, "latest")
    assert done(events).text == "Fine." and done(events).stopped == "done"
    assert len(primary.calls) == 2
    assert store.get_summary(cid) is not None
    assert "SUMMARY OF EARLIER" in primary.calls[1][0][0].content
    assert len(primary.calls[1][0]) < len(primary.calls[0][0])


async def test_bad_request_twice_is_an_error(store):
    primary = ScriptedLLM([LLMBadRequest("a"), LLMBadRequest("b")])
    agent = make_agent(store, primary, fast=ScriptedLLM(["s"]))
    assert done(await run(agent)).stopped == "error"


async def test_empty_reply_retried_once_with_unpersisted_nudge(store):
    llm = ScriptedLLM(["", "Here you go."])
    agent = make_agent(store, llm)
    events = await run(agent)
    assert done(events).text == "Here you go."
    assert "empty" in llm.calls[1][0][-1].content
    assert all("empty" not in m.content for m in stored(store, agent))


async def test_empty_reply_twice_gives_apology(store):
    agent = make_agent(store, ScriptedLLM(["", "  "]))
    d = done(await run(agent))
    assert "sorry" in d.text.lower() and d.stopped == "done"


async def test_cancel_mid_tool_leaves_valid_transcript(store):
    llm = ScriptedLLM([calls(("echo", {"text": "a"}), ("stuck", {})), "never reached"])
    agent = make_agent(store, llm)
    asyncio.get_running_loop().call_later(0.15, agent.cancel)
    started = asyncio.get_running_loop().time()
    events = await run(agent)
    assert asyncio.get_running_loop().time() - started < 5
    assert done(events).stopped == "cancelled"
    msgs = stored(store, agent)
    assert_valid(msgs)
    assert msgs[-1].role == "tool" and "cancelled" in msgs[-1].content


async def test_cancel_mid_stream(store):
    agent = make_agent(store, ScriptedLLM([hang]))
    asyncio.get_running_loop().call_later(0.1, agent.cancel)
    events = await run(agent)
    assert done(events).stopped == "cancelled"
    assert_valid(stored(store, agent))


async def test_history_is_sent_on_next_turn(store):
    llm = ScriptedLLM([calls(("echo", {"text": "a"})), "First answer.", "Second answer."])
    agent = make_agent(store, llm)
    await run(agent, "first question")
    await run(agent, "second question")
    sent = llm.calls[-1][0]
    roles = [m.role for m in sent]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert sent[1].content == "first question" and sent[3].content == "echo a"
    assert sent[4].content == "First answer." and sent[5].content == "second question"


async def test_unexpected_exception_never_raises(store):
    def boom(messages, tools):
        raise RuntimeError("bug")

    agent = make_agent(store, ScriptedLLM([boom]))
    d = done(await run(agent))
    assert d.stopped == "error" and d.text


async def test_recalled_facts_reach_prompt(store):
    store.facts = [FakeFact("Prefers metric units")]
    llm = ScriptedLLM(["ok"])
    agent = make_agent(store, llm)
    await run(agent, "what's the weather")
    assert "Prefers metric units" in llm.calls[0][0][0].content


# -- context --------------------------------------------------------------------
def fill(store: FakeStore, cid: str, turns: int, tool_size: int = 100) -> None:
    for i in range(turns):
        store.append_message(cid, Message("user", f"question {i} " + "x" * 200))
        store.append_message(cid, Message("assistant", "", [ToolCall(f"t{i}", "echo", {"text": "y" * 50})]))
        store.append_message(cid, Message("tool", "r" * tool_size, tool_call_id=f"t{i}", name="echo"))
        store.append_message(cid, Message("assistant", f"answer {i} " + "z" * 200))


async def test_compaction_never_splits_pairs_and_reduces_tokens(store):
    cid = store.start_conversation()
    fill(store, cid, 12)
    fast = ScriptedLLM(["Goals: X. Files: C:/a.txt. Unfinished: none."])
    cm = ContextManager(store, make_settings(), fast)
    before, _ = await cm.build(cid)
    assert await cm.compact(cid)
    after, summary = await cm.build(cid)
    assert summary and "C:/a.txt" in summary
    assert after[0].role == "user"
    assert_valid(after)
    assert sum(m.role == "user" for m in after) == 6
    assert estimate_tokens(after) < estimate_tokens(before)
    # The summariser saw the old content and had no tools.
    assert "question 0" in fast.calls[0][0][1].content and fast.calls[0][1] is None
    assert await cm.compact(cid, aggressive=True)
    assert sum(m.role == "user" for m in (await cm.build(cid))[0]) == 2


async def test_compact_noop_when_short(store):
    cid = store.start_conversation()
    fill(store, cid, 3)
    cm = ContextManager(store, make_settings(), ScriptedLLM([]))
    assert not await cm.compact(cid)


async def test_summariser_failure_falls_back_to_dropping(store):
    cid = store.start_conversation()
    fill(store, cid, 10)
    cm = ContextManager(store, make_settings(), ScriptedLLM([LLMUnavailable("no")]))
    assert await cm.compact(cid)
    after, summary = await cm.build(cid)
    assert summary and "trimmed" in summary
    assert after[0].role == "user"
    assert_valid(after)


async def test_old_tool_results_truncated_in_send_not_in_store(store):
    cid = store.start_conversation()
    fill(store, cid, 5, tool_size=10000)
    cm = ContextManager(store, make_settings(), ScriptedLLM([]))
    msgs, _ = await cm.build(cid)
    tool_lens = [len(m.content) for m in msgs if m.role == "tool"]
    assert tool_lens[0] < 3000 and tool_lens[-1] == 10000
    assert all(len(m.content) == 10000 for _, m in store.load_messages(cid) if m.role == "tool")


async def test_interrupted_history_is_repaired(store):
    cid = store.start_conversation()
    store.append_message(cid, Message("user", "do it"))
    store.append_message(
        cid, Message("assistant", "", [ToolCall("a", "echo", {}), ToolCall("b", "echo", {})])
    )
    cm = ContextManager(store, make_settings(), ScriptedLLM([]))
    msgs, _ = await cm.build(cid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "tool"]
    assert "interrupted" in msgs[-1].content
    assert_valid(msgs)
    assert cm.repair_stored(cid) == 2  # persisted
    assert len(store.load_messages(cid)) == 4


def test_repair_in_the_middle_of_history():
    msgs = [
        Message("user", "a"),
        Message("assistant", "", [ToolCall("a", "echo", {})]),
        Message("user", "b"),
    ]
    fixed = repair_history(msgs)
    assert [m.role for m in fixed] == ["user", "assistant", "tool", "user"]
    assert_valid(fixed)


def test_estimate_tokens_counts_tool_arguments():
    plain = Message("assistant", "hello")
    with_call = Message("assistant", "hello", [ToolCall("1", "echo", {"text": "q" * 400})])
    assert estimate_tokens([with_call]) >= estimate_tokens([plain]) + 100


async def test_resume_latest_repairs(store):
    agent = make_agent(store, ScriptedLLM(["ok"]))
    cid = agent.conversation_id
    store.append_message(cid, Message("user", "x"))
    store.append_message(cid, Message("assistant", "", [ToolCall("z", "echo", {})]))
    other = make_agent(store, ScriptedLLM(["ok"]))
    store.messages[other.conversation_id] = []
    # latest_conversation in the fake returns the newest id, so point it back explicitly
    store.latest_conversation = lambda max_age_hours=None: cid  # type: ignore[method-assign]
    assert other.resume_latest()
    assert other.conversation_id == cid
    assert_valid(stored(store, other))


# -- prompt ---------------------------------------------------------------------
def test_system_prompt_contents():
    settings = make_settings(user_name="Tony")
    now = datetime(2025, 3, 4, 14, 30)
    text = build_system_prompt(
        settings=settings,
        facts=[FakeFact("Likes dark roast coffee")],  # type: ignore[list-item]
        summary="Earlier we planned a trip to Oslo.",
        output_mode="voice",
        is_local=False,
        now=now,
    )
    assert "Tuesday 04 March 2025" in text and "14:30" in text
    assert "Likes dark roast coffee" in text and "Oslo" in text
    assert "Tony" in text and "remember" in text
    assert "spoken aloud" in text and "small local model" not in text
    assert "Desktop" in text and "Downloads" in text
    assert len(text.split()) < 700


def test_system_prompt_local_and_text_variants():
    text = build_system_prompt(
        settings=make_settings(), facts=[], summary=None, output_mode="text", is_local=True
    )
    assert "small local model" in text and "terminal" in text and "sir" in text


def test_tool_context_import_is_used():
    assert ToolContext is not None


async def test_cancel_before_first_read_is_honoured_and_idle_cancel_is_forgotten(store):
    agent = make_agent(store, ScriptedLLM(["one", "two"]))
    agent.cancel()  # nothing is running: this must not leak into the next turn
    assert done(await run(agent, "a")).stopped == "done"
    stream = agent.run_turn("b")
    agent.cancel()  # after the turn was started, before its stream is first read
    events = [ev async for ev in stream]
    assert done(events).stopped == "cancelled"


async def test_hosted_alternate_is_tried_before_the_local_model(store):
    primary = ScriptedLLM([LLMUnavailable("stalled")], name="big")
    alternate = ScriptedLLM(["From the other hosted model."], name="small")
    local = ScriptedLLM(["Local here."], name="local", is_local=True)
    agent = make_agent(store, primary, fallback=local, alternates=[primary, alternate])
    events = await run(agent)
    assert done(events).text == "From the other hosted model." and done(events).model == "small-model"
    assert not any(isinstance(e, Notice) for e in events)  # hosted to hosted is not news
    assert "shelly" in alternate.tool_names[0]  # full toolset, not the local subset
    assert local.calls == []


async def test_local_model_is_the_last_resort_when_every_hosted_model_fails(store):
    primary = ScriptedLLM([LLMUnavailable("down")], name="big")
    alternate = ScriptedLLM([LLMUnavailable("down too")], name="small")
    local = ScriptedLLM(["Local here."], name="local", is_local=True)
    agent = make_agent(store, primary, fallback=local, alternates=[primary, alternate])
    events = await run(agent)
    assert done(events).text == "Local here."
    assert sum(isinstance(e, Notice) for e in events) == 1


async def test_the_voice_model_falls_back_to_the_main_hosted_model(store):
    main = ScriptedLLM(["Main model answering."], name="big")
    voice = ScriptedLLM([LLMUnavailable("stalled")], name="small")
    agent = make_agent(store, main, alternates=[main, voice])
    agent.primary = voice  # what App.use_voice_model() does
    events = await run(agent)
    assert done(events).model == "big-model"


async def test_empty_reply_switches_to_the_other_hosted_model_without_a_nudge(store):
    primary = ScriptedLLM([""], name="big")
    alternate = ScriptedLLM(["From the other hosted model."], name="small")
    local = ScriptedLLM(["Local here."], name="local", is_local=True)
    agent = make_agent(store, primary, fallback=local, alternates=[primary, alternate])
    events = await run(agent)
    assert done(events).text == "From the other hosted model." and done(events).model == "small-model"
    assert all("empty" not in m.content for m in alternate.calls[0][0])  # no nudge was needed
    assert not any(isinstance(e, Notice) for e in events) and local.calls == []


async def test_empty_reply_mid_task_hands_the_transcript_to_the_other_hosted_model(store):
    primary = ScriptedLLM([calls(("echo", {"text": "a"})), ""], name="big")
    alternate = ScriptedLLM(["Finished."], name="small")
    agent = make_agent(store, primary, alternates=[primary, alternate])
    d = done(await run(agent))
    assert d.text == "Finished." and d.tool_calls == 1
    assert any(m.role == "tool" for m in alternate.calls[0][0])  # it saw the tool result


async def test_empty_from_every_hosted_model_nudges_then_uses_the_local_model(store):
    primary = ScriptedLLM([""], name="big")
    alternate = ScriptedLLM(["", ""], name="small")
    local = ScriptedLLM(["Local here."], name="local", is_local=True)
    agent = make_agent(store, primary, fallback=local, alternates=[primary, alternate])
    events = await run(agent)
    assert done(events).text == "Local here."
    assert "empty" in alternate.calls[1][0][-1].content  # the nudge came before the local model
    assert sum(isinstance(e, Notice) for e in events) == 1


async def test_a_model_that_just_failed_does_not_lead_the_next_turns(store, monkeypatch):
    primary = ScriptedLLM([LLMUnavailable("down"), "Back again."], name="big")
    alternate = ScriptedLLM(["One.", "Two."], name="small")
    agent = make_agent(store, primary, alternates=[primary, alternate])
    assert done(await run(agent)).model == "small-model"
    assert done(await run(agent)).model == "small-model"  # straight to the working model
    assert len(primary.calls) == 1
    agent._benched["big-model"] = 0.0  # the bench has expired
    assert done(await run(agent)).text == "Back again."


async def test_an_empty_reply_also_benches_the_model(store):
    primary = ScriptedLLM([""], name="big")
    alternate = ScriptedLLM(["One.", "Two."], name="small")
    agent = make_agent(store, primary, alternates=[primary, alternate])
    await run(agent)
    assert done(await run(agent)).model == "small-model" and len(primary.calls) == 1


async def test_when_every_hosted_model_is_benched_the_chosen_one_still_leads(store):
    primary = ScriptedLLM(["Still me."], name="big")
    alternate = ScriptedLLM([], name="small")
    agent = make_agent(store, primary, alternates=[primary, alternate])
    agent._bench(primary)
    agent._bench(alternate)
    assert done(await run(agent)).model == "big-model"


async def test_a_declined_action_is_not_put_to_the_user_again(store):
    asked: list[str] = []

    async def no(description: str) -> bool:
        asked.append(description)
        return False

    again = calls(("danger", {"target":"a"}))
    llm = ScriptedLLM([again, again, calls(("danger", {"target":"b"})), "Not done."])
    events = await run(make_agent(store, llm), confirm=no)
    results = [e.result for e in events if isinstance(e, ToolFinished)]
    assert len(asked) == 2  # once per distinct action, not once per attempt
    assert all(not r.ok and "declined" in r.content for r in results)
    assert done(events).text == "Not done."


async def test_after_three_refusals_nothing_more_is_asked_this_turn(store):
    asked: list[str] = []

    async def no(description: str) -> bool:
        asked.append(description)
        return False

    steps = [calls(("danger", {"target": p})) for p in "abc"]
    llm = ScriptedLLM([*steps, "You declined those, so nothing was changed."])
    events = await run(make_agent(store, llm), confirm=no)
    assert len(asked) == 3
    assert sum(isinstance(e, ToolFinished) and not e.result.ok for e in events) == 3
    # the turn is then wrapped up: one last call with no tools, told why
    assert llm.calls[-1][1] is None and "declined actions" in llm.calls[-1][0][-1].content
    assert done(events).text == "You declined those, so nothing was changed." and done(events).stopped == "max_steps"
    asked.clear()  # a new turn starts with a clean slate
    agent = make_agent(store, ScriptedLLM([calls(("danger", {"target":"a"})), "ok"]))
    await run(agent, confirm=no)
    assert len(asked) == 1


async def test_a_model_stuck_repeating_a_call_is_stopped_and_made_to_report(store):
    same = calls(("echo", {"text": "x"}))
    llm = ScriptedLLM([same] * 5 + ["I could not get past that step."])
    events = await run(make_agent(store, llm))
    d = done(events)
    assert d.text == "I could not get past that step." and d.stopped == "max_steps"
    assert len(llm.calls) == 6 and llm.calls[-1][1] is None  # the last call offered no tools
    assert "repeated calls" in llm.calls[-1][0][-1].content
