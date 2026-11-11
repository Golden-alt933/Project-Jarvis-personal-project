"""The agent loop: the only control flow in Jarvis.

The model decides what to do; this code streams it, runs the tools it asks for,
keeps the context healthy and enforces safety limits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from jarvis.agent.context import ContextManager, interrupted_result
from jarvis.agent.events import AgentEvent, Notice, TextChunk, ToolFinished, ToolStarted, TurnDone
from jarvis.agent.prompt import build_system_prompt
from jarvis.llm.types import (
    LLMBadRequest,
    LLMClient,
    LLMError,
    LLMResponse,
    LLMUnavailable,
    Message,
    ResponseDone,
    TextDelta,
    ToolCall,
    Usage,
)
from jarvis.tools.base import Risk, ToolContext, ToolResult
from jarvis.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.memory.store import Fact, MemoryStore

log = logging.getLogger(__name__)

MAX_IDENTICAL_CALLS = 3  # the third identical call in a turn is not executed
RECALL_LIMIT = 8
# A hosted model that just failed (unreachable, rate limited, or an empty reply) is not
# the first choice for new turns for this long: on the free tier a model that is down
# stays down for minutes, and every retry costs the user seconds.
BENCH_SECONDS = 300.0
MAX_BLOCKED_CALLS = 3  # refused repeats in one turn before the turn is wrapped up
MAX_REFUSALS = 3  # after this many refusals in one turn, stop asking and let the model report
CANCELLED_TEXT = "Error: cancelled by the user before this call finished."
EMPTY_NUDGE = "(Your last reply was empty. Reply to the user, or call a tool if you need one.)"
WRAPUP_NOTE = (
    "(You have reached the limit for this turn ({reason}). Do not call any more tools. "
    "Tell the user what has been done, what has not, and what they could ask next.)"
)
EMPTY_APOLOGY = "I'm sorry, I didn't manage to produce a reply to that. Could you try asking again?"
_END = object()


class _AskOnce:
    """The turn's confirm callback: the same question is never put to the user twice, and
    after MAX_REFUSALS refusals nothing more is asked this turn (every answer is then no).

    A model that is told "the user declined" sometimes simply calls the tool again; without
    this the user is asked the identical question until the repeat guard steps in.
    """

    def __init__(self, ask: Callable[[str], Awaitable[bool]]) -> None:
        self._ask = ask
        self._refused: set[str] = set()

    @property
    def exhausted(self) -> bool:
        """True once the user has said no MAX_REFUSALS times this turn."""
        return len(self._refused) >= MAX_REFUSALS

    async def __call__(self, description: str) -> bool:
        if description in self._refused or self.exhausted:
            return False
        approved = bool(await self._ask(description))
        if not approved:
            self._refused.add(description)
        return approved


class _Cancelled(Exception):
    """Internal: the user cancelled the turn."""


@dataclass
class _Turn:
    client: LLMClient
    output_mode: str
    ctx: ToolContext
    history: list[Message]
    summary: str | None
    facts: "list[Fact]"
    started: float = field(default_factory=time.monotonic)
    steps: int = 0
    tool_calls: int = 0
    usage: Usage | None = None
    call_counts: dict[str, int] = field(default_factory=dict)
    tried: list[str] = field(default_factory=list)  # models that failed this turn
    blocked: int = 0  # calls refused as exact repeats

    def add_usage(self, usage: Usage | None) -> None:
        if usage is None:
            return
        total = self.usage or Usage()
        total.prompt_tokens += usage.prompt_tokens
        total.completion_tokens += usage.completion_tokens
        self.usage = total


class Agent:
    def __init__(
        self,
        primary: LLMClient,
        registry: ToolRegistry,
        store: "MemoryStore",
        settings: "Settings",
        fallback: LLMClient | None = None,
        fast: LLMClient | None = None,
        alternates: list[LLMClient] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        # Other hosted models to try, in order, before the local fallback. A stalled
        # free-tier request is far more common than a real outage, and switching to a
        # local model means loading gigabytes onto the user's machine.
        self.alternates: list[LLMClient] = list(alternates or [])
        self.fast = fast or primary
        self.registry = registry
        self.store = store
        self.settings = settings
        self.context = ContextManager(store, settings, self.fast)
        self._force_local = False
        self._benched: dict[str, float] = {}  # model -> monotonic time it may lead again
        self._cancel = asyncio.Event()
        self._cancel_waiter: asyncio.Future[Any] | None = None
        self.conversation_id: str = store.start_conversation()

    # -- conversation management -------------------------------------------
    def new_conversation(self) -> str:
        self.conversation_id = self.store.start_conversation()
        return self.conversation_id

    def resume_latest(self, max_age_hours: float = 12) -> bool:
        latest = self.store.latest_conversation(max_age_hours)
        if latest is None:
            return False
        self.conversation_id = latest
        self.context.repair_stored(latest)
        return True

    def use_local(self, on: bool) -> None:
        if on and self._local_client() is None:
            log.warning("use_local requested but no local model is configured")
            return
        self._force_local = on

    @property
    def using_local(self) -> bool:
        """True when the local model is forced for new turns."""
        return self._force_local

    def cancel(self) -> None:
        self._cancel.set()

    def _local_client(self) -> LLMClient | None:
        if self.primary.is_local:
            return self.primary
        if self.fallback is not None and self.fallback.is_local:
            return self.fallback
        return None

    def _initial_client(self) -> LLMClient:
        if self._force_local:
            local = self._local_client()
            if local is not None:
                return local
        if self._is_benched(self.primary):
            for alt in self.alternates:
                if alt.model != self.primary.model and not self._is_benched(alt):
                    return alt
        return self.primary

    def _bench(self, client: LLMClient) -> None:
        if not client.is_local:
            self._benched[client.model] = time.monotonic() + BENCH_SECONDS

    def _is_benched(self, client: LLMClient) -> bool:
        until = self._benched.get(client.model)
        if until is None:
            return False
        if time.monotonic() >= until:
            del self._benched[client.model]
            return False
        return True

    # -- the turn -------------------------------------------------------------
    def run_turn(
        self,
        user_text: str,
        *,
        confirm: Callable[[str], Awaitable[bool]] | None = None,
        notify: Callable[[str], Awaitable[None]] | None = None,
        output_mode: str = "text",
    ) -> AsyncIterator[AgentEvent]:
        """Start a turn and return its event stream.

        Not itself a generator: the cancel flag is armed here, at call time, so a
        cancel() that arrives before the stream is first read still stops the turn,
        while one that arrived earlier, when nothing was running, is forgotten.
        """
        self._cancel = asyncio.Event()
        return self._turn_events(user_text, confirm, notify, output_mode)

    async def _turn_events(
        self,
        user_text: str,
        confirm: Callable[[str], Awaitable[bool]] | None,
        notify: Callable[[str], Awaitable[None]] | None,
        output_mode: str,
    ) -> AsyncIterator[AgentEvent]:
        started = time.monotonic()
        self._cancel_waiter = asyncio.ensure_future(self._cancel.wait())
        turn: _Turn | None = None
        try:
            turn = await self._start_turn(user_text, confirm, notify, output_mode)
            async for event in self._run(turn):
                yield event
        except _Cancelled:
            yield self._done(turn, started, "", "cancelled")
        except Exception as exc:
            log.exception("turn failed")
            yield self._done(turn, started, self._error_text(exc), "error")
        finally:
            if self._cancel_waiter is not None:
                self._cancel_waiter.cancel()

    def _done(self, turn: _Turn | None, started: float, text: str, stopped: str) -> TurnDone:
        client = turn.client if turn else self.primary
        return TurnDone(
            text=text,
            steps=turn.steps if turn else 0,
            tool_calls=turn.tool_calls if turn else 0,
            seconds=time.monotonic() - started,
            model=client.model,
            usage=turn.usage if turn else None,
            stopped=stopped,
        )

    @staticmethod
    def _error_text(exc: Exception) -> str:
        if isinstance(exc, LLMUnavailable):
            return "I couldn't reach the language model, and nothing else was available to try. Please check the connection and ask again."
        if isinstance(exc, LLMError):
            return "The language model rejected the request, so I couldn't finish. Starting a fresh conversation may help."
        return "Something went wrong on my side before I could finish. Nothing further was done."

    async def _start_turn(
        self,
        user_text: str,
        confirm: Callable[[str], Awaitable[bool]] | None,
        notify: Callable[[str], Awaitable[None]] | None,
        output_mode: str,
    ) -> _Turn:
        ctx_kwargs: dict[str, Any] = {}
        if confirm is not None:
            ctx_kwargs["confirm"] = _AskOnce(confirm)
        if notify is not None:
            ctx_kwargs["notify"] = notify
        ctx = ToolContext(
            settings=self.settings,
            memory=self.store,
            cancelled=self._cancel,
            conversation_id=self.conversation_id,
            **ctx_kwargs,
        )
        self.store.append_message(self.conversation_id, Message("user", user_text))
        facts = self._recall(user_text)
        history, summary = await self._compacted_history()
        return _Turn(self._initial_client(), output_mode, ctx, history, summary, facts)

    def _recall(self, text: str) -> "list[Fact]":
        try:
            return list(self.store.recall(text, RECALL_LIMIT))
        except Exception:
            log.exception("fact recall failed")
            return []

    async def _compacted_history(self) -> tuple[list[Message], str | None]:
        history, summary = await self.context.build(self.conversation_id)
        if self.context.over_budget(history):
            for aggressive in (False, True):
                if not await self.context.compact(self.conversation_id, aggressive=aggressive):
                    continue
                history, summary = await self.context.build(self.conversation_id)
                if not self.context.over_budget(history):
                    break
        return history, summary

    def _messages(self, turn: _Turn, extra: Message | None = None) -> list[Message]:
        system = build_system_prompt(
            settings=self.settings,
            facts=turn.facts,
            summary=turn.summary,
            output_mode=turn.output_mode,
            is_local=turn.client.is_local,
            tool_groups=self._tool_groups(turn),
        )
        msgs = [Message("system", system), *turn.history]
        if extra is not None:
            msgs.append(extra)
        return msgs

    def _tools(self, turn: _Turn) -> list[dict[str, Any]]:
        if turn.client.is_local:
            return self.registry.schemas(self.settings.local_tool_groups)
        return self.registry.schemas()

    def _tool_groups(self, turn: _Turn) -> set[str]:
        """Groups of the tools offered this turn (the prompt adds guidance for some of them)."""
        groups = self.settings.local_tool_groups if turn.client.is_local else None
        return {s.group for s in self.registry.specs(groups)}

    def _persist(self, turn: _Turn, message: Message) -> None:
        turn.history.append(message)
        self.store.append_message(self.conversation_id, message)

    # -- main loop ------------------------------------------------------------
    async def _run(self, turn: _Turn) -> AsyncIterator[AgentEvent]:
        nudged = False
        while True:
            self._check_cancel()
            limit = self._limit_reached(turn)
            if limit:
                async for ev in self._wrap_up(turn, limit):
                    yield ev
                return

            extra = Message("user", EMPTY_NUDGE) if nudged else None
            response: LLMResponse | None = None
            async for item in self._model_call(turn, tools=True, extra=extra):
                if isinstance(item, LLMResponse):
                    response = item
                else:
                    yield item
            assert response is not None
            turn.steps += 1
            turn.add_usage(response.usage)

            if not response.tool_calls:
                if not response.content.strip():
                    # An empty reply is a failed model as far as the user is concerned. Another
                    # hosted model is the first remedy; the nudge and the local model come after.
                    successor = self._next_client(turn, hosted_only=not nudged)
                    if successor is not None:
                        notice = self._switch(turn, successor, "returned an empty reply")
                        if notice is not None:
                            yield notice
                        nudged = False
                        continue
                    if not nudged:
                        nudged = True
                        continue
                    self._persist(turn, Message("assistant", EMPTY_APOLOGY))
                    yield TextChunk(EMPTY_APOLOGY)
                    yield self._finish(turn, EMPTY_APOLOGY, "done")
                    return
                self._persist(turn, Message("assistant", response.content))
                yield self._finish(turn, response.content, "done")
                return

            nudged = False
            calls = self._with_ids(response.tool_calls)
            self._persist(turn, Message("assistant", response.content, calls))
            async for ev in self._run_tools(turn, calls):
                yield ev

    def _finish(self, turn: _Turn, text: str, stopped: str) -> TurnDone:
        return self._done(turn, turn.started, text, stopped)

    def _limit_reached(self, turn: _Turn) -> str | None:
        if turn.steps >= self.settings.max_steps:
            return "max_steps"
        if turn.blocked >= MAX_BLOCKED_CALLS:
            return "repeated_calls"  # the model is going round in circles: stop and report
        if getattr(turn.ctx.confirm, "exhausted", False):
            return "declined_actions"  # the user keeps saying no: stop trying other ways
        if time.monotonic() - turn.started > self.settings.turn_time_budget:
            return "time_budget"
        return None

    @staticmethod
    def _with_ids(calls: list[ToolCall]) -> list[ToolCall]:
        return [c if c.id else ToolCall(f"call_{uuid.uuid4().hex[:12]}", c.name, c.arguments) for c in calls]

    async def _wrap_up(self, turn: _Turn, reason: str) -> AsyncIterator[AgentEvent]:
        note = Message("user", WRAPUP_NOTE.format(reason=reason.replace("_", " ")))
        response: LLMResponse | None = None
        async for item in self._model_call(turn, tools=False, extra=note):
            if isinstance(item, LLMResponse):
                response = item
            else:
                yield item
        turn.steps += 1
        text = response.content.strip() if response else ""
        if response:
            turn.add_usage(response.usage)
        if not text:
            text = (
                f"I stopped because I hit the {reason.replace('_', ' ')} limit for one turn. "
                "Some of the work may be unfinished; ask me to continue if you want."
            )
            yield TextChunk(text)
        # Tool calls requested during wrap-up are dropped, so the transcript stays valid.
        self._persist(turn, Message("assistant", text))
        yield self._finish(turn, text, "max_steps" if reason in ("repeated_calls", "declined_actions") else reason)

    # -- model call with failure handling -----------------------------------
    async def _model_call(
        self, turn: _Turn, *, tools: bool, extra: Message | None
    ) -> AsyncIterator[AgentEvent | LLMResponse]:
        bad_request_retried = False
        while True:
            emitted = False
            try:
                async for ev in self._stream(
                    turn.client, self._messages(turn, extra), self._tools(turn) if tools else None
                ):
                    if isinstance(ev, TextDelta):
                        if ev.text:
                            emitted = True
                            yield TextChunk(ev.text)
                    elif isinstance(ev, ResponseDone):
                        yield ev.response
                        return
                raise LLMUnavailable("stream ended without a final response")
            except LLMBadRequest:
                if bad_request_retried or emitted:
                    raise
                bad_request_retried = True
                log.warning("bad request; compacting harder and retrying")
                await self.context.compact(self.conversation_id, aggressive=True)
                turn.history, turn.summary = await self.context.build(self.conversation_id)
            except LLMUnavailable as exc:
                self._bench(turn.client)
                successor = None if emitted else self._next_client(turn)
                if successor is None:
                    raise
                notice = self._switch(turn, successor, f"unavailable ({exc})")
                if notice is not None:
                    yield notice

    def _switch(self, turn: _Turn, successor: LLMClient, why: str) -> Notice | None:
        """Continue the turn on `successor`. Returns a notice when that is worth telling."""
        log.warning("%s %s; switching to %s", turn.client.model, why, successor.model)
        self._bench(turn.client)
        turn.tried.append(turn.client.model)
        turn.client = successor
        if not successor.is_local:
            return None  # hosted to hosted: same tools, same speed, not news
        return Notice(
            f"{self.primary.name} is unreachable; continuing on the local model "
            f"({successor.model}) with a reduced toolset."
        )

    def _next_client(self, turn: _Turn, *, hosted_only: bool = False) -> LLMClient | None:
        """The next model to try this turn: hosted alternates first, the local one last."""
        for candidate in (*self.alternates, self.fallback):
            if candidate is None or candidate is turn.client:
                continue
            if candidate.model == turn.client.model or candidate.model in turn.tried:
                continue
            if hosted_only and candidate.is_local:
                continue
            return candidate
        return None

    async def _stream(
        self, client: LLMClient, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> AsyncIterator[Any]:
        it = client.stream(
            messages,
            tools or None,
            max_tokens=self.settings.max_output_tokens,
            temperature=self.settings.temperature,
        ).__aiter__()

        async def next_event() -> Any:
            try:
                return await it.__anext__()
            except StopAsyncIteration:
                return _END

        try:
            while True:
                ev = await self._race(next_event())
                if ev is _END:
                    return
                yield ev
        finally:
            aclose = getattr(it, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    log.debug("stream close failed", exc_info=True)

    # -- cancellation ---------------------------------------------------------
    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled

    async def _race(self, awaitable: Awaitable[Any]) -> Any:
        """Await `awaitable`, abandoning it promptly if the turn is cancelled."""
        if self._cancel.is_set():
            for discard in ("close", "cancel"):  # avoid a never-awaited warning
                if hasattr(awaitable, discard):
                    getattr(awaitable, discard)()
                    break
            raise _Cancelled
        task = asyncio.ensure_future(awaitable)
        waiter = self._cancel_waiter
        assert waiter is not None
        try:
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            task.cancel()
            raise
        if task.done():
            return task.result()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise _Cancelled

    # -- tools ----------------------------------------------------------------
    def _blocked(self, turn: _Turn, call: ToolCall) -> bool:
        key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
        turn.call_counts[key] = turn.call_counts.get(key, 0) + 1
        if turn.call_counts[key] < MAX_IDENTICAL_CALLS:
            return False
        turn.blocked += 1
        return True

    def _is_safe(self, call: ToolCall) -> bool:
        spec = self.registry.get(call.name)
        if spec is None:
            return True  # fails fast as "unknown tool"
        try:
            return spec.risk_for(call.arguments or {}) is Risk.SAFE
        except Exception:
            return False

    async def _execute(self, turn: _Turn, call: ToolCall, blocked: bool) -> tuple[ToolResult, float]:
        t0 = time.monotonic()
        if blocked:
            result = ToolResult.fail(
                f"{call.name} was already called with these exact arguments {MAX_IDENTICAL_CALLS - 1} times this turn; "
                "the result will not change",
                "Try different arguments or another approach, or tell the user what is blocking you.",
            )
        else:
            result = await self.registry.execute(call, turn.ctx)
        return result, time.monotonic() - t0

    async def _run_tools(self, turn: _Turn, calls: list[ToolCall]) -> AsyncIterator[AgentEvent]:
        blocked = [self._blocked(turn, c) for c in calls]
        done: dict[int, tuple[ToolResult, float]] = {}
        parallel = len(calls) > 1 and all(b or self._is_safe(c) for b, c in zip(blocked, calls))
        try:
            if parallel:
                async for ev in self._tools_parallel(turn, calls, blocked, done):
                    yield ev
            else:
                async for ev in self._tools_sequential(turn, calls, blocked, done):
                    yield ev
        except BaseException:
            # Cancelled, or the consumer went away: keep the transcript valid.
            for i, call in enumerate(calls):
                if i not in done:
                    self._persist(turn, interrupted_result(call.id, call.name, CANCELLED_TEXT))
            raise
        turn.tool_calls += len(calls)

    async def _tools_sequential(
        self, turn: _Turn, calls: list[ToolCall], blocked: list[bool],
        done: dict[int, tuple[ToolResult, float]],
    ) -> AsyncIterator[AgentEvent]:
        for i, call in enumerate(calls):
            yield ToolStarted(call)
            result, secs = await self._race(self._execute(turn, call, blocked[i]))
            done[i] = (result, secs)
            self._persist(turn, self._tool_message(call, result))
            yield ToolFinished(call, result, secs)

    async def _tools_parallel(
        self, turn: _Turn, calls: list[ToolCall], blocked: list[bool],
        done: dict[int, tuple[ToolResult, float]],
    ) -> AsyncIterator[AgentEvent]:
        for call in calls:
            yield ToolStarted(call)
        results = await self._race(
            asyncio.gather(*(self._execute(turn, c, b) for c, b in zip(calls, blocked)))
        )
        for i, (call, (result, secs)) in enumerate(zip(calls, results)):
            done[i] = (result, secs)
            self._persist(turn, self._tool_message(call, result))
            yield ToolFinished(call, result, secs)

    @staticmethod
    def _tool_message(call: ToolCall, result: ToolResult) -> Message:
        return Message("tool", result.content or "(no output)", tool_call_id=call.id, name=call.name)
