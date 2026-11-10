"""History assembly, repair and compaction.

Invariants: history sent to a model never splits an assistant tool-call message from
its tool results, and compaction cuts only immediately before a user message.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from jarvis.llm.types import LLMClient, Message

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.memory.store import MemoryStore

log = logging.getLogger(__name__)

KEEP_USER_TURNS = 6
KEEP_USER_TURNS_AGGRESSIVE = 2
OLD_TOOL_RESULT_LIMIT = 2000  # chars sent for tool results in older turns
RECENT_TURNS_UNTRUNCATED = 2
TRANSCRIPT_TOOL_LIMIT = 1500
TRANSCRIPT_MAX_CHARS = 60000

INTERRUPTED_TEXT = "Error: this tool call was interrupted before it finished; its result is unknown."

_SUMMARY_SYSTEM = (
    "You condense a conversation between a user and their assistant Jarvis into notes the "
    "assistant will rely on later. Preserve: the user's goals, decisions made, facts learned "
    "(including about the user), exact file paths, names and numbers, and any unfinished "
    "work or open questions. Drop pleasantries and raw tool output. Write compact plain "
    "text, no preamble."
)


def estimate_tokens(messages: list[Message]) -> int:
    chars = 0
    for m in messages:
        chars += len(m.content) + 16
        for c in m.tool_calls:
            chars += len(c.name) + len(json.dumps(c.arguments, default=str))
    return chars // 4


def repair_history(messages: list[Message]) -> list[Message]:
    """Insert synthetic results for tool calls that have none (app killed mid-turn)."""
    out: list[Message] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        out.append(m)
        i += 1
        if m.role != "assistant" or not m.tool_calls:
            continue
        answered: set[str | None] = set()
        while i < len(messages) and messages[i].role == "tool":
            answered.add(messages[i].tool_call_id)
            out.append(messages[i])
            i += 1
        for call in m.tool_calls:
            if call.id not in answered:
                out.append(interrupted_result(call.id, call.name))
    return out


def interrupted_result(call_id: str, name: str, text: str = INTERRUPTED_TEXT) -> Message:
    return Message("tool", text, tool_call_id=call_id, name=name)


def _user_indices(entries: list) -> list[int]:
    return [i for i, (_, m) in enumerate(entries) if m.role == "user"]


def _format_transcript(entries: list) -> str:
    lines: list[str] = []
    for _, m in entries:
        if m.role == "tool":
            body = m.content
            if len(body) > TRANSCRIPT_TOOL_LIMIT:
                body = body[:TRANSCRIPT_TOOL_LIMIT] + " [truncated]"
            lines.append(f"TOOL RESULT ({m.name}): {body}")
        elif m.role == "assistant":
            if m.content:
                lines.append(f"JARVIS: {m.content}")
            for c in m.tool_calls:
                lines.append(f"JARVIS CALLED {c.name}({json.dumps(c.arguments, default=str)})")
        else:
            lines.append(f"USER: {m.content}")
    text = "\n".join(lines)
    if len(text) > TRANSCRIPT_MAX_CHARS:
        text = "[start omitted]\n" + text[-TRANSCRIPT_MAX_CHARS:]
    return text


class ContextManager:
    def __init__(
        self, store: "MemoryStore", settings: "Settings", fast_client: LLMClient
    ) -> None:
        self.store = store
        self.settings = settings
        self.fast_client = fast_client

    # -- assembling ---------------------------------------------------------
    async def build(self, conversation_id: str) -> tuple[list[Message], str | None]:
        summary_row = self.store.get_summary(conversation_id)
        summary, upto = summary_row if summary_row else (None, 0)
        entries = self.store.load_messages(conversation_id, after_id=upto)
        messages = repair_history([m for _, m in entries])
        return self._truncate_old_results(messages), summary

    def repair_stored(self, conversation_id: str) -> int:
        """Persist synthetic results if the stored history ends mid tool-call."""
        summary_row = self.store.get_summary(conversation_id)
        upto = summary_row[1] if summary_row else 0
        stored = [m for _, m in self.store.load_messages(conversation_id, after_id=upto)]
        repaired = repair_history(stored)
        added = repaired[len(stored):] if repaired[: len(stored)] == stored else []
        for m in added:
            self.store.append_message(conversation_id, m)
        return len(added)

    def _truncate_old_results(self, messages: list[Message]) -> list[Message]:
        users = [i for i, m in enumerate(messages) if m.role == "user"]
        if len(users) <= RECENT_TURNS_UNTRUNCATED:
            return messages
        boundary = users[-RECENT_TURNS_UNTRUNCATED]
        out: list[Message] = []
        for i, m in enumerate(messages):
            if i < boundary and m.role == "tool" and len(m.content) > OLD_TOOL_RESULT_LIMIT:
                omitted = len(m.content) - OLD_TOOL_RESULT_LIMIT
                m = replace(
                    m, content=m.content[:OLD_TOOL_RESULT_LIMIT] + f"\n[... {omitted} chars omitted from older turn]"
                )
            out.append(m)
        return out

    def over_budget(self, messages: list[Message]) -> bool:
        return estimate_tokens(messages) > self.settings.history_token_budget

    # -- compaction ---------------------------------------------------------
    async def compact(self, conversation_id: str, *, aggressive: bool = False) -> bool:
        """Summarise the oldest part of the history. Returns True if anything was cut."""
        summary_row = self.store.get_summary(conversation_id)
        old_summary, upto = summary_row if summary_row else (None, 0)
        entries = self.store.load_messages(conversation_id, after_id=upto)
        users = _user_indices(entries)
        keep = KEEP_USER_TURNS_AGGRESSIVE if aggressive else KEEP_USER_TURNS
        if len(users) <= keep:
            return False
        cut = users[-keep]  # index of a user message: never inside a tool-call group
        if cut == 0:
            return False
        head = entries[:cut]
        new_summary = await self._summarise(old_summary, head)
        self.store.set_summary(conversation_id, new_summary, head[-1][0])
        log.info("compacted %d messages (aggressive=%s)", len(head), aggressive)
        return True

    async def _summarise(self, old_summary: str | None, head: list) -> str:
        prompt = ""
        if old_summary:
            prompt += f"Existing notes about the earlier conversation:\n{old_summary}\n\n"
        prompt += f"New conversation to fold into the notes:\n{_format_transcript(head)}"
        try:
            reply = await self.fast_client.complete(
                [Message("system", _SUMMARY_SYSTEM), Message("user", prompt)],
                max_tokens=1024,
                temperature=0.2,
            )
            text = reply.content.strip()
            if text:
                return text
            log.warning("summariser returned nothing; dropping old turns")
        except Exception:
            log.exception("summariser failed; dropping old turns")
        placeholder = f"(Earlier conversation, {len(head)} messages, was trimmed and not summarised.)"
        return f"{old_summary}\n{placeholder}" if old_summary else placeholder
