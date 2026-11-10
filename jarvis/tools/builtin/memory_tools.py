"""Tools that let the model read and write long-term memory."""

from __future__ import annotations

from typing import Literal

from jarvis.tools.base import ToolContext, ToolResult, tool

def _no_memory() -> ToolResult:
    return ToolResult.fail("long-term memory is not available in this session")


@tool(group="memory")
def remember(
    fact: str,
    ctx: ToolContext,
    category: Literal["profile", "preference", "project", "person", "general"] = "general",
) -> ToolResult:
    """Save a durable fact to long-term memory so it is available in future conversations.

    Use when the user states a lasting preference, a personal detail (name, job, location,
    routine), information about a person or project in their life, or explicitly asks you
    to remember something. Write the fact as one short self-contained sentence. Do not use
    it for one-off task details, temporary state, or things only relevant to this conversation.

    Args:
        fact: The fact to store, e.g. "The user's sister is called Maya and lives in Pune."
        category: profile = who the user is (name, job, location); preference = how they like
            things done; project = ongoing work; person = someone they know; general = anything else.
    """
    if ctx.memory is None:
        return _no_memory()
    if not fact.strip():
        return ToolResult.fail("fact is empty", "Provide the sentence to remember.")
    saved = ctx.memory.remember(fact, category)
    return ToolResult.success(
        f"Remembered (id {saved.id}, {saved.category}): {saved.text}", data={"id": saved.id}
    )


@tool(group="memory")
def recall(query: str, ctx: ToolContext, limit: int = 8) -> ToolResult:
    """Search long-term memory for saved facts about the user and their world.

    Use when the answer may depend on something the user told you before (preferences,
    people, projects, personal details) and it is not already in the conversation. Each
    result shows an id, which can be passed to forget.

    Args:
        query: Keywords to look for, e.g. "sister birthday" or "favourite editor".
        limit: Maximum number of facts to return.
    """
    if ctx.memory is None:
        return _no_memory()
    facts = ctx.memory.recall(query, limit=max(1, limit))
    if not facts:
        return ToolResult.success("No saved facts yet.", data=[])
    lines = [f"[{f.id}] ({f.category}) {f.text}" for f in facts]
    return ToolResult.success("\n".join(lines), data=[f.id for f in facts])


@tool(group="memory")
def forget(fact_id: int, ctx: ToolContext) -> ToolResult:
    """Delete one saved fact from long-term memory.

    Use when the user asks you to forget something or says a saved fact is wrong or
    outdated. Find the id with recall first; if the fact changed, remember the corrected
    version afterwards.

    Args:
        fact_id: The id shown in square brackets in recall results.
    """
    if ctx.memory is None:
        return _no_memory()
    if ctx.memory.forget(fact_id):
        return ToolResult.success(f"Forgot fact {fact_id}.")
    return ToolResult.fail(f"no saved fact with id {fact_id}", "Use recall to list current ids.")


@tool(group="memory")
def search_past_conversations(query: str, ctx: ToolContext, limit: int = 6) -> ToolResult:
    """Search earlier conversations with the user for what was said.

    Use when the user refers to something discussed before ("what did we decide about...",
    "that script from last week") and it is not in the current conversation. This searches
    only what the user and assistant said, not tool output. For durable preferences and
    personal details use recall instead.

    Args:
        query: Keywords likely to appear in the earlier discussion.
        limit: Maximum number of matching messages to return.
    """
    if ctx.memory is None:
        return _no_memory()
    hits = ctx.memory.search_messages(query, limit=max(1, limit))
    if not hits:
        return ToolResult.success("No matching past messages.", data=[])
    lines = [f"{h['created_at'][:16]} {h['role']}: {h['text']}" for h in hits]
    return ToolResult.success("\n".join(lines), data=hits)
