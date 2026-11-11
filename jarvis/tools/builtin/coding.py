"""Code review: a focused second pass over the user's code.

The model driving a turn sees ninety tools and a conversation; asked "what is wrong with my
code" in the middle of that, it tends to stop at the first defect. The same model given
only the task and the code, and told to trace every requirement and example, finds the rest
(measured on the free NVIDIA tier: 6 of 6 two-defect solutions fully corrected in 3 to 9
seconds, against roughly half in the middle of an agent turn). This tool is that focused
call. It never runs the code.
"""

from __future__ import annotations

import logging

from jarvis.llm.types import LLMClient, LLMError, Message
from jarvis.tools.base import ToolContext, ToolResult, tool

log = logging.getLogger(__name__)

MAX_CODE_CHARS = 30000
MAX_TASK_CHARS = 12000
REVIEW_SYSTEM = (
    "You are a meticulous code reviewer. You are given what a piece of code is meant to do and the "
    "code itself. Find every defect, not just the first: read each requirement and each example, "
    "trace the code against them one by one, and check the edge cases (empty input, duplicates, "
    "ordering of the output, off-by-one, wrong data structure). Keep the user's structure, names and "
    "signature; change only what is needed. Reply in exactly this form:\n"
    "What is wrong: one short line per defect.\n"
    "Corrected code: the complete corrected code in a single fenced code block.\n"
    "If the code is already correct, say so in one line and repeat it unchanged in the block."
)


def _reviewers(ctx: ToolContext) -> list[LLMClient]:
    """Models to ask, in order: the fast hosted model first (quick, and reliable at this
    narrow job), then the other hosted models."""
    from jarvis.llm.providers import build_alternates, build_fast_client

    clients: list[LLMClient] = []
    try:
        clients.append(build_fast_client(ctx.settings))
    except Exception:
        log.debug("no fast model for code review", exc_info=True)
    try:
        for client in build_alternates(ctx.settings):
            if all(client.model != seen.model for seen in clients):
                clients.append(client)
    except Exception:
        log.debug("no alternate models for code review", exc_info=True)
    return clients


@tool(group="core", timeout=150)
async def review_code(code: str, task: str, ctx: ToolContext, language: str = "", error: str = "") -> ToolResult:
    """Find everything that is wrong with a piece of code and get the corrected version.

    Call this for every "what is wrong with my code", "why does this fail", "fix my solution"
    or "check my code" request, before you answer: it traces the code against each
    requirement and example of the task, which catches defects a quick read misses. Pass the
    code exactly as it is and everything known about what it should do. It only reads the
    code; nothing is run or changed.

    Args:
        code: The user's code, exactly as written.
        task: What the code is meant to do: the full problem statement with its examples and constraints, or the user's own description.
        language: The programming language, if known.
        error: The error message or wrong output the user is seeing, if any.
    """
    if not code or not code.strip():
        return ToolResult.fail("there is no code to review", "Get the user's code first, for example with copy_text.")
    if len(code) > MAX_CODE_CHARS:
        return ToolResult.fail(f"the code is longer than {MAX_CODE_CHARS} characters", "Review it one function or file at a time.")
    task = (task or "").strip()[:MAX_TASK_CHARS]
    parts = [f"What the code should do:\n{task or '(not stated: judge from the code itself)'}"]
    if error and error.strip():
        parts.append(f"What goes wrong when it runs:\n{error.strip()[:3000]}")
    parts.append(f"Code:\n```{language.strip().lower()}\n{code.rstrip()}\n```")
    messages = [Message("system", REVIEW_SYSTEM), Message("user", "\n\n".join(parts))]
    problems: list[str] = []
    for client in _reviewers(ctx):
        try:
            reply = await client.complete(messages, None, max_tokens=4096, temperature=0.1)
        except LLMError as exc:
            problems.append(f"{client.model}: {type(exc).__name__}")
            continue
        text = (reply.content or "").strip()
        if "```" not in text:
            problems.append(f"{client.model}: no corrected code in the reply")
            continue
        return ToolResult.success(
            f"Review by {client.model}:\n{text}\n\n"
            "Tell the user what was wrong in a sentence or two and show them the corrected code in a "
            "fenced block. Do not change their editor unless they asked you to write or fix it there; "
            "when they did ask, call paste_text with exactly this corrected code and replace_all=true. "
            "The code was not run.",
            data={"review": text, "model": client.model},
        )
    detail = "; ".join(problems) or "no model is configured"
    return ToolResult.fail(
        f"the review could not be done ({detail})",
        "Review the code yourself, carefully, against every requirement and example.",
    )
