"""review_code: a focused review call, with the next model as a fallback."""

from __future__ import annotations

import pytest

from jarvis.config import Settings
from jarvis.llm.types import LLMResponse, LLMUnavailable
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import coding
from jarvis.tools.registry import ToolRegistry

CODE = "def add(a, b):\n    return a - b\n"
GOOD = "What is wrong: it subtracts.\nCorrected code:\n```python\ndef add(a, b):\n    return a + b\n```"


class Reviewer:
    def __init__(self, model: str, answer) -> None:
        self.model, self.answer, self.calls = model, answer, []

    async def complete(self, messages, tools=None, *, max_tokens=None, temperature=None):
        self.calls.append((messages, tools))
        if isinstance(self.answer, Exception):
            raise self.answer
        return LLMResponse(self.answer, [])


@pytest.fixture
def ctx():
    return ToolContext(settings=Settings())


async def test_review_sends_the_task_and_the_code_alone_and_returns_the_fix(ctx, monkeypatch):
    fast = Reviewer("fast", GOOD)
    monkeypatch.setattr(coding, "_reviewers", lambda c: [fast])
    result = await coding.review_code(CODE, "add two numbers; add(2, 3) is 5", ctx, language="Python", error="add(2, 3) gives -1")
    assert result.ok and "return a + b" in result.content and "unless they asked" in result.content
    messages, tools = fast.calls[0]
    assert tools is None  # a focused call: no tool menu
    assert "every defect" in messages[0].content
    user = messages[1].content
    assert "add(2, 3) is 5" in user and "```python\ndef add(a, b):\n    return a - b\n```" in user and "gives -1" in user
    assert result.data == {"review": GOOD, "model": "fast"}


async def test_review_moves_to_the_next_model_when_one_fails_or_gives_no_code(ctx, monkeypatch):
    down, vague, good = Reviewer("a", LLMUnavailable("down")), Reviewer("b", "Looks fine to me."), Reviewer("c", GOOD)
    monkeypatch.setattr(coding, "_reviewers", lambda c: [down, vague, good])
    result = await coding.review_code(CODE, "add two numbers", ctx)
    assert result.ok and result.data["model"] == "c"
    monkeypatch.setattr(coding, "_reviewers", lambda c: [down, vague])
    result = await coding.review_code(CODE, "add two numbers", ctx)
    assert not result.ok and "a: LLMUnavailable" in result.content and "Review the code yourself" in result.content


async def test_review_refuses_nothing_and_too_much(ctx, monkeypatch):
    monkeypatch.setattr(coding, "_reviewers", lambda c: [Reviewer("fast", GOOD)])
    assert not (await coding.review_code("  ", "x", ctx)).ok
    assert not (await coding.review_code("x" * (coding.MAX_CODE_CHARS + 1), "x", ctx)).ok
    assert (await coding.review_code(CODE, "", ctx)).ok  # a missing task is allowed


def test_review_code_is_a_core_tool_and_the_prompt_points_at_it():
    from jarvis.agent.prompt import build_system_prompt

    registry = ToolRegistry().load_builtin()
    spec = registry.get("review_code")
    assert spec is not None and spec.group == "core" and spec.risk.value == "safe"
    prompt = build_system_prompt(settings=Settings(), facts=[], summary=None, output_mode="text",
                                 is_local=False, tool_groups={"core", "desktop"})
    assert "review_code" in prompt and "copy_text" in prompt
