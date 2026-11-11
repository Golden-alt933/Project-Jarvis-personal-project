"""Optional arguments a model fills with "none" or null mean "left out"."""

from __future__ import annotations

from typing import Literal

from jarvis.config import Settings
from jarvis.llm.types import ToolCall
from jarvis.tools.base import ToolContext, ToolResult, tool
from jarvis.tools.registry import ToolRegistry


@tool(group="core")
def look(text: str, filter: str | None = None, wait_for: str = "", mode: Literal["fast", "none"] = "fast",
         count: int = 3) -> ToolResult:
    """Echo the arguments as received.

    Args:
        text: Required text.
        filter: Optional filter.
        wait_for: Optional text to wait for.
        mode: A choice that really includes "none".
        count: A number.
    """
    return ToolResult.success(repr((text, filter, wait_for, mode, count)))


async def call(**arguments) -> ToolResult:
    registry = ToolRegistry()
    registry.register(look.__tool_spec__)
    return await registry.execute(ToolCall("x", "look", arguments), ToolContext(settings=Settings()))


async def test_placeholders_in_optional_arguments_become_the_default():
    r = await call(text="hi", filter="none", wait_for="NULL", count=None)
    assert r.content == repr(("hi", None, "", "fast", 3))
    r = await call(text="hi", filter=" None ", wait_for="undefined")
    assert r.content == repr(("hi", None, "", "fast", 3))


async def test_real_values_required_arguments_and_listed_choices_are_left_alone():
    r = await call(text="none", filter="n/a", wait_for="nothing", mode="none")
    assert r.content == repr(("none", "n/a", "nothing", "none", 3))
    assert not (await call(text=None)).ok  # a required argument is still required
