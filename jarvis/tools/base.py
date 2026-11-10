"""The tool contract.

A tool is a plain typed function. The JSON schema the model sees is generated
from the signature and docstring, so there is no hand-written schema to drift.

    @tool(group="files", risk=Risk.CONFIRM)
    async def delete_file(path: str, ctx: ToolContext) -> ToolResult:
        '''Move a file to the Recycle Bin.

        Args:
            path: Absolute path of the file to delete.
        '''

Rules for tool authors:
- Parameters are JSON-friendly types (str, int, float, bool, list[...], Literal[...],
  X | None). A parameter named ``ctx`` is injected and hidden from the model.
- Return ``ToolResult.success(text, data=...)`` or ``ToolResult.fail(error, hint=...)``.
  ``content`` is what the model reads: factual, compact, no prose padding. On failure
  say what went wrong and, where possible, what to try instead.
- Never parse the user's sentence. The model has already turned it into arguments.
- Raise nothing on expected failures; unexpected exceptions are caught by the registry.
"""

from __future__ import annotations

import asyncio
import enum
import inspect
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.memory.store import MemoryStore


class Risk(enum.Enum):
    SAFE = "safe"  # runs without asking
    CONFIRM = "confirm"  # the user must approve this call


@dataclass
class ToolResult:
    ok: bool
    content: str
    data: Any = None

    @classmethod
    def success(cls, content: str, data: Any = None) -> "ToolResult":
        return cls(True, content, data)

    @classmethod
    def fail(cls, error: str, hint: str | None = None) -> "ToolResult":
        text = f"Error: {error}"
        if hint:
            text += f"\nHint: {hint}"
        return cls(False, text)


async def _deny(_: str) -> bool:
    return False


async def _ignore(_: str) -> None:
    return None


@dataclass
class ToolContext:
    """Everything a tool may touch besides its arguments."""

    settings: "Settings"
    memory: "MemoryStore | None" = None
    # Ask the user to approve something; returns True if approved.
    confirm: Callable[[str], Awaitable[bool]] = _deny
    # Show a short progress line to the user while a long tool runs.
    notify: Callable[[str], Awaitable[None]] = _ignore
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    conversation_id: str | None = None


RiskRule = Risk | Callable[[dict[str, Any]], Risk]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema (object)
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    risk: RiskRule = Risk.SAFE
    group: str = "core"
    timeout: float = 30.0
    # Human-readable line shown when asking for confirmation.
    describe: Callable[[dict[str, Any]], str] | None = None
    wants_ctx: bool = False

    def risk_for(self, args: dict[str, Any]) -> Risk:
        return self.risk(args) if callable(self.risk) else self.risk

    def confirmation_text(self, args: dict[str, Any]) -> str:
        if self.describe:
            return self.describe(args)
        shown = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        return f"{self.name}({shown})"

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _short(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


_ARG_LINE = re.compile(r"^\s{0,8}(\w+)\s*(?:\([^)]*\))?:\s*(.+)$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (description, {param: description})."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    desc: list[str] = []
    params: dict[str, str] = {}
    section = "desc"
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            section, current = "args", None
            continue
        if stripped in ("Returns:", "Raises:", "Example:", "Examples:", "Note:", "Notes:"):
            section, current = "other", None
            continue
        if section == "desc":
            desc.append(stripped)
        elif section == "args":
            m = _ARG_LINE.match(line)
            if m and not line.startswith(" " * 9):
                current = m.group(1)
                params[current] = m.group(2).strip()
            elif current and stripped:
                params[current] += " " + stripped
    return " ".join(p for p in desc if p).strip(), params


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $defs so providers that reject $ref still accept the schema."""
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs.get(node["$ref"].split("/")[-1], {})
                return walk(dict(target))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def build_spec(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    risk: RiskRule = Risk.SAFE,
    group: str = "core",
    timeout: float = 30.0,
    describe: Callable[[dict[str, Any]], str] | None = None,
) -> ToolSpec:
    description, param_docs = _parse_docstring(fn.__doc__)
    if not description:
        raise ValueError(f"tool {fn.__name__} needs a docstring: the model reads it")
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    wants_ctx = False
    for pname, param in inspect.signature(fn).parameters.items():
        if pname == "ctx":
            wants_ctx = True
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ValueError(f"tool {fn.__name__}: *args/**kwargs are not supported")
        annotation = hints.get(pname, str)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (annotation, Field(default, description=param_docs.get(pname)))
    args_model = create_model(  # type: ignore[call-overload]
        f"{fn.__name__}_args", __config__=ConfigDict(extra="forbid"), **fields
    )
    schema = _strip_titles(_inline_refs(args_model.model_json_schema()))
    schema.setdefault("properties", {})
    schema["type"] = "object"
    return ToolSpec(
        name=name or fn.__name__,
        description=description,
        parameters=schema,
        fn=fn,
        args_model=args_model,
        risk=risk,
        group=group,
        timeout=timeout,
        describe=describe,
        wants_ctx=wants_ctx,
    )


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    risk: RiskRule = Risk.SAFE,
    group: str = "core",
    timeout: float = 30.0,
    describe: Callable[[dict[str, Any]], str] | None = None,
) -> Any:
    """Mark a function as a tool. Usable bare (``@tool``) or with options."""

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        f.__tool_spec__ = build_spec(  # type: ignore[attr-defined]
            f, name=name, risk=risk, group=group, timeout=timeout, describe=describe
        )
        return f

    return wrap(fn) if fn is not None else wrap
