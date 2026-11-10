"""Tool registry: discovery, schemas, and the single guarded execution path.

Every tool call from the model goes through ``execute``. It never raises: any
problem comes back as a failed ToolResult the model can read and react to.
"""

from __future__ import annotations

import asyncio
import difflib
import importlib
import inspect
import logging
import pkgutil
from typing import Any, Iterable

from pydantic import ValidationError

from jarvis.llm.types import ToolCall
from jarvis.tools.base import Risk, ToolContext, ToolResult, ToolSpec

log = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, result_max_chars: int = 12000) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._result_max_chars = result_max_chars

    # -- registration -------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._tools[spec.name] = spec

    def register_module(self, module: Any) -> int:
        count = 0
        for _, obj in inspect.getmembers(module):
            spec = getattr(obj, "__tool_spec__", None)
            if isinstance(spec, ToolSpec) and getattr(obj, "__module__", None) == module.__name__:
                self.register(spec)
                count += 1
        return count

    def load_builtin(self) -> "ToolRegistry":
        """Import every module in jarvis.tools.builtin. A module whose optional
        dependency is missing is skipped with a warning instead of breaking startup."""
        import jarvis.tools.builtin as pkg

        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.startswith("_"):
                continue
            try:
                module = importlib.import_module(f"{pkg.__name__}.{info.name}")
            except ImportError as exc:
                log.warning("tool module %s skipped: %s", info.name, exc)
                continue
            self.register_module(module)
        return self

    # -- lookup -------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def specs(self, groups: Iterable[str] | None = None) -> list[ToolSpec]:
        wanted = set(groups) if groups is not None else None
        return [
            s
            for s in sorted(self._tools.values(), key=lambda s: (s.group, s.name))
            if wanted is None or s.group in wanted
        ]

    def schemas(self, groups: Iterable[str] | None = None) -> list[dict[str, Any]]:
        return [s.to_openai() for s in self.specs(groups)]

    def groups(self) -> list[str]:
        return sorted({s.group for s in self._tools.values()})

    # -- execution ----------------------------------------------------------
    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        spec = self._tools.get(call.name)
        if spec is None:
            close = difflib.get_close_matches(call.name, self._tools, n=3, cutoff=0.5)
            hint = f"Did you mean: {', '.join(close)}?" if close else "Use only the listed tools."
            return ToolResult.fail(f"unknown tool '{call.name}'", hint)

        try:
            parsed = spec.args_model.model_validate(call.arguments or {})
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()
            )
            return ToolResult.fail(
                f"invalid arguments for {spec.name}: {problems}",
                "Fix the arguments and call the tool again.",
            )
        args = {name: getattr(parsed, name) for name in spec.args_model.model_fields}

        if spec.risk_for(args) is Risk.CONFIRM:
            try:
                approved = await ctx.confirm(spec.confirmation_text(args))
            except Exception:  # a broken frontend must not approve by accident
                log.exception("confirm callback failed")
                approved = False
            if not approved:
                return ToolResult.fail(
                    "the user declined this action",
                    "Do not retry it. Tell the user it was not done, or ask what they want instead.",
                )

        kwargs = dict(args)
        if spec.wants_ctx:
            kwargs["ctx"] = ctx
        try:
            if inspect.iscoroutinefunction(spec.fn):
                result = await asyncio.wait_for(spec.fn(**kwargs), spec.timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(spec.fn, **kwargs), spec.timeout
                )
        except asyncio.TimeoutError:
            return ToolResult.fail(
                f"{spec.name} timed out after {spec.timeout:.0f}s",
                "The action may or may not have completed; check before retrying.",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("tool %s crashed", spec.name)
            return ToolResult.fail(f"{spec.name} failed: {type(exc).__name__}: {exc}")

        if not isinstance(result, ToolResult):
            result = ToolResult.success("" if result is None else str(result))
        if len(result.content) > self._result_max_chars:
            omitted = len(result.content) - self._result_max_chars
            result.content = (
                result.content[: self._result_max_chars]
                + f"\n[... {omitted} more characters truncated. Narrow the request to see the rest.]"
            )
        return result
