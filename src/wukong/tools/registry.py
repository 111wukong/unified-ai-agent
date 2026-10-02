"""Tool registry."""

from __future__ import annotations

from typing import Any, Iterable

from wukong.errors import ToolError
from wukong.tools.base import Tool, ValidationFailure, validate_against
from wukong.types import EffectClass


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ToolError(f"duplicate tool name {name!r}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(
                f"unknown tool {name!r}. Available: {', '.join(sorted(self._tools)) or '(none)'}"
            ) from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def maybe_get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return [self._tools[k] for k in sorted(self._tools)]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec.as_openai_tool() for t in self.list()]

    def prompt_catalog(self) -> str:
        return "\n".join(t.spec.as_prompt_line() for t in self.list())

    def by_effect(self, effect: EffectClass) -> list[Tool]:
        """Tools in one effect class.

        The reason this exists rather than a filter at the call site: "what
        can this runtime actually execute / write / reach over the network"
        is the question you ask when reviewing the permission surface, and it
        should be answerable from one place.
        """
        return [t for t in self.list() if t.spec.effect_class is effect]

    def validate_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        return validate_against(tool.spec.parameters, args)

    def describe(self, *, effect: EffectClass | None = None) -> list[dict[str, Any]]:
        tools = self.by_effect(effect) if effect is not None else self.list()
        return [
            {
                "name": t.spec.name,
                "effect": t.spec.effect_class.value,
                "source": t.spec.source,
                "idempotent": t.spec.idempotent,
                "requires_confirmation": t.spec.requires_confirmation,
                "description": t.spec.description,
            }
            for t in tools
        ]


__all__ = ["ToolRegistry", "ValidationFailure"]
