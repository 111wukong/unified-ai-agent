"""Tool protocol + a small JSON Schema validator.

Why not pull in `jsonschema`: the validator here exists to produce *repair
hints* for the model. `jsonschema`'s messages are written for humans
reading tracebacks, not for feeding back into a prompt, and we only need
the subset that tool schemas actually use.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from wukong.types import EffectClass, ToolResult

# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Everything a tool is allowed to know about the run.

    Note what is *not* here: no ambient `os.environ`, no global cwd. A tool
    that wants either must go through the context, which is what makes the
    permission engine enforceable rather than advisory.
    """

    task_id: str
    session_id: str
    step_id: str
    workspace: Path
    home: Path
    artifact_dir: Path
    tool_call_id: str | None = None
    approved_effects: frozenset[EffectClass] = field(default_factory=frozenset)
    dry_run: bool = False
    # Subprocess environment, already scrubbed of credentials.
    env: dict[str, str] = field(default_factory=dict)

    def is_approved(self, effect: EffectClass) -> bool:
        return effect in self.approved_effects


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    effect_class: EffectClass = EffectClass.READ_ONLY
    # Static requirement; the policy engine may still force confirmation.
    requires_confirmation: bool = False
    # Can this be safely re-run after a crash that left its outcome unknown?
    idempotent: bool = True
    timeout_s: float | None = None
    source: str = "builtin"  # builtin | mcp:<server> | skill:<name>
    tags: list[str] = Field(default_factory=list)
    #: Argument names whose values are files this tool will overwrite.
    #:
    #: Declared rather than inferred, so "which files can this tool damage"
    #: is answerable from the tool's own definition. The runtime snapshots
    #: each of these *before* execution, which is what makes a rewind
    #: possible; a tool that writes files without declaring them here is a
    #: tool whose changes cannot be undone.
    snapshot_paths: list[str] = Field(default_factory=list)

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def as_prompt_line(self) -> str:
        """Compact rendering for the text-protocol fallback."""
        props = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        args = ", ".join(
            f"{name}: {spec.get('type', 'any')}{'' if name in required else '?'}"
            for name, spec in props.items()
        )
        return f"- {self.name}({args}) — {self.description.strip().splitlines()[0]}"


class Tool(ABC):
    """A capability the model can invoke."""

    spec: ToolSpec

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

    # -- optional hooks ---------------------------------------------------
    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate + normalize. Raises ValidationFailure with a repair hint."""
        return validate_against(self.spec.parameters, args)

    def dry_run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            output=f"[dry-run] would execute {self.spec.name}({json.dumps(args, default=str)})",
            metadata={"dry_run": True},
        )

    def preview(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        """Human summary shown in confirmation prompts.

        `ctx` is optional, and only some tools want it: a write needs the
        workspace to diff against the file that is actually on disk. The
        default ignores it and dumps the arguments, which is the honest answer
        for a tool whose whole effect is described by those arguments.
        """
        return f"{self.spec.name}({json.dumps(args, ensure_ascii=False, default=str)})"


class ValidationFailure(Exception):
    """Tool arguments did not match the schema.

    `hint` is written to be pasted straight back into the conversation.
    """

    def __init__(self, hint: str) -> None:
        super().__init__(hint)
        self.hint = hint


# ---------------------------------------------------------------------------
# minimal JSON Schema validation (draft 2020-12 subset)
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def validate_against(schema: dict[str, Any], args: Any, *, path: str = "$") -> Any:
    if not isinstance(schema, dict) or not schema:
        return args

    # composition
    if "oneOf" in schema or "anyOf" in schema:
        options = schema.get("oneOf") or schema.get("anyOf") or []
        errors: list[str] = []
        for option in options:
            try:
                return validate_against(option, args, path=path)
            except ValidationFailure as exc:
                errors.append(exc.hint)
        raise ValidationFailure(
            f"{path}: value did not match any allowed variant. " + " | ".join(errors)
        )

    if "enum" in schema and args not in schema["enum"]:
        raise ValidationFailure(
            f"{path}: {args!r} is not one of {schema['enum']!r}"
        )

    if "const" in schema and args != schema["const"]:
        raise ValidationFailure(f"{path}: expected the constant {schema['const']!r}")

    expected = schema.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(args, t) for t in types):
            raise ValidationFailure(
                f"{path}: expected {'/'.join(types)}, got {type(args).__name__}"
            )

    if isinstance(args, str):
        if "maxLength" in schema and len(args) > schema["maxLength"]:
            raise ValidationFailure(
                f"{path}: string is {len(args)} chars, max {schema['maxLength']}"
            )
        if "minLength" in schema and len(args) < schema["minLength"]:
            raise ValidationFailure(
                f"{path}: string is {len(args)} chars, min {schema['minLength']}"
            )
        if pattern := schema.get("pattern"):
            import re

            if not re.search(pattern, args):
                raise ValidationFailure(f"{path}: {args!r} does not match /{pattern}/")

    if isinstance(args, (int, float)) and not isinstance(args, bool):
        if "minimum" in schema and args < schema["minimum"]:
            raise ValidationFailure(f"{path}: {args} < minimum {schema['minimum']}")
        if "maximum" in schema and args > schema["maximum"]:
            raise ValidationFailure(f"{path}: {args} > maximum {schema['maximum']}")

    if isinstance(args, list):
        if "minItems" in schema and len(args) < schema["minItems"]:
            raise ValidationFailure(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(args) > schema["maxItems"]:
            raise ValidationFailure(f"{path}: at most {schema['maxItems']} items allowed")
        item_schema = schema.get("items")
        if item_schema:
            return [validate_against(item_schema, v, path=f"{path}[{i}]") for i, v in enumerate(args)]
        return args

    if isinstance(args, dict):
        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            known = ", ".join(sorted(properties)) or "(none)"
            raise ValidationFailure(
                f"{path}: missing required argument(s) {missing}. Accepted keys: {known}"
            )
        if schema.get("additionalProperties") is False:
            extra = [k for k in args if k not in properties]
            if extra:
                raise ValidationFailure(
                    f"{path}: unexpected argument(s) {extra}. Accepted keys: "
                    f"{', '.join(sorted(properties)) or '(none)'}"
                )
        out: dict[str, Any] = {}
        for key, value in args.items():
            sub = properties.get(key)
            out[key] = validate_against(sub, value, path=f"{path}.{key}") if sub else value
        return out

    return args


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    expected = _TYPE_MAP.get(type_name)
    return isinstance(value, expected) if expected else True
