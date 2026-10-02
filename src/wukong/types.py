"""Core value types shared by every layer.

Kept dependency-light (pydantic only) so models/, tools/ and storage/ can
all import from here without cycles.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EffectClass(str, Enum):
    """What a tool call can do to the world.

    This is the axis the permission engine reasons over. Ordered from
    harmless to dangerous; `rank` is used for "approve everything up to X".
    """

    READ_ONLY = "read_only"
    WRITE_LOCAL = "write_local"
    EXECUTE_LOCAL = "execute_local"
    NETWORK = "network"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    SYSTEM_ADMIN = "system_admin"

    @property
    def rank(self) -> int:
        return _EFFECT_ORDER.index(self)


_EFFECT_ORDER: list[EffectClass] = [
    EffectClass.READ_ONLY,
    EffectClass.WRITE_LOCAL,
    EffectClass.EXECUTE_LOCAL,
    EffectClass.NETWORK,
    EffectClass.EXTERNAL_SIDE_EFFECT,
    EffectClass.SYSTEM_ADMIN,
]


class Decision(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class ToolCall(BaseModel):
    """A model's request to run a tool. `arguments` is already parsed."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def canonical_arguments(self) -> str:
        """Stable serialization, used for the idempotency key.

        Sorted keys, no whitespace, no float formatting surprises.
        """
        return json.dumps(self.arguments, sort_keys=True, separators=(",", ":"), default=str)


class ToolResult(BaseModel):
    success: bool
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Set when output was truncated and the full body was offloaded.
    artifact_path: str | None = None
    truncated: bool = False

    def as_observation(self) -> str:
        if self.success:
            body = self.output
        else:
            body = f"ERROR: {self.error or 'unknown error'}"
            if self.output:
                body += f"\n{self.output}"
        if self.truncated and self.artifact_path:
            body += f"\n\n[output truncated; full body at {self.artifact_path}]"
        return body


class Message(BaseModel):
    """Provider-neutral chat message.

    `tool_calls` on an assistant message and `tool_call_id` on a tool
    message are the only structural coupling to native tool calling.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    # Bookkeeping for context budgeting; never sent to the provider.
    pinned: bool = False
    tokens: int | None = None

    def to_provider_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.role == "assistant" and self.tool_calls:
            out["content"] = self.content or None
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.canonical_arguments()},
                }
                for tc in self.tool_calls
            ]
            return out
        if self.role == "tool":
            out["content"] = self.content
            out["tool_call_id"] = self.tool_call_id
            if self.name:
                out["name"] = self.name
            return out
        out["content"] = self.content
        return out


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
        )


class ModelResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    provider: str = ""
    finish_reason: str | None = None
    # True when tool calls were recovered from a text protocol rather than
    # the provider's native tool-calling field.
    via_text_protocol: bool = False

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def idempotency_key(task_id: str, step_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
    """Stable identity for "this exact call, in this exact task and step".

    **What reads it.** Only the ledger column and the AG-UI encoder (which
    falls back to it for a `call_id`). It is *not* what resume uses to decide
    whether a call already ran -- resume keys off the ledger row's `status`,
    because "was this started and never finished" is a fact the row records
    directly and a hash cannot.

    It was previously documented as letting "the resume logic detect a
    re-run". That was never true, and the mistake was load-bearing: a reader
    trusting the docstring would have assumed crash-recovery was hash-based
    and left the status check out.

    **It collides by design.** `step_id` is the *plan* step, so every call
    made while a plan step is in progress shares a step id. One real run
    produced 120 rows and 38 distinct keys, one of them 15 times over. So do
    not reach for this as a dedup key -- repetition is counted by
    `repetition.call_signature`, which deliberately excludes task and step.
    """
    payload = json.dumps(
        {"task": task_id, "step": step_id, "tool": tool_name, "args": arguments},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
