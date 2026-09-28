"""Event vocabulary.

Events are the append-only source of truth. Anything not represented here
cannot be replayed, so new behaviour must add an event type before it can
be made durable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # lifecycle
    TASK_CREATED = "task_created"
    TASK_RESUMED = "task_resumed"
    TASK_CANCELLED = "task_cancelled"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    STATE_TRANSITION = "state_transition"
    BUDGET_EXCEEDED = "budget_exceeded"

    # planning
    PLAN_CREATED = "plan_created"
    PLAN_REVISED = "plan_revised"
    PLAN_INVALID = "plan_invalid"

    # model
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    MODEL_ERROR = "model_error"
    MODEL_RETRY = "model_retry"

    # tools
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    TOOL_REPLAYED = "tool_replayed"
    TOOL_AMBIGUOUS = "tool_ambiguous"

    # human in the loop
    CONFIRMATION_REQUESTED = "confirmation_requested"
    CONFIRMATION_GRANTED = "confirmation_granted"
    CONFIRMATION_DENIED = "confirmation_denied"

    # context
    CONTEXT_COMPACTED = "context_compacted"

    # state (drives replay; the TOOL_* events above are for humans)
    LOG_APPENDED = "log_appended"

    # knowledge
    MEMORY_WRITTEN = "memory_written"
    MEMORY_SEARCHED = "memory_searched"
    SKILL_LOADED = "skill_loaded"

    # orchestration. A workflow run is a task, so its progress lands in the
    # same event stream as everything else -- and `uaa task events` shows the
    # whole run without a second viewer.
    WORKFLOW_STARTED = "workflow_started"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    SKILL_CANDIDATE = "skill_candidate"


class Event(BaseModel):
    """A durable fact. `seq` is per-task and gapless."""

    seq: int
    task_id: str
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""

    def summary(self) -> str:
        t = self.type.value
        p = self.payload
        if self.type == EventType.TOOL_STARTED:
            return f"{t}: {p.get('name')}({_short(p.get('arguments'))})"
        if self.type in (EventType.TOOL_COMPLETED, EventType.TOOL_FAILED):
            return f"{t}: {p.get('name')} ok={p.get('success')}"
        if self.type == EventType.MODEL_RESPONSE:
            return (
                f"{t}: {p.get('model')} tools={len(p.get('tool_calls') or [])} "
                f"tokens={p.get('usage', {}).get('total_tokens', 0)}"
            )
        if self.type == EventType.STATE_TRANSITION:
            return f"{t}: {p.get('from')} -> {p.get('to')}"
        if self.type == EventType.CONFIRMATION_REQUESTED:
            return f"{t}: {p.get('tool')} ({p.get('effect')})"
        if self.type == EventType.BUDGET_EXCEEDED:
            return f"{t}: {p.get('kind')} ({p.get('used')}/{p.get('limit')})"
        if self.type == EventType.TOOL_AMBIGUOUS:
            return f"{t}: {p.get('name')} outcome unknown after crash"
        return f"{t}: {_short(p)}"


def _short(value: Any, limit: int = 120) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
