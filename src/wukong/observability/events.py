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
    #: A call that was refused because the agent had already made the exact
    #: same call and would have learned nothing new. Distinct from TOOL_FAILED
    #: because nothing went wrong -- the runtime declined to spend the tokens.
    #: Recorded so a task that ends on a step budget can be told apart from
    #: one that ended because the agent would not stop re-reading.
    TOOL_REPEATED = "tool_repeated"
    #: A pre-image of a file, captured *before* a write tool touches it.
    #:
    #: Recorded before execution, not after, for the same reason the tool
    #: ledger is write-ahead: if the process dies mid-write, the only copy of
    #: the original is the one taken beforehand. This is what makes
    #: `wukong task rewind` possible -- without it the runtime can replay
    #: *state*, but it cannot put a file back.
    FILE_CHECKPOINT = "file_checkpoint"

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
    # same event stream as everything else -- and `wukong task events` shows the
    # whole run without a second viewer.
    WORKFLOW_STARTED = "workflow_started"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    SKILL_CANDIDATE = "skill_candidate"

    # multi-agent delegation. A sub-agent is a child task, so its own events
    # live under its own task_id; these are the parent's record of having
    # delegated, which is what makes the fan-out visible from one place.
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_COMPLETED = "subagent_completed"
    SUBAGENT_FAILED = "subagent_failed"


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
