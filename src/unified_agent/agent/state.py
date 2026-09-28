"""Agent state + event replay.

The original spec had `AgentState` and an `events` table without saying
which one is authoritative. Here: **events are truth, state is a fold over
them**. `tasks.state` in SQLite is a projection kept for fast reads and is
recomputed by `replay()` on resume. If the two disagree, the projection is
wrong by construction -- which removes an entire class of "resumed into a
state that never existed" bugs.

The other thing this file decides: the conversation sent to the model is
*rebuilt* from state each step, not appended to. That makes compaction and
resume trivially correct, and avoids the provider-specific failure where a
rebuilt history has a dangling `tool_call_id`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from unified_agent.observability.events import EventType
from unified_agent.types import TokenUsage


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
RESUMABLE_STATUSES = {
    TaskStatus.PENDING,
    TaskStatus.PLANNING,
    TaskStatus.RUNNING,
    TaskStatus.WAITING_CONFIRMATION,
}


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """An *intention*, not a bound call.

    Deliberately no `tool_name` / `arguments` fields (the spec had them).
    A plan that pre-binds arguments is stale the moment the first tool
    result comes back, and models then follow the stale plan instead of
    the evidence. Arguments are chosen at execution time; the plan records
    only what the step is for and which tools it expects to need.
    """

    id: str
    description: str
    expected_tools: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    note: str = ""


class LogEntry(BaseModel):
    """One line of the agent's scratchpad.

    `kind="tool"` is a tool result; `kind="note"` is the model's own
    interim text; `kind="system"` is the runtime telling the model
    something (a failure, a policy denial, an ambiguity warning).
    """

    index: int
    step_id: str = ""
    kind: Literal["tool", "note", "system"] = "tool"
    assistant_note: str = ""
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    text: str = ""
    artifact_path: str | None = None
    attempt: int = 1
    replayed: bool = False
    approved: bool = False
    ambiguous: bool = False

    def render(self, *, max_chars: int = 6_000) -> str:
        text = self.text
        if len(text) > max_chars:
            text = text[: max_chars // 2] + "\n...\n" + text[-max_chars // 2 :]
        if self.kind == "tool":
            status = "ok" if self.success else "FAILED"
            flags = []
            if self.approved:
                flags.append("ran after approval")
            if self.replayed:
                flags.append("re-run after interrupt")
            if self.ambiguous:
                flags.append("OUTCOME UNKNOWN")
            if self.attempt > 1:
                flags.append(f"attempt {self.attempt}")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            head = f"[{self.index}] {self.tool}({_compact_args(self.arguments)}) -> {status}{suffix}"
            body = f"{head}\n{text}"
            if self.assistant_note:
                body = f"{head}\n(assistant said: {self.assistant_note[:400]})\n{text}"
            return body
        if self.kind == "system":
            return f"[{self.index}] SYSTEM: {text}"
        return f"[{self.index}] assistant: {text}"


def _compact_args(args: dict[str, Any], limit: int = 160) -> str:
    import json

    text = json.dumps(args, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class PendingConfirmation(BaseModel):
    request_id: str
    step_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    effect: str = ""
    reason: str = ""
    preview: str = ""


class AgentState(BaseModel):
    task_id: str
    session_id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    plan: list[PlanStep] = Field(default_factory=list)
    current_step: int = 0
    log: list[LogEntry] = Field(default_factory=list)
    # Number of log entries dropped by compaction. Keeps display indices
    # monotonic so "[12]" still refers to the same thing after a compaction.
    log_offset: int = 0
    compacted_summary: str | None = None
    steps_used: int = 0
    model_calls: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    pending_confirmation: PendingConfirmation | None = None
    # A call the user approved (or refused) while the task was paused. Kept
    # separately from `pending_confirmation` so that resume() can execute the
    # *exact* call that was approved instead of asking the model again --
    # otherwise the model re-decides and the approved call silently vanishes.
    approved_pending: PendingConfirmation | None = None
    denied_pending: PendingConfirmation | None = None
    answer: str | None = None
    error: str | None = None
    started_at: float = 0.0
    deadline: float = 0.0
    #: The directory this task operates in, recorded at creation.
    #:
    #: Part of the task's identity, not of the process that happens to be
    #: running it: the tool context, the sandbox wrap and the path fence all
    #: derive from it, and resuming in a different directory would move every
    #: side effect while the model still believed it was somewhere else.
    workspace: str = ""
    loaded_skills: list[str] = Field(default_factory=list)
    # Effects the user pre-approved for this task (from the CLI or an
    # earlier `approve`). Kept on the state so a resume keeps the grant.
    approved_effects: list[str] = Field(default_factory=list)

    # -- convenience ------------------------------------------------------
    def next_log_index(self) -> int:
        return self.log_offset + len(self.log)

    def append_log(self, entry: LogEntry) -> LogEntry:
        entry.index = self.next_log_index()
        self.log.append(entry)
        return entry

    def sync_current_step(self) -> int:
        """Derive the cursor from the checklist instead of tracking it separately.

        Two counters that must agree (a cursor and a status per step) is a
        classic source of drift; deriving one from the other removes it.
        """
        for i, step in enumerate(self.plan):
            if step.status not in {StepStatus.COMPLETED, StepStatus.SKIPPED}:
                self.current_step = i
                return i
        self.current_step = len(self.plan)
        return self.current_step

    def active_step(self) -> PlanStep | None:
        self.sync_current_step()
        if 0 <= self.current_step < len(self.plan):
            return self.plan[self.current_step]
        return None

    def step_id(self) -> str:
        step = self.active_step()
        return step.id if step else f"step_{self.current_step}"

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def replay(events: list, *, task_id: str, session_id: str = "") -> AgentState:
    """Fold an event stream back into state.

    Deterministic: no wall clock, no randomness, no I/O. That is what makes
    `resume` safe -- the same events always produce the same state, so the
    projection in SQLite can be treated as a cache.
    """
    state = AgentState(task_id=task_id, session_id=session_id, goal="")

    for event in events:
        payload = event.payload
        kind = event.type

        if kind is EventType.TASK_CREATED:
            state.goal = payload.get("goal", "")
            state.session_id = payload.get("session_id", session_id)
            budgets = payload.get("budgets") or {}
            state.deadline = float(budgets.get("deadline") or 0.0)
            state.started_at = float(budgets.get("started_at") or 0.0)
            state.workspace = str(budgets.get("workspace") or "")
            state.approved_effects = list(budgets.get("approved_effects") or [])
            state.status = TaskStatus.PENDING

        elif kind is EventType.PLAN_CREATED:
            state.plan = [PlanStep(**s) for s in payload.get("steps") or []]
            state.current_step = 0

        elif kind is EventType.PLAN_REVISED:
            state.plan = [PlanStep(**s) for s in payload.get("steps") or []]
            state.current_step = int(payload.get("current_step") or 0)

        elif kind is EventType.LOG_APPENDED:
            entry = LogEntry(**payload["entry"])
            state.log.append(entry)

        elif kind is EventType.MODEL_RESPONSE:
            state.model_calls += 1
            usage = payload.get("usage") or {}
            state.usage = state.usage + TokenUsage(**usage)
            # Derive the step counter from events rather than trusting the
            # in-memory counter. Otherwise `resume` resets it to zero and the
            # step budget can be reset indefinitely by resuming in a loop.
            if payload.get("phase") == "step":
                state.steps_used += 1

        elif kind is EventType.CONTEXT_COMPACTED:
            state.compacted_summary = payload.get("summary")
            dropped = int(payload.get("kept_from", 0))
            state.log_offset += dropped
            state.log = state.log[dropped:]

        elif kind is EventType.CONFIRMATION_REQUESTED:
            state.pending_confirmation = PendingConfirmation(
                request_id=payload["request_id"],
                step_id=payload.get("step_id", ""),
                tool=payload.get("tool", ""),
                arguments=payload.get("arguments") or {},
                effect=payload.get("effect", ""),
                reason=payload.get("reason", ""),
                preview=payload.get("preview", ""),
            )
            state.status = TaskStatus.WAITING_CONFIRMATION

        elif kind is EventType.CONFIRMATION_GRANTED:
            request = state.pending_confirmation
            if request and request.request_id == payload.get("request_id"):
                if request.effect and request.effect not in state.approved_effects:
                    state.approved_effects.append(request.effect)
                state.approved_pending = request
                state.pending_confirmation = None
                state.status = TaskStatus.RUNNING

        elif kind is EventType.CONFIRMATION_DENIED:
            request = state.pending_confirmation
            if request and request.request_id == payload.get("request_id"):
                state.denied_pending = request
            state.pending_confirmation = None
            state.status = TaskStatus.RUNNING

        elif kind is EventType.SKILL_LOADED:
            # Folded rather than kept in memory: a resume rebuilds state from
            # events alone, so a loaded-skill list that only lived in RAM
            # would silently vanish and the task-end attribution would be
            # missing exactly the skills that were used before the crash.
            name = payload.get("name")
            if name and name not in state.loaded_skills:
                state.loaded_skills.append(name)

        elif kind is EventType.STATE_TRANSITION:
            target = payload.get("to")
            if target:
                state.status = TaskStatus(target)

        elif kind is EventType.BUDGET_EXCEEDED:
            state.status = TaskStatus.FAILED
            state.error = f"budget exceeded: {payload.get('kind')}"

        elif kind is EventType.TASK_COMPLETED:
            state.answer = payload.get("answer")
            state.status = TaskStatus.COMPLETED

        elif kind is EventType.TASK_FAILED:
            state.error = payload.get("error")
            state.status = TaskStatus.FAILED

        elif kind is EventType.TASK_CANCELLED:
            state.status = TaskStatus.CANCELLED
            state.error = "cancelled by user"

    return state
