"""The agent loop.

Differences from the spec's loop that matter:

* **Budgets are enforced, not reported.** Steps, tokens, cost and wall
  clock are all checked at the top of every iteration. The spec listed them
  under observability; a metric you only record after the fact does not
  stop a runaway loop.
* **Cancellation is a file, not a flag.** A task started in one process can
  be cancelled from another (`uaa task cancel`), so the check has to cross
  a process boundary.
* **In-flight tool calls are resolved explicitly on resume.** See
  `_resolve_unfinished_calls`. This is the failure mode that makes naive
  "resume" dangerous, and it is the reason the ledger is write-ahead.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable

from pydantic import BaseModel, Field

from unified_agent.agent.context import ContextBuilder
from unified_agent.agent.executor import FINISH_TOOL, PLAN_TOOL, ToolRunner
from unified_agent.agent.planner import Planner, apply_plan_update
from unified_agent.agent.state import (
    AgentState,
    LogEntry,
    PendingConfirmation,
    TaskStatus,
    replay,
)
from unified_agent.config import Settings
from unified_agent.errors import ConfirmationRequired, ModelError
from unified_agent.models.registry import ModelRegistry
from unified_agent.observability.events import EventType
from unified_agent.tools.base import ToolContext
from unified_agent.tools.permissions import PermissionEngine, scrub_env
from unified_agent.tools.registry import ToolRegistry
from unified_agent.types import EffectClass, TokenUsage, ToolCall

ProgressHook = Callable[[str, dict[str, Any]], None]


class AgentResult(BaseModel):
    task_id: str
    session_id: str
    status: str
    answer: str | None = None
    error: str | None = None
    steps: int = 0
    model_calls: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    pending_confirmation: PendingConfirmation | None = None
    duration_s: float = 0.0
    plan: list[str] = Field(default_factory=list)

    @property
    def needs_approval(self) -> bool:
        return self.status == TaskStatus.WAITING_CONFIRMATION.value


class AgentRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        store: Any,
        registry: ToolRegistry,
        engine: PermissionEngine,
        models: ModelRegistry,
        skills: Any = None,
        summarizer: Any = None,
        on_progress: ProgressHook | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.registry = registry
        self.engine = engine
        self.models = models
        self.skills = skills
        self.on_progress = on_progress
        self._runner = ToolRunner(
            registry=registry,
            engine=engine,
            store=store,
            settings=settings,
            redactor=store.redactor,
        )
        self._context: ContextBuilder | None = None
        self._model: Any = None
        self._summarizer = summarizer

    # -- public API -------------------------------------------------------
    async def run(
        self,
        goal: str,
        *,
        session_id: str,
        model_alias: str | None = None,
        approved_effects: list[EffectClass] | None = None,
        max_steps: int | None = None,
    ) -> AgentResult:
        model = self.models.get(model_alias)
        self._model = model
        effects = [e.value for e in (approved_effects or [])]

        budgets = {
            "max_steps": max_steps or self.settings.agent.max_steps,
            "max_tokens": self.settings.agent.max_tokens,
            "max_cost_usd": self.settings.agent.max_cost_usd,
            "started_at": time.time(),
            "deadline": time.time() + self.settings.agent.wall_clock_s,
            "approved_effects": effects,
            "model": model_alias or self.settings.default_model,
        }
        task_id = self.store.create_task(session_id=session_id, goal=goal, budgets=budgets)
        state = AgentState(
            task_id=task_id,
            session_id=session_id,
            goal=goal,
            started_at=budgets["started_at"],
            deadline=budgets["deadline"],
            approved_effects=effects,
        )
        self._context = ContextBuilder(
            settings=self.settings,
            model=model,
            tool_catalog=self.registry.prompt_catalog(),
            skills_index=self.skills.index_prompt() if self.skills else "",
            store=self.store,
            summarizer=self._summarizer or self.models.summarizer(),
        )
        return await self._drive(state, max_steps=max_steps or self.settings.agent.max_steps)

    async def resume(self, task_id: str, *, max_steps: int | None = None) -> AgentResult:
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id}")
        events = self.store.events(task_id)
        state = replay(events, task_id=task_id, session_id=task["session_id"])
        if state.status.terminal:
            return self._result(state, duration=0.0)

        model_alias = None
        for event in events:
            if event.type is EventType.TASK_CREATED:
                model_alias = (event.payload.get("budgets") or {}).get("model")
                break
        model = self.models.get(model_alias)
        self._model = model
        self._context = ContextBuilder(
            settings=self.settings,
            model=model,
            tool_catalog=self.registry.prompt_catalog(),
            skills_index=self.skills.index_prompt() if self.skills else "",
            store=self.store,
            summarizer=self._summarizer or self.models.summarizer(),
        )

        self._transition(state, TaskStatus.RUNNING)
        self.store.append(task_id, EventType.TASK_RESUMED, {"last_seq": self.store.last_seq(task_id)})
        self._progress("resumed", {"task_id": task_id, "status": state.status.value})

        # A human decision taken while the task was paused. Both branches
        # re-enter the loop with the *same* call the model already chose --
        # asking the model again would let it silently pick something else.
        if state.approved_pending is not None:
            pending = state.approved_pending
            state.approved_pending = None
            self._progress("approval_granted", {"tool": pending.tool, "effect": pending.effect})
            # Execute the *same* call the model chose. `approved=True` only
            # affects the scratchpad label -- the call itself is a first
            # execution, not a replay.
            await self._runner.execute(
                ToolCall(id=pending.request_id, name=pending.tool, arguments=pending.arguments),
                state=state,
                ctx=self._tool_context(state),
                approved=True,
            )
        elif state.denied_pending is not None:
            pending = state.denied_pending
            state.denied_pending = None
            state.append_log(
                LogEntry(
                    index=0,
                    kind="system",
                    text=(
                        f"The user REFUSED to approve {pending.tool} ({pending.effect}). "
                        "Do not attempt this call or any other route to the same effect. "
                        "Either finish with what you have, or report the blockage."
                    ),
                )
            )
            self.store.append(
                task_id, EventType.LOG_APPENDED, {"entry": state.log[-1].model_dump(mode="json")}
            )
            self._persist(state)

        await self._resolve_unfinished_calls(state)
        return await self._drive(state, max_steps=max_steps or self.settings.agent.max_steps)

    async def approve(self, task_id: str, *, request_id: str | None = None) -> AgentResult:
        state, _ = self._load(task_id)
        pending = state.pending_confirmation
        if pending is None:
            return self._result(state, duration=0.0)
        if request_id and request_id != pending.request_id:
            raise ValueError(f"request {request_id!r} is not pending; {pending.request_id!r} is")
        self.store.append(
            task_id,
            EventType.CONFIRMATION_GRANTED,
            {"request_id": pending.request_id, "tool": pending.tool, "effect": pending.effect},
        )
        return await self.resume(task_id)

    async def deny(self, task_id: str, *, request_id: str | None = None, note: str = "") -> AgentResult:
        state, _ = self._load(task_id)
        pending = state.pending_confirmation
        if pending is None:
            return self._result(state, duration=0.0)
        if request_id and request_id != pending.request_id:
            raise ValueError(f"request {request_id!r} is not pending; {pending.request_id!r} is")
        self.store.append(
            task_id,
            EventType.CONFIRMATION_DENIED,
            {"request_id": pending.request_id, "tool": pending.tool, "note": note},
        )
        return await self.resume(task_id)

    def cancel(self, task_id: str) -> bool:
        marker = self.settings.state_dir / f"{task_id}.cancel"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(time.time()), encoding="utf-8")
        return True

    def is_cancelled(self, task_id: str) -> bool:
        return (self.settings.state_dir / f"{task_id}.cancel").exists()

    # -- loop -------------------------------------------------------------
    async def _drive(self, state: AgentState, *, max_steps: int) -> AgentResult:
        started = time.time()
        model = self._model

        # planning
        if not state.plan:
            self._transition(state, TaskStatus.PLANNING)
            self._progress("planning", {})
            planner = Planner(model=model, settings=self.settings, store=self.store)
            try:
                steps = await planner.plan(state, tool_names=self.registry.names())
            except ModelError as exc:
                return self._fail(state, f"planning failed: {exc}", started)
            state.plan = steps
            state.sync_current_step()
            self.store.append(
                state.task_id,
                EventType.PLAN_CREATED,
                {"steps": [s.model_dump(mode="json") for s in steps]},
            )
            self._progress("plan", {"steps": [s.description for s in steps]})

        self._transition(state, TaskStatus.RUNNING)

        while True:
            # --- budget gate -------------------------------------------
            over = self._check_budget(state, max_steps)
            if over:
                self.store.append(
                    state.task_id,
                    EventType.BUDGET_EXCEEDED,
                    {"kind": over[0], "used": over[1], "limit": over[2]},
                )
                return self._fail(
                    state,
                    f"stopped: {over[0]} budget exhausted ({over[1]} / {over[2]})",
                    started,
                )
            if self.is_cancelled(state.task_id):
                self.store.append(state.task_id, EventType.TASK_CANCELLED, {})
                state.status = TaskStatus.CANCELLED
                state.error = "cancelled"
                self._persist(state)
                return self._result(state, duration=time.time() - started)

            state.steps_used += 1
            await self._maybe_compact(state)
            memory_block = self._memory_block(state)
            messages = self._context.build(state, memory_block=memory_block)

            # --- model call --------------------------------------------
            try:
                resp = await self._call_model(messages)
            except ModelError as exc:
                return self._fail(state, f"model call failed: {exc}", started)
            state.model_calls += 1
            state.usage = state.usage + resp.usage
            self.store.append(
                state.task_id,
                EventType.MODEL_RESPONSE,
                {
                    "phase": "step",
                    "model": resp.model,
                    "usage": resp.usage.model_dump(),
                    "tool_calls": [tc.name for tc in resp.tool_calls],
                    "via_text_protocol": resp.via_text_protocol,
                    "content_preview": (resp.content or "")[:300],
                },
            )
            if resp.content:
                self._progress("assistant", {"text": resp.content})

            # --- no tool call: the answer ------------------------------
            if not resp.tool_calls:
                return self._complete(state, resp.content or "(the model returned no content)", started)

            # --- execute ------------------------------------------------
            try:
                finished = await self._run_tool_calls(resp.tool_calls, state)
            except ConfirmationRequired as need:
                return self._pause_for_confirmation(state, need, started)
            self._persist(state)
            if finished is not None:
                return finished

    async def _run_tool_calls(
        self, calls: list[ToolCall], state: AgentState
    ) -> AgentResult | None:
        for call in calls:
            if call.name == PLAN_TOOL:
                await self._apply_plan_tool(call, state)
                continue
            if call.name == FINISH_TOOL:
                result = self._finish_from_tool(call, state)
                if result is not None:
                    return result
                continue
            entry = await self._runner.execute(call, state=state, ctx=self._tool_context(state))
            self._progress("tool", {"name": call.name, "success": entry.success})
        return None

    async def _apply_plan_tool(self, call: ToolCall, state: AgentState) -> None:
        try:
            args = self.registry.validate_args(PLAN_TOOL, call.arguments)
        except Exception as exc:  # noqa: BLE001
            state.append_log(
                LogEntry(index=0, kind="system", text=f"update_plan rejected: {exc}")
            )
            self.store.append(
                state.task_id,
                EventType.LOG_APPENDED,
                {"entry": state.log[-1].model_dump(mode="json")},
            )
            return
        apply_plan_update(state, args["steps"])
        self.store.append(
            state.task_id,
            EventType.PLAN_REVISED,
            {
                "steps": [s.model_dump(mode="json") for s in state.plan],
                "current_step": state.current_step,
                "reason": args.get("reason"),
            },
        )
        self._progress("plan_revised", {"steps": [s.description for s in state.plan]})

    def _finish_from_tool(self, call: ToolCall, state: AgentState) -> AgentResult | None:
        """Returns a result to end the task, or None to keep going.

        A malformed `finish` must not end the task with a bogus answer --
        that is worse than another turn. The validation error goes back into
        the scratchpad and the loop continues.
        """
        try:
            args = self.registry.validate_args(FINISH_TOOL, call.arguments)
        except Exception as exc:  # noqa: BLE001
            state.append_log(
                LogEntry(
                    index=0,
                    kind="system",
                    text=f"finish rejected — {exc}. Fix the arguments or answer in plain text.",
                )
            )
            self.store.append(
                state.task_id,
                EventType.LOG_APPENDED,
                {"entry": state.log[-1].model_dump(mode="json")},
            )
            return None
        answer = args["answer"]
        if args.get("status") == "blocked":
            state.answer = answer
            state.error = "the agent reported it was blocked"
            state.status = TaskStatus.FAILED
            self.store.append(
                state.task_id, EventType.TASK_FAILED, {"error": state.error, "answer": answer}
            )
            self._persist(state)
            return self._result(state, duration=0.0)
        return self._complete(state, answer, time.time())

    async def _call_model(self, messages: list) -> Any:
        model = self._model
        attempts = max(1, self.settings.agent.model_retry_attempts)
        last: ModelError | None = None
        for attempt in range(attempts):
            try:
                return await model.chat(messages, tools=self.registry.specs())
            except ModelError as exc:
                last = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                delay = min(2**attempt, 8)
                self._progress("model_retry", {"attempt": attempt + 1, "error": str(exc)})
                await asyncio.sleep(delay)
        raise last or ModelError("model call failed")

    # -- resume semantics -------------------------------------------------
    async def _resolve_unfinished_calls(self, state: AgentState) -> None:
        """Decide what to do about tool calls whose outcome we never learned.

        This is the part naive `resume` implementations get wrong. A row with
        status='started' means the process died between "about to touch the
        world" and "here is what happened". Two cases:

        * tool declares `idempotent=True` -> safe to re-run; do it and hand
          the model the fresh result.
        * tool declares `idempotent=False` (any shell command) -> we do NOT
          know whether it ran. Re-running `git commit` or a deploy would be
          a silent double-execution. Instead the model is told the outcome
          is unknown and must verify the state before continuing.
        """
        unfinished = self.store.unfinished_calls(state.task_id)
        for record in unfinished:
            tool = self.registry.maybe_get(record.name)
            if tool is not None and tool.spec.idempotent:
                self.store.finish_tool_call(record.id, status="retry", error="interrupted")
                text = (
                    f"{record.name}({record.arguments}) was interrupted before it finished. "
                    "It is idempotent, so it is being re-run now."
                )
                state.append_log(LogEntry(index=0, kind="system", text=text))
                self.store.append(
                    state.task_id,
                    EventType.LOG_APPENDED,
                    {"entry": state.log[-1].model_dump(mode="json")},
                )
                await self._runner.execute(
                    ToolCall(id=record.id, name=record.name, arguments=record.arguments),
                    state=state,
                    ctx=self._tool_context(state),
                    attempt=int(record.attempt or 1) + 1,
                    replay_reason="resumed",
                )
                continue

            self.store.finish_tool_call(record.id, status="ambiguous", error="outcome unknown")
            self.store.append(
                state.task_id,
                EventType.TOOL_AMBIGUOUS,
                {"name": record.name, "arguments": record.arguments, "step_id": record.step_id},
            )
            text = (
                f"WARNING: {record.name}({record.arguments}) was in flight when the process "
                "stopped. Its outcome is UNKNOWN — it may have taken effect, or not at all. "
                "Do not assume either way: inspect the current state (files, git status, "
                "process output) before you continue. If it was a side-effecting command, "
                "tell the user it may have run twice."
            )
            state.append_log(LogEntry(index=0, kind="system", text=text))
            self.store.append(
                state.task_id,
                EventType.LOG_APPENDED,
                {"entry": state.log[-1].model_dump(mode="json")},
            )

    # -- helpers ----------------------------------------------------------
    def _tool_context(self, state: AgentState) -> ToolContext:
        effects = {EffectClass(e) for e in state.approved_effects}
        return ToolContext(
            task_id=state.task_id,
            session_id=state.session_id,
            step_id=state.step_id(),
            workspace=self.settings.workspace,
            home=self.settings.home,
            artifact_dir=self.settings.artifact_dir,
            approved_effects=frozenset(effects),
            env=scrub_env(known_secrets=self.settings.secret_values()),
        )

    def _memory_block(self, state: AgentState) -> str:
        hits = self.store.search_memories(state.goal[:60], limit=5)
        if not hits:
            return ""
        lines = ["# Recalled memories", ""]
        lines += [f"- [{h['scope']}] {h['content'][:400]}" for h in hits]
        return "\n".join(lines)

    async def _maybe_compact(self, state: AgentState) -> None:
        if self._context is None:
            return
        if await self._context.maybe_compact(state):
            self._progress("compacted", {"summary": (state.compacted_summary or "")[:200]})

    def _check_budget(self, state: AgentState, max_steps: int) -> tuple[str, Any, Any] | None:
        if state.steps_used >= max_steps:
            return ("steps", state.steps_used, max_steps)
        if state.usage.total_tokens >= self.settings.agent.max_tokens:
            return ("tokens", state.usage.total_tokens, self.settings.agent.max_tokens)
        if (
            self.settings.agent.max_cost_usd > 0
            and state.usage.cost_usd >= self.settings.agent.max_cost_usd
        ):
            return ("cost_usd", round(state.usage.cost_usd, 4), self.settings.agent.max_cost_usd)
        if state.deadline and time.time() > state.deadline:
            return ("wall_clock_s", int(time.time() - state.started_at), int(state.deadline - state.started_at))
        return None

    def _pause_for_confirmation(
        self, state: AgentState, need: ConfirmationRequired, started: float
    ) -> AgentResult:
        request_id = f"req_{uuid.uuid4().hex[:10]}"
        tool = self.registry.maybe_get(need.tool)
        preview = tool.preview(need.arguments) if tool else f"{need.tool}({need.arguments})"
        pending = PendingConfirmation(
            request_id=request_id,
            step_id=state.step_id(),
            tool=need.tool,
            arguments=need.arguments,
            effect=need.effect,
            reason=str(need),
            preview=preview,
        )
        state.pending_confirmation = pending
        state.status = TaskStatus.WAITING_CONFIRMATION
        self.store.append(
            state.task_id,
            EventType.CONFIRMATION_REQUESTED,
            {
                "request_id": request_id,
                "step_id": pending.step_id,
                "tool": pending.tool,
                "arguments": pending.arguments,
                "effect": pending.effect,
                "reason": pending.reason,
                "preview": preview,
            },
        )
        self._persist(state)
        self._progress("confirmation", {"preview": preview, "effect": pending.effect})
        return self._result(state, duration=time.time() - started)

    def _complete(self, state: AgentState, answer: str, started: float) -> AgentResult:
        state.answer = answer
        state.status = TaskStatus.COMPLETED
        self.store.append(
            state.task_id,
            EventType.TASK_COMPLETED,
            {"answer": answer, "steps": state.steps_used, "usage": state.usage.model_dump()},
        )
        self._persist(state)
        return self._result(state, duration=max(0.0, time.time() - started))

    def _fail(self, state: AgentState, error: str, started: float) -> AgentResult:
        state.error = error
        state.status = TaskStatus.FAILED
        self.store.append(state.task_id, EventType.TASK_FAILED, {"error": error})
        self._persist(state)
        return self._result(state, duration=max(0.0, time.time() - started))

    def _transition(self, state: AgentState, to: TaskStatus) -> None:
        previous = state.status
        if previous is to:
            return
        state.status = to
        self.store.append(
            state.task_id,
            EventType.STATE_TRANSITION,
            {"from": previous.value, "to": to.value},
        )

    def _persist(self, state: AgentState) -> None:
        self.store.save_projection(
            state.task_id,
            status=state.status.value,
            state=state.snapshot(),
            steps_used=state.steps_used,
            tokens_in=state.usage.prompt_tokens,
            tokens_out=state.usage.completion_tokens,
            cost_usd=state.usage.cost_usd,
            result={"answer": state.answer, "error": state.error},
            pending_confirmation=state.pending_confirmation.model_dump(mode="json")
            if state.pending_confirmation
            else None,
            error=state.error,
        )

    def _result(self, state: AgentState, *, duration: float) -> AgentResult:
        return AgentResult(
            task_id=state.task_id,
            session_id=state.session_id,
            status=state.status.value,
            answer=state.answer,
            error=state.error,
            steps=state.steps_used,
            model_calls=state.model_calls,
            usage=state.usage,
            pending_confirmation=state.pending_confirmation,
            duration_s=round(duration, 3),
            plan=[f"[{s.status.value}] {s.description}" for s in state.plan],
        )

    def _load(self, task_id: str) -> tuple[AgentState, dict[str, Any]]:
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id}")
        return replay(self.store.events(task_id), task_id=task_id, session_id=task["session_id"]), task

    def _progress(self, kind: str, payload: dict[str, Any]) -> None:
        if self.on_progress:
            try:
                self.on_progress(kind, payload)
            except Exception:  # noqa: BLE001 - UI must never break a task
                pass


__all__ = ["AgentRuntime", "AgentResult"]
