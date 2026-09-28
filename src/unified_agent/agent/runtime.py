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
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from unified_agent.agent.context import ContextBuilder
from unified_agent.agent.executor import FINISH_TOOL, PLAN_TOOL, SKILL_TOOL, ToolRunner
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
from unified_agent.observability.bus import EventBus
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
        bus: EventBus | None = None,
        memory: Any = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.registry = registry
        self.engine = engine
        self.models = models
        self.skills = skills
        self.on_progress = on_progress
        self.bus = bus
        self.memory = memory
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
        task_id: str | None = None,
        parent_task_id: str | None = None,
        reflect: bool | None = None,
    ) -> AgentResult:
        """`task_id` lets a caller subscribe to the event stream *before* the
        task exists. Without it the SSE endpoint races the first events.

        `reflect` overrides `agent.reflect` for this run only. Sub-agents pass
        `False`: a fan-out of five sub-agents would otherwise pay five extra
        extraction calls, and the orchestrator already reflects once over the
        combined result.
        """
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
            # Recorded so a resume can refuse to operate somewhere else. The
            # session carries it too, but a task that describes itself does
            # not depend on a join to answer "where does this act".
            "workspace": str(self.settings.workspace),
        }
        task_id = self.store.create_task(
            session_id=session_id,
            goal=goal,
            budgets=budgets,
            task_id=task_id,
            parent_task_id=parent_task_id,
        )
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
        result = await self._drive(state, max_steps=max_steps or self.settings.agent.max_steps)
        await self._reflect(state, result, enabled=reflect)
        return result

    async def resume(self, task_id: str, *, max_steps: int | None = None) -> AgentResult:
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id}")
        events = self.store.events(task_id)
        state = replay(events, task_id=task_id, session_id=task["session_id"])
        if state.status.terminal:
            return self._result(state, duration=0.0)

        # A task acts where it was created, not where the resuming process
        # happens to have been launched. Without this check, `uaa task approve`
        # run from a different directory silently moved every subsequent side
        # effect -- and the path fence would have allowed it, because the fence
        # is built from *this* process's workspace while the model's view of
        # "the project" came from the recorded one. Two answers to the same
        # question, and the wrong one wins.
        recorded = state.workspace
        if recorded and Path(recorded).resolve() != Path(self.settings.workspace).resolve():
            raise ValueError(
                f"task {task_id} was created in {recorded!r} but this process "
                f"is working in {str(self.settings.workspace)!r}. Resuming it "
                "here would move every side effect. Re-run with "
                f"`--workspace {recorded}`."
            )

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
        result = await self._drive(state, max_steps=max_steps or self.settings.agent.max_steps)
        await self._reflect(state, result)
        return result

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
                self._record_skill_runs(state)
                return self._result(state, duration=time.time() - started)

            state.steps_used += 1
            await self._maybe_compact(state)
            memory_block = await self._memory_block(state)
            messages = self._context.build(state, memory_block=memory_block)

            # --- model call --------------------------------------------
            # One id per assistant turn, so the AG-UI encoder can correlate
            # the ephemeral text deltas with the durable MODEL_RESPONSE event.
            message_id = f"msg_{uuid.uuid4().hex[:12]}"
            try:
                resp = await self._call_model(
                    state,
                    messages,
                    stream=self._stream_callback(state.task_id, message_id),
                )
            except ModelError as exc:
                return self._fail(state, f"model call failed: {exc}", started)
            state.model_calls += 1
            state.usage = state.usage + resp.usage
            self.store.append(
                state.task_id,
                EventType.MODEL_RESPONSE,
                {
                    "phase": "step",
                    "message_id": message_id,
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
            if not resp.wants_tools:
                if not (resp.content or "").strip():
                    # An empty response is not an answer. Reporting `completed`
                    # here is how a task that did nothing looks like a task
                    # that succeeded -- and for a reasoning model it is the
                    # common shape of "the output budget ran out before the
                    # model got to its answer".
                    return self._fail(state, _empty_response_reason(resp), started)
                return self._complete(state, resp.content, started)

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
            if call.name == SKILL_TOOL and entry.success:
                self._note_skill_loaded(entry, state)
            self._progress("tool", {"name": call.name, "success": entry.success})
        return None

    def _note_skill_loaded(self, entry: LogEntry, state: AgentState) -> None:
        """Record which skills a task actually pulled in.

        `state.loaded_skills` is what the task-end `skill_runs` row is built
        from, so without this the whole skill-effectiveness signal is empty.
        """
        name = str(entry.arguments.get("name") or "").strip()
        if not name:
            return
        if name not in state.loaded_skills:
            state.loaded_skills.append(name)
        self.store.append(
            state.task_id,
            EventType.SKILL_LOADED,
            {
                "name": name,
                "step_id": entry.step_id,
                "allowed_tools": (entry.arguments or {}).get("allowed_tools"),
            },
        )

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

    def _stream_callback(self, task_id: str, message_id: str):
        """Only stream when someone is listening.

        Without the subscriber check, every run pays the SSE parsing cost in
        the provider adapter for output nobody reads.
        """
        bus = self.bus
        if bus is None or not bus.has_subscribers(task_id):
            return None

        def on_delta(text: str) -> None:
            bus.publish_delta(task_id, message_id=message_id, text=text)

        return on_delta

    async def _call_model(self, state: AgentState, messages: list, *, stream: Any = None) -> Any:
        model = self._model
        attempts = max(1, self.settings.agent.model_retry_attempts)
        last: ModelError | None = None
        self.store.append(
            state.task_id,
            EventType.MODEL_REQUEST,
            {
                "phase": "step",
                "messages": len(messages),
                "attempt": 1,
                "tools": len(self.registry.specs()),
            },
        )
        for attempt in range(attempts):
            try:
                return await model.chat(
                    messages, tools=self.registry.specs(), stream=stream
                )
            except ModelError as exc:
                last = exc
                self.store.append(
                    state.task_id,
                    EventType.MODEL_ERROR,
                    {
                        "phase": "step",
                        "attempt": attempt + 1,
                        "retryable": exc.retryable,
                        "error": str(exc)[:500],
                    },
                )
                if not exc.retryable or attempt == attempts - 1:
                    raise
                delay = min(2**attempt, 8)
                self.store.append(
                    state.task_id,
                    EventType.MODEL_RETRY,
                    {"attempt": attempt + 1, "delay_s": delay, "error": str(exc)[:500]},
                )
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
        # The task's recorded workspace, not the process's. `resume` refuses a
        # mismatch, so these agree -- reading the task's value here is what
        # makes that agreement explicit rather than incidental.
        workspace = Path(state.workspace) if state.workspace else self.settings.workspace
        return ToolContext(
            task_id=state.task_id,
            session_id=state.session_id,
            step_id=state.step_id(),
            workspace=workspace,
            home=self.settings.home,
            artifact_dir=self.settings.artifact_dir,
            approved_effects=frozenset(effects),
            env=scrub_env(known_secrets=self.settings.secret_values()),
        )

    async def _memory_block(self, state: AgentState) -> str:
        """Recalled memories, hybrid when a provider is configured.

        The query is the goal rather than the last observation: at this point
        in the loop the goal is what defines relevance, and embedding a
        changing query every step would make the recalled set flicker.
        """
        mode = "fts"
        if self.memory is not None:
            try:
                hits = await self.memory.recall(state.goal[:200], limit=5)
                mode = "hybrid" if self.memory.embeddings.available else "fts"
            except Exception:  # noqa: BLE001 - recall must never break a task
                hits = []
        else:
            hits = self.store.search_memories(state.goal[:60], limit=5)
        # Recorded even when nothing matched: "recall ran and found nothing"
        # and "recall never ran" are different, and only the event tells them
        # apart when a task behaves as though it had no context.
        self.store.append(
            state.task_id,
            EventType.MEMORY_SEARCHED,
            {
                "query": state.goal[:200],
                "mode": mode,
                "hits": len(hits),
                "ids": [h.get("id") for h in hits],
            },
        )
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
        self._record_skill_runs(state)
        return self._result(state, duration=max(0.0, time.time() - started))

    def _fail(self, state: AgentState, error: str, started: float) -> AgentResult:
        state.error = error
        state.status = TaskStatus.FAILED
        self.store.append(state.task_id, EventType.TASK_FAILED, {"error": error})
        self._persist(state)
        self._record_skill_runs(state)
        return self._result(state, duration=max(0.0, time.time() - started))

    def _record_skill_runs(self, state: AgentState) -> None:
        """Attribute the task's outcome to every skill it loaded.

        This is the only feedback loop a skill has: without it `deprecate`
        is a decision made on vibes, and `skill_runs` stays an empty table
        that the schema promised would answer "does this skill ever help?".
        """
        if not state.loaded_skills:
            return
        outcome = state.status.value
        for name in state.loaded_skills:
            self.store.record_skill_run(
                skill_name=name, task_id=state.task_id, outcome=outcome
            )

    async def _reflect(
        self, state: AgentState, result: AgentResult, *, enabled: bool | None = None
    ) -> None:
        """Post-task learning: one extra model call, at the very end.

        This lives in the runtime rather than in a front end because it is
        part of the kernel. It used to run only from `uaa run --reflect`, so
        a task started over HTTP never wrote a memory and never produced a
        skill candidate -- the learning path existed and was reachable from
        exactly one of the two entry points.
        """
        if enabled is None:
            enabled = self.settings.agent.reflect
        if not enabled or result.status != TaskStatus.COMPLETED.value:
            return
        from unified_agent.agent.reflector import Reflector

        try:
            reflector = Reflector(
                model=self._model,
                settings=self.settings,
                store=self.store,
                memory=self.memory,
            )
            if not reflector.worth_running(state):
                return
            await reflector.run_and_persist(state)
        except Exception as exc:  # noqa: BLE001 - learning must never fail a task
            self._progress(
                "reflection_failed", {"error": f"{type(exc).__name__}: {exc}"[:300]}
            )

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


def _empty_response_reason(resp: Any) -> str:
    """Why an empty response is a failure, in terms the user can act on.

    `finish_reason == "length"` is the case worth naming precisely: reasoning
    models spend output tokens on their reasoning, so a budget that looks
    generous for prose can be exhausted before any content is emitted. The
    message says which setting to raise rather than leaving the user to guess
    why their agent "completed" without doing anything.
    """
    if resp.finish_reason == "length":
        return (
            "the model used its entire output budget before producing an answer "
            "(finish_reason=length). Nothing was accomplished. Raise "
            "`models.<alias>.max_output_tokens`, or reduce the reasoning effort "
            "if the provider exposes one."
        )
    return (
        "the model returned no content and no tool calls "
        f"(finish_reason={resp.finish_reason!r}). Nothing was accomplished, so "
        "this is a failure rather than an empty answer."
    )


__all__ = ["AgentRuntime", "AgentResult"]
