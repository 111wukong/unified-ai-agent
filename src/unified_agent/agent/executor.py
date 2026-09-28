"""Tool execution: policy, write-ahead ledger, truncation, artifact offload.

The ordering here is the whole point and is easy to get subtly wrong:

    validate -> decide -> **ledger(begin)** -> execute -> ledger(end)

The ledger row is written *before* the tool touches anything. If the
process dies between begin and end, the row survives with status='started'
and `runtime` can tell the difference between "definitely did not run" and
"may or may not have run". Skipping that write -- the obvious optimisation
-- is what makes crash recovery unsound in most agent runtimes.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from unified_agent.agent.state import AgentState, LogEntry
from unified_agent.config import Settings
from unified_agent.errors import ConfirmationRequired, PermissionDenied, ToolError
from unified_agent.observability.events import EventType
from unified_agent.observability.redact import Redactor
from unified_agent.tools.base import ToolContext, ValidationFailure
from unified_agent.tools.permissions import PermissionEngine
from unified_agent.tools.registry import ToolRegistry
from unified_agent.types import ToolCall, ToolResult, idempotency_key

# Tools whose result the runtime intercepts instead of just logging.
PLAN_TOOL = "update_plan"
FINISH_TOOL = "finish"


class ToolRunner:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        engine: PermissionEngine,
        store: Any,
        settings: Settings,
        redactor: Redactor | None = None,
    ) -> None:
        self.registry = registry
        self.engine = engine
        self.store = store
        self.settings = settings
        self.redactor = redactor

    async def execute(
        self,
        call: ToolCall,
        *,
        state: AgentState,
        ctx: ToolContext,
        attempt: int = 1,
        replay_reason: str | None = None,
        approved: bool = False,
    ) -> LogEntry:
        step_id = ctx.step_id
        key = idempotency_key(state.task_id, step_id, call.name, call.arguments)

        # 1. tool exists?
        try:
            tool = self.registry.get(call.name)
        except ToolError as exc:
            return self._fail(state, step_id, call, str(exc), attempt=attempt)

        # 2. arguments valid?
        try:
            args = tool.validate(call.arguments)
        except ValidationFailure as exc:
            return self._fail(
                state,
                step_id,
                call,
                f"invalid arguments — {exc.hint}",
                attempt=attempt,
                args=call.arguments,
            )

        # 3. policy
        verdict = self.engine.decide(tool, args)
        if verdict.denied:
            entry = self._fail(
                state,
                step_id,
                call,
                f"PERMISSION DENIED — {verdict.reason}. This is a hard stop; "
                "do not attempt a different route to the same effect.",
                attempt=attempt,
                args=args,
            )
            self.store.append(
                state.task_id,
                EventType.TOOL_FAILED,
                {
                    "name": call.name,
                    "effect": tool.spec.effect_class.value,
                    "denied": True,
                    "reason": verdict.reason,
                    "step_id": step_id,
                },
            )
            return entry
        if verdict.needs_confirmation and not ctx.is_approved(tool.spec.effect_class):
            raise ConfirmationRequired(
                verdict.reason or f"{tool.spec.effect_class.value} requires approval",
                tool=call.name,
                arguments=args,
                effect=tool.spec.effect_class.value,
            )

        # 4. write-ahead
        call_id = self.store.begin_tool_call(
            task_id=state.task_id,
            step_id=step_id,
            attempt=attempt,
            name=call.name,
            arguments=args,
            effect_class=tool.spec.effect_class.value,
            idempotency_key=key,
        )
        ctx.tool_call_id = call_id
        self.store.append(
            state.task_id,
            EventType.TOOL_STARTED,
            {
                "call_id": call_id,
                "name": call.name,
                "arguments": args,
                "effect": tool.spec.effect_class.value,
                "step_id": step_id,
                "attempt": attempt,
                "idempotency_key": key,
            },
        )

        # 5. execute
        started = time.monotonic()
        timeout = tool.spec.timeout_s or self.settings.permissions.shell.default_timeout_s
        try:
            result = await asyncio.wait_for(tool.run(args, ctx), timeout=timeout)
        except asyncio.TimeoutError:
            result = ToolResult(
                success=False, error=f"{call.name} exceeded its {timeout:.0f}s timeout"
            )
        except PermissionDenied as exc:
            result = ToolResult(success=False, error=f"PERMISSION DENIED — {exc}")
        except Exception as exc:  # noqa: BLE001 - a tool crash is data, not a task failure
            result = ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
        duration_ms = int((time.monotonic() - started) * 1000)

        # 6. truncate + offload (the artifact keeps the *full* body, so the
        #    truncation is lossy for the prompt and lossless on disk)
        full_output = result.output
        rendered, truncated = truncate(full_output, self.settings.agent.tool_output_chars)
        result.output = rendered
        result.truncated = truncated
        if truncated:
            result.artifact_path = self.store.save_artifact(
                task_id=state.task_id,
                tool_call_id=call_id,
                content=full_output,
                artifact_dir=self.settings.artifact_dir,
                sha256=hashlib.sha256(full_output.encode("utf-8")).hexdigest()[:16],
                suffix="txt",
            )

        # 7. close the ledger
        self.store.finish_tool_call(
            call_id,
            status="succeeded" if result.success else "failed",
            result={"output": rendered, "artifact": result.artifact_path, "metadata": result.metadata},
            error=result.error,
            duration_ms=duration_ms,
        )
        self.store.append(
            state.task_id,
            EventType.TOOL_COMPLETED if result.success else EventType.TOOL_FAILED,
            {
                "call_id": call_id,
                "name": call.name,
                "success": result.success,
                "error": result.error,
                "step_id": step_id,
                "attempt": attempt,
                "duration_ms": duration_ms,
                "truncated": truncated,
                "artifact_path": result.artifact_path,
            },
        )

        entry = LogEntry(
            index=0,
            step_id=step_id,
            kind="tool",
            tool=call.name,
            arguments=args,
            success=result.success,
            text=result.as_observation(),
            artifact_path=result.artifact_path,
            attempt=attempt,
            replayed=bool(replay_reason),
            approved=approved,
        )
        state.append_log(entry)
        self.store.append(
            state.task_id, EventType.LOG_APPENDED, {"entry": entry.model_dump(mode="json")}
        )
        return entry

    # -- helpers ----------------------------------------------------------
    def _fail(
        self,
        state: AgentState,
        step_id: str,
        call: ToolCall,
        message: str,
        *,
        attempt: int,
        args: dict[str, Any] | None = None,
    ) -> LogEntry:
        entry = LogEntry(
            index=0,
            step_id=step_id,
            kind="tool",
            tool=call.name,
            arguments=args if args is not None else call.arguments,
            success=False,
            text=message,
            attempt=attempt,
        )
        state.append_log(entry)
        self.store.append(
            state.task_id, EventType.LOG_APPENDED, {"entry": entry.model_dump(mode="json")}
        )
        return entry


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """Head+tail truncation.

    Tail matters: stack traces, `pytest` summaries and `git diff` hunks all
    put the answer at the *end*. Head-only truncation is why agents
    "cannot see" the failing assertion.
    """
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.6)
    tail = limit - head - 80
    return (
        f"{text[:head]}\n\n[... {len(text) - head - tail} characters omitted "
        f"(full output saved as an artifact) ...]\n\n{text[-tail:]}",
        True,
    )
