"""AG-UI encoder: internal events -> the AG-UI wire protocol.

AG-UI (ag-ui.com, CopilotKit + LangGraph / Mastra / Pydantic AI / Microsoft
Agent Framework) is an open event protocol for agent<->frontend. Implementing
it instead of inventing a WebSocket format costs nothing and means the
console can later be swapped for CopilotKit's React components without
touching the backend.

Three event patterns, all of which map cleanly onto what this runtime
already produces:

* **Start-Content-End** for streaming content -- TEXT_MESSAGE_* and
  TOOL_CALL_*.
* **Snapshot-Delta** for state -- STATE_SNAPSHOT / ACTIVITY_SNAPSHOT. The
  plan maps onto `activityType: "PLAN"` exactly.
* **Lifecycle** -- RUN_STARTED / STEP_* / RUN_FINISHED / RUN_ERROR.

Two details that are easy to get wrong:

1. **Human-in-the-loop is not a separate event.** AG-UI expresses a pause as
   `RUN_FINISHED` with `outcome: {type: "interrupt", interrupts: [...]}`.
   That means the frontend has exactly one terminal event to handle, rather
   than "wait for RUN_FINISHED, but what if it never comes". This maps onto
   `PendingConfirmation` directly.

2. **Deltas and the final message must not both be sent.** If tokens were
   already streamed for a `messageId`, the durable `MODEL_RESPONSE` must
   only close the message (TEXT_MESSAGE_END); re-sending the full text would
   duplicate it. A late subscriber that missed the deltas needs the full
   text instead. The encoder tracks which message ids were streamed.

Compatibility note, stated honestly: event names, base fields and the
interrupt *container* follow the published spec. The per-interrupt field set
below is ours, because the reference docs enumerate `outcome.interrupts`
without defining the interrupt object's fields.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from unified_agent.observability.bus import BusItem
from unified_agent.observability.events import Event, EventType

# ---------------------------------------------------------------------------
# event constructors
# ---------------------------------------------------------------------------


def _base(event_type: str) -> dict[str, Any]:
    return {"type": event_type, "timestamp": int(time.time() * 1000)}


def run_started(
    thread_id: str, run_id: str, *, input_text: str = "", task_id: str | None = None
) -> dict[str, Any]:
    """The run is beginning.

    `taskId` is an extension to the AG-UI shape, and it is here because a client
    that started a run otherwise has no way to learn which task it belongs to --
    `threadId` is the client's own id. Without it the console had to guess, and
    it guessed wrong: approving an approval prompt posted to whatever task the
    list happened to have selected, which is a 409 the moment the user has run
    anything before. A consumer that does not know the field ignores it.
    """
    event = _base("RUN_STARTED")
    event.update({"threadId": thread_id, "runId": run_id})
    if task_id:
        event["taskId"] = task_id
    if input_text:
        event["input"] = {"messages": [{"role": "user", "content": input_text}]}
    return event


def run_finished(*, result: dict[str, Any] | None = None) -> dict[str, Any]:
    event = _base("RUN_FINISHED")
    event["outcome"] = {"type": "success"}
    if result is not None:
        event["result"] = result
    return event


def run_interrupted(interrupts: list[dict[str, Any]]) -> dict[str, Any]:
    event = _base("RUN_FINISHED")
    event["outcome"] = {"type": "interrupt", "interrupts": interrupts}
    return event


def run_error(message: str, code: str | None = None) -> dict[str, Any]:
    event = _base("RUN_ERROR")
    event["message"] = message
    if code:
        event["code"] = code
    return event


def text_message_start(message_id: str, role: str = "assistant") -> dict[str, Any]:
    event = _base("TEXT_MESSAGE_START")
    event.update({"messageId": message_id, "role": role})
    return event


def text_message_content(message_id: str, delta: str) -> dict[str, Any]:
    event = _base("TEXT_MESSAGE_CONTENT")
    event.update({"messageId": message_id, "delta": delta})
    return event


def text_message_end(message_id: str) -> dict[str, Any]:
    event = _base("TEXT_MESSAGE_END")
    event["messageId"] = message_id
    return event


def tool_call_start(tool_call_id: str, name: str, parent_message_id: str = "") -> dict[str, Any]:
    event = _base("TOOL_CALL_START")
    event.update({"toolCallId": tool_call_id, "toolCallName": name})
    if parent_message_id:
        event["parentMessageId"] = parent_message_id
    return event


def tool_call_args(tool_call_id: str, delta: str) -> dict[str, Any]:
    event = _base("TOOL_CALL_ARGS")
    event.update({"toolCallId": tool_call_id, "delta": delta})
    return event


def tool_call_end(tool_call_id: str) -> dict[str, Any]:
    event = _base("TOOL_CALL_END")
    event["toolCallId"] = tool_call_id
    return event


def tool_call_result(
    tool_call_id: str, content: str, message_id: str, *, success: bool = True
) -> dict[str, Any]:
    event = _base("TOOL_CALL_RESULT")
    event.update(
        {
            "messageId": message_id,
            "toolCallId": tool_call_id,
            "content": content,
            "role": "tool",
        }
    )
    event["metadata"] = {"success": success}
    return event


def step_started(step_name: str) -> dict[str, Any]:
    event = _base("STEP_STARTED")
    event["stepName"] = step_name
    return event


def step_finished(step_name: str) -> dict[str, Any]:
    event = _base("STEP_FINISHED")
    event["stepName"] = step_name
    return event


def state_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    event = _base("STATE_SNAPSHOT")
    event["snapshot"] = snapshot
    return event


def activity_snapshot(
    message_id: str, activity_type: str, content: Any, *, replace: bool = True
) -> dict[str, Any]:
    event = _base("ACTIVITY_SNAPSHOT")
    event.update(
        {
            "messageId": message_id,
            "activityType": activity_type,
            "content": content,
            "replace": replace,
        }
    )
    return event


def custom(name: str, value: Any) -> dict[str, Any]:
    event = _base("CUSTOM")
    event.update({"name": name, "value": value})
    return event


def sse(payload: dict[str, Any]) -> str:
    """SSE frame. AG-UI carries the event name inside the JSON payload."""
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------


@dataclass
class AgUiEncoder:
    task_id: str
    thread_id: str
    run_id: str
    _streamed: set[str] = field(default_factory=set)
    _open_steps: set[str] = field(default_factory=set)
    _final: bool = False

    def on_delta(self, item: BusItem) -> list[dict[str, Any]]:
        """Ephemeral token delta. Not durable, so it is never replayed."""
        if not item.text:
            return []
        out: list[dict[str, Any]] = []
        if item.message_id not in self._streamed:
            self._streamed.add(item.message_id)
            out.append(text_message_start(item.message_id))
        out.append(text_message_content(item.message_id, item.text))
        return out

    def on_event(self, event: Event) -> list[dict[str, Any]]:
        """Always returns a list, even when the handler returns one event.

        Normalising here is not cosmetic: a handler that returns a bare dict
        and a caller that does `for x in result` iterates the dict's *keys*,
        so the real event never goes out and the wire carries strings like
        "type" instead. That is a silent protocol corruption, and it is
        fixed once, here, rather than in every caller.
        """
        handler = getattr(self, f"_on_{event.type.value}", None)
        if handler is None:
            return []
        result = handler(event)
        if result is None:
            return []
        if isinstance(result, dict):
            return [result]
        return list(result)

    def finalize(self) -> list[dict[str, Any]]:
        """Close any message that was streamed but never closed."""
        out = [text_message_end(mid) for mid in sorted(self._streamed)]
        self._streamed.clear()
        return out

    # -- handlers ---------------------------------------------------------
    def _on_plan_created(self, event: Event) -> list[dict[str, Any]]:
        steps = event.payload.get("steps") or []
        return [
            activity_snapshot(
                f"plan_{self.task_id}",
                "PLAN",
                {
                    "steps": [
                        {
                            "id": s.get("id"),
                            "description": s.get("description"),
                            "status": s.get("status", "pending"),
                        }
                        for s in steps
                    ]
                },
            )
        ]

    def _on_plan_revised(self, event: Event) -> list[dict[str, Any]]:
        # `replace=true` is the documented default, so a second snapshot is a
        # valid way to publish the new plan without computing a JSON Patch.
        return self._on_plan_created(event)

    def _on_tool_started(self, event: Event) -> list[dict[str, Any]]:
        call_id = event.payload.get("call_id") or event.payload.get("idempotency_key", "")
        name = event.payload.get("name", "tool")
        args = event.payload.get("arguments") or {}
        self._open_steps.add(name)
        return [
            step_started(name),
            tool_call_start(call_id, name),
            tool_call_args(call_id, json.dumps(args, ensure_ascii=False, default=str)),
            tool_call_end(call_id),
        ]

    def _on_tool_completed(self, event: Event) -> list[dict[str, Any]]:
        return self._tool_result(event, success=True)

    def _on_tool_failed(self, event: Event) -> list[dict[str, Any]]:
        return self._tool_result(event, success=False)

    def _tool_result(self, event: Event, *, success: bool) -> list[dict[str, Any]]:
        call_id = event.payload.get("call_id") or ""
        name = event.payload.get("name", "tool")
        content = (
            event.payload.get("observation")
            or event.payload.get("error")
            or event.payload.get("output")
            or ""
        )
        out = [tool_call_result(call_id, str(content)[:4000], f"tool_{call_id}", success=success)]
        if name in self._open_steps:
            self._open_steps.discard(name)
            out.append(step_finished(name))
        return out

    def _on_tool_ambiguous(self, event: Event) -> list[dict[str, Any]]:
        # Worth surfacing loudly: the user may need to check whether a
        # side-effecting command actually ran.
        return [
            custom(
                "tool_ambiguous",
                {
                    "tool": event.payload.get("name"),
                    "arguments": event.payload.get("arguments"),
                    "message": "outcome unknown after a restart",
                },
            )
        ]

    def _on_model_response(self, event: Event) -> list[dict[str, Any]]:
        payload = event.payload
        if payload.get("phase") != "step":
            return []
        message_id = payload.get("message_id") or f"msg_{event.seq}"
        content = payload.get("content_preview") or ""
        tool_names = payload.get("tool_calls") or []

        out: list[dict[str, Any]] = []
        if message_id in self._streamed:
            # Deltas already delivered the text; only close it.
            self._streamed.discard(message_id)
            out.append(text_message_end(message_id))
        elif content:
            # No live stream (late subscriber, or a model without streaming).
            out.append(text_message_start(message_id))
            out.append(text_message_content(message_id, content))
            out.append(text_message_end(message_id))

        if tool_names:
            out.append(custom("tool_calls_requested", {"names": tool_names}))
        usage = payload.get("usage") or {}
        if usage:
            out.append(
                custom(
                    "usage",
                    {
                        "model": payload.get("model"),
                        "totalTokens": usage.get("total_tokens"),
                        "costUsd": usage.get("cost_usd"),
                    },
                )
            )
        return out

    def _on_confirmation_requested(self, event: Event) -> list[dict[str, Any]]:
        # Deferred: the interrupt is published as the terminal RUN_FINISHED.
        return []

    def _on_context_compacted(self, event: Event) -> list[dict[str, Any]]:
        return [
            custom(
                "context_compacted",
                {
                    "droppedEntries": event.payload.get("dropped_entries"),
                    "summary": (event.payload.get("summary") or "")[:1000],
                },
            )
        ]

    def _on_memory_written(self, event: Event) -> list[dict[str, Any]]:
        return [custom("memory_written", {"content": event.payload.get("content")})]

    def _on_skill_candidate(self, event: Event) -> list[dict[str, Any]]:
        return [custom("skill_candidate", event.payload)]

    def _on_budget_exceeded(self, event: Event) -> list[dict[str, Any]]:
        return [custom("budget_exceeded", event.payload)]

    def _on_task_completed(self, event: Event) -> list[dict[str, Any]]:
        self._final = True
        return run_finished(result={"answer": event.payload.get("answer")})

    def _on_task_failed(self, event: Event) -> list[dict[str, Any]]:
        self._final = True
        return run_error(event.payload.get("error") or "task failed")

    def _on_task_cancelled(self, event: Event) -> list[dict[str, Any]]:
        self._final = True
        return run_finished(result={"status": "cancelled"})

    @property
    def finished(self) -> bool:
        return self._final


def interrupt_for(pending: Any) -> dict[str, Any]:
    """Shape a PendingConfirmation as an AG-UI interrupt.

    The container (`outcome.interrupts`) is spec; the field set is ours.
    """
    return {
        "id": pending.request_id,
        "reason": "approval_required",
        "tool": pending.tool,
        "effect": pending.effect,
        "arguments": pending.arguments,
        "preview": pending.preview,
        "detail": pending.reason,
    }


def interrupts_for(state: Any) -> list[dict[str, Any]]:
    if getattr(state, "pending_confirmation", None) is None:
        return []
    return [interrupt_for(state.pending_confirmation)]


def interrupt_from_event(event: Event) -> dict[str, Any]:
    """Build the interrupt straight from the CONFIRMATION_REQUESTED payload.

    Preferred over replaying state: the payload is already in hand, and a
    replay would need a database round-trip inside the streaming loop.
    """
    payload = event.payload
    return {
        "id": payload.get("request_id"),
        "reason": "approval_required",
        "tool": payload.get("tool"),
        "effect": payload.get("effect"),
        "arguments": payload.get("arguments") or {},
        "preview": payload.get("preview") or "",
        "detail": payload.get("reason") or "",
    }


TERMINAL_TYPES = {
    EventType.TASK_COMPLETED,
    EventType.TASK_FAILED,
    EventType.TASK_CANCELLED,
}

__all__ = [
    "AgUiEncoder",
    "interrupt_for",
    "interrupts_for",
    "sse",
    "run_started",
    "run_finished",
    "run_interrupted",
    "run_error",
    "custom",
    "state_snapshot",
    "TERMINAL_TYPES",
]
