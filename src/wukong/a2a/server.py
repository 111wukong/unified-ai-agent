"""A2A JSON-RPC binding.

Four methods, which is what the spec's minimal server needs:

    message/send      -> run a task, return it
    message/stream    -> run a task, stream status updates over SSE
    tasks/get         -> fetch a task by id
    tasks/cancel      -> cancel a running task

The binding is JSON-RPC 2.0 because that is the spec's default and the
cheapest to implement correctly: a single `POST` endpoint, one envelope
shape, and error codes the peer already knows how to handle.

**A remote task is an ordinary task.** It goes through the same
`AgentRuntime.run` as a local one, so budgets, permissions, the event log,
the approval gate and resume all apply without a second path. `contextId` is
the session id, so a peer that sends several messages in one context shares
memory and history the way a local session does.

The one thing that is genuinely new here is *whose* task it is: the session
records the remote peer in its metadata, because a task an outside agent
started has to be attributable after the fact.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator

from wukong.a2a.card import AgentCard, build_card
from wukong.a2a.security import check_parts
from wukong.a2a.types import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatusObject,
    TextPart,
    now_iso,
    parts_to_text,
    state_for,
)
from wukong.agent.state import replay
from wukong.observability.events import EventType
from wukong.storage.store import new_id

# JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

#: Which A2A states end the stream. `input-required` is one of them: the run
#: is paused for a human, and the peer resumes by sending the answer, which is
#: a new request rather than more of this stream. This set is the single place
#: that decides `final`, so a new interrupt cannot be added to the mapping
#: without also deciding whether it ends the stream.
_STREAM_TERMINAL = frozenset(
    {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELED,
        TaskState.REJECTED,
        TaskState.INPUT_REQUIRED,
        TaskState.AUTH_REQUIRED,
    }
)


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class A2AServer:
    """Serves one agent over the A2A JSON-RPC binding."""

    def __init__(self, agent: Any, *, token_required: bool = False) -> None:
        self.agent = agent
        self.token_required = token_required

    # -- discovery --------------------------------------------------------
    def card(self, *, url: str, provider: dict[str, str] | None = None) -> AgentCard:
        return build_card(
            self.agent, url=url, token_required=self.token_required, provider=provider
        )

    # -- JSON-RPC ---------------------------------------------------------
    async def handle(self, request: Any) -> dict[str, Any]:
        """Dispatch one JSON-RPC request. Never raises: errors are responses."""
        if not isinstance(request, dict):
            return _error(None, INVALID_REQUEST, "request must be a JSON object")
        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return _error(request_id, INVALID_REQUEST, "jsonrpc must be '2.0'")
        method = request.get("method")
        if not isinstance(method, str):
            return _error(request_id, INVALID_REQUEST, "method must be a string")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            return _error(request_id, INVALID_PARAMS, "params must be an object")

        try:
            if method == "message/send":
                return _result(request_id, await self._send(params))
            if method == "tasks/get":
                return _result(request_id, self._get(params))
            if method == "tasks/cancel":
                return _result(request_id, self._cancel(params))
            if method == "message/stream":
                # Streaming is served by the HTTP layer, which knows how to
                # write SSE; reaching here means the caller used the
                # non-streaming endpoint.
                raise JsonRpcError(
                    METHOD_NOT_FOUND,
                    "message/stream must be called on the streaming endpoint",
                )
        except JsonRpcError as exc:
            return _error(request_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # noqa: BLE001 - a peer gets a response, not a traceback
            return _error(
                request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}"[:400]
            )
        return _error(request_id, METHOD_NOT_FOUND, f"unknown method {method!r}")

    # -- methods ----------------------------------------------------------
    async def _send(self, params: dict[str, Any]) -> dict[str, Any]:
        goal, peer, context_id = self._parse_message(params)
        session_id = self._session_for(peer, context_id)
        result = await self.agent.runtime.run(goal, session_id=session_id)
        return self._to_task(result.task_id, session_id=session_id, message=params.get("message"))

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise JsonRpcError(INVALID_PARAMS, "params.id is required")
        row = self.agent.store.get_task(task_id)
        if row is None:
            raise JsonRpcError(INVALID_PARAMS, f"unknown task {task_id!r}")
        return self._to_task(task_id, session_id=row.get("session_id") or "")

    def _cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise JsonRpcError(INVALID_PARAMS, "params.id is required")
        row = self.agent.store.get_task(task_id)
        if row is None:
            raise JsonRpcError(INVALID_PARAMS, f"unknown task {task_id!r}")
        self.agent.runtime.cancel(task_id)
        state = replay(self.agent.store.events(task_id), task_id=task_id)
        # Cancellation is a file the loop checks at the top of its next
        # iteration, so a task that is not running cannot be cancelled --
        # saying so beats returning a task the peer believes is stopping.
        if state.status.terminal:
            raise JsonRpcError(
                INVALID_PARAMS, f"task {task_id!r} is already {state.status.value}"
            )
        return self._to_task(task_id, session_id=row.get("session_id") or "")

    # -- streaming --------------------------------------------------------
    async def stream(
        self, params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Run a task and yield A2A status updates until it stops.

        The task id is generated here and handed to `run()`, because the
        subscription has to exist *before* the run starts: subscribing
        afterwards races the first events, which is the bug the AG-UI stream
        already documents. That is what `run(task_id=...)` is for.
        """
        goal, peer, context_id = self._parse_message(params)
        session_id = self._session_for(peer, context_id)
        bus = self.agent.bus
        task_id = new_id("task")

        yield _update(task_id, TaskState.SUBMITTED)

        if bus is None:  # pragma: no cover - build_agent always provides one
            result = await self.agent.runtime.run(
                goal, session_id=session_id, task_id=task_id
            )
            yield _update(task_id, state_for(result.status), final=True)
            return

        async with bus.subscribe(task_id) as sub:
            runner = asyncio.create_task(
                self.agent.runtime.run(goal, session_id=session_id, task_id=task_id)
            )
            try:
                while True:
                    item = await sub.get(timeout=1.0)
                    if item is None:
                        # Nothing arrived. Stop once the run is done; before
                        # that, keep waiting rather than closing a live stream.
                        if runner.done():
                            break
                        continue
                    if item.kind != "event" or item.event is None:
                        continue
                    update = _status_update(item.event, task_id=task_id)
                    if update is None:
                        continue
                    yield update
                    if update["final"]:
                        return
            finally:
                if not runner.done():
                    runner.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await runner

        # The loop can end without a terminal event -- a crash between the
        # last write and the terminal append. Report the real state rather
        # than closing the stream as though it finished.
        final = self._to_task(task_id, session_id=session_id)
        yield {
            "kind": "status-update",
            "taskId": task_id,
            "state": final["status"]["state"],
            "final": True,
            "timestamp": now_iso(),
        }

    # -- helpers ----------------------------------------------------------
    def _parse_message(self, params: dict[str, Any]) -> tuple[str, str, str]:
        raw = params.get("message")
        if not isinstance(raw, dict):
            raise JsonRpcError(INVALID_PARAMS, "params.message is required")
        parts, report = check_parts(raw.get("parts") or [])
        if not report.ok:
            raise JsonRpcError(
                INVALID_PARAMS, "message rejected", {"problems": report.errors}
            )
        goal = parts_to_text(parts)
        if not goal.strip():
            raise JsonRpcError(
                INVALID_PARAMS,
                "message carries no text or data content",
            )
        peer = str(params.get("metadata", {}).get("peer") or "unknown")
        context_id = str(raw.get("contextId") or "")
        return goal, peer, context_id

    def _session_for(self, peer: str, context_id: str) -> str:
        """One session per (peer, context), so a peer's tasks share history.

        The session metadata records who is calling. A task an outside agent
        started has to be attributable after the fact -- otherwise the audit
        log shows work nobody can trace to a counterparty.
        """
        session_id = self.agent.store.ensure_session(
            name=f"a2a:{peer}",
            working_dir=str(self.agent.settings.workspace),
            model_alias=self.agent.settings.default_model,
            metadata={"a2a_peer": peer, "a2a_context_id": context_id or "default"},
        )
        return session_id

    def _to_task(
        self, task_id: str, *, session_id: str, message: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Render an internal task as an A2A Task."""
        row = self.agent.store.get_task(task_id) or {}
        events = self.agent.store.events(task_id)
        state = replay(events, task_id=task_id, session_id=session_id)
        task_state = state_for(state.status)

        artifacts: list[dict[str, Any]] = []
        if state.answer:
            artifacts.append(
                Artifact(
                    artifactId=f"{task_id}-answer",
                    name="answer",
                    parts=[TextPart(text=state.answer)],
                ).model_dump()
            )

        history: list[dict[str, Any]] = []
        if isinstance(message, dict):
            parsed, _ = check_parts(message.get("parts") or [])
            history.append(
                Message(
                    role="user",
                    parts=parsed,
                    messageId=str(message.get("messageId") or ""),
                    taskId=task_id,
                    contextId=session_id,
                ).model_dump()
            )
        if state.answer:
            history.append(
                Message(
                    role="agent",
                    parts=[TextPart(text=state.answer)],
                    messageId=f"{task_id}-answer",
                    taskId=task_id,
                    contextId=session_id,
                ).model_dump()
            )

        pending = state.pending_confirmation
        status_message = None
        if task_state is TaskState.INPUT_REQUIRED and pending is not None:
            # The peer is told *what* is needed, not merely that something is.
            # A bare "input-required" leaves a remote orchestrator with no way
            # to answer except to guess.
            status_message = Message(
                role="agent",
                parts=[
                    TextPart(
                        text=(
                            f"Approval required for `{pending.tool}` "
                            f"({pending.effect}). Preview: {pending.preview}"
                        )
                    )
                ],
                messageId=f"{task_id}-input-required",
                taskId=task_id,
                contextId=session_id,
            ).model_dump()

        task = Task(
            id=task_id,
            contextId=session_id,
            status=TaskStatusObject(
                state=task_state,
                timestamp=now_iso(),
                message=Message(**status_message) if status_message else None,
            ),
            artifacts=[Artifact(**a) for a in artifacts],
            history=[Message(**m) for m in history],
            metadata={
                "steps": state.steps_used,
                "tokens": state.usage.total_tokens,
                "error": state.error,
                "internalStatus": row.get("status") or state.status.value,
            },
        )
        return task.model_dump(mode="json")


# ---------------------------------------------------------------------------
# streaming translation
# ---------------------------------------------------------------------------


def _update(task_id: str, state: TaskState, *, final: bool | None = None) -> dict[str, Any]:
    """One status update. `final` defaults to "does this state end a stream".

    Defaulting it from `_STREAM_TERMINAL` rather than passing it per call site
    is what keeps the set load-bearing: a new interrupt added to the mapping
    gets the right `final` without anyone remembering to update a branch.
    """
    return {
        "kind": "status-update",
        "taskId": task_id,
        "state": state.value,
        "final": state in _STREAM_TERMINAL if final is None else final,
        "timestamp": now_iso(),
    }


def _status_update(event: Any, *, task_id: str) -> dict[str, Any] | None:
    """Translate one internal event into an A2A status update, or None.

    Only the events a peer can act on are forwarded. Streaming the whole
    internal event log would expose the runtime's internals as an interface
    and commit this project to keeping them stable for outside consumers.
    """
    kind = event.type
    if kind is EventType.STATE_TRANSITION:
        target = event.payload.get("to")
        return _update(task_id, state_for(target) if target else TaskState.WORKING)
    if kind is EventType.CONFIRMATION_REQUESTED:
        return {
            **_update(task_id, TaskState.INPUT_REQUIRED, final=True),
            "message": {
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": (
                            f"Approval required for `{event.payload.get('tool')}` "
                            f"({event.payload.get('effect')})."
                        ),
                    }
                ],
            },
        }
    if kind in _TERMINAL_EVENTS:
        return _update(task_id, _TERMINAL_EVENTS[kind])
    return None


#: Internal terminal event -> the wire state it reports.
_TERMINAL_EVENTS = {
    EventType.TASK_COMPLETED: TaskState.COMPLETED,
    EventType.TASK_FAILED: TaskState.FAILED,
    EventType.TASK_CANCELLED: TaskState.CANCELED,
}


def _update(task_id: str, state: TaskState, *, final: bool = False) -> dict[str, Any]:
    return {
        "kind": "status-update",
        "taskId": task_id,
        "state": state.value,
        "final": final,
        "timestamp": now_iso(),
    }


def _result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


__all__ = [
    "A2AServer",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "JsonRpcError",
    "PARSE_ERROR",
    "_STREAM_TERMINAL",
]
