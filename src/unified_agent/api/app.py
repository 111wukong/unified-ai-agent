"""FastAPI service layer.

Two transports over one event source:

* **SSE at `POST /agui`** speaking the AG-UI protocol -- this is what the
  bundled console uses, and what any AG-UI-compatible frontend can consume.
* **REST** for scripts, CI and the A2A layer. REST is not a fallback here;
  it is genuinely better for non-interactive callers.

The runtime is a single in-process object. That is a deliberate limit: a
local-first tool does not need horizontal scaling, and keeping one agent
means the event bus stays a plain in-process fan-out instead of requiring
Redis.

`POST /agui` subscribes *before* starting the run. Starting first and then
subscribing loses the early events (task_created, plan_created) and the
console renders an empty plan -- a race that only shows up on fast models.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextlib as _contextlib
import secrets
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from unified_agent import __version__
from unified_agent.agent.factory import Agent, build_agent
from unified_agent.agent.state import replay
from unified_agent.api import agui
from unified_agent.config import Settings, load_settings
from unified_agent.observability.events import EventType
from unified_agent.storage.store import new_id
from unified_agent.types import EffectClass

CONSOLE_DIR = Path(__file__).parent / "console"

# How long a stream waits with no events before checking whether the task
# is already finished. Short enough that a late client is not held open,
# long enough not to spam notices during a slow model call.
IDLE_TIMEOUT_S = 30.0

# Loopback only. A localhost service that can execute commands must reject a
# request whose Host header points elsewhere: that is DNS rebinding -- a page
# you visit resolves its own domain to 127.0.0.1 and drives your local agent
# through your browser.
DEFAULT_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

# Paths that need the guard. The console and its assets do not: a browser
# cannot read them cross-origin, and the console must load before it has a
# token to present.
PROTECTED_PREFIXES = ("/api/",)
PROTECTED_EXACT = {"/agui"}

# Effects a UI can pre-approve via the request body. SYSTEM_ADMIN is
# deliberately excluded: it must never be grantable by a web request.
GRANTABLE = {e for e in EffectClass if e is not EffectClass.SYSTEM_ADMIN}


# ---------------------------------------------------------------------------
# request / response models
# ---------------------------------------------------------------------------


class SessionIn(BaseModel):
    name: str = "default"
    working_dir: str | None = None
    model: str | None = None


class TaskIn(BaseModel):
    goal: str = Field(min_length=1)
    session_id: str | None = None
    model: str | None = None
    max_steps: int | None = None
    approve: list[str] = Field(default_factory=list)


class AgUiMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool", "developer"]
    content: str = ""
    id: str | None = None


class AgUiRunInput(BaseModel):
    """Subset of AG-UI's `RunAgentInput` that this runtime actually uses."""

    threadId: str | None = None  # noqa: N815 - wire field names are camelCase
    runId: str | None = None  # noqa: N815
    messages: list[AgUiMessage] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    resume: list[dict[str, Any]] = Field(default_factory=list)
    approve: list[str] = Field(default_factory=list)
    model: str | None = None
    max_steps: int | None = None


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


class Service:
    """Holds the single agent plus the running-task registry."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.agent: Agent | None = None
        self.running: dict[str, asyncio.Task] = {}

    async def startup(self) -> None:
        self.agent = await build_agent(settings=self.settings)

    async def shutdown(self) -> None:
        for task in list(self.running.values()):
            task.cancel()
        for task in list(self.running.values()):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.running.clear()
        if self.agent:
            self.agent.close()

    @property
    def a(self) -> Agent:
        if self.agent is None:
            raise HTTPException(status_code=503, detail="service not ready")
        return self.agent

    def track(self, task_id: str, coro: Any) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.running[task_id] = task

        def _done(_: asyncio.Task) -> None:
            self.running.pop(task_id, None)

        task.add_done_callback(_done)
        return task


@_contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    svc: Service = app.state.svc
    if svc.agent is None:
        await svc.startup()
    try:
        yield
    finally:
        await svc.shutdown()


def create_app(
    settings: Settings | None = None,
    *,
    service: Service | None = None,
    token: str | None = None,
    allowed_hosts: Iterable[str] | None = None,
) -> FastAPI:
    """Build the app.

    `token` turns on the session-token requirement (the desktop launcher
    generates one per launch). `allowed_hosts` overrides the loopback
    allowlist; tests pass their own because the test client's Host header is
    not a loopback name.
    """
    hosts = frozenset(allowed_hosts) if allowed_hosts is not None else DEFAULT_ALLOWED_HOSTS
    app = FastAPI(
        lifespan=_lifespan,
        title="unified-ai-agent",
        version=__version__,
        description=(
            "Local-first agent runtime. `POST /agui` speaks the AG-UI protocol "
            "over SSE; the REST routes are for scripts and the A2A layer."
        ),
    )
    svc = service or Service(settings or load_settings(create_if_missing=True))
    app.state.svc = svc
    app.state.token = token
    app.state.allowed_hosts = hosts

    @app.middleware("http")
    async def _guard(request: Request, call_next):  # noqa: ANN001, ANN202
        if _needs_guard(request.url.path):
            problem = _guard_problem(request, hosts=hosts, token=token)
            if problem:
                return JSONResponse({"detail": problem}, status_code=403)
        return await call_next(request)

    # -- meta -------------------------------------------------------------
    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        # Surface sink failures: a broken sink means the event stream is
        # silently empty, which is worth knowing before debugging anything else.
        sink_problems: list[str] = []
        sink = getattr(svc.a.store, "sink", None)
        sink_problems = list(getattr(sink, "problems", []) or [])
        return {
            "status": "ok",
            "version": __version__,
            "workspace": str(svc.settings.workspace),
            "default_model": svc.settings.default_model,
            "running_tasks": len(svc.running),
            "sink_problems": sink_problems,
            # Reported so a client can tell whether it must authenticate
            # before attempting a mutating call.
            "token_required": bool(token),
        }

    @app.get("/api/v1/sandbox")
    async def sandbox_status() -> dict[str, Any]:
        selection = svc.a.sandbox_selection
        return {
            "requested": getattr(selection, "requested", "unknown"),
            "backend": svc.a.sandbox.name if svc.a.sandbox else "unknown",
            "isolation": getattr(svc.a.sandbox, "isolation", "unknown"),
            "fell_back": getattr(selection, "fell_back", False),
            "notes": getattr(selection, "notes", []),
            "probe": getattr(svc.a.sandbox, "probe_detail", ""),
            "caveats": svc.a.sandbox.caveats() if svc.a.sandbox else [],
        }

    # -- sessions ---------------------------------------------------------
    @app.post("/api/v1/sessions")
    async def create_session(payload: SessionIn) -> dict[str, Any]:
        agent = svc.a
        sid = agent.store.ensure_session(
            name=payload.name,
            working_dir=payload.working_dir or str(svc.settings.workspace),
            model_alias=payload.model or svc.settings.default_model,
        )
        return {"id": sid, **{k: v for k, v in (agent.store.get_session(sid) or {}).items()}}

    @app.get("/api/v1/sessions")
    async def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
        return svc.a.store.list_sessions(limit=limit)

    # -- tasks ------------------------------------------------------------
    @app.post("/api/v1/tasks")
    async def create_task(payload: TaskIn) -> dict[str, Any]:
        agent = svc.a
        session_id = payload.session_id or agent.store.ensure_session(
            name="default",
            working_dir=str(svc.settings.workspace),
            model_alias=payload.model or svc.settings.default_model,
        )
        effects = _grantable(payload.approve)
        task_id = new_id("task")
        svc.track(
            task_id,
            agent.runtime.run(
                payload.goal,
                session_id=session_id,
                model_alias=payload.model,
                approved_effects=effects,
                max_steps=payload.max_steps,
                task_id=task_id,
            ),
        )
        return {"id": task_id, "session_id": session_id, "status": "pending"}

    @app.post("/api/v1/tasks/{task_id}/wait")
    async def wait_task(task_id: str, timeout_s: float = 120.0) -> dict[str, Any]:
        """Block until a tracked task finishes. Useful for scripts and tests."""
        task = svc.running.get(task_id)
        if task is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        return _task_view(svc, task_id)

    @app.get("/api/v1/tasks")
    async def list_tasks(session_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        return svc.a.store.list_tasks(session_id=session_id, limit=limit)

    @app.get("/api/v1/tasks/{task_id}")
    async def get_task(task_id: str) -> dict[str, Any]:
        return _task_view(svc, task_id)

    @app.get("/api/v1/tasks/{task_id}/events")
    async def task_events(task_id: str, since_seq: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        events = svc.a.store.events(task_id, since_seq=since_seq, limit=limit)
        return [
            {"seq": e.seq, "type": e.type.value, "payload": e.payload, "createdAt": e.created_at}
            for e in events
        ]

    @app.post("/api/v1/tasks/{task_id}/approve")
    async def approve(task_id: str, request_id: str | None = None) -> dict[str, Any]:
        _require_pending(svc, task_id)
        svc.track(task_id, svc.a.runtime.approve(task_id, request_id=request_id))
        return {"id": task_id, "status": "running"}

    @app.post("/api/v1/tasks/{task_id}/deny")
    async def deny(task_id: str, note: str = "") -> dict[str, Any]:
        _require_pending(svc, task_id)
        svc.track(task_id, svc.a.runtime.deny(task_id, note=note))
        return {"id": task_id, "status": "running"}

    @app.post("/api/v1/tasks/{task_id}/cancel")
    async def cancel(task_id: str) -> dict[str, Any]:
        svc.a.runtime.cancel(task_id)
        return {"id": task_id, "status": "cancelling"}

    @app.post("/api/v1/tasks/{task_id}/resume")
    async def resume(task_id: str) -> dict[str, Any]:
        svc.track(task_id, svc.a.runtime.resume(task_id))
        return {"id": task_id, "status": "running"}

    # -- inventory --------------------------------------------------------
    @app.get("/api/v1/tools")
    async def list_tools() -> list[dict[str, Any]]:
        return svc.a.registry.describe()

    @app.get("/api/v1/skills")
    async def list_skills() -> list[dict[str, Any]]:
        return svc.a.store.list_skills()

    @app.get("/api/v1/memory")
    async def list_memory(scope: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return svc.a.store.list_memories(scope=scope, limit=limit)

    @app.get("/api/v1/memory/search")
    async def search_memory(q: str, limit: int = 8) -> list[dict[str, Any]]:
        return svc.a.store.search_memories(q, limit=limit)

    # -- AG-UI over SSE ---------------------------------------------------
    @app.post("/agui")
    async def agui_endpoint(payload: AgUiRunInput) -> StreamingResponse:
        return StreamingResponse(
            _agui_stream(svc, payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/tasks/{task_id}/stream")
    async def attach_stream(task_id: str, thread_id: str | None = None) -> StreamingResponse:
        """Attach to an already-created task and stream AG-UI events."""
        if svc.a.store.get_task(task_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
        return StreamingResponse(
            _attach_stream(svc, task_id, thread_id or task_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    # -- WebSocket --------------------------------------------------------
    @app.websocket("/ws/tasks/{task_id}")
    async def task_ws(websocket: WebSocket, task_id: str) -> None:
        """Same events as SSE, for clients that prefer a socket.

        The AG-UI spec lists HTTP and WebSockets as the foundational
        transports; SSE is just the simpler default. History is replayed on
        connect, so attaching after the fact still shows the whole run.
        """
        # HTTP middleware does not run for a WebSocket upgrade, so the same
        # guard has to be applied here or this becomes the way around it.
        problem = _guard_problem(websocket, hosts=hosts, token=token)
        if problem:
            await websocket.close(code=1008, reason=problem)
            return
        await websocket.accept()
        if svc.a.store.get_task(task_id) is None:
            await websocket.close(code=1008, reason=f"unknown task {task_id}")
            return
        try:
            async for event in _agui_events(
                svc,
                task_id=task_id,
                thread_id=task_id,
                run_id=new_id("run"),
                replay=True,
                announce_start=True,
            ):
                await websocket.send_json(event)
        except WebSocketDisconnect:
            return
        finally:
            with contextlib.suppress(RuntimeError):
                await websocket.close()

    # -- console ----------------------------------------------------------
    if CONSOLE_DIR.is_dir():
        app.mount("/console", StaticFiles(directory=str(CONSOLE_DIR), html=True), name="console")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse((CONSOLE_DIR / "index.html").read_text(encoding="utf-8"))

    return app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _needs_guard(path: str) -> bool:
    return path in PROTECTED_EXACT or path.startswith(PROTECTED_PREFIXES)


def _guard_problem(request: Any, *, hosts: frozenset[str], token: str | None) -> str | None:
    """Return a refusal reason, or None to allow.

    Two independent checks, because they stop different attackers:

    * **Host** defeats DNS rebinding. A browser will happily POST to
      127.0.0.1 on behalf of any page it loads, but it cannot forge the Host
      header -- so requiring a loopback Host rejects the rebinding case.
    * **Token** defeats another *local* process, which can discover the port
      but not the per-launch token. The desktop launcher passes it to the
      window in the URL fragment, which is never sent to the server and never
      appears in a Referer.
    """
    raw_host = request.headers.get("host") or ""
    hostname = raw_host.rsplit(":", 1)[0] if raw_host.count(":") == 1 else raw_host
    if hostname and hostname not in hosts:
        return (
            f"host {hostname!r} is not allowed; this service only answers "
            f"loopback requests ({', '.join(sorted(hosts))})"
        )

    if not token:
        return None

    supplied = (
        request.headers.get("x-uaa-token") or request.query_params.get("token") or ""
    )
    if not supplied:
        authorization = request.headers.get("authorization") or ""
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
    if not secrets.compare_digest(supplied, token):
        return "missing or invalid session token"
    return None


def _grantable(names: list[str]) -> list[EffectClass]:
    out: list[EffectClass] = []
    for name in names:
        try:
            effect = EffectClass(name)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"unknown effect {name!r}; known: {[e.value for e in EffectClass]}",
            ) from None
        if effect not in GRANTABLE:
            raise HTTPException(
                status_code=400,
                detail=f"{effect.value} cannot be pre-approved over HTTP",
            )
        out.append(effect)
    return out


def _task_view(svc: Service, task_id: str) -> dict[str, Any]:
    task = svc.a.store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    state = replay(svc.a.store.events(task_id), task_id=task_id, session_id=task["session_id"])
    return {
        "id": task_id,
        "session_id": task["session_id"],
        "goal": task["goal"],
        "status": state.status.value,
        "steps": state.steps_used,
        "model_calls": state.model_calls,
        "tokens": state.usage.total_tokens,
        "cost_usd": state.usage.cost_usd,
        "answer": state.answer,
        "error": state.error,
        "plan": [
            {"id": s.id, "description": s.description, "status": s.status.value}
            for s in state.plan
        ],
        "pending_confirmation": state.pending_confirmation.model_dump()
        if state.pending_confirmation
        else None,
        "events": svc.a.store.last_seq(task_id),
    }


def _require_pending(svc: Service, task_id: str) -> None:
    task = svc.a.store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    if not task["pending_confirmation"]:
        raise HTTPException(status_code=409, detail=f"task {task_id} has no pending approval")


def _encode(encoder: agui.AgUiEncoder, item: Any) -> list[dict[str, Any]]:
    if item.kind == "text_delta":
        return encoder.on_delta(item)
    if item.kind == "notice":
        return [agui.custom("stream_notice", {"notice": item.notice, "dropped": item.dropped})]
    if item.kind == "event" and item.event is not None:
        return encoder.on_event(item.event)
    return []


async def _agui_events(
    svc: Service,
    *,
    task_id: str,
    thread_id: str,
    run_id: str,
    replay: bool,
    announce_start: bool,
    launch: Any = None,
) -> AsyncIterator[dict[str, Any]]:
    """The single source of truth for both transports.

    SSE and WebSocket differ only in framing, so they share this generator.
    Having two copies is how the WebSocket endpoint ended up not replaying
    history and hanging forever on a late connection.

    `launch` is an optional coroutine to start *after* the subscription
    exists. Subscribing after starting races the first events.
    """
    agent = svc.a
    bus = agent.bus
    if bus is None:  # pragma: no cover - build_agent always provides one
        yield agui.run_error("event bus unavailable", code="no_bus")
        return

    encoder = agui.AgUiEncoder(task_id=task_id, thread_id=thread_id, run_id=run_id)
    launched: asyncio.Task | None = None

    async with bus.subscribe(task_id) as sub:
        if announce_start:
            yield agui.run_started(thread_id, run_id)
        if launch is not None:
            launched = svc.track(task_id, launch)

        if replay:
            for event in agent.store.events(task_id):
                for encoded in encoder.on_event(event):
                    yield encoded
            if encoder.finished:
                for encoded in encoder.finalize():
                    yield encoded
                return

        while True:
            item = await sub.get(timeout=IDLE_TIMEOUT_S)
            if item is None:
                # Nothing arrived. Stop rather than holding the connection
                # open for an hour -- but first distinguish "finished" from
                # "the run died without saying so", because the second one
                # must be visible, not a hang.
                exhausted = _task_terminal(svc, task_id) or (
                    launched is not None and launched.done()
                )
                if exhausted and sub.pending() == 0:
                    if not encoder.finished:
                        detail = _task_failure(svc, launched)
                        if detail:
                            yield agui.run_error(detail, code="no_terminal_event")
                    for encoded in encoder.finalize():
                        yield encoded
                    return
                yield agui.custom("stream_idle", {"seconds": IDLE_TIMEOUT_S})
                continue

            for encoded in _encode(encoder, item):
                yield encoded

            if item.kind != "event" or item.event is None:
                continue
            kind = item.event.type
            if kind is EventType.CONFIRMATION_REQUESTED:
                # A pause is a terminal event for this run: the client
                # resumes by starting a new run.
                yield agui.run_interrupted([agui.interrupt_from_event(item.event)])
                for encoded in encoder.finalize():
                    yield encoded
                return
            if kind in agui.TERMINAL_TYPES:
                for encoded in encoder.finalize():
                    yield encoded
                return


async def _agui_stream(svc: Service, payload: AgUiRunInput) -> AsyncIterator[str]:
    thread_id = payload.threadId or new_id("thread")
    run_id = payload.runId or new_id("run")

    if payload.resume:
        task_id = _resume_task_id(payload)
        if not task_id or svc.a.store.get_task(task_id) is None:
            yield agui.sse(agui.run_error("resume referenced an unknown task", code="bad_resume"))
            return
        async for event in _agui_events(
            svc, task_id=task_id, thread_id=thread_id, run_id=run_id,
            replay=True, announce_start=True,
        ):
            yield agui.sse(event)
        return

    goal = _last_user_message(payload)
    if not goal:
        yield agui.sse(agui.run_error("no user message in the request", code="empty_input"))
        return

    agent = svc.a
    session_id = agent.store.ensure_session(
        name=thread_id,
        working_dir=str(svc.settings.workspace),
        model_alias=payload.model or svc.settings.default_model,
    )
    task_id = new_id("task")
    launch = agent.runtime.run(
        goal,
        session_id=session_id,
        model_alias=payload.model,
        approved_effects=_grantable(payload.approve),
        max_steps=payload.max_steps,
        task_id=task_id,
    )
    async for event in _agui_events(
        svc, task_id=task_id, thread_id=thread_id, run_id=run_id,
        replay=False, announce_start=True, launch=launch,
    ):
        yield agui.sse(event)


async def _attach_stream(svc: Service, task_id: str, thread_id: str) -> AsyncIterator[str]:
    async for event in _agui_events(
        svc, task_id=task_id, thread_id=thread_id, run_id=new_id("run"),
        replay=True, announce_start=True,
    ):
        yield agui.sse(event)


def _resume_task_id(payload: AgUiRunInput) -> str:
    entry = payload.resume[0] if payload.resume else {}
    return str(entry.get("taskId") or entry.get("id") or "")


def _task_failure(svc: Service, launched: asyncio.Task | None) -> str:
    """Explain why a run produced no terminal event, if it did not."""
    if launched is not None:
        if launched.cancelled():
            return "the run was cancelled before it finished"
        exc = launched.exception()
        if exc is not None:
            return f"the run raised {type(exc).__name__}: {exc}"
    return "the run ended without a terminal event"


def _task_terminal(svc: Service, task_id: str) -> bool:
    task = svc.a.store.get_task(task_id)
    if task is None:
        return True
    return task["status"] in {"completed", "failed", "cancelled"}


def _last_user_message(payload: AgUiRunInput) -> str:
    for message in reversed(payload.messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return ""


def _state_snapshot_for(svc: Service, task_id: str) -> dict[str, Any]:
    task = svc.a.store.get_task(task_id)
    state = replay(svc.a.store.events(task_id), task_id=task_id, session_id=task["session_id"])
    return {
        "status": state.status.value,
        "plan": [{"id": s.id, "description": s.description, "status": s.status.value} for s in state.plan],
        "stepsUsed": state.steps_used,
        "tokens": state.usage.total_tokens,
    }


__all__ = ["create_app", "Service", "app"]


def app() -> FastAPI:  # pragma: no cover - uvicorn factory target
    return create_app()
