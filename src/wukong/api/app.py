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
import hashlib
import json
import secrets
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from wukong import __version__
from wukong.agent.factory import Agent, build_agent
from wukong.agent.state import replay
from wukong.api import agui
from wukong.config import Settings, load_settings
from wukong.observability.events import EventType
from wukong.storage.store import new_id
from wukong.types import EffectClass

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
PROTECTED_PREFIXES = ("/api/", "/a2a")
# The Agent Card is guarded too, which is a deliberate deviation from A2A's
# public-discovery convention: the card lists this agent's capabilities and
# skills, and this is a local-first tool where the whole surface is behind
# the same loopback + token check. A peer therefore has to be configured with
# credentials rather than discovering them -- which is the safer direction,
# and the card advertises the scheme it needs.
PROTECTED_EXACT = {"/agui", "/.well-known/agent-card.json"}

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


class FileWrite(BaseModel):
    """A save from the console's file editor."""

    path: str
    text: str
    #: The digest the editor was handed when it opened the file. Sending it
    #: back is what turns "save" into "save, if nothing else moved first".
    base_sha: str | None = None


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


class SkillPromotion(BaseModel):
    status: Literal["candidate", "validated", "approved", "active", "deprecated"]


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
            # Wake any subscriber still waiting on this task so it finalises
            # instead of sitting out the idle timeout. Uses `self.agent`
            # directly, not the `a` property: that raises an HTTPException
            # when the service is not ready, and raising inside a done
            # callback is an unraisable error rather than a 503.
            agent = self.agent
            if agent is not None and agent.bus is not None:
                agent.bus.close_task(task_id)

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
        title="wukong",
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

    # -- workspace --------------------------------------------------------
    @app.get("/api/v1/files")
    async def list_files(path: str = "") -> dict[str, Any]:
        """One directory of the workspace, for the console's file tree.

        Scoped to the workspace on purpose: the console is a window onto the
        project the agent is working in, not a file manager for the machine.
        A path that escapes is refused rather than silently clamped -- clamping
        is how someone ends up looking at a different file than the one they
        asked for and never finds out.
        """
        root = Path(svc.settings.workspace).resolve()
        target = (root / path).resolve() if path else root
        if target != root and root not in target.parents:
            raise HTTPException(status_code=400, detail="path escapes the workspace")
        if not target.is_dir():
            raise HTTPException(status_code=404, detail=f"not a directory: {path or '.'}")

        entries: list[dict[str, Any]] = []
        try:
            for child in sorted(target.iterdir(), key=_file_sort_key):
                if child.name.startswith("."):
                    continue
                try:
                    is_dir = child.is_dir()
                    size = None if is_dir else child.stat().st_size
                except OSError:
                    continue
                entries.append(
                    {"name": child.name, "type": "dir" if is_dir else "file", "size": size}
                )
        except PermissionError:
            raise HTTPException(status_code=403, detail="permission denied") from None

        return {
            "path": str(target.relative_to(root)) if target != root else "",
            "entries": entries[:500],
        }

    @app.get("/api/v1/files/content")
    async def read_file(path: str) -> dict[str, Any]:
        """A text file from the workspace, for the console's preview pane.

        Read-only and size-capped. This is a *window*, not an editor: nothing
        here can change a byte, so it cannot race the agent. The cap is not a
        nicety either -- a preview that tries to render a 200 MB log freezes
        the tab, and the failure looks like the console being broken rather
        than like a file being too big.
        """
        root = Path(svc.settings.workspace).resolve()
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            raise HTTPException(status_code=400, detail="path escapes the workspace")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"not a file: {path}")

        size = target.stat().st_size
        if size > _PREVIEW_MAX_BYTES:
            return {
                "path": path,
                "size": size,
                "text": "",
                "binary": False,
                "reason": f"文件 {size} 字节，超过预览上限 {_PREVIEW_MAX_BYTES}",
            }

        raw = target.read_bytes()
        # A NUL in the head is the cheap, reliable binary tell -- the same one
        # git uses. Checking the whole file would mean reading it all to decide
        # whether to read it.
        if b"\x00" in raw[:8192]:
            return {
                "path": path,
                "size": size,
                "text": "",
                "binary": True,
                "reason": "二进制文件，不预览",
            }
        return {
            "path": path,
            "size": size,
            "text": raw.decode("utf-8", errors="replace"),
            "binary": False,
            "reason": "",
            # Returned so an editor can send it back on save. Without it there
            # is no way to tell "the file I opened" from "the file as it is
            # now", and a save silently discards whatever happened between.
            "sha": _digest(raw),
        }

    @app.put("/api/v1/files/content")
    async def write_file(payload: FileWrite) -> dict[str, Any]:
        """Write a text file from the console's editor.

        **Refused outright while a task is running**, and that refusal *is* the
        consistency design. The agent holds observations of these files in its
        context; a file changed underneath it turns those observations into a
        lie it will act on. `apply_patch` fails loudly on a mismatch, but a
        decision already made from stale content does not -- by the time the
        write lands, the reasoning that produced it is already wrong.

        So the rule is not "who wins", it is "there is only ever one writer".
        While the agent has the workspace, it has it.
        """
        if svc.running:
            raise HTTPException(
                status_code=409,
                detail="有任务正在运行，工作区归它使用。等它结束后再编辑。",
            )

        root = Path(svc.settings.workspace).resolve()
        target = (root / payload.path).resolve()
        if target != root and root not in target.parents:
            raise HTTPException(status_code=400, detail="path escapes the workspace")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"not a file: {payload.path}")

        encoded = payload.text.encode("utf-8")
        if len(encoded) > _PREVIEW_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"内容 {len(encoded)} 字节，超过编辑上限 {_PREVIEW_MAX_BYTES}",
            )

        current = _digest(target.read_bytes())
        if payload.base_sha and payload.base_sha != current:
            raise HTTPException(
                status_code=409,
                detail="这个文件在打开之后被改过。重新打开再编辑，否则会覆盖掉那次改动。",
            )

        target.write_text(payload.text, encoding="utf-8")
        return {"path": payload.path, "size": len(encoded), "sha": _digest(encoded)}

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
        """Cancel a task. Reports what happened rather than assuming.

        A paused task is cancelled outright (nothing is running to read a
        marker); a running one is asked to stop at its next step boundary.
        Answering "cancelling" for both is how a Cancel button that does
        nothing looks like a Cancel button that worked.
        """
        outcome = svc.a.runtime.cancel(task_id)
        if outcome == "not_found":
            raise HTTPException(status_code=404, detail=f"no such task: {task_id}")
        row = svc.a.runtime.store.get_task(task_id) or {}
        return {
            "id": task_id,
            "outcome": outcome,
            "status": row.get("status", "unknown"),
        }

    @app.post("/api/v1/tasks/{task_id}/resume")
    async def resume(task_id: str) -> dict[str, Any]:
        svc.track(task_id, svc.a.runtime.resume(task_id))
        return {"id": task_id, "status": "running"}

    # -- inventory --------------------------------------------------------
    @app.get("/api/v1/tools")
    async def list_tools(effect: str | None = None) -> list[dict[str, Any]]:
        """Registered tools, optionally filtered to one effect class.

        The filter matters for reviewing the permission surface: "what can
        this runtime execute, or reach over the network" is one request
        rather than a scan of the whole catalogue.
        """
        wanted: EffectClass | None = None
        if effect is not None:
            try:
                wanted = EffectClass(effect)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"unknown effect {effect!r}; expected one of "
                        f"{', '.join(e.value for e in EffectClass)}"
                    ),
                ) from exc
        return svc.a.registry.describe(effect=wanted)

    @app.get("/api/v1/skills")
    async def list_skills() -> list[dict[str, Any]]:
        """Every skill the registry knows, including the agent's own candidates.

        This used to read the `skills` table, which only holds skills that have
        been *promoted* at least once. So a candidate the agent wrote was
        invisible here -- and the human gate became unreachable, because the
        one thing that needed a decision was the one thing the console could
        not show. Reading the registry fixes the direction of the problem: the
        list is what the runtime can actually see.
        """
        return [
            {
                "name": skill.name,
                "status": skill.status,
                "source": skill.source,
                "description": skill.description,
                "path": str(getattr(skill, "path", "")),
                "allowed_tools": sorted(skill.allowed_tools),
            }
            for skill in svc.a.skills.list()
        ]

    @app.get("/api/v1/skills/pending")
    async def pending_skills() -> list[dict[str, Any]]:
        """The ones waiting on a human, nearest-to-running first.

        Its own endpoint rather than a filter on the list, because the console
        polls this to decide whether to draw attention -- and "is anything
        waiting for me" should be one cheap question, not a client-side scan
        that has to know which statuses count.
        """
        return [
            {"name": skill.name, "status": skill.status, "source": skill.source}
            for skill in svc.a.skills.pending_review()
        ]

    @app.post("/api/v1/skills/{name}/promote")
    async def promote_skill(name: str, payload: SkillPromotion) -> dict[str, Any]:
        """Move a skill one rung up the ladder.

        Exposed over HTTP because the ladder is the safety mechanism, and a
        safety mechanism reachable only from the terminal is one that gets
        bypassed -- by editing the database, or by moving files around.
        """
        try:
            skill = svc.a.skills.promote(name, payload.status)
        except Exception as exc:  # noqa: BLE001 - SkillError and unknown-name alike
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "name": skill.name,
            "status": skill.status,
            "source": skill.source,
        }

    @app.get("/api/v1/skills/runs")
    async def skill_runs(name: str | None = None) -> list[dict[str, Any]]:
        return svc.a.store.list_skill_runs(skill_name=name)

    # -- A2A v1.0 ---------------------------------------------------------
    def _a2a_server() -> Any:
        from wukong.a2a import A2AServer

        return A2AServer(svc.a, token_required=bool(token))

    def _base_url(request: Request) -> str:
        """Where this agent says it lives.

        A proxy deployment must set `a2a.public_url`: the request's own
        base_url is whatever the proxy forwarded, and a card advertising an
        internal address is a card peers cannot use.
        """
        if svc.settings.a2a.public_url:
            return svc.settings.a2a.public_url.rstrip("/")
        return str(request.base_url).rstrip("/")

    @app.get("/.well-known/agent-card.json")
    async def agent_card(request: Request) -> dict[str, Any]:
        if not svc.settings.a2a.enabled:
            raise HTTPException(
                status_code=404,
                detail="A2A is not enabled; set a2a.enabled to publish this agent",
            )
        card = _a2a_server().card(url=f"{_base_url(request)}/a2a")
        return card.model_dump(mode="json")

    @app.post("/a2a")
    async def a2a_endpoint(request: Request) -> Any:
        """The JSON-RPC binding, on one endpoint.

        `message/stream` answers with SSE rather than a JSON-RPC response,
        which is what the spec's HTTP binding says: the streaming method
        returns a stream, and the JSON-RPC envelope only frames the
        unary methods.
        """
        if not svc.settings.a2a.enabled:
            raise HTTPException(
                status_code=404,
                detail="A2A is not enabled; set a2a.enabled to accept peers",
            )
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 - a malformed body is a JSON-RPC parse error
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "request body is not JSON"},
            }
        if isinstance(payload, dict) and payload.get("method") == "message/stream":
            return StreamingResponse(
                _a2a_sse(_a2a_server(), payload),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return await _a2a_server().handle(payload)

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
        # Mounted at the **root**, not under a prefix.
        #
        # The page uses relative URLs (`style.css`, `app.js`), which resolve
        # against the directory the document was served from. Serving the HTML
        # from `/` while the assets lived under `/console/` meant the browser
        # asked for `/style.css` and got a 404: the page arrived with no
        # stylesheet and no script, so it rendered as unstyled HTML and every
        # control was dead. The status code for `/` was 200 the whole time,
        # which is why checking the entry point proved nothing.
        #
        # Registered last so every API route above matches first; `html=True`
        # serves index.html for `/`.
        app.mount("/", StaticFiles(directory=str(CONSOLE_DIR), html=True), name="console")

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
        request.headers.get("x-wukong-token") or request.query_params.get("token") or ""
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
        "budgets": _budgets(svc, task_id),
    }


def _budgets(svc: Service, task_id: str) -> dict[str, Any]:
    """The limits this task was created with, for the console's progress bars.

    Read from the TASK_CREATED event rather than the projection: the budgets
    are part of the task's own record, and the projection is a convenience view
    that does not carry them. The first event of a task is always its creation,
    so this is one indexed row, not a scan.
    """
    for event in svc.a.store.events(task_id, limit=1):
        if getattr(event.type, "value", event.type) == "task_created":
            return dict((event.payload or {}).get("budgets") or {})
    return {}


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
    if item.is_durable and item.event is not None:
        return encoder.on_event(item.event)
    return []


async def _a2a_sse(server: Any, payload: dict[str, Any]) -> AsyncIterator[str]:
    """Frame A2A status updates as SSE.

    Each frame carries the JSON-RPC id, so a client multiplexing several
    requests on one connection can attribute them -- and so the stream is
    still a JSON-RPC binding rather than a second, undocumented protocol.
    """
    request_id = payload.get("id")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    try:
        async for update in server.stream(params):
            frame = {"jsonrpc": "2.0", "id": request_id, "result": update}
            yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"
    except Exception as exc:  # noqa: BLE001 - the peer gets an error frame, not a dropped socket
        error = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"{type(exc).__name__}: {exc}"[:300],
            },
        }
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"


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
            yield agui.run_started(thread_id, run_id, task_id=task_id)
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

            if not item.is_durable or item.event is None:
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
        prior_context=_prior_context(agent.store, session_id),
    )
    async for event in _agui_events(
        svc, task_id=task_id, thread_id=thread_id, run_id=run_id,
        replay=False, announce_start=True, launch=launch,
    ):
        yield agui.sse(event)


#: Above this, the preview pane says how big the file is instead of rendering
#: it. Half a megabyte is roughly where a `<pre>` stops being scrollable and
#: starts being a frozen tab.
_PREVIEW_MAX_BYTES = 512 * 1024


def _digest(raw: bytes) -> str:
    """A short content digest, for the editor's optimistic lock.

    Short on purpose: it is shown to nobody and compared only for equality, so
    sixteen hex characters are as good as sixty-four and cheaper to carry
    around in a request body.
    """
    return hashlib.sha256(raw).hexdigest()[:16]


def _file_sort_key(entry: Path) -> tuple[int, str]:
    """Directories first, then case-insensitive by name.

    A tree that mixes them alphabetically makes the reader hunt for the folder
    they want. Grouping is what every file browser does, and it costs nothing.
    """
    try:
        is_dir = entry.is_dir()
    except OSError:
        is_dir = False
    return (0 if is_dir else 1, entry.name.lower())


def _prior_context(store: Any, session_id: str) -> str:
    """Earlier turns of this conversation, rendered for the system prompt.

    A session is a thread: the user asked something, got an answer, and is now
    asking a follow-up. Every task starts from a blank state by design -- that
    is what makes one auditable on its own -- so the thread has to be handed
    back explicitly. Without it "expand on the second point" arrives with no
    second point to expand on, and the model reasonably starts over.

    The most recent turns are kept, not the earliest: a follow-up refers to
    what was just said, and a long thread would otherwise push the thing being
    followed off the top.
    """
    turns = store.session_turns(session_id, limit=6)
    if not turns:
        return ""
    lines = ["# Earlier in this conversation", ""]
    for i, turn in enumerate(turns, start=1):
        lines.append(f"## 第 {i} 轮")
        lines.append(f"用户：{turn['goal'][:600]}")
        lines.append("")
        lines.append(f"你：{turn['answer'][:1200]}")
        lines.append("")
    lines.append(
        "The request below is a follow-up in this thread. Read it against the "
        "turns above — do not start over, and do not redo work those answers "
        "already cover."
    )
    return "\n".join(lines)


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


__all__ = ["create_app", "Service", "app"]


def app() -> FastAPI:  # pragma: no cover - uvicorn factory target
    return create_app()
