"""Durable store: event log, task projection, tool-call ledger, artifacts, memory, skills."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from unified_agent.observability.events import Event, EventType
from unified_agent.observability.redact import DEFAULT_REDACTOR, Redactor
from unified_agent.storage.db import connect


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ToolCallRecord:
    __slots__ = (
        "id",
        "task_id",
        "step_id",
        "attempt",
        "name",
        "arguments",
        "effect_class",
        "idempotency_key",
        "status",
        "result",
        "error",
        "started_at",
        "finished_at",
        "duration_ms",
    )

    def __init__(self, row: sqlite3.Row | dict[str, Any]) -> None:
        keys = set(row.keys())
        for key in self.__slots__:
            setattr(self, key, row[key] if key in keys else None)
        if isinstance(self.arguments, str):
            try:
                self.arguments = json.loads(self.arguments)
            except json.JSONDecodeError:
                self.arguments = {}
        if isinstance(self.result, str) and self.result:
            try:
                self.result = json.loads(self.result)
            except json.JSONDecodeError:
                pass

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ToolCallRecord {self.name} {self.status} attempt={self.attempt}>"


class Store:
    """All persistence. Single connection, guarded by a lock.

    Deliberately not async: SQLite calls here are sub-millisecond, and
    pushing them onto a thread pool would buy nothing but cancellation
    bugs in the middle of a transaction.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        redactor: Redactor | None = None,
        sink: Any = None,
    ) -> None:
        self.conn = connect(db_path)
        self.redactor = redactor or DEFAULT_REDACTOR
        # Optional JsonlSink. Every durable event is mirrored to the append-only
        # log file, because the DB is mutable in principle and the JSONL is not.
        self.sink = sink
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    def create_session(
        self, *, name: str, working_dir: str, model_alias: str, metadata: dict | None = None
    ) -> str:
        sid = new_id("session")
        with self._lock:
            self.conn.execute(
                "INSERT INTO sessions(id,name,working_dir,model_alias,created_at,metadata)"
                " VALUES(?,?,?,?,?,?)",
                (sid, name, working_dir, model_alias, now_iso(), json.dumps(metadata or {})),
            )
        return sid

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def ensure_session(self, *, name: str, working_dir: str, model_alias: str) -> str:
        """Reuse the session for a given (working_dir, name) pair if it exists."""
        row = self.conn.execute(
            "SELECT id FROM sessions WHERE name=? AND working_dir=? ORDER BY created_at DESC LIMIT 1",
            (name, working_dir),
        ).fetchone()
        if row:
            return row["id"]
        return self.create_session(name=name, working_dir=working_dir, model_alias=model_alias)

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # events  (append-only, gapless per task)
    # ------------------------------------------------------------------

    def append(self, task_id: str, type_: EventType, payload: dict[str, Any]) -> Event:
        safe = self.redactor.deep(payload)
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS s FROM events WHERE task_id=?", (task_id,)
                ).fetchone()
                seq = int(row["s"]) + 1
                created = now_iso()
                self.conn.execute(
                    "INSERT INTO events(task_id,seq,type,payload,created_at) VALUES(?,?,?,?,?)",
                    (task_id, seq, type_.value, json.dumps(safe, default=str), created),
                )
                self.conn.execute(
                    "UPDATE tasks SET last_event_seq=?, updated_at=? WHERE id=?",
                    (seq, created, task_id),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        event = Event(seq=seq, task_id=task_id, type=type_, payload=safe, created_at=created)
        if self.sink is not None:
            try:
                self.sink.emit(event)
            except Exception:  # noqa: BLE001 - logging must never break a task
                pass
        return event

    def events(self, task_id: str, *, since_seq: int = 0, limit: int = 5000) -> list[Event]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE task_id=? AND seq>? ORDER BY seq ASC LIMIT ?",
            (task_id, since_seq, limit),
        ).fetchall()
        return [
            Event(
                seq=r["seq"],
                task_id=r["task_id"],
                type=EventType(r["type"]),
                payload=json.loads(r["payload"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def last_seq(self, task_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0) AS s FROM events WHERE task_id=?", (task_id,)
        ).fetchone()
        return int(row["s"])

    # ------------------------------------------------------------------
    # tasks
    # ------------------------------------------------------------------

    def create_task(
        self, *, session_id: str, goal: str, budgets: dict[str, Any] | None = None
    ) -> str:
        tid = new_id("task")
        ts = now_iso()
        with self._lock:
            self.conn.execute(
                "INSERT INTO tasks(id,session_id,goal,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (tid, session_id, goal, "pending", ts, ts),
            )
        self.append(
            tid,
            EventType.TASK_CREATED,
            {"session_id": session_id, "goal": goal, "budgets": budgets or {}},
        )
        return tid

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def list_tasks(self, *, session_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if session_id:
            rows = self.conn.execute(
                "SELECT id,session_id,goal,status,steps_used,tokens_in,tokens_out,cost_usd,"
                "created_at,updated_at FROM tasks WHERE session_id=? "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id,session_id,goal,status,steps_used,tokens_in,tokens_out,cost_usd,"
                "created_at,updated_at FROM tasks ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_projection(
        self,
        task_id: str,
        *,
        status: str,
        state: dict[str, Any],
        steps_used: int,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        result: dict[str, Any] | None = None,
        pending_confirmation: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE tasks SET status=?, state=?, steps_used=?, tokens_in=?, tokens_out=?,"
                " cost_usd=?, result=?, pending_confirmation=?, error=?, version=version+1,"
                " updated_at=? WHERE id=?",
                (
                    status,
                    json.dumps(self.redactor.deep(state), default=str),
                    steps_used,
                    tokens_in,
                    tokens_out,
                    cost_usd,
                    json.dumps(result, default=str) if result is not None else None,
                    json.dumps(self.redactor.deep(pending_confirmation), default=str)
                    if pending_confirmation
                    else None,
                    error,
                    now_iso(),
                    task_id,
                ),
            )

    def find_resumable(self, task_id: str) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if task and task["status"] in {"pending", "planning", "running", "waiting_confirmation"}:
            return task
        return None

    # ------------------------------------------------------------------
    # tool-call ledger (write-ahead)
    # ------------------------------------------------------------------

    def begin_tool_call(
        self,
        *,
        task_id: str,
        step_id: str,
        attempt: int,
        name: str,
        arguments: dict[str, Any],
        effect_class: str,
        idempotency_key: str,
    ) -> str:
        """Insert the ledger row *before* execution. Returns the call id."""
        call_id = new_id("call")
        with self._lock:
            self.conn.execute(
                "INSERT INTO tool_calls(id,task_id,step_id,attempt,name,arguments,effect_class,"
                "idempotency_key,status,started_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    call_id,
                    task_id,
                    step_id,
                    attempt,
                    name,
                    json.dumps(self.redactor.deep(arguments), default=str),
                    effect_class,
                    idempotency_key,
                    "started",
                    now_iso(),
                ),
            )
        return call_id

    def finish_tool_call(
        self,
        call_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE tool_calls SET status=?, result=?, error=?, finished_at=?, duration_ms=?"
                " WHERE id=?",
                (
                    status,
                    json.dumps(self.redactor.deep(result), default=str) if result else None,
                    error,
                    now_iso(),
                    duration_ms,
                    call_id,
                ),
            )

    def tool_call(self, call_id: str) -> ToolCallRecord | None:
        row = self.conn.execute("SELECT * FROM tool_calls WHERE id=?", (call_id,)).fetchone()
        return ToolCallRecord(row) if row else None

    def calls_by_key(self, key: str) -> list[ToolCallRecord]:
        rows = self.conn.execute(
            "SELECT * FROM tool_calls WHERE idempotency_key=? ORDER BY attempt ASC, rowid ASC", (key,)
        ).fetchall()
        return [ToolCallRecord(r) for r in rows]

    def unfinished_calls(self, task_id: str) -> list[ToolCallRecord]:
        """Calls that were written-ahead but never reached a terminal state.

        These are the dangerous ones: the process died between "I am about
        to touch the world" and "here is what happened".
        """
        rows = self.conn.execute(
            "SELECT * FROM tool_calls WHERE task_id=? AND status='started' "
            "ORDER BY started_at ASC, rowid ASC",
            (task_id,),
        ).fetchall()
        return [ToolCallRecord(r) for r in rows]

    def list_tool_calls(self, task_id: str, limit: int = 500) -> list[ToolCallRecord]:
        rows = self.conn.execute(
            "SELECT * FROM tool_calls WHERE task_id=? ORDER BY started_at ASC, rowid ASC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [ToolCallRecord(r) for r in rows]

    # ------------------------------------------------------------------
    # artifacts
    # ------------------------------------------------------------------

    def save_artifact(
        self,
        *,
        task_id: str,
        tool_call_id: str | None,
        content: str,
        artifact_dir: Path,
        sha256: str,
        suffix: str = "txt",
    ) -> str:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        aid = new_id("art")
        path = artifact_dir / f"{aid}.{suffix}"
        # Redact on the way in. An offloaded tool output is still a log line.
        path.write_text(self.redactor(content), encoding="utf-8")
        with self._lock:
            self.conn.execute(
                "INSERT INTO artifacts(id,task_id,tool_call_id,path,bytes,sha256,created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (aid, task_id, tool_call_id, str(path), len(content), sha256, now_iso()),
            )
        return str(path)

    def list_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM artifacts WHERE task_id=? ORDER BY created_at ASC, rowid ASC", (task_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # memory
    # ------------------------------------------------------------------

    def add_memory(
        self,
        *,
        content: str,
        session_id: str | None = None,
        scope: str = "project",
        tags: Iterable[str] = (),
        importance: float = 0.5,
        source: str = "agent",
    ) -> str:
        mid = new_id("mem")
        ts = now_iso()
        safe = self.redactor(content)
        with self._lock:
            self.conn.execute(
                "INSERT INTO memories(id,session_id,scope,content,tags,importance,source,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (mid, session_id, scope, safe, " ".join(tags), importance, source, ts, ts),
            )
        return mid

    def search_memories(
        self, query: str, *, scope: str | None = None, limit: int = 8
    ) -> list[dict[str, Any]]:
        """FTS5 trigram search with a LIKE fallback for short queries.

        Trigram tokenization needs >=3 characters. Chinese two-character
        terms ("认证", "部署") are extremely common, so falling back to LIKE
        is not an optimisation -- it is required for correctness.
        """
        q = query.strip()
        if not q:
            return []
        rows: list[sqlite3.Row] = []
        if len(q) >= 3:
            try:
                sql = (
                    "SELECT m.*, bm25(memories_fts) AS score FROM memories_fts f"
                    " JOIN memories m ON m.rowid = f.rowid"
                    " WHERE memories_fts MATCH ? AND m.superseded_by IS NULL"
                )
                params: list[Any] = [_fts_query(q)]
                if scope:
                    sql += " AND m.scope=?"
                    params.append(scope)
                sql += " ORDER BY score LIMIT ?"
                params.append(limit)
                rows = self.conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            sql = (
                "SELECT *, 0.0 AS score FROM memories WHERE superseded_by IS NULL"
                " AND (content LIKE ? OR tags LIKE ?)"
            )
            like = f"%{q}%"
            params = [like, like]
            if scope:
                sql += " AND scope=?"
                params.append(scope)
            sql += " ORDER BY importance DESC, created_at DESC, rowid DESC LIMIT ?"
            params.append(limit)
            rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def list_memories(self, *, scope: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if scope:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE scope=? AND superseded_by IS NULL"
                " ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (scope, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE superseded_by IS NULL"
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # skills registry
    # ------------------------------------------------------------------

    def upsert_skill(
        self,
        *,
        name: str,
        path: str,
        description: str,
        source: str,
        status: str,
        allowed_tools: str,
        sha256: str,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO skills(name,source,path,description,status,allowed_tools,"
                "installed_at,sha256) VALUES(?,?,?,?,?,?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET source=excluded.source, path=excluded.path,"
                " description=excluded.description, status=excluded.status,"
                " allowed_tools=excluded.allowed_tools, sha256=excluded.sha256",
                (name, source, path, description, status, allowed_tools, now_iso(), sha256),
            )

    def list_skills(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM skills ORDER BY name ASC").fetchall()
        return [dict(r) for r in rows]

    def record_skill_run(self, *, skill_name: str, task_id: str, outcome: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO skill_runs(id,skill_name,task_id,outcome,created_at) VALUES(?,?,?,?,?)",
                (new_id("skillrun"), skill_name, task_id, outcome, now_iso()),
            )


def _fts_query(text: str) -> str:
    """Quote each whitespace-separated term so FTS5 syntax chars are literal."""
    terms = [t for t in text.replace('"', " ").split() if t]
    return " AND ".join(f'"{t}"' for t in terms) if terms else '""'
