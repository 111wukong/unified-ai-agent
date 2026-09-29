"""Durable store: event log, task projection, tool-call ledger, artifacts, memory, skills."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from unified_agent.memory.vector import VectorIndex

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
        read_only: bool = False,
    ) -> None:
        self.conn = connect(db_path, read_only=read_only)
        self.redactor = redactor or DEFAULT_REDACTOR
        # Optional JsonlSink. Every durable event is mirrored to the append-only
        # log file, because the DB is mutable in principle and the JSONL is not.
        self.sink = sink
        self._lock = threading.RLock()
        self._vector_index: Any = None

    @classmethod
    def readonly(cls, db_path: Path | str) -> "Store":
        """A store that cannot write, for read-only commands.

        A read path that can write is a read path whose bug corrupts the
        store, and the CLI opens the database for every listing it prints.

        Falls back to a normal open when the file does not exist yet: a
        read-only connection cannot create the file, and "nothing has been
        recorded yet" must not be an error.
        """
        path = Path(db_path)
        if not path.exists():
            return cls(path)
        return cls(path, read_only=True)

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

    def ensure_session(
        self,
        *,
        name: str,
        working_dir: str,
        model_alias: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Reuse the session for a given (working_dir, name) pair if it exists.

        `metadata` is only applied when the session is created: overwriting it
        on reuse would let the most recent caller silently rewrite the
        provenance of everything already in the session.
        """
        row = self.conn.execute(
            "SELECT id FROM sessions WHERE name=? AND working_dir=? ORDER BY created_at DESC LIMIT 1",
            (name, working_dir),
        ).fetchone()
        if row:
            return row["id"]
        return self.create_session(
            name=name,
            working_dir=working_dir,
            model_alias=model_alias,
            metadata=metadata,
        )

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
        self,
        *,
        session_id: str,
        goal: str,
        budgets: dict[str, Any] | None = None,
        task_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> str:
        tid = task_id or new_id("task")
        ts = now_iso()
        with self._lock:
            self.conn.execute(
                "INSERT INTO tasks(id,session_id,parent_task_id,goal,status,created_at,"
                "updated_at) VALUES(?,?,?,?,?,?,?)",
                (tid, session_id, parent_task_id, goal, "pending", ts, ts),
            )
        self.append(
            tid,
            EventType.TASK_CREATED,
            {
                "session_id": session_id,
                "goal": goal,
                "budgets": budgets or {},
                "parent_task_id": parent_task_id,
            },
        )
        return tid

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def task_workspace(self, task_id: str) -> str | None:
        """The directory a task was created in, via its session.

        Two consumers, both about not acting in the wrong place: `resume`
        refuses a mismatch, and the CLI defaults `--workspace` from it so
        `uaa task approve <id>` does the right thing without the user having
        to remember which directory the task was started in.

        Read from the session rather than a new column: the session already
        records it, and a second copy is a second thing that can disagree.
        """
        row = self.conn.execute(
            "SELECT s.working_dir AS working_dir FROM tasks t"
            " JOIN sessions s ON s.id = t.session_id WHERE t.id=?",
            (task_id,),
        ).fetchone()
        return row["working_dir"] if row else None

    def list_children(self, parent_task_id: str) -> list[dict[str, Any]]:
        """Tasks a workflow node started, so a run's tree is walkable."""
        rows = self.conn.execute(
            "SELECT * FROM tasks WHERE parent_task_id=? ORDER BY created_at ASC, rowid ASC",
            (parent_task_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_tasks(
        self,
        *,
        session_id: str | None = None,
        statuses: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Newest first. `statuses` filters on the projection's status column.

        Filtering here rather than by a point lookup on a single task is the
        honest shape: the projection is what a list view is for, whereas
        `resume` must decide from the event fold (see `AgentRuntime.resume`).
        A point lookup that read the projection would be a second, staler
        answer to "can this be resumed".
        """
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id=?")
            params.append(session_id)
        if statuses is not None:
            wanted = list(statuses)
            if not wanted:
                return []
            clauses.append(f"status IN ({','.join('?' * len(wanted))})")
            params.extend(wanted)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            "SELECT id,session_id,parent_task_id,goal,status,steps_used,tokens_in,"
            "tokens_out,cost_usd,created_at,updated_at FROM tasks"
            f"{where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
            tuple(params),
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
        redact: bool = True,
    ) -> str:
        """Persist a blob. `redact=False` only for file checkpoints.

        Everything else goes through the redactor on the way in -- an
        offloaded tool output is still a log line. A checkpoint is the
        exception, and it has to be: it exists to be written back to disk
        byte-for-byte, so redacting it produces a *corrupted* restore. The
        content is a copy of a file that is already in the workspace, so it
        is in the same trust domain either way; the artifact directory is not
        a new exposure.
        """
        artifact_dir.mkdir(parents=True, exist_ok=True)
        aid = new_id("art")
        path = artifact_dir / f"{aid}.{suffix}"
        path.write_text(self.redactor(content) if redact else content, encoding="utf-8")
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

    def search_tasks(
        self,
        query: str,
        *,
        limit: int = 8,
        include_children: bool = False,
    ) -> list[dict[str, Any]]:
        """Find past tasks by goal or final answer.

        Sub-agent and workflow-child tasks are excluded by default. They are
        this runtime's own fan-out rather than history: one multi-agent run
        writes a task per worker, and including them buries the handful of tasks
        the *user* started under the ones the runtime invented. Hermes hides
        them from its session search for the same reason.

        The query is quoted before it reaches FTS5. A raw string can contain
        `"`, `*` or `NEAR`, each of which is syntax, and a search box that
        throws on a stray quote is worse than one that finds nothing.
        """
        text = (query or "").strip()
        if not text:
            return []
        rows = self._search_tasks_fts(text, limit=limit, include_children=include_children)
        if rows:
            return rows
        # The `trigram` tokenizer indexes three-character sequences, so a query
        # shorter than that matches nothing -- and two-character words are the
        # common case in Chinese ("边界", "缓存", "测试"). Falling back to a
        # scan keeps a short query from silently finding nothing, which is
        # indistinguishable from "there is no such task".
        return self._search_tasks_like(text, limit=limit, include_children=include_children)

    def _search_tasks_fts(
        self, text: str, *, limit: int, include_children: bool
    ) -> list[dict[str, Any]]:
        phrase = '"' + text.replace('"', '""') + '"'
        sql = (
            "SELECT t.id, t.session_id, t.parent_task_id, t.goal, t.status,"
            " t.created_at, t.updated_at, t.steps_used, t.tokens_in, t.tokens_out,"
            " bm25(tasks_fts) AS rank"
            " FROM tasks_fts JOIN tasks t ON t.rowid = tasks_fts.rowid"
            " WHERE tasks_fts MATCH ?"
        )
        if not include_children:
            sql += " AND t.parent_task_id IS NULL"
        sql += " ORDER BY rank LIMIT ?"
        try:
            rows = self.conn.execute(sql, (phrase, limit)).fetchall()
        except sqlite3.OperationalError:
            # Unbalanced quotes and the like reach FTS5 as syntax. A search box
            # that raises on a stray character is worse than one that finds
            # nothing, and the LIKE pass below will still try.
            return []
        return [dict(row) for row in rows]

    def _search_tasks_like(
        self, text: str, *, limit: int, include_children: bool
    ) -> list[dict[str, Any]]:
        pattern = f"%{text}%"
        sql = (
            "SELECT t.id, t.session_id, t.parent_task_id, t.goal, t.status,"
            " t.created_at, t.updated_at, t.steps_used, t.tokens_in, t.tokens_out,"
            " 0.0 AS rank"
            " FROM tasks t WHERE (t.goal LIKE ? OR t.result LIKE ?)"
        )
        params: list[Any] = [pattern, pattern]
        if not include_children:
            sql += " AND t.parent_task_id IS NULL"
        sql += " ORDER BY t.created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def search_memories(
        self,
        query: str,
        *,
        scope: str | None = None,
        limit: int = 8,
        mode: str = "fts",
        query_vector: list[float] | None = None,
        vector_model: str | None = None,
        min_similarity: float = 0.0,
        vector_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories. `mode` is `fts`, `vector`, `hybrid` or `auto`.

        `auto` uses hybrid when a query vector is supplied and fts otherwise,
        so callers that cannot embed (no provider configured) get the old
        behaviour without special-casing.

        Hybrid merges the two rankings with reciprocal rank fusion rather than
        a weighted score sum. FTS5's `bm25` and a cosine similarity are not on
        the same scale and have no shared meaning, so any linear blend needs a
        normalisation step that is itself a tuning problem. RRF only looks at
        *positions*, which is why it works without tuning and why it is the
        right default here.

        `vector_limit` is how deep the vector branch reads before fusion, and
        is floored at `limit`: a list shorter than the final result set cannot
        distinguish "rank 1 here, absent there" from "rank 1 in both", which is
        the only signal RRF has. None keeps the `limit * 3` heuristic for
        callers with no configuration to read.
        """
        q = query.strip()
        if mode == "auto":
            mode = "hybrid" if query_vector else "fts"
        if mode == "fts" and not q:
            return []
        if mode == "vector" and not query_vector:
            mode = "fts"

        if mode == "fts":
            return self._search_fts(q, scope=scope, limit=limit)
        if mode == "vector":
            return self.search_vectors(
                query_vector or [],
                model=vector_model or "",
                limit=limit,
                scope=scope,
                min_similarity=min_similarity,
            )

        ranked: list[list[str]] = []
        fts_rows = self._search_fts(q, scope=scope, limit=limit * 3) if q else []
        ranked.append([r["id"] for r in fts_rows])
        depth = max(limit, vector_limit) if vector_limit else limit * 3
        vector_rows = (
            self.search_vectors(
                query_vector or [],
                model=vector_model or "",
                limit=depth,
                scope=scope,
                min_similarity=min_similarity,
            )
            if query_vector
            else []
        )
        ranked.append([r["id"] for r in vector_rows])
        order = _reciprocal_rank_fusion(ranked)[:limit]
        if not order:
            return []
        by_id = {r["id"]: r for r in [*fts_rows, *vector_rows]}
        out: list[dict[str, Any]] = []
        for memory_id in order:
            row = by_id.get(memory_id)
            if row is None:
                row = self.get_memory(memory_id)
            if row:
                out.append(row)
        return out

    def _search_fts(self, q: str, *, scope: str | None, limit: int) -> list[dict[str, Any]]:
        """FTS5 trigram, with a LIKE fallback for short queries.

        Trigram tokenization needs >=3 characters. Chinese two-character
        terms ("认证", "部署") are extremely common, so falling back to LIKE
        is not an optimisation -- it is required for correctness.
        """
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

    def search_memories_any(
        self, terms: Sequence[str], *, scope: str | None = None, limit: int = 8
    ) -> list[dict[str, Any]]:
        """FTS5 OR over terms, ranked by how many of them matched.

        Needed because `search_memories` quotes the whole query as one term.
        That is right for a user typing a phrase, and wrong for "find things
        like this paragraph": a candidate sentence is never a substring of a
        stored one, so a literal search finds nothing and the contradiction
        check silently has no neighbours to look at.
        """
        usable = [t.strip() for t in terms if len(t.strip()) >= 2]
        if not usable:
            return []
        seen: dict[str, dict[str, Any]] = {}
        for term in usable[:12]:
            for row in self._search_fts(term, scope=scope, limit=limit * 4):
                entry = seen.setdefault(row["id"], {**row, "matched": 0})
                entry["matched"] += 1
        if not seen:
            return []
        ranked = sorted(
            seen.values(),
            key=lambda r: (r["matched"], r.get("importance", 0.0)),
            reverse=True,
        )
        return ranked[:limit]

    # -- vectors ----------------------------------------------------------
    @property
    def vectors(self) -> "VectorIndex":
        if self._vector_index is None:
            from unified_agent.memory.vector import VectorIndex

            self._vector_index = VectorIndex(self.conn)
        return self._vector_index

    def put_vector(
        self, memory_id: str, vector: list[float], *, model: str, dim: int | None = None
    ) -> None:
        """Store a vector under `model`. Pass `dim` to assert the expected size."""
        with self._lock:
            self.vectors.upsert(memory_id, vector, model=model, dim=dim)

    def put_vectors(
        self, rows: list[tuple[str, list[float]]], *, model: str, dim: int | None = None
    ) -> int:
        """Batch form of `put_vector`, for the reindex path.

        One lock acquisition for the whole batch rather than one per row, so a
        reindex of a few thousand memories does not hold and release the lock
        a few thousand times.
        """
        with self._lock:
            return self.vectors.upsert_many(rows, model=model, dim=dim)

    def search_vectors(
        self,
        query_vector: list[float],
        *,
        model: str,
        limit: int = 10,
        scope: str | None = None,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Nearest memories by cosine similarity. Superseded rows are excluded."""
        if not query_vector:
            return []
        hits = self.vectors.search(
            query_vector, model=model, limit=limit * 2, min_similarity=min_similarity
        )
        if not hits:
            return []
        by_id = {hit.memory_id: hit.similarity for hit in hits}
        placeholders = ",".join("?" for _ in by_id)
        sql = (
            f"SELECT * FROM memories WHERE id IN ({placeholders})"
            " AND superseded_by IS NULL"
        )
        params: list[Any] = list(by_id)
        if scope:
            sql += " AND scope=?"
            params.append(scope)
        rows = self.conn.execute(sql, params).fetchall()
        out = [{**dict(r), "similarity": by_id[r["id"]]} for r in rows]
        out.sort(key=lambda r: r["similarity"], reverse=True)
        return out[:limit]

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return dict(row) if row else None

    # -- invalidation -----------------------------------------------------
    def invalidate_memory(
        self,
        memory_id: str,
        *,
        replaced_by: str,
        reason: str = "",
        source: str = "curator",
    ) -> bool:
        """Mark a memory as superseded rather than deleting it.

        Zep's temporal model, at the cost of one column and one row: the old
        fact stops appearing in search results but the store can still answer
        "what did this used to be", which is the question a plain overwrite
        destroys.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT id FROM memories WHERE id=? AND superseded_by IS NULL", (memory_id,)
            ).fetchone()
            if row is None:
                return False
            self.conn.execute(
                "UPDATE memories SET superseded_by=?, updated_at=? WHERE id=?",
                (replaced_by, now_iso(), memory_id),
            )
            self.conn.execute(
                "INSERT INTO memory_revisions(id,memory_id,replaced_by,reason,source,created_at)"
                " VALUES(?,?,?,?,?,?)",
                (new_id("rev"), memory_id, replaced_by, reason, source, now_iso()),
            )
        return True

    def memory_history(self, memory_id: str) -> list[dict[str, Any]]:
        """The whole chain a memory belongs to, oldest first.

        Walks `superseded_by` forward and `memory_revisions` backward, so the
        caller sees "was X, then Y, now Z" regardless of which link it started
        from.
        """
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Backwards to the origin.
        cursor = memory_id
        while cursor and cursor not in seen:
            seen.add(cursor)
            row = self.conn.execute(
                "SELECT * FROM memory_revisions WHERE replaced_by=? ORDER BY created_at DESC",
                (cursor,),
            ).fetchone()
            if row is None:
                break
            cursor = row["memory_id"]
            chain.append({"id": cursor, "via": "revision"})

        chain.reverse()
        chain.append({"id": memory_id, "via": "self"})

        # Forwards through the supersessions.
        cursor = memory_id
        while True:
            row = self.conn.execute(
                "SELECT superseded_by FROM memories WHERE id=?", (cursor,)
            ).fetchone()
            if row is None or not row["superseded_by"] or row["superseded_by"] in seen:
                break
            cursor = row["superseded_by"]
            seen.add(cursor)
            chain.append({"id": cursor, "via": "superseded"})

        out: list[dict[str, Any]] = []
        for entry in chain:
            memory = self.get_memory(entry["id"])
            if memory is None:
                continue
            revision = self.conn.execute(
                "SELECT reason, source, created_at FROM memory_revisions WHERE memory_id=?",
                (entry["id"],),
            ).fetchone()
            out.append(
                {
                    **memory,
                    "relation": entry["via"],
                    "replaced_reason": revision["reason"] if revision else "",
                }
            )
        return out

    def memory_stats(self) -> dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        active = self.conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE superseded_by IS NULL"
        ).fetchone()["n"]
        by_scope = {
            r["scope"]: r["n"]
            for r in self.conn.execute(
                "SELECT scope, COUNT(*) AS n FROM memories WHERE superseded_by IS NULL"
                " GROUP BY scope"
            ).fetchall()
        }
        by_source = {
            r["source"]: r["n"]
            for r in self.conn.execute(
                "SELECT source, COUNT(*) AS n FROM memories WHERE superseded_by IS NULL"
                " GROUP BY source"
            ).fetchall()
        }
        return {
            "total": int(total),
            "active": int(active),
            "superseded": int(total - active),
            "by_scope": by_scope,
            "by_source": by_source,
            "vectors": self.vectors.describe(),
        }

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
                " ORDER BY created_at DESC, rowid DESC LIMIT ?",
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

    def skill_statuses(self) -> dict[str, str]:
        """name -> status, for seeding the registry's ladder.

        This is the read half of the ladder. Without it the table was
        write-only: `promote` could be called all day and the next process
        still saw whatever the directory implied.

        The write half is `upsert_skill`, not a bare UPDATE -- a promotion
        can be the first thing that ever touches a freshly written candidate,
        and an UPDATE against a missing row succeeds while changing nothing.
        """
        rows = self.conn.execute("SELECT name, status FROM skills").fetchall()
        return {r["name"]: r["status"] for r in rows}

    def record_skill_run(self, *, skill_name: str, task_id: str, outcome: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO skill_runs(id,skill_name,task_id,outcome,created_at) VALUES(?,?,?,?,?)",
                (new_id("skillrun"), skill_name, task_id, outcome, now_iso()),
            )

    def list_skill_runs(self, *, skill_name: str | None = None) -> list[dict[str, Any]]:
        """Skill usage history, newest first.

        The feedback half of the loop: `promote` to `active` is a guess until
        there is evidence, and this is the only place that evidence lives.

        Ordered by `created_at` *and* `rowid`. `created_at` is millisecond
        precision, so two runs recorded in the same millisecond tie and the
        order becomes whatever SQLite happens to return -- the same defect
        that already bit the task list. `rowid` is monotonic for inserts, so
        it breaks the tie by real insertion order.
        """
        if skill_name:
            rows = self.conn.execute(
                "SELECT * FROM skill_runs WHERE skill_name=?"
                " ORDER BY created_at DESC, rowid DESC",
                (skill_name,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM skill_runs ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def skill_run_counts(self) -> dict[str, dict[str, int]]:
        """name -> {outcome: count}. One query, for the listing."""
        rows = self.conn.execute(
            "SELECT skill_name, outcome, COUNT(*) AS n FROM skill_runs GROUP BY skill_name, outcome"
        ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            out.setdefault(row["skill_name"], {})[row["outcome"]] = row["n"]
        return out


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], *, k: int = 60) -> list[str]:
    """Merge rankings by position, not by score.

    bm25 and cosine similarity are not comparable quantities, and normalising
    them against each other is a tuning problem with no right answer. RRF only
    asks where an item placed, which is why it needs no weight to tune.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            if not item:
                continue
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


def _fts_query(text: str) -> str:
    """Quote each whitespace-separated term so FTS5 syntax chars are literal."""
    terms = [t for t in text.replace('"', " ").split() if t]
    return " AND ".join(f'"{t}"' for t in terms) if terms else '""'
