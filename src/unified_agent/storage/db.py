"""SQLite connection + schema.

Schema decisions worth defending:

* `events` is the source of truth. `tasks.state` is a *projection* kept for
  fast reads (list/show) and is rebuilt from events on resume. The original
  spec kept both without saying which wins; when they disagree, the
  projection is wrong by definition.
* `tool_calls` is a write-ahead ledger. The row is inserted with
  status='started' *before* the tool runs, so a crash mid-execution leaves
  a durable trace that the tool may have had an effect.
* FTS5 uses the `trigram` tokenizer. `unicode61` treats a whole run of CJK
  as one token, so "认证" would never match "认证模块". Trigram needs >=3
  characters, hence the LIKE fallback in memory/store.py.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    working_dir TEXT NOT NULL DEFAULT '',
    model_alias TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL,
    goal                 TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending',
    version              INTEGER NOT NULL DEFAULT 0,
    last_event_seq       INTEGER NOT NULL DEFAULT 0,
    state                TEXT,
    result               TEXT,
    pending_confirmation TEXT,
    steps_used           INTEGER NOT NULL DEFAULT 0,
    tokens_in            INTEGER NOT NULL DEFAULT 0,
    tokens_out           INTEGER NOT NULL DEFAULT 0,
    cost_usd             REAL NOT NULL DEFAULT 0,
    error                TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tasks_session ON tasks(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_tasks_status  ON tasks(status);

CREATE TABLE IF NOT EXISTS events (
    task_id    TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_id, seq)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    step_id         TEXT NOT NULL DEFAULT '',
    attempt         INTEGER NOT NULL DEFAULT 1,
    name            TEXT NOT NULL,
    arguments       TEXT NOT NULL DEFAULT '{}',
    effect_class    TEXT NOT NULL DEFAULT 'read_only',
    idempotency_key TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'started',
    result          TEXT,
    error           TEXT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    duration_ms     INTEGER
);
CREATE INDEX IF NOT EXISTS ix_tool_calls_key    ON tool_calls(idempotency_key);
CREATE INDEX IF NOT EXISTS ix_tool_calls_task   ON tool_calls(task_id, started_at);
CREATE INDEX IF NOT EXISTS ix_tool_calls_status ON tool_calls(status);

CREATE TABLE IF NOT EXISTS artifacts (
    id           TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL,
    tool_call_id TEXT,
    path         TEXT NOT NULL,
    bytes        INTEGER NOT NULL DEFAULT 0,
    sha256       TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_artifacts_task ON artifacts(task_id);

CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    session_id    TEXT,
    scope         TEXT NOT NULL DEFAULT 'project',
    content       TEXT NOT NULL,
    tags          TEXT NOT NULL DEFAULT '',
    importance    REAL NOT NULL DEFAULT 0.5,
    source        TEXT NOT NULL DEFAULT 'agent',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS ix_memories_scope ON memories(scope, created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    tags,
    content='memories',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.rowid, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.rowid, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS skills (
    name          TEXT PRIMARY KEY,
    source        TEXT NOT NULL DEFAULT 'local',
    path          TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'active',
    allowed_tools TEXT NOT NULL DEFAULT '',
    installed_at  TEXT NOT NULL,
    sha256        TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS skill_runs (
    id         TEXT PRIMARY KEY,
    skill_name TEXT NOT NULL,
    task_id    TEXT,
    outcome    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def connect(db_path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    uri = f"file:{path}?mode=ro" if read_only else None
    conn = sqlite3.connect(
        uri or str(path),
        check_same_thread=False,
        isolation_level=None,  # explicit transactions
        uri=uri is not None,
    )
    conn.row_factory = sqlite3.Row
    if not read_only:
        conn.executescript(_DDL)
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
    return conn
