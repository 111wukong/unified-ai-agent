"""JSONL event sink.

One file per day under $UAA_HOME/logs. Written with a lock because the
runtime can be resumed from a second process while the first is still
shutting down, and interleaved half-lines make the log unparseable.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from unified_agent.observability.events import Event
from unified_agent.observability.redact import DEFAULT_REDACTOR, Redactor


class JsonlSink:
    def __init__(self, log_dir: Path, *, redactor: Redactor | None = None) -> None:
        self.log_dir = Path(log_dir)
        self.redactor = redactor or DEFAULT_REDACTOR
        self._lock = threading.Lock()
        self._fh = None
        self._open_day = ""

    def _handle(self):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._fh is None or day != self._open_day:
            if self._fh is not None:
                self._fh.close()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"events-{day}.jsonl"
            self._fh = path.open("a", encoding="utf-8")
            self._open_day = day
        return self._fh

    def emit(self, event: Event) -> None:
        record = {
            "seq": event.seq,
            "task_id": event.task_id,
            "type": event.type.value,
            "payload": self.redactor.deep(event.payload),
            "created_at": event.created_at,
            "pid": os.getpid(),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            fh = self._handle()
            fh.write(line + "\n")
            fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
