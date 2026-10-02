"""Cosine search over memory vectors.

sqlite-vec is used as an **accelerator, not a schema owner**: it contributes
the `vec_distance_cosine` scalar function, and the rows stay in our own
`memory_vectors` table. Two consequences that are the whole reason for this
shape:

* The accelerated path and the pure-Python fallback compute the same metric
  over the same rows, so they can be checked against each other. If the
  extension owned the schema, "sqlite-vec is installed" would mean a
  *different database* rather than a faster one, and the two paths would
  drift apart in ways no test would catch.
* Vectors are keyed by `(model, dim)`. A memory embedded with model A is not
  comparable to one embedded with model B, so a search filters to a single
  model and the rest are simply absent rather than silently wrong.

Vectors are L2-normalised on the way in, which makes cosine similarity a dot
product and lets the fallback skip a division per row.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Sequence

from wukong.memory.embeddings import (
    cosine_similarity,
    pack_vector,
    unpack_vector,
)


def _normalise(vector: list[float]) -> list[float]:
    import math

    norm = math.sqrt(sum(v * v for v in vector))
    return vector if norm == 0 else [v / norm for v in vector]

# Above this many vectors, a Python scan starts to be noticeable in a chat
# turn. Not a hard limit -- just the point where the log says so.
PYTHON_SCAN_WARN = 20_000


@dataclass
class VectorHit:
    memory_id: str
    similarity: float

    def __repr__(self) -> str:  # pragma: no cover
        return f"<VectorHit {self.memory_id} {self.similarity:.4f}>"


class VectorIndex:
    def __init__(self, conn: sqlite3.Connection, *, accelerator: str | None = None) -> None:
        self.conn = conn
        self.accelerator = accelerator or self._detect_accelerator()

    def _detect_accelerator(self) -> str:
        try:
            self.conn.execute("SELECT vec_version()").fetchone()
        except sqlite3.Error:
            return "python"
        return "sqlite-vec"

    @property
    def accelerated(self) -> bool:
        return self.accelerator == "sqlite-vec"

    # -- writes -----------------------------------------------------------
    def upsert(
        self, memory_id: str, vector: Sequence[float], *, model: str, dim: int | None = None
    ) -> None:
        """Store a vector, normalised.

        Normalisation happens here rather than being a caller obligation.
        The two search paths compute cosine differently -- sqlite-vec
        normalises internally, the Python fallback does a dot product -- so an
        unnormalised vector makes them disagree by exactly the amount the
        input was off unit length. That is a silent ranking difference, and
        the whole reason for keeping both paths on the same metric is that
        they must not differ.
        """
        if not vector:
            return
        resolved_dim = dim or len(vector)
        if resolved_dim != len(vector):
            raise ValueError(f"dim={resolved_dim} does not match vector length {len(vector)}")
        normalised = _normalise(list(vector))
        self.conn.execute(
            "INSERT INTO memory_vectors(memory_id, model, dim, vec, created_at)"
            " VALUES(?,?,?,?,?)"
            " ON CONFLICT(memory_id) DO UPDATE SET model=excluded.model,"
            " dim=excluded.dim, vec=excluded.vec, created_at=excluded.created_at",
            (memory_id, model, resolved_dim, pack_vector(normalised), _now()),
        )

    def upsert_many(
        self,
        rows: Sequence[tuple[str, Sequence[float]]],
        *,
        model: str,
        dim: int | None = None,
    ) -> int:
        written = 0
        for memory_id, vector in rows:
            if vector:
                self.upsert(memory_id, vector, model=model, dim=dim)
                written += 1
        return written

    def delete(self, memory_id: str) -> None:
        self.conn.execute("DELETE FROM memory_vectors WHERE memory_id=?", (memory_id,))

    # -- reads ------------------------------------------------------------
    def search(
        self,
        query_vector: Sequence[float],
        *,
        model: str,
        limit: int = 10,
        exclude: Sequence[str] = (),
        min_similarity: float = 0.0,
    ) -> list[VectorHit]:
        """Nearest neighbours above `min_similarity`.

        The floor exists because a cosine similarity of exactly 0 means the
        two vectors share nothing, and returning those as "results" is worse
        than returning fewer. It is a floor, not a relevance gate: with a
        semantic model, unrelated text still scores well above zero, so this
        cannot be tuned into a quality threshold and is not presented as one.
        """
        if not query_vector:
            return []
        dim = len(query_vector)
        excluded = set(exclude)
        # Fetch a few extra so exclusions and the floor do not silently
        # shrink the result below what was asked for.
        fetch = limit + len(excluded)
        if self.accelerated:
            hits = self._search_sqlite_vec(query_vector, model=model, dim=dim, limit=fetch)
        else:
            hits = self._search_python(query_vector, model=model, dim=dim, limit=fetch)
        return [
            hit
            for hit in hits
            if hit.memory_id not in excluded and hit.similarity > min_similarity
        ][:limit]

    def _search_sqlite_vec(
        self, query_vector: Sequence[float], *, model: str, dim: int, limit: int
    ) -> list[VectorHit]:
        rows = self.conn.execute(
            "SELECT memory_id, vec_distance_cosine(vec, ?) AS distance"
            " FROM memory_vectors WHERE model=? AND dim=?"
            " ORDER BY distance ASC LIMIT ?",
            (pack_vector(list(query_vector)), model, dim, limit),
        ).fetchall()
        # vec_distance_cosine returns a distance (0 = identical); similarity
        # is its complement. Doing the conversion here keeps both paths
        # returning the same quantity.
        return [VectorHit(r["memory_id"], 1.0 - float(r["distance"])) for r in rows]

    def _search_python(
        self, query_vector: Sequence[float], *, model: str, dim: int, limit: int
    ) -> list[VectorHit]:
        query = list(query_vector)
        rows = self.conn.execute(
            "SELECT memory_id, vec FROM memory_vectors WHERE model=? AND dim=?",
            (model, dim),
        ).fetchall()
        scored = [
            VectorHit(row["memory_id"], cosine_similarity(query, unpack_vector(row["vec"])))
            for row in rows
        ]
        scored.sort(key=lambda hit: hit.similarity, reverse=True)
        return scored[:limit]

    # -- housekeeping -----------------------------------------------------
    def count(self, *, model: str | None = None, dim: int | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM memory_vectors"
        clauses: list[str] = []
        params: list[object] = []
        if model is not None:
            clauses.append("model=?")
            params.append(model)
        if dim is not None:
            clauses.append("dim=?")
            params.append(dim)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return int(self.conn.execute(sql, params).fetchone()["n"])

    def models(self) -> list[tuple[str, int, int]]:
        rows = self.conn.execute(
            "SELECT model, dim, COUNT(*) AS n FROM memory_vectors GROUP BY model, dim"
            " ORDER BY n DESC"
        ).fetchall()
        return [(r["model"], int(r["dim"]), int(r["n"])) for r in rows]

    def missing(self, *, model: str, dim: int | None = None) -> list[str]:
        """Memory ids with no vector for this model. Drives `memory reindex`."""
        sql = (
            "SELECT m.id FROM memories m"
            " LEFT JOIN memory_vectors v ON v.memory_id = m.id AND v.model = ?"
            " WHERE v.memory_id IS NULL"
        )
        params: list[object] = [model]
        if dim is not None:
            sql += " AND (v.dim IS NULL OR v.dim = ?)"
            params.append(dim)
        sql += " ORDER BY m.created_at ASC"
        return [r["id"] for r in self.conn.execute(sql, params).fetchall()]

    def stale_model(self, *, model: str, dim: int) -> list[str]:
        """Ids whose vector was built with a different model or dimension."""
        rows = self.conn.execute(
            "SELECT memory_id FROM memory_vectors WHERE model != ? OR dim != ?",
            (model, dim),
        ).fetchall()
        return [r["memory_id"] for r in rows]

    def describe(self) -> dict[str, object]:
        return {
            "accelerator": self.accelerator,
            "vectors": self.count(),
            "models": [
                {"model": model, "dim": dim, "count": n} for model, dim, n in self.models()
            ],
            "python_scan_warn_threshold": PYTHON_SCAN_WARN,
        }


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


__all__ = ["PYTHON_SCAN_WARN", "VectorHit", "VectorIndex", "time"]
