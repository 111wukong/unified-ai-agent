"""Memory.

Three layers, and it is worth being explicit about which does what:

* **FTS5** (`storage/store.py`) — exact terms, and the only thing that works
  for two-character Chinese queries. Always available.
* **Vectors** (`memory/vector.py`) — cosine search, accelerated by sqlite-vec
  when it loads. Semantic only if a real embedding model is configured; the
  offline fallback is a lexical hash.
* **The curator** (`memory/extract.py`) — extraction and, more importantly,
  reconciliation: a new fact that contradicts an old one *invalidates* it
  rather than sitting beside it.

Why all three rather than picking one: the research on Mem0 / Letta / Zep
converges on the same conclusion -- the hard part of agent memory is not
retrieval, it is contradiction. A store that only appends will happily return
"uses pytest" and "migrated to unittest" together, and the model then has to
guess which is current.

Phase 1 shipped FTS5 alone and deferred vectors on the grounds that there was
nothing worth embedding yet. That was the right call then; this is the point
where it stops being true, because the contradiction check needs a candidate
set to look at and full-text search cannot supply one for a paraphrased fact.
"""

from __future__ import annotations

from typing import Any, Sequence

from unified_agent.memory.embeddings import (
    EmbeddingProvider,
    HashingEmbeddings,
    NullEmbeddings,
    OpenAICompatEmbeddings,
    build_embeddings,
)
from unified_agent.memory.extract import (
    Candidate,
    Decision,
    IngestResult,
    MemoryCurator,
    Verdict,
)
from unified_agent.memory.vector import VectorHit, VectorIndex
from unified_agent.storage.store import Store


class MemoryService:
    """The one thing the rest of the runtime talks to."""

    def __init__(
        self,
        store: Store,
        *,
        embeddings: EmbeddingProvider | None = None,
        curator: MemoryCurator | None = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings or NullEmbeddings()
        self.curator = curator
        # A floor, not a relevance gate: cosine 0 means "nothing in common",
        # and returning those as results is worse than returning fewer.
        self.min_similarity = 0.0
        # How deep the vector branch reads before rank fusion. Set from
        # `memory.vector_limit`; `search_memories` floors it at `limit`.
        self.vector_limit = 10

    # -- reads ------------------------------------------------------------
    @property
    def vector_key(self) -> str:
        from unified_agent.memory.embeddings import vector_key_for

        return vector_key_for(self.embeddings)

    async def recall(
        self,
        query: str,
        *,
        scope: str | None = None,
        limit: int = 8,
        mode: str = "auto",
    ) -> list[dict[str, Any]]:
        """Hybrid search. `auto` embeds the query only when a provider can."""
        query_vector: list[float] | None = None
        vector_model: str | None = None
        if mode in {"auto", "hybrid", "vector"} and self.embeddings.available:
            try:
                query_vector = await self.embeddings.embed_query(query)
                vector_model = self.vector_key
            except Exception:  # noqa: BLE001 - degrade to FTS rather than fail the task
                query_vector = None
        return self.store.search_memories(
            query,
            scope=scope,
            limit=limit,
            mode=mode,
            query_vector=query_vector,
            vector_model=vector_model,
            min_similarity=self.min_similarity,
            vector_limit=self.vector_limit,
        )

    def stats(self) -> dict[str, Any]:
        data = self.store.memory_stats()
        data["embeddings"] = self.embeddings.describe()
        data["semantic"] = self.embeddings.semantic
        data["vector_key"] = self.vector_key if self.embeddings.available else None
        return data

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        return self.store.memory_history(memory_id)

    # -- writes -----------------------------------------------------------
    async def remember(
        self,
        candidates: Sequence[Candidate],
        *,
        scope: str = "project",
        session_id: str | None = None,
        source: str = "agent",
    ) -> IngestResult:
        if self.curator is None:
            # No reconciliation, but still embed: vector search must work
            # whether or not a judge is configured.
            result = IngestResult()
            for candidate in candidates:
                memory_id = self.store.add_memory(
                    content=candidate.content,
                    session_id=session_id,
                    scope=scope,
                    tags=candidate.tags,
                    importance=candidate.importance,
                    source=source,
                )
                result.added.append(memory_id)
                if self.embeddings.available:
                    await self._embed(memory_id, candidate.content)
            return result
        return await self.curator.ingest(
            list(candidates), scope=scope, session_id=session_id, source=source
        )

    async def remember_text(
        self,
        content: str,
        *,
        scope: str = "project",
        session_id: str | None = None,
        source: str = "user",
        tags: Sequence[str] = (),
    ) -> IngestResult:
        return await self.remember(
            [Candidate(content=content, tags=list(tags))],
            scope=scope,
            session_id=session_id,
            source=source,
        )

    async def _embed(self, memory_id: str, content: str) -> None:
        try:
            vector = await self.embeddings.embed_query(content)
        except Exception:  # noqa: BLE001 - vectors are an optimisation
            return
        if vector:
            self.store.put_vector(memory_id, vector, model=self.vector_key)

    # -- maintenance ------------------------------------------------------
    async def reindex(self, *, scope: str | None = None, batch_size: int = 32) -> dict[str, Any]:
        """Embed every memory that has no vector for the current key.

        Reports what it did rather than a bare count: "0 embedded" with
        "provider: none" is a different situation from "0 embedded because
        everything was already indexed", and the user has to be able to tell
        them apart.
        """
        if not self.embeddings.available:
            return {
                "embedded": 0,
                "pending": 0,
                "provider": self.embeddings.describe(),
                "note": "no embedding provider configured; FTS5 search still works",
            }
        model = self.vector_key
        pending = self.store.vectors.missing(model=model)
        if scope:
            allowed = {row["id"] for row in self.store.list_memories(scope=scope, limit=10_000)}
            pending = [memory_id for memory_id in pending if memory_id in allowed]

        embedded = 0
        for start in range(0, len(pending), batch_size):
            batch_ids = pending[start : start + batch_size]
            contents: list[str] = []
            for memory_id in batch_ids:
                memory = self.store.get_memory(memory_id)
                if memory:
                    contents.append(memory["content"])
            if not contents:
                continue
            vectors = await self.embeddings.embed(contents)
            for memory_id, vector in zip(batch_ids, vectors, strict=False):
                if vector:
                    self.store.put_vector(memory_id, vector, model=model)
                    embedded += 1
        return {
            "embedded": embedded,
            "pending": len(pending),
            "provider": self.embeddings.describe(),
            "vector_key": model,
        }


def build_memory_service(
    *,
    settings: Any,
    store: Store,
    model: Any = None,
) -> MemoryService:
    """Wire memory from configuration. Never raises on a missing key."""
    import os

    config = settings.memory
    api_key = os.environ.get(config.embedding_api_key_env, "") if config.embedding_model else ""
    if config.embedding_model and not api_key and not config.embedding_base_url:
        # Configured but unusable: fall back rather than fail every task.
        config = config.model_copy(update={"embedding_model": ""})

    embeddings = build_embeddings(
        model=config.embedding_model or None,
        base_url=config.embedding_base_url,
        api_key=api_key or None,
        fallback_dim=config.hashing_dim,
    )
    curator = MemoryCurator(
        store=store,
        embeddings=embeddings,
        model=model if config.reconcile else None,
        neighbour_limit=config.neighbour_limit,
    )
    service = MemoryService(store, embeddings=embeddings, curator=curator)
    service.min_similarity = config.min_similarity
    service.vector_limit = config.vector_limit
    return service


__all__ = [
    "Candidate",
    "Decision",
    "EmbeddingProvider",
    "HashingEmbeddings",
    "IngestResult",
    "MemoryCurator",
    "MemoryService",
    "NullEmbeddings",
    "OpenAICompatEmbeddings",
    "Store",
    "VectorHit",
    "VectorIndex",
    "Verdict",
    "build_embeddings",
    "build_memory_service",
]
