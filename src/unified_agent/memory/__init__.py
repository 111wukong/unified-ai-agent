"""Memory.

Phase 1-3 scope is FTS5 only: SQLite ships with FTS5 and it is enough for
a few thousand entries. Vector search is deliberately deferred -- the
original spec's own advice ("不要一开始同时引入 PostgreSQL、Redis、Milvus、
Elasticsearch") applies with equal force to bolting on an embedding store
before there is anything worth embedding.
"""

from unified_agent.storage.store import Store


class MemoryStore:
    """Thin facade so the runtime does not depend on the whole Store."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def remember(
        self,
        content: str,
        *,
        session_id: str | None = None,
        scope: str = "project",
        tags: list[str] | None = None,
        importance: float = 0.5,
        source: str = "agent",
    ) -> str:
        return self.store.add_memory(
            content=content,
            session_id=session_id,
            scope=scope,
            tags=tags or [],
            importance=importance,
            source=source,
        )

    def recall(self, query: str, *, scope: str | None = None, limit: int = 8) -> list[dict]:
        return self.store.search_memories(query, scope=scope, limit=limit)

    def all(self, *, scope: str | None = None, limit: int = 50) -> list[dict]:
        return self.store.list_memories(scope=scope, limit=limit)


__all__ = ["MemoryStore"]
