"""Turning task output into durable facts, and reconciling contradictions.

The research conclusion this implements: **the hard part of agent memory is
not retrieval, it is contradiction.** A vector store that only ever appends
will happily return "the project uses pytest" and "the project migrated to
unittest" side by side, and the model then has to guess which is current.

So the write path is a two-stage pipeline, borrowed from Mem0 for the
extraction and from Zep for the reconciliation:

1. **Extract** atomic facts from a finished task (one LLM call, already part
   of the reflection step).
2. **Reconcile** each fact against its nearest neighbours, deciding
   `add` / `update` / `duplicate` / `none`.

`update` does not overwrite. The superseded row stays with `superseded_by`
pointing forward and a `memory_revisions` row pointing back, so the store can
still answer "what did this used to be" -- the reason Zep invalidates rather
than replaces.

Cost note, because it decides whether this is usable: reconciliation only
calls the model when there *are* neighbours. A genuinely new fact costs
nothing extra, and that is the common case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from unified_agent.memory.embeddings import EmbeddingProvider
from unified_agent.types import Message

CURATOR_PROMPT = """\
You maintain the long-term memory of a coding agent.

A new fact has been proposed. Below it are the existing memories most similar
to it, each with an id.

Choose exactly one action:

- `add` — the fact is new and does not conflict with anything listed.
- `update` — the fact supersedes one of the listed memories. Put that
  memory's id in `target_id`. Prefer `update` over `add` whenever the new
  fact corrects, narrows or extends an existing one: the old memory is kept
  as history, not deleted, so there is no cost to being decisive.
- `duplicate` — a listed memory already says this. No write happens.
- `none` — not worth remembering at all.

Rules:

- A changed preference is an `update`, not an `add`. "uses pytest" followed
  by "migrated to unittest" must not leave both in the store.
- Never invent a `target_id`. Use only ids from the list, or null.
- Prefer `none` for anything transient (this task's output, a file you read)
  and for anything the repository already states plainly.
- Return JSON only.
"""

CURATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["add", "update", "duplicate", "none"],
        },
        "target_id": {"type": ["string", "null"]},
        "reason": {"type": "string", "maxLength": 400},
        "content": {
            "type": "string",
            "description": "For `update`, the full replacement text. Omit to keep the candidate.",
        },
        "tags": {"type": "array", "items": {"type": "string"}},
        "importance": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["action"],
    "additionalProperties": False,
}


class Verdict(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DUPLICATE = "duplicate"
    NONE = "none"


@dataclass
class Candidate:
    content: str
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5


@dataclass
class Decision:
    candidate: Candidate
    verdict: Verdict
    target_id: str | None = None
    reason: str = ""
    #: Ids of the memories the model was shown, for auditability.
    considered: list[str] = field(default_factory=list)

    @property
    def wrote(self) -> bool:
        return self.verdict in {Verdict.ADD, Verdict.UPDATE}


@dataclass
class IngestResult:
    added: list[str] = field(default_factory=list)
    updated: list[tuple[str, str]] = field(default_factory=list)  # (old_id, new_id)
    duplicated: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    model_calls: int = 0

    @property
    def summary(self) -> str:
        return (
            f"{len(self.added)} added, {len(self.updated)} updated, "
            f"{len(self.duplicated)} duplicate, {len(self.rejected)} rejected "
            f"({self.model_calls} model call(s))"
        )


class MemoryCurator:
    """Reconciles candidates against what is already stored.

    `model` is optional. Without it the curator still runs, using a
    deterministic rule instead of the judge: an exact-normalised duplicate is
    dropped, everything else is added. That keeps the pipeline testable and
    usable offline -- a memory feature that silently stops working without an
    API key is worse than one that degrades predictably.
    """

    def __init__(
        self,
        *,
        store: Any,
        embeddings: EmbeddingProvider | None = None,
        model: Any = None,
        neighbour_limit: int = 5,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.model = model
        self.neighbour_limit = neighbour_limit

    # -- neighbours --------------------------------------------------------
    async def neighbours(self, candidate: Candidate, *, scope: str = "project") -> list[dict]:
        """Most similar stored memories, by vector if possible, else FTS."""
        found: list[dict] = []
        if self.embeddings is not None and self.embeddings.available:
            vector = await self.embeddings.embed_query(candidate.content)
            if vector:
                hits = self.store.search_vectors(
                    vector,
                    model=self.embeddings_key(),
                    limit=self.neighbour_limit,
                    scope=scope,
                )
                found = [dict(hit) for hit in hits]
        if not found:
            # Term-wise, not literal. A candidate sentence is never a
            # substring of a stored one, so a phrase search finds nothing and
            # the contradiction check would silently have no neighbours.
            found = self.store.search_memories_any(
                candidate_terms(candidate.content), scope=scope, limit=self.neighbour_limit
            )
        return found

    # -- judging -----------------------------------------------------------
    async def judge(self, candidate: Candidate, neighbours: list[dict]) -> Decision:
        considered = [n["id"] for n in neighbours]
        if not neighbours:
            # Nothing similar exists, so there is nothing to contradict and no
            # reason to spend a call asking.
            return Decision(candidate, Verdict.ADD, reason="no similar memories", considered=[])

        if self.model is None:
            return self._judge_offline(candidate, neighbours)

        listing = "\n".join(
            f"- id={n['id']} | {n['content'][:400]}" for n in neighbours
        )
        prompt = (
            f"# Proposed fact\n\n{candidate.content}\n\n"
            f"# Existing memories\n\n{listing}\n"
        )
        try:
            response = await self.model.chat(
                [
                    Message(role="system", content=CURATOR_PROMPT),
                    Message(role="user", content=prompt),
                ],
                temperature=0.0,
                response_format=_response_format(self.model),
                max_output_tokens=600,
            )
        except Exception:  # noqa: BLE001 - a judge failure must not lose the fact
            return Decision(
                candidate,
                Verdict.ADD,
                reason="judge unavailable; added without reconciliation",
                considered=considered,
            )

        payload = _extract_json(response.content or "")
        action = str(payload.get("action") or "").strip().lower()
        if action not in {v.value for v in Verdict}:
            return Decision(
                candidate,
                Verdict.ADD,
                reason=f"judge returned an unusable action {action!r}; added",
                considered=considered,
            )

        verdict = Verdict(action)
        target = payload.get("target_id")
        # A model that names a target that was never shown is hallucinating;
        # treat it as an add rather than pointing at a random row.
        if verdict is Verdict.UPDATE and target not in considered:
            return Decision(
                candidate,
                Verdict.ADD,
                reason=f"judge named an unknown target {target!r}; added instead",
                considered=considered,
            )

        if replacement := (payload.get("content") or "").strip():
            candidate.content = replacement
        if tags := payload.get("tags"):
            candidate.tags = [str(t) for t in tags]
        if isinstance(payload.get("importance"), (int, float)):
            candidate.importance = max(0.0, min(1.0, float(payload["importance"])))

        # Keep the target for `duplicate` as well as `update`: the caller
        # reports which memory was already known, and dropping it turns a
        # useful answer into an empty string.
        keep_target = verdict in {Verdict.UPDATE, Verdict.DUPLICATE}
        return Decision(
            candidate,
            verdict,
            target_id=target if keep_target else None,
            reason=str(payload.get("reason") or "")[:400],
            considered=considered,
        )

    def _judge_offline(self, candidate: Candidate, neighbours: list[dict]) -> Decision:
        """No model available: exact duplicates are dropped, the rest is added.

        Deliberately conservative. Guessing at contradictions without a
        judge would corrupt the store, and a corrupted memory store is worse
        than a slightly redundant one.
        """
        considered = [n["id"] for n in neighbours]
        key = _fingerprint(candidate.content)
        for neighbour in neighbours:
            if _fingerprint(neighbour["content"]) == key:
                return Decision(
                    candidate,
                    Verdict.DUPLICATE,
                    target_id=neighbour["id"],
                    reason="identical text already stored",
                    considered=considered,
                )
        return Decision(
            candidate,
            Verdict.ADD,
            reason="no judge available; added without contradiction check",
            considered=considered,
        )

    # -- orchestration -----------------------------------------------------
    async def ingest(
        self,
        candidates: list[Candidate],
        *,
        scope: str = "project",
        session_id: str | None = None,
        source: str = "curator",
    ) -> IngestResult:
        result = IngestResult()
        for candidate in candidates:
            content = candidate.content.strip()
            if len(content) < 8:
                continue
            candidate.content = content

            neighbours = await self.neighbours(candidate, scope=scope)
            if neighbours and self.model is not None:
                result.model_calls += 1
            decision = await self.judge(candidate, neighbours)
            result.decisions.append(decision)

            if decision.verdict is Verdict.DUPLICATE:
                result.duplicated.append(decision.target_id or "")
                continue
            if decision.verdict is Verdict.NONE:
                result.rejected.append(candidate.content[:80])
                continue

            new_id = self.store.add_memory(
                content=candidate.content,
                session_id=session_id,
                scope=scope,
                tags=candidate.tags,
                importance=candidate.importance,
                source=source,
            )
            if decision.verdict is Verdict.UPDATE and decision.target_id:
                self.store.invalidate_memory(
                    decision.target_id, replaced_by=new_id, reason=decision.reason, source=source
                )
                result.updated.append((decision.target_id, new_id))
            else:
                result.added.append(new_id)

            if self.embeddings is not None and self.embeddings.available:
                await self._embed_and_store(new_id, candidate.content)
        return result

    async def _embed_and_store(self, memory_id: str, content: str) -> None:
        try:
            vector = await self.embeddings.embed_query(content)
        except Exception:  # noqa: BLE001 - vectors are an optimisation
            return
        if vector:
            self.store.put_vector(memory_id, vector, model=self.embeddings_key())

    def embeddings_key(self) -> str:
        from unified_agent.memory.embeddings import vector_key_for

        assert self.embeddings is not None
        return vector_key_for(self.embeddings)


def _response_format(model: Any) -> dict[str, Any] | None:
    caps = getattr(model, "capabilities", None)
    if caps is None:
        return None
    if getattr(caps, "json_schema", False):
        return {
            "type": "json_schema",
            "json_schema": {"name": "memory_decision", "schema": CURATOR_SCHEMA, "strict": True},
        }
    if getattr(caps, "json_object", False):
        return {"type": "json_object"}
    return None


def _extract_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "into", "are", "was",
        "has", "have", "not", "but", "its", "it's", "you", "your", "our", "when",
        "use", "used", "uses", "using", "project", "code",
    }
)


def candidate_terms(text: str, *, limit: int = 12) -> list[str]:
    """Significant terms for a similarity probe.

    Longest first, because a long token is more discriminating than a short
    one and the caller only uses the first handful. CJK is emitted as bigrams
    since it has no word boundaries.
    """
    import re

    lowered = text.lower()
    terms: list[str] = []
    for match in re.finditer(r"[a-z0-9_]{3,}", lowered):
        token = match.group(0)
        if token not in _STOPWORDS:
            terms.append(token)
    for match in re.finditer(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}", lowered):
        run = match.group(0)
        terms.extend(run[i : i + 2] for i in range(len(run) - 1))
    seen: set[str] = set()
    unique = [t for t in terms if not (t in seen or seen.add(t))]
    unique.sort(key=len, reverse=True)
    return unique[:limit]


def _fingerprint(text: str) -> str:
    """Whitespace- and case-insensitive identity for duplicate detection."""
    import re

    return re.sub(r"\s+", " ", text.strip().lower())


__all__ = [
    "CURATOR_PROMPT",
    "CURATOR_SCHEMA",
    "Candidate",
    "candidate_terms",
    "Decision",
    "IngestResult",
    "MemoryCurator",
    "Verdict",
]
