"""Vector memory, hybrid retrieval, and contradiction handling.

The tests that matter most here are the ones guarding against *silent* memory
corruption: vectors that change between processes, a superseded fact that
still surfaces, a judge that names a row it was never shown.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from unified_agent.memory import (
    Candidate,
    MemoryCurator,
    MemoryService,
    Verdict,
    build_embeddings,
    build_memory_service,
)
from unified_agent.memory.embeddings import (
    HashingEmbeddings,
    NullEmbeddings,
    OpenAICompatEmbeddings,
    cosine_similarity,
    pack_vector,
    unpack_vector,
)
from unified_agent.storage.store import Store


@pytest.fixture
def store(tmp_path: Path):  # noqa: ANN001
    s = Store(tmp_path / "mem.db")
    yield s
    s.close()


class TestHashingEmbeddings:
    """The offline fallback. Lexical, deterministic, and honest about it."""

    async def test_vectors_are_l2_normalised(self) -> None:
        provider = HashingEmbeddings(dim=128)
        vector = (await provider.embed(["the project uses pytest"]))[0]
        norm = math.sqrt(sum(v * v for v in vector))
        assert norm == pytest.approx(1.0, abs=1e-6)

    async def test_similar_text_scores_higher_than_unrelated(self) -> None:
        provider = HashingEmbeddings(dim=512)
        vectors = await provider.embed(
            [
                "the project uses pytest for testing",
                "the project uses pytest to run tests",
                "completely unrelated sentence about weather",
            ]
        )
        near = cosine_similarity(vectors[0], vectors[1])
        far = cosine_similarity(vectors[0], vectors[2])
        assert near > far

    async def test_chinese_is_tokenised_into_bigrams(self) -> None:
        """CJK has no spaces, so character bigrams are the words."""
        provider = HashingEmbeddings(dim=512)
        vectors = await provider.embed(
            ["认证模块的测试覆盖率不足", "认证模块需要补充用例", "今天天气不错"]
        )
        near = cosine_similarity(vectors[0], vectors[1])
        far = cosine_similarity(vectors[0], vectors[2])
        assert near > far

    async def test_empty_string_does_not_crash(self) -> None:
        provider = HashingEmbeddings(dim=32)
        vector = (await provider.embed([""]))[0]
        assert len(vector) == 32
        assert all(v == 0.0 for v in vector)

    def test_is_not_advertised_as_semantic(self) -> None:
        """Calling this semantic would be the dishonest part."""
        provider = HashingEmbeddings()
        assert provider.semantic is False
        assert "not meaning" in provider.describe()

    def test_vectors_are_stable_across_processes(self) -> None:
        """The trap this guards: `hash()` on a str is salted per process.

        Using the builtin would make a reindex in a new process produce
        entirely different vectors, so every stored vector becomes noise.
        The symptom is "search got worse", with nothing pointing at the cause.
        """
        script = (
            "import asyncio, sys;"
            "sys.path.insert(0, %r);"
            "from unified_agent.memory.embeddings import HashingEmbeddings;"
            "p = HashingEmbeddings(dim=64);"
            "v = asyncio.run(p.embed(['deterministic across processes']))[0];"
            "print(sum(v))"
        ) % str(Path(__file__).resolve().parents[1])

        outputs = []
        for seed in ("0", "1", "12345"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            outputs.append(result.stdout.decode().strip())
        assert len(set(outputs)) == 1, f"vectors differ per process: {outputs}"


class TestProviderSelection:
    def test_no_model_means_the_lexical_fallback(self) -> None:
        provider = build_embeddings(model=None, base_url=None, api_key=None)
        assert isinstance(provider, HashingEmbeddings)
        assert provider.available

    def test_a_hosted_model_without_a_key_degrades(self) -> None:
        """A missing key must not make memory stop working."""
        provider = build_embeddings(
            model="text-embedding-3-small", base_url=None, api_key=None
        )
        assert isinstance(provider, HashingEmbeddings)

    def test_a_local_endpoint_needs_no_key(self) -> None:
        provider = build_embeddings(
            model="nomic-embed-text", base_url="http://127.0.0.1:11434/v1", api_key=None
        )
        assert isinstance(provider, OpenAICompatEmbeddings)
        assert provider.semantic is True

    def test_null_provider_reports_unavailable(self) -> None:
        assert NullEmbeddings().available is False


class TestVectorIndex:
    def test_round_trip(self, store: Store) -> None:
        store.put_vector("m1", [1.0, 0.0, 0.0], model="test")
        hits = store.vectors.search([1.0, 0.0, 0.0], model="test", limit=5)
        assert [h.memory_id for h in hits] == ["m1"]
        assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)

    def test_orders_by_similarity(self, store: Store) -> None:
        store.put_vector("near", [0.99, 0.1, 0.0], model="t")
        store.put_vector("far", [0.0, 1.0, 0.0], model="t")
        hits = store.vectors.search([1.0, 0.0, 0.0], model="t", limit=5)
        assert [h.memory_id for h in hits] == ["near"]

    def test_the_two_paths_agree(self, store: Store) -> None:
        """sqlite-vec is an accelerator; if it changed the answer it would be
        a different database rather than a faster one."""
        for i, vector in enumerate(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.7]]
        ):
            store.put_vector(f"m{i}", vector, model="t")

        accelerated = store.vectors
        accelerated.accelerator = "sqlite-vec"
        fast = accelerated.search([1.0, 0.0, 0.0], model="t", limit=3)

        accelerated.accelerator = "python"
        slow = accelerated.search([1.0, 0.0, 0.0], model="t", limit=3)

        assert [h.memory_id for h in fast] == [h.memory_id for h in slow]
        for a, b in zip(fast, slow, strict=True):
            assert a.similarity == pytest.approx(b.similarity, abs=1e-5)

    def test_models_are_not_mixed(self, store: Store) -> None:
        """Vectors from two models are not comparable, so they must not meet."""
        store.put_vector("a", [1.0, 0.0], model="model-a")
        store.put_vector("b", [1.0, 0.0], model="model-b")
        assert [h.memory_id for h in store.vectors.search([1.0, 0.0], model="model-a", limit=5)] == [
            "a"
        ]

    def test_dimension_mismatch_is_rejected(self, store: Store) -> None:
        """An explicit dim that disagrees with the vector is a caller bug."""
        with pytest.raises(ValueError, match="does not match"):
            store.put_vector("m", [1.0, 0.0], model="t", dim=3)

    def test_vectors_are_normalised_on_write(self, store: Store) -> None:
        """Otherwise the accelerated and fallback paths disagree by exactly
        how far the input was off unit length."""
        store.put_vector("m", [3.0, 4.0], model="t")
        hits = store.vectors.search([1.0, 0.0], model="t", limit=1)
        assert hits[0].similarity == pytest.approx(0.6, abs=1e-5)

    def test_min_similarity_drops_orthogonal_hits(self, store: Store) -> None:
        """Cosine 0 means "nothing in common"; returning it is noise."""
        store.put_vector("same", [1.0, 0.0], model="t")
        store.put_vector("orthogonal", [0.0, 1.0], model="t")
        hits = store.vectors.search([1.0, 0.0], model="t", limit=5, min_similarity=0.0)
        assert [h.memory_id for h in hits] == ["same"]

    def test_delete_removes_the_vector(self, store: Store) -> None:
        store.put_vector("m", [1.0, 0.0], model="t")
        store.vectors.delete("m")
        assert store.vectors.count() == 0

    def test_missing_finds_unembedded_memories(self, store: Store) -> None:
        first = store.add_memory(content="one")
        store.add_memory(content="two")
        store.put_vector(first, [1.0, 0.0], model="t")
        missing = store.vectors.missing(model="t")
        assert len(missing) == 1 and missing[0] != first

    def test_pack_unpack_round_trip(self) -> None:
        vector = [0.1, -0.5, 1.0, 0.0]
        assert unpack_vector(pack_vector(vector)) == pytest.approx(vector, abs=1e-6)

    def test_describe_reports_the_accelerator(self, store: Store) -> None:
        described = store.vectors.describe()
        assert described["accelerator"] in {"sqlite-vec", "python"}
        assert "vectors" in described


class TestHybridSearch:
    def _seed(self, store: Store) -> None:
        store.add_memory(content="项目用 .venv/bin/pytest 跑测试", tags=["testing"])
        store.add_memory(content="部署需要先通过 CI 检查", tags=["ci"])
        store.add_memory(content="数据库迁移用 alembic 管理", tags=["db"])

    async def test_fts_mode_is_unchanged(self, store: Store) -> None:
        self._seed(store)
        hits = store.search_memories("部署", mode="fts")
        assert len(hits) == 1 and "CI" in hits[0]["content"]

    async def test_two_character_chinese_still_works(self, store: Store) -> None:
        """Trigram needs 3 chars; the LIKE fallback is load-bearing."""
        self._seed(store)
        assert store.search_memories("测试", mode="fts")

    async def test_hybrid_merges_without_duplicating(self, store: Store) -> None:
        self._seed(store)
        provider = HashingEmbeddings(dim=256)
        vector = await provider.embed_query("pytest 测试")
        for memory in store.list_memories():
            embedded = await provider.embed_query(memory["content"])
            store.put_vector(memory["id"], embedded, model="hashing:256")

        hits = store.search_memories(
            "测试", mode="hybrid", query_vector=vector, vector_model="hashing:256", limit=5
        )
        ids = [h["id"] for h in hits]
        assert len(ids) == len(set(ids)), "fusion must not emit the same row twice"
        assert any("pytest" in h["content"] for h in hits)

    async def test_hybrid_falls_back_when_there_is_no_vector(self, store: Store) -> None:
        self._seed(store)
        hits = store.search_memories("部署", mode="hybrid", query_vector=None)
        assert len(hits) == 1

    async def test_auto_mode_picks_fts_without_a_vector(self, store: Store) -> None:
        self._seed(store)
        assert store.search_memories("部署", mode="auto") != []

    async def test_superseded_rows_never_surface(self, store: Store) -> None:
        old = store.add_memory(content="项目用 pytest 跑测试")
        new = store.add_memory(content="项目改用 unittest 跑测试")
        store.invalidate_memory(old, replaced_by=new, reason="migrated")
        store.put_vector(old, [1.0, 0.0], model="t")
        store.put_vector(new, [1.0, 0.0], model="t")

        vector_hits = store.search_vectors([1.0, 0.0], model="t", limit=5)
        assert [h["id"] for h in vector_hits] == [new]

        fts_hits = store.search_memories("pytest", mode="fts")
        assert all(h["id"] != old for h in fts_hits)


class TestInvalidation:
    def test_invalidate_keeps_the_row(self, store: Store) -> None:
        old = store.add_memory(content="uses pytest")
        new = store.add_memory(content="uses unittest")
        assert store.invalidate_memory(old, replaced_by=new, reason="migrated")

        kept = store.get_memory(old)
        assert kept is not None, "the old fact must survive for the timeline"
        assert kept["superseded_by"] == new

    def test_invalidating_twice_is_a_no_op(self, store: Store) -> None:
        old = store.add_memory(content="a")
        new = store.add_memory(content="b")
        other = store.add_memory(content="c")
        assert store.invalidate_memory(old, replaced_by=new)
        assert store.invalidate_memory(old, replaced_by=other) is False
        assert store.get_memory(old)["superseded_by"] == new

    def test_history_walks_the_whole_chain(self, store: Store) -> None:
        first = store.add_memory(content="uses pytest")
        second = store.add_memory(content="uses nose")
        third = store.add_memory(content="uses unittest")
        store.invalidate_memory(first, replaced_by=second, reason="step 1")
        store.invalidate_memory(second, replaced_by=third, reason="step 2")

        # From the middle, both directions are visible.
        chain = store.memory_history(second)
        assert [entry["id"] for entry in chain] == [first, second, third]
        assert chain[0]["content"] == "uses pytest"
        assert chain[2]["content"] == "uses unittest"
        # Each entry carries the reason *it* was replaced, not the reason its
        # successor was.
        assert chain[0]["replaced_reason"] == "step 1"
        assert chain[1]["replaced_reason"] == "step 2"
        assert chain[2]["replaced_reason"] == ""

    def test_history_from_the_latest_also_reaches_back(self, store: Store) -> None:
        first = store.add_memory(content="v1")
        second = store.add_memory(content="v2")
        store.invalidate_memory(first, replaced_by=second)
        chain = store.memory_history(second)
        assert [entry["id"] for entry in chain] == [first, second]

    def test_stats_count_superseded(self, store: Store) -> None:
        first = store.add_memory(content="v1")
        second = store.add_memory(content="v2")
        store.invalidate_memory(first, replaced_by=second)
        stats = store.memory_stats()
        assert stats["total"] == 2
        assert stats["active"] == 1
        assert stats["superseded"] == 1


class _Judge:
    """A scripted model that returns a fixed curator decision."""

    def __init__(self, payload: dict) -> None:
        import json

        self.payload = payload
        self.calls = 0

        class _Caps:
            json_schema = False
            json_object = False

        self.capabilities = _Caps()
        self._json = json

    async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN201
        from unified_agent.types import ModelResponse, TokenUsage

        self.calls += 1
        return ModelResponse(
            content=self._json.dumps(self.payload),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )


class TestCurator:
    async def test_no_neighbours_means_no_model_call(self, store: Store) -> None:
        """The common case -- a genuinely new fact -- must be free."""
        judge = _Judge({"action": "add"})
        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=judge)
        result = await curator.ingest([Candidate(content="the sky is blue today")])
        assert result.added
        assert judge.calls == 0, "nothing to contradict, so nothing to ask"
        assert result.model_calls == 0

    async def test_exact_duplicate_is_dropped_without_a_model(self, store: Store) -> None:
        store.add_memory(content="The project uses pytest")
        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=None)
        result = await curator.ingest([Candidate(content="the project uses pytest")])
        assert result.duplicated
        assert not result.added
        assert len(store.list_memories()) == 1

    async def test_without_a_judge_a_paraphrase_is_added_not_merged(
        self, store: Store
    ) -> None:
        """Conservative on purpose: guessing at contradictions without a judge
        would corrupt the store, and that is worse than being redundant."""
        store.add_memory(content="uses pytest for testing")
        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=None)
        result = await curator.ingest([Candidate(content="testing is done with pytest")])
        assert result.added

    async def test_update_invalidates_the_old_fact(self, store: Store) -> None:
        old = store.add_memory(content="项目用 pytest 跑测试")
        judge = _Judge(
            {"action": "update", "target_id": old, "reason": "migrated to unittest"}
        )
        curator = MemoryCurator(
            store=store, embeddings=NullEmbeddings(), model=judge, allow_supersede=True
        )
        result = await curator.ingest([Candidate(content="项目改用 unittest 跑测试")])

        assert len(result.updated) == 1
        superseded_id, new_id = result.updated[0]
        assert superseded_id == old
        assert store.get_memory(old)["superseded_by"] == new_id
        assert store.memory_stats()["superseded"] == 1

    async def test_a_hallucinated_target_falls_back_to_add(self, store: Store) -> None:
        """A judge that names a row it was never shown must not point at it."""
        # The candidate must share terms with something, or there is no
        # neighbour to hallucinate a target from and the guard is not reached.
        store.add_memory(content="the project uses pytest for its tests")
        judge = _Judge({"action": "update", "target_id": "mem_does_not_exist"})
        curator = MemoryCurator(
            store=store, embeddings=NullEmbeddings(), model=judge, allow_supersede=True
        )
        result = await curator.ingest(
            [Candidate(content="the project uses pytest with coverage enabled")]
        )

        assert result.added
        assert not result.updated
        assert result.decisions[0].verdict is Verdict.ADD
        assert "unknown target" in result.decisions[0].reason

    async def test_an_unusable_action_falls_back_to_add(self, store: Store) -> None:
        store.add_memory(content="the deployment goes through the CI pipeline")
        judge = _Judge({"action": "explode"})
        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=judge)
        result = await curator.ingest(
            [Candidate(content="the deployment pipeline now runs on every commit")]
        )
        assert result.added
        assert "unusable action" in result.decisions[0].reason

    async def test_a_broken_judge_does_not_lose_the_fact(self, store: Store) -> None:
        store.add_memory(content="the test command is pytest -q")

        class Broken:
            capabilities = _Judge({"action": "add"}).capabilities

            async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN201
                raise RuntimeError("provider exploded")

        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=Broken())
        result = await curator.ingest(
            [Candidate(content="the test command is pytest -q --no-header")]
        )
        assert result.added, "an unreachable judge must not drop the memory"
        assert "judge unavailable" in result.decisions[0].reason

    async def test_duplicate_verdict_writes_nothing(self, store: Store) -> None:
        existing = store.add_memory(content="the deploy target is the staging cluster")
        judge = _Judge({"action": "duplicate", "target_id": existing})
        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=judge)
        result = await curator.ingest(
            [Candidate(content="the staging cluster is the deploy target")]
        )
        assert result.duplicated == [existing]
        assert len(store.list_memories()) == 1

    async def test_none_verdict_rejects(self, store: Store) -> None:
        store.add_memory(content="the migration tool is alembic")
        judge = _Judge({"action": "none", "reason": "transient"})
        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=judge)
        result = await curator.ingest(
            [Candidate(content="this task read the alembic migration file")]
        )
        assert result.rejected
        assert not result.added

    async def test_short_candidates_are_skipped(self, store: Store) -> None:
        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=None)
        result = await curator.ingest([Candidate(content="ok")])
        assert not result.added and not result.duplicated

    async def test_update_can_rewrite_the_text(self, store: Store) -> None:
        old = store.add_memory(content="the command is pytest")
        judge = _Judge(
            {"action": "update", "target_id": old, "content": "the command is python -m pytest"}
        )
        curator = MemoryCurator(
            store=store, embeddings=NullEmbeddings(), model=judge, allow_supersede=True
        )
        result = await curator.ingest([Candidate(content="the command is pytest -q")])
        new_id = result.updated[0][1]
        assert "python -m pytest" in store.get_memory(new_id)["content"]


class TestMemoryService:
    async def test_remember_writes_and_embeds(self, store: Store) -> None:
        service = MemoryService(store, embeddings=HashingEmbeddings(dim=64))
        result = await service.remember_text("the project uses pytest")
        assert result.added
        assert store.vectors.count(model=service.vector_key) == 1

    async def test_recall_returns_the_matching_memory(self, store: Store) -> None:
        service = MemoryService(store, embeddings=HashingEmbeddings(dim=256))
        await service.remember_text("the test command is pytest -q")
        await service.remember_text("deployment goes through CI")
        hits = await service.recall("pytest", limit=5)
        assert hits and "pytest" in hits[0]["content"]

    async def test_recall_survives_a_broken_provider(self, store: Store) -> None:
        class Exploding(HashingEmbeddings):
            async def embed(self, texts):  # noqa: ANN001, ANN201
                raise RuntimeError("no endpoint")

        store.add_memory(content="the test command is pytest -q")
        service = MemoryService(store, embeddings=Exploding(dim=64))
        hits = await service.recall("pytest", limit=5)
        assert hits, "a broken embedder must degrade to FTS, not return nothing"

    async def test_stats_admit_the_fallback_is_not_semantic(self, store: Store) -> None:
        service = MemoryService(store, embeddings=HashingEmbeddings(dim=64))
        stats = service.stats()
        assert stats["semantic"] is False
        assert "not meaning" in stats["embeddings"]

    async def test_reindex_reports_honestly_with_no_provider(self, store: Store) -> None:
        store.add_memory(content="a memory with no vector")
        service = MemoryService(store, embeddings=NullEmbeddings())
        result = await service.reindex()
        assert result["embedded"] == 0
        assert "no embedding provider" in result["note"]

    async def test_reindex_is_idempotent(self, store: Store) -> None:
        # Written directly, bypassing the service, so there is genuinely
        # something left to index.
        store.add_memory(content="a memory written before embeddings existed")
        service = MemoryService(store, embeddings=HashingEmbeddings(dim=64))
        first = await service.reindex()
        second = await service.reindex()
        assert first["embedded"] == 1
        assert second["pending"] == 0
        assert second["embedded"] == 0

    async def test_build_memory_service_never_raises(self, settings) -> None:  # noqa: ANN001
        store = Store(settings.db_path)
        try:
            service = build_memory_service(settings=settings, store=store, model=None)
            assert service.embeddings.available
            assert service.recall is not None
        finally:
            store.close()

    def test_build_memory_service_falls_back_when_the_key_is_missing(
        self, settings, monkeypatch
    ) -> None:  # noqa: ANN001
        """Configured but unusable must degrade, not fail every task."""
        monkeypatch.delenv("UAA_TEST_EMBED_KEY", raising=False)
        settings.memory.embedding_model = "text-embedding-3-small"
        settings.memory.embedding_api_key_env = "UAA_TEST_EMBED_KEY"
        settings.memory.embedding_base_url = None
        store = Store(settings.db_path)
        try:
            service = build_memory_service(settings=settings, store=store)
            assert service.embeddings.semantic is False
        finally:
            store.close()


class TestSupersedeIsOffByDefault:
    """A model judgement must not be able to hide a stored fact.

    Superseding takes a memory out of search results. The judgement behind it
    -- "this new fact replaces that old one" -- has no reliable prior: two
    facts are often complementary rather than contradictory, and the failure
    is silent and effectively permanent. A contradiction someone can see is a
    smaller problem than a fact that quietly disappeared.

    mem0 measured +26 points on LongMemEval by removing write-time
    reconciliation altogether, and this project's six-tier permission model
    already refuses to let a model approve its own tools. Same principle.
    """

    async def test_the_default_adds_instead_of_superseding(self, store: Store) -> None:
        old = store.add_memory(content="项目用 pytest 跑测试")
        judge = _Judge({"action": "update", "target_id": old, "reason": "migrated"})
        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=judge)

        result = await curator.ingest([Candidate(content="项目改用 unittest 跑测试")])

        assert result.added, "the fact must still be kept"
        assert not result.updated
        assert result.decisions[0].verdict is Verdict.ADD
        assert "allow_supersede" in result.decisions[0].reason
        # Nothing was hidden: both facts are live.
        assert store.get_memory(old)["superseded_by"] is None
        assert store.memory_stats()["superseded"] == 0
        assert store.memory_stats()["active"] == 2

    async def test_the_prompt_and_schema_do_not_offer_the_action(self) -> None:
        """Narrowing the enum rather than rejecting the answer afterwards: a
        model that is offered `update` will use it."""
        from unified_agent.memory.extract import (
            CURATOR_PROMPT,
            CURATOR_PROMPT_SUPERSEDE,
            curator_schema,
        )

        assert "update" not in curator_schema(allow_supersede=False)["properties"]["action"]["enum"]
        assert "update" in curator_schema(allow_supersede=True)["properties"]["action"]["enum"]
        assert "may not supersede" in CURATOR_PROMPT
        assert "may not supersede" not in CURATOR_PROMPT_SUPERSEDE

    async def test_a_provider_that_ignores_the_schema_is_still_held_to_it(
        self, store: Store
    ) -> None:
        """Defence in depth. A gateway that ignores `response_format` can
        still return `update`, and the restriction has to hold anyway -- the
        point is that a fact cannot be hidden by a model judgement."""
        old = store.add_memory(content="the deploy target is staging")
        # A judge that always answers `update`, whatever the schema says.
        judge = _Judge({"action": "update", "target_id": old})
        curator = MemoryCurator(store=store, embeddings=NullEmbeddings(), model=judge)

        result = await curator.ingest([Candidate(content="the deploy target is production")])

        assert result.added
        assert store.get_memory(old)["superseded_by"] is None

    async def test_turning_it_on_restores_the_old_behaviour(self, store: Store) -> None:
        """The mechanism is still there for a store that must stay small; it
        is simply not what happens unless asked."""
        old = store.add_memory(content="the command is pytest")
        judge = _Judge({"action": "update", "target_id": old, "reason": "moved"})
        curator = MemoryCurator(
            store=store, embeddings=NullEmbeddings(), model=judge, allow_supersede=True
        )
        result = await curator.ingest([Candidate(content="the command is python -m pytest")])

        assert len(result.updated) == 1
        assert store.get_memory(old)["superseded_by"] is not None


class TestSaveMemoryToolIntegration:
    async def test_the_tool_reports_which_verdict_it_got(self, store: Store) -> None:
        """'saved' and 'replaced an older fact' are different outcomes."""
        from unified_agent.tools.base import ToolContext
        from unified_agent.tools.memory_tools import SaveMemoryTool

        old = store.add_memory(content="项目用 pytest 跑测试")
        judge = _Judge({"action": "update", "target_id": old, "reason": "migrated"})
        service = MemoryService(
            store,
            embeddings=NullEmbeddings(),
            curator=MemoryCurator(
                store=store, embeddings=NullEmbeddings(), model=judge, allow_supersede=True
            ),
        )
        ctx = ToolContext(
            task_id="t",
            session_id="s",
            step_id="step_1",
            workspace=Path("/tmp"),
            home=Path("/tmp"),
            artifact_dir=Path("/tmp"),
        )
        result = await SaveMemoryTool(store, service).run(
            {"content": "项目改用 unittest 跑测试"}, ctx
        )
        assert result.success
        assert result.metadata["verdict"] == "update"
        assert "replaced" in result.output


class TestStoreReadPathsSmoke:
    """Call every public read method once.

    A missing space in a concatenated SQL literal -- `"IS NULL"` +
    `"ORDER BY"` -> `"IS NULLORDER BY"` -- is a syntax error that only
    surfaces when that exact branch runs. Enumerating the read paths catches
    the whole class at once instead of waiting for a feature to reach it.
    """

    def test_every_read_path_executes(self, store: Store) -> None:
        store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        session = store.list_sessions()[0]
        task = store.create_task(session_id=session["id"], goal="smoke")
        memory = store.add_memory(content="a fact about deployments", scope="project")
        store.add_memory(content="a fact about testing", scope="user")

        store.get_session(session["id"])
        store.ensure_session(name="s", working_dir="/tmp", model_alias="mock")
        store.get_task(task)
        store.list_tasks(session_id=session["id"])
        store.list_tasks()
        store.events(task)
        store.last_seq(task)
        store.list_tool_calls(task)
        store.unfinished_calls(task)
        store.list_artifacts(task)
        store.list_memories()
        store.list_memories(scope="project")
        store.list_memories(limit=1)
        store.search_memories("deploy")
        store.search_memories("deploy", scope="project")
        store.search_memories("部署", mode="fts")
        store.search_memories_any(["deploy", "testing"])
        store.get_memory(memory)
        store.memory_stats()
        store.memory_history(memory)
        store.vectors.describe()
        store.vectors.count()
        store.vectors.models()
        store.vectors.missing(model="hashing")
        store.vectors.stale_model(model="hashing", dim=256)
        store.list_skills()
        store.skill_statuses()
        store.list_skill_runs()
        store.skill_run_counts()
        store.list_tasks(statuses=["completed"])

    def test_an_empty_store_answers_without_erroring(self, tmp_path: Path) -> None:
        empty = Store(tmp_path / "empty.db")
        try:
            assert empty.list_memories() == []
            assert empty.search_memories("anything") == []
            assert empty.memory_history("mem_nope") == []
            assert empty.memory_stats()["active"] == 0
        finally:
            empty.close()
