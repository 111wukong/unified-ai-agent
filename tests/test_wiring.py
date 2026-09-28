"""Wiring: a declared knob or event must reach behaviour.

This file covers one failure mode, and it is the one that is invisible from
both ends:

* a `Settings` field nobody reads still renders in `uaa config show`, so the
  user changes it, sees no error, and believes it took effect;
* an `EventType` nobody emits still imports and type-checks, so the audit
  trail looks complete while the one thing you need to see is missing.

Neither shows up as an exception, a wrong answer, or a red test. Both were
present in this codebase: three config fields and seven event types were
declared and dead. The rule this file enforces is that anything named here
has an assertion that its value changed an observable outcome -- adding a
knob or an event without one is how the previous batch got in.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import ScriptedModel, ScriptedModels

from unified_agent.agent.factory import build_agent
from unified_agent.agent.state import replay
from unified_agent.config import ModelSpec
from unified_agent.errors import ModelError
from unified_agent.models.mock import MockModel
from unified_agent.observability.events import EventType
from unified_agent.storage.store import Store
from unified_agent.tools.base import ToolContext
from unified_agent.tools import net as net_module
from unified_agent.tools.fs import FS_TOOLS
from unified_agent.tools.net import HttpGetTool
from unified_agent.tools.shell import RunCommandTool, build_shell_tools
from unified_agent.types import EffectClass, idempotency_key


def ctx_for(workspace: Path, home: Path) -> ToolContext:
    return ToolContext(
        task_id="t",
        session_id="s",
        step_id="step_1",
        workspace=workspace,
        home=home,
        artifact_dir=home,
    )


# ---------------------------------------------------------------------------
# configured limits
# ---------------------------------------------------------------------------


class TestShellOutputLimit:
    async def test_the_configured_ceiling_is_what_cuts_the_output(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        """`permissions.shell.max_output_bytes`, not the module constant.

        The constant and the default config value were both 200_000, so a
        limit that did nothing looked identical to one that worked.
        """
        tool = RunCommandTool(max_output_bytes=120)
        result = await tool.run({"command": "seq 1 4000"}, ctx_for(workspace, tmp_path))

        assert result.success
        assert "output cut at 120 chars" in result.output
        assert len(result.output) < 400, "the ceiling did not apply"

    async def test_a_generous_ceiling_does_not_cut(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        tool = RunCommandTool(max_output_bytes=1_000_000)
        result = await tool.run({"command": "seq 1 10"}, ctx_for(workspace, tmp_path))

        assert result.success
        assert "output cut" not in result.output

    def test_the_builder_threads_it_to_every_shell_tool(self) -> None:
        tools = build_shell_tools(max_output_bytes=4_242)
        assert [t.spec.name for t in tools] == ["run_command", "run_tests", "run_linter"]
        for tool in tools:
            runner = getattr(tool, "_runner", tool)
            assert runner.max_output_bytes == 4_242


class TestNetworkResponseLimit:
    async def test_the_configured_ceiling_is_what_cuts_the_body(
        self, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = b"z" * 5_000

        class FakeResponse:
            content = body
            status_code = 200
            encoding = "utf-8"
            headers: dict[str, str] = {"content-type": "text/plain"}
            is_redirect = False
            is_success = True

        class FakeClient:
            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *exc: Any) -> bool:
                return False

            async def request(self, method: str, url: str, json: Any = None) -> FakeResponse:
                return FakeResponse()

        monkeypatch.setattr(net_module.httpx, "AsyncClient", lambda **kw: FakeClient())

        tool = HttpGetTool(max_response_bytes=64)
        result = await tool.run({"url": "https://example.com/big"}, ctx_for(workspace, tmp_path))

        assert result.success
        assert result.truncated, "the response ceiling was not reported"
        assert result.metadata["bytes"] == 5_000
        assert "z" * 64 in result.output
        assert "z" * 65 not in result.output


class TestSettingsReachTheTools:
    async def test_every_limit_arrives_at_the_object_that_enforces_it(
        self, settings, scripted
    ) -> None:  # noqa: ANN001
        """The end-to-end form of the two tests above.

        Asserting the tool's own attribute is the only check that survives a
        future refactor of `build_agent`: it fails if the value stops being
        passed, not merely if the truncation logic breaks.
        """
        settings.permissions.shell.max_output_bytes = 3_333
        settings.permissions.network.max_response_bytes = 4_444

        agent, _ = await scripted([])

        assert agent.registry.get("run_command").max_output_bytes == 3_333
        assert agent.registry.get("http_get").max_response_bytes == 4_444
        assert agent.registry.get("http_post").max_response_bytes == 4_444


class TestVectorLimit:
    """`memory.vector_limit` is the depth the vector branch reads before fusion."""

    @staticmethod
    def _store(tmp_path: Path) -> Store:
        return Store(tmp_path / "mem.db")

    @staticmethod
    def _seed(store: Store) -> None:
        for i in range(40):
            store.add_memory(
                content=f"fact number {i} about deployments",
                session_id="s",
                scope="project",
                tags=[],
                importance=0.5,
                source="test",
            )

    def _depth(self, store: Store, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> int:
        captured: dict[str, Any] = {}
        original = store.search_vectors

        def spy(vector, **call_kwargs):  # noqa: ANN001, ANN202
            captured.update(call_kwargs)
            return original(vector, **call_kwargs)

        monkeypatch.setattr(store, "search_vectors", spy)
        store.search_memories(
            "deployments",
            mode="hybrid",
            query_vector=[1.0, 0.0],
            vector_model="hashing:2",
            **kwargs,
        )
        return captured["limit"]

    def test_a_larger_limit_reads_deeper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = self._store(tmp_path)
        self._seed(store)
        assert self._depth(store, monkeypatch, limit=5, vector_limit=20) == 20

    def test_it_is_floored_at_the_result_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ranked list shorter than the result set gives RRF nothing to
        compare, so a small `vector_limit` must not shrink below `limit`."""
        store = self._store(tmp_path)
        self._seed(store)
        assert self._depth(store, monkeypatch, limit=8, vector_limit=2) == 8

    def test_none_keeps_the_heuristic_for_callers_without_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = self._store(tmp_path)
        self._seed(store)
        assert self._depth(store, monkeypatch, limit=5) == 15

    def test_the_service_carries_the_configured_value(self, settings) -> None:  # noqa: ANN001
        from unified_agent.memory import build_memory_service

        store = Store(settings.db_path)
        service = build_memory_service(settings=settings, store=store, model=None)
        assert service.vector_limit == settings.memory.vector_limit
        store.close()


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


class _FailingModel(MockModel):
    """Fails the loop's first N calls, so the retry path is exercised."""

    def __init__(self, spec, failures: int, *, retryable: bool = True) -> None:  # noqa: ANN001
        super().__init__(spec)
        self.remaining = failures
        self.retryable = retryable

    async def _chat(
        self,
        messages,  # noqa: ANN001
        *,
        tools,  # noqa: ANN001
        temperature: float,
        response_format: Any,
        max_output_tokens: int,
        stream: Any = None,
    ) -> Any:
        if response_format:
            return self._structured(messages, response_format)
        if self.remaining > 0:
            self.remaining -= 1
            raise ModelError("provider hiccup", retryable=self.retryable)
        return self._final(messages)


class TestModelEvents:
    """A retry the user cannot see is a retry that looks like latency."""

    async def test_a_retried_call_is_recorded(
        self, settings, session_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The backoff is real seconds; the test does not need to wait them out.
        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("unified_agent.agent.runtime.asyncio.sleep", no_sleep)

        model = _FailingModel(settings.models["scripted"], failures=1, retryable=True)
        agent = await build_agent(settings=settings, models=ScriptedModels(model))
        try:
            result = await agent.runtime.run("do the thing", session_id=session_id)
            events = agent.store.events(result.task_id)
        finally:
            agent.close()

        kinds = [e.type for e in events]
        assert result.status == "completed", result.error
        assert EventType.MODEL_ERROR in kinds
        assert EventType.MODEL_RETRY in kinds

        # The planning call also emits MODEL_RESPONSE, so filter to the loop.
        loop_calls = [
            e
            for e in events
            if e.type is EventType.MODEL_RESPONSE and e.payload.get("phase") == "step"
        ]
        assert len(loop_calls) == 1, "the retry produced a second call"

    async def test_a_permanent_failure_is_recorded(
        self, settings, session_id: str
    ) -> None:
        model = _FailingModel(settings.models["scripted"], failures=1, retryable=False)
        agent = await build_agent(settings=settings, models=ScriptedModels(model))
        try:
            result = await agent.runtime.run("do the thing", session_id=session_id)
            errors = [
                e
                for e in agent.store.events(result.task_id)
                if e.type is EventType.MODEL_ERROR
            ]
        finally:
            agent.close()

        assert result.status == "failed"
        assert errors and errors[0].payload["retryable"] is False


class TestMemorySearchedEvent:
    async def test_recall_is_visible_even_when_it_finds_nothing(
        self, scripted, session_id: str
    ) -> None:
        """`recall ran, found nothing` and `recall never ran` are different
        diagnoses, and only the event separates them."""
        agent, _ = await scripted([{"content": "done"}])
        result = await agent.runtime.run("a goal nothing matches", session_id=session_id)
        searched = [
            e for e in agent.store.events(result.task_id) if e.type is EventType.MEMORY_SEARCHED
        ]

        assert searched, "the memory lookup left no trace"
        assert searched[0].payload["hits"] == 0
        # `mode` is recorded because it is the difference between "the FTS
        # index found nothing" and "the vector branch was never consulted".
        # Offline this is `hybrid`: the hashing fallback is available, it is
        # just not semantic.
        assert searched[0].payload["mode"] in {"fts", "hybrid"}


class TestToolReplayedEvent:
    async def test_a_rerun_after_an_interrupt_is_its_own_event(
        self, scripted, session_id: str
    ) -> None:
        """`TOOL_STARTED` cannot say whether a call is a first run or a
        replay; before this event the only trace was a flag inside a log
        entry, which is not queryable."""
        agent, _ = await scripted([{"content": "done"}])
        task_id = agent.store.create_task(
            session_id=session_id, goal="read it", budgets={"model": "scripted"}
        )
        agent.store.begin_tool_call(
            task_id=task_id,
            step_id="step_1",
            attempt=1,
            name="read_file",
            arguments={"path": "app.py"},
            effect_class="read_only",
            idempotency_key=idempotency_key(task_id, "step_1", "read_file", {"path": "app.py"}),
        )
        # No finish_tool_call: this is what a crash leaves behind.

        await agent.runtime.resume(task_id)

        replayed = [e for e in agent.store.events(task_id) if e.type is EventType.TOOL_REPLAYED]
        assert replayed, "the re-run was not distinguishable from a first run"
        assert replayed[0].payload["name"] == "read_file"
        assert replayed[0].payload["reason"] == "resumed"


class TestSkillLoadedEvent:
    @staticmethod
    def _skills_root(tmp_path: Path) -> Path:
        """A directory *containing* skill directories, which is what
        `discover()` globs -- passing the skill's own directory finds nothing."""
        root = tmp_path / "skills"
        directory = root / "reviewer"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(
            "---\nname: reviewer\ndescription: Review code and report findings.\n---\n\n"
            "# Steps\n\n1. Read the diff.\n",
            encoding="utf-8",
        )
        return root

    async def test_loading_a_skill_is_recorded_and_attributed(
        self, settings, session_id: str, tmp_path: Path
    ) -> None:
        """The only feedback loop a skill has.

        Without both halves -- the event and the `skill_runs` row -- the
        `deprecate` decision has no evidence behind it and the `skill_runs`
        table stays empty while the schema promises it answers "does this
        skill ever help?".
        """
        model = ScriptedModel(
            settings.models["scripted"],
            [
                {"tool_calls": [{"name": "load_skill", "arguments": {"name": "reviewer"}}]},
                {"content": "done"},
            ],
        )
        agent = await build_agent(
            settings=settings,
            models=ScriptedModels(model),
            skill_dirs=[self._skills_root(tmp_path)],
        )
        try:
            result = await agent.runtime.run("review the code", session_id=session_id)
            loaded = [
                e
                for e in agent.store.events(result.task_id)
                if e.type is EventType.SKILL_LOADED
            ]
            runs = [dict(r) for r in agent.store.conn.execute("SELECT * FROM skill_runs")]
        finally:
            agent.close()

        assert result.status == "completed", result.error
        assert loaded and loaded[0].payload["name"] == "reviewer"

        assert len(runs) == 1, "the skill run was never attributed"
        assert runs[0]["skill_name"] == "reviewer"
        assert runs[0]["outcome"] == "completed"

    async def test_a_resumed_task_still_knows_which_skills_it_used(
        self, settings, session_id: str, tmp_path: Path
    ) -> None:
        """`loaded_skills` is folded from events, not kept in memory.

        A resume rebuilds state from the event stream alone, so a list that
        only lived in RAM would lose exactly the skills loaded before the
        crash -- and the attribution would be wrong for the tasks that need
        it most.
        """
        model = ScriptedModel(
            settings.models["scripted"],
            [
                {"tool_calls": [{"name": "load_skill", "arguments": {"name": "reviewer"}}]},
                {"content": "done"},
            ],
        )
        agent = await build_agent(
            settings=settings,
            models=ScriptedModels(model),
            skill_dirs=[self._skills_root(tmp_path)],
        )
        try:
            result = await agent.runtime.run("review the code", session_id=session_id)
            state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        finally:
            agent.close()

        assert state.loaded_skills == ["reviewer"]


# ---------------------------------------------------------------------------
# capability matrix
# ---------------------------------------------------------------------------


class TestCapabilityMatrix:
    """Every field in `ModelCapabilities` is read by something.

    `vision` used to sit in that list with no reader. A capability flag
    nothing consults can only mislead: the runtime cannot put an image in a
    message, so a model marked vision-capable still cannot be asked to look
    at one.
    """

    def test_vision_was_removed_rather_than_left_dangling(self, settings) -> None:  # noqa: ANN001
        from unified_agent.models.base import ModelCapabilities

        assert "vision" not in ModelCapabilities.model_fields
        caps = ModelCapabilities()
        with pytest.raises(ValueError, match="unknown capability overrides"):
            caps.merged({"vision": True})

    def test_the_output_ceiling_clamps_the_request(self) -> None:
        """`ModelSpec.max_output_tokens` is the budget; the capability is
        what the model can physically emit.

        Asking for more than the ceiling is a config mistake whose failure
        mode is a provider 400 in the middle of a task, so the request is
        clamped instead.
        """
        from unified_agent.models.base import ChatModel, ModelCapabilities
        from unified_agent.types import ModelResponse

        class Recording(ChatModel):
            provider = "recording"
            seen: int | None = None

            def declared_capabilities(self) -> ModelCapabilities:
                return ModelCapabilities(max_output_tokens=1_000)

            async def _chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
                Recording.seen = kwargs["max_output_tokens"]
                return ModelResponse(content="ok")

        spec = ModelSpec(provider="mock", model="m", max_output_tokens=9_999)
        model = Recording(spec)
        asyncio.run(model.chat([]))
        assert Recording.seen == 1_000

        # Under the ceiling, the spec wins.
        Recording.seen = None
        asyncio.run(model.chat([], max_output_tokens=250))
        assert Recording.seen == 250

    def test_parallel_tool_calls_reaches_the_wire(self) -> None:
        """The flag is what a user override actually changes."""
        from unified_agent.models.openai_compat import OpenAICompatModel

        captured: dict[str, Any] = {}

        class FakeResponse:
            status_code = 200
            text = "{}"
            headers: dict[str, str] = {}

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": "hi"}}]}

        class FakeClient:
            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *exc: Any) -> bool:
                return False

            async def post(self, url: str, *, headers: Any, json: Any) -> FakeResponse:
                captured.update(json)
                return FakeResponse()

        import unified_agent.models.openai_compat as module

        original = module.httpx.AsyncClient
        module.httpx.AsyncClient = lambda **kw: FakeClient()  # type: ignore[assignment]
        try:
            tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
            # Real OpenAI (no base_url) declares the capability.
            openai_like = OpenAICompatModel(ModelSpec(provider="openai_compat", model="gpt"))
            asyncio.run(openai_like.chat([], tools=tools))
            assert captured.get("parallel_tool_calls") is True

            captured.clear()
            # A gateway does not, and must not receive an unknown field.
            gateway = OpenAICompatModel(
                ModelSpec(provider="openai_compat", model="qwen", base_url="http://x/v1")
            )
            asyncio.run(gateway.chat([], tools=tools))
            assert "parallel_tool_calls" not in captured
        finally:
            module.httpx.AsyncClient = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# read-only store
# ---------------------------------------------------------------------------


class TestReadOnlyStore:
    """Read commands open the database read-only.

    A read path that *can* write is a read path whose bug corrupts the store,
    and every CLI listing opens the database.
    """

    def test_a_read_only_store_refuses_to_write(self, tmp_path: Path) -> None:
        import sqlite3

        path = tmp_path / "uaa.db"
        writable = Store(path)
        writable.close()

        readonly = Store.readonly(path)
        try:
            assert readonly.list_tasks() == []
            with pytest.raises(sqlite3.OperationalError):
                readonly.conn.execute("INSERT INTO sessions(id,name,working_dir,"
                                      "model_alias,created_at,metadata) VALUES('x','x','x','x','x','{}')")
        finally:
            readonly.close()

    def test_a_missing_database_is_created_instead_of_failing(self, tmp_path: Path) -> None:
        """"Nothing recorded yet" must not be an error.

        A read-only connection cannot create the file, so `readonly` falls
        back to a normal open when it is absent.
        """
        path = tmp_path / "fresh" / "uaa.db"
        store = Store.readonly(path)
        try:
            assert store.list_tasks() == []
        finally:
            store.close()
        assert path.exists()


# ---------------------------------------------------------------------------
# tool inventory
# ---------------------------------------------------------------------------


class TestToolInventoryFilter:
    def test_describe_can_filter_to_one_effect_class(self) -> None:
        """Reviewing the permission surface means asking "what can this
        runtime execute", not scanning the whole catalogue."""
        from unified_agent.tools.registry import ToolRegistry
        from unified_agent.tools.shell import RunCommandTool

        registry = ToolRegistry()
        registry.register(RunCommandTool())
        for tool in FS_TOOLS:
            registry.register(tool)

        executable = registry.describe(effect=EffectClass.EXECUTE_LOCAL)
        assert [item["name"] for item in executable] == ["run_command"]
        assert all(item["effect"] == "execute_local" for item in executable)

        # No filter is still the whole catalogue.
        assert len(registry.describe()) > len(executable)


# ---------------------------------------------------------------------------
# memory: batch writes and stale vectors
# ---------------------------------------------------------------------------


class TestVectorMaintenance:
    def test_reindex_writes_the_batch_and_reports_stale_vectors(self, settings) -> None:  # noqa: ANN001
        """Vectors built under a previous embedding model are invisible to
        search, so changing `memory.embedding_model` silently narrows
        retrieval to whatever was indexed since."""
        from unified_agent.memory import build_memory_service

        store = Store(settings.db_path)
        try:
            for i in range(5):
                store.add_memory(
                    content=f"fact number {i} about deployments",
                    session_id="s",
                    scope="project",
                    tags=[],
                    importance=0.5,
                    source="test",
                )
            service = build_memory_service(settings=settings, store=store, model=None)

            result = asyncio.run(service.reindex())
            assert result["embedded"] == 5
            assert store.vectors.count() == 5
            assert service.stats()["stale_vectors"] == 0

            # Pretend the embedding model changed: same rows, different key.
            store.conn.execute("UPDATE memory_vectors SET model='old:model:512'")
            assert service.stats()["stale_vectors"] == 5
        finally:
            store.close()


# ---------------------------------------------------------------------------
# projection consistency
# ---------------------------------------------------------------------------


class TestProjectionHealth:
    async def test_doctor_notices_a_projection_that_stopped_being_updated(
        self, settings, scripted  # noqa: ANN001
    ) -> None:
        """`tasks.state` is written every step and read by nothing, while the
        project claims the projection is a cache that is wrong by
        construction if it disagrees. A claim nothing checks is not a claim.
        """
        from unified_agent.cli import _store_health

        agent, _ = await scripted([{"content": "done"}])
        session = agent.store.ensure_session(
            name="t", working_dir=str(settings.workspace), model_alias="scripted"
        )
        result = await agent.runtime.run("say done", session_id=session)
        task_id = result.task_id

        assert dict(_store_health(settings))["projection"].startswith("[green]")

        # Corrupt the projection the way a crash between the event append and
        # the projection write would.
        agent.store.conn.execute(
            "UPDATE tasks SET status='running' WHERE id=?", (task_id,)
        )
        detail = dict(_store_health(settings))["projection"]
        assert detail.startswith("[red]")
        assert "disagree" in detail
