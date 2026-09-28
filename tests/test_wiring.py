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

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import ScriptedModel, ScriptedModels

from unified_agent.agent.factory import build_agent
from unified_agent.agent.state import replay
from unified_agent.errors import ModelError
from unified_agent.models.mock import MockModel
from unified_agent.observability.events import EventType
from unified_agent.storage.store import Store
from unified_agent.tools.base import ToolContext
from unified_agent.tools import net as net_module
from unified_agent.tools.net import HttpGetTool
from unified_agent.tools.shell import RunCommandTool, build_shell_tools
from unified_agent.types import idempotency_key


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
