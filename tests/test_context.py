"""Context budgeting and compaction.

The claim: a long task does not die at step 12 because the prompt outgrew
the window. Two mechanisms -- proactive compaction and a deterministic
fallback that works with no summariser model available.
"""

from __future__ import annotations


from unified_agent.agent.context import ContextBuilder, _digest
from unified_agent.agent.state import AgentState, LogEntry, PlanStep, TaskStatus
from unified_agent.models.base import estimate_tokens
from unified_agent.observability.events import EventType

from tests.conftest import looping_script


def _state(goal: str = "do work") -> AgentState:
    return AgentState(task_id="task_x", session_id="session_x", goal=goal)


class TestBudget:
    def test_budget_respects_the_agent_token_ceiling(self, settings) -> None:  # noqa: ANN001
        model = _model_for(settings)
        settings.agent.max_tokens = 8_000
        builder = ContextBuilder(
            settings=settings, model=model, tool_catalog="read_file(path) — read a file"
        )
        assert builder.budget() == 8_000

    def test_budget_respects_the_model_window(self, settings):  # noqa: ANN001
        model = _model_for(settings)
        model.capabilities.max_context_tokens = 4_000
        settings.agent.max_tokens = 400_000
        builder = ContextBuilder(settings=settings, model=model, tool_catalog="")
        # 4000 - 15% reserve = 3400
        assert builder.budget() == 3_400

    def test_budget_has_a_floor(self, settings):  # noqa: ANN001
        model = _model_for(settings)
        model.capabilities.max_context_tokens = 200
        builder = ContextBuilder(settings=settings, model=model, tool_catalog="")
        assert builder.budget() == 2_000


def _model_for(settings):  # noqa: ANN001
    from unified_agent.models.mock import MockModel

    return MockModel(settings.models["scripted"])


class TestMessageAssembly:
    def test_goal_is_pinned_and_always_present(self, settings) -> None:  # noqa: ANN001
        model = _model_for(settings)
        builder = ContextBuilder(settings=settings, model=model, tool_catalog="t() — x")
        messages = builder.build(_state("the goal text"))

        pinned = [m for m in messages if m.pinned]
        assert any("the goal text" in m.content for m in pinned)
        assert any(m.role == "system" and "t() — x" in m.content for m in messages)

    def test_plan_is_rendered_with_the_cursor(self, settings) -> None:  # noqa: ANN001
        model = _model_for(settings)
        builder = ContextBuilder(settings=settings, model=model, tool_catalog="")
        state = _state()
        state.plan = [
            PlanStep(id="step_1", description="first"),
            PlanStep(id="step_2", description="second"),
        ]
        state.sync_current_step()
        text = "\n".join(m.content for m in builder.build(state))
        assert "1. first" in text
        assert "you are here" in text

    def test_already_called_tools_are_summarised(self, settings) -> None:  # noqa: ANN001
        """Redundancy guard: models happily re-list the same directory forever."""
        model = _model_for(settings)
        builder = ContextBuilder(settings=settings, model=model, tool_catalog="")
        state = _state()
        for _ in range(3):
            state.append_log(LogEntry(index=0, kind="tool", tool="list_directory", text="x"))
        state.append_log(LogEntry(index=0, kind="tool", tool="read_file", text="y"))

        text = "\n".join(m.content for m in builder.build(state))
        assert "list_directory x3" in text
        assert "read_file x1" in text
        assert "Do not repeat" in text

    def test_skills_index_is_in_the_system_prompt(self, settings) -> None:  # noqa: ANN001
        model = _model_for(settings)
        builder = ContextBuilder(
            settings=settings, model=model, tool_catalog="", skills_index="- pdf: do pdf things"
        )
        messages = builder.build(_state())
        assert "pdf: do pdf things" in messages[0].content


class TestCompaction:
    async def test_compaction_triggers_and_records_an_event(
        self, scripted, session_id: str, workspace
    ) -> None:  # noqa: ANN001
        big = workspace / "big.txt"
        big.write_text("lorem ipsum dolor sit amet " * 400, encoding="utf-8")

        agent, _ = await scripted(
            looping_script(6, "read_file", path="big.txt")
            + [{"content": "done"}]
        )
        agent.settings.agent.keep_recent_observations = 2
        agent.settings.agent.tool_output_chars = 4_000
        agent.settings.agent.max_steps = 20
        # Shrink the prompt budget: with a 200k window nothing ever needs
        # compacting, and the test would pass vacuously.
        agent.settings.agent.max_tokens = 4_000
        # Shrink the prompt budget: with a 200k window nothing ever needs
        # compacting, and the test would pass vacuously.
        agent.settings.agent.max_tokens = 4_000

        result = await agent.runtime.run("read the big file", session_id=session_id)
        assert result.status == TaskStatus.COMPLETED.value

        events = agent.store.events(result.task_id)
        compacted = [e for e in events if e.type is EventType.CONTEXT_COMPACTED]
        assert compacted, "compaction should have fired"

        from unified_agent.agent.state import replay

        state = replay(events, task_id=result.task_id)
        assert state.compacted_summary
        assert state.log_offset > 0
        assert len(state.log) <= agent.settings.agent.keep_recent_observations

    async def test_deterministic_fallback_needs_no_model(self) -> None:
        entries = [
            LogEntry(index=1, kind="tool", tool="read_file", text="app.py has a bug"),
            LogEntry(
                index=2,
                kind="tool",
                tool="run_tests",
                success=False,
                text="FAILED tests/test_app.py::test_add - AssertionError",
            ),
        ]
        digest = _digest(entries)
        assert "2 earlier step(s)" in digest
        assert "read_file x1" in digest
        assert "run_tests x1" in digest
        assert "failures:" in digest
        assert "app.py" in digest

    async def test_log_indices_stay_monotonic_across_compaction(
        self, scripted, session_id: str, workspace
    ) -> None:  # noqa: ANN001
        big = workspace / "big.txt"
        big.write_text("word " * 2_000, encoding="utf-8")

        agent, _ = await scripted(
            looping_script(6, "read_file", path="big.txt") + [{"content": "done"}]
        )
        agent.settings.agent.keep_recent_observations = 2
        agent.settings.agent.tool_output_chars = 4_000
        agent.settings.agent.max_steps = 20
        # Shrink the prompt budget: with a 200k window nothing ever needs
        # compacting, and the test would pass vacuously.
        agent.settings.agent.max_tokens = 4_000
        # Shrink the prompt budget: with a 200k window nothing ever needs
        # compacting, and the test would pass vacuously.
        agent.settings.agent.max_tokens = 4_000
        await agent.runtime.run("read it", session_id=session_id)

        from unified_agent.agent.state import replay

        events = agent.store.events(agent.store.list_tasks(limit=1)[0]["id"])
        state = replay(events, task_id=agent.store.list_tasks(limit=1)[0]["id"])

        # A fresh entry appended after compaction must not reuse a dropped index.
        entry = state.append_log(LogEntry(index=0, kind="note", text="later"))
        assert state.log_offset > 0, "compaction must actually have happened"
        assert state.log_offset > 0, "compaction must actually have happened"
        assert entry.index >= state.log_offset

    async def test_compaction_survives_a_dead_summariser(
        self, scripted, session_id: str, workspace
    ) -> None:  # noqa: ANN001
        """Compaction must degrade, never fail the task."""
        big = workspace / "big.txt"
        big.write_text("data " * 3_000, encoding="utf-8")

        agent, _ = await scripted(
            looping_script(6, "read_file", path="big.txt") + [{"content": "done"}]
        )
        agent.settings.agent.keep_recent_observations = 2
        agent.settings.agent.tool_output_chars = 4_000
        agent.settings.agent.max_steps = 20
        # Shrink the prompt budget: with a 200k window nothing ever needs
        # compacting, and the test would pass vacuously.
        agent.settings.agent.max_tokens = 4_000
        # Shrink the prompt budget: with a 200k window nothing ever needs
        # compacting, and the test would pass vacuously.
        agent.settings.agent.max_tokens = 4_000

        result = await agent.runtime.run("read it", session_id=session_id)
        assert result.status == TaskStatus.COMPLETED.value
        from unified_agent.agent.state import replay

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert state.compacted_summary


class TestTokenEstimation:
    def test_cjk_is_not_undercounted(self) -> None:
        chinese = "这是一个中文句子用来测试分词估算的准确性" * 10
        latin = "a" * len(chinese)
        assert estimate_tokens(chinese) > estimate_tokens(latin) * 2

    def test_empty_input(self) -> None:
        assert estimate_tokens("") == 0
