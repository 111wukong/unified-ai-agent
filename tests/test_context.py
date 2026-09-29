"""Context budgeting and compaction.

The claim: a long task does not die at step 12 because the prompt outgrew
the window. Two mechanisms -- proactive compaction and a deterministic
fallback that works with no summariser model available.
"""

from __future__ import annotations

from pathlib import Path

from unified_agent.agent.context import ContextBuilder, _digest
from unified_agent.agent.state import AgentState, LogEntry, PlanStep, TaskStatus
from unified_agent.config import AgentConfig, ModelSpec, Settings
from unified_agent.models.base import estimate_tokens
from unified_agent.models.mock import MockModel
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
        """Redundancy guard: models happily re-list the same directory forever.

        The block names the *call*, not the tool. It used to say
        `list_directory x3`, which a real run could not act on -- it could not
        tell which directory it was looping on, so it listed the same four
        directories again for another 40 steps.
        """
        model = _model_for(settings)
        builder = ContextBuilder(settings=settings, model=model, tool_catalog="")
        state = _state()
        for _ in range(3):
            entry = LogEntry(index=0, kind="tool", tool="list_directory", text="x")
            entry.arguments = {"path": "."}
            state.append_log(entry)
            state.record_call("list_directory", entry.arguments)
        read = LogEntry(index=0, kind="tool", tool="read_file", text="y")
        read.arguments = {"path": "app.py"}
        state.append_log(read)
        state.record_call("read_file", read.arguments)

        text = "\n".join(m.content for m in builder.build(state))
        assert 'list_directory({"path": "."}) x3' in text
        assert 'read_file({"path": "app.py"})' in text
        assert "REPEATED" in text
        assert "Do not make any of these calls again" in text
        # Only the call that actually repeated is flagged. A call made once is
        # context, not a loop.
        repeated_line = next(
            line for line in text.splitlines() if line.startswith("REPEATED")
        )
        assert "list_directory" in repeated_line
        assert "read_file" not in repeated_line

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

        result = await agent.runtime.run("read the big file", session_id=session_id)
        assert result.status == TaskStatus.COMPLETED.value

        events = agent.store.events(result.task_id)
        compacted = [e for e in events if e.type is EventType.CONTEXT_COMPACTED]
        assert compacted, "compaction should have fired"

        from unified_agent.agent.state import replay

        state = replay(events, task_id=result.task_id)
        assert state.compacted_summary
        assert state.log_offset > 0
        # `keep_recent_observations` is how much is *retained when compaction
        # fires*, not a hard ceiling: entries appended after the last fire stay
        # until the next one. The exact invariant is therefore
        # `keep + everything appended since the last compaction`, and asserting
        # a bare `<= keep` only passed because the old test happened to end on
        # a compaction.
        last_compaction = max(e.seq for e in compacted)
        after = sum(
            1
            for e in events
            if e.type is EventType.LOG_APPENDED and e.seq > last_compaction
        )
        assert len(state.log) == agent.settings.agent.keep_recent_observations + after
        # And the log really is a small fraction of everything recorded.
        total = sum(1 for e in events if e.type is EventType.LOG_APPENDED)
        assert len(state.log) < total

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


class TestOldEntriesKeepTheirHead:
    """Two tiers, and the free one carries the history.

    Squeezing an old tool result to a smaller body still spends the tokens,
    and when the budget runs out the oldest entries are dropped whole -- so
    the model loses the record of what it already did and re-runs calls whose
    results it can no longer see. A digest costs a few dozen characters, so
    the entire action history fits.
    """

    def test_a_digest_names_the_call_the_status_and_the_size(self) -> None:
        entry = LogEntry(
            index=7,
            kind="tool",
            tool="read_file",
            arguments={"path": "app.py"},
            success=True,
            text="x" * 4_000,
        )
        line = entry.digest()
        assert "read_file" in line
        assert "app.py" in line
        assert "-> ok" in line
        assert "4000 chars elided" in line
        assert len(line) < 200, "a digest that is not small defeats the purpose"

    def test_a_digest_keeps_the_ambiguous_marker(self) -> None:
        """The one flag the model must not lose: this call's outcome is
        unknown, so re-running it may double-execute."""
        entry = LogEntry(
            index=3, kind="tool", tool="run_command", arguments={"command": "git commit"}, success=False
        )
        entry.ambiguous = True
        assert "OUTCOME UNKNOWN" in entry.digest()

    def test_old_entries_are_digested_and_recent_ones_are_not(self) -> None:
        builder = _builder(keep=2)
        state = _state()
        for i in range(6):
            state.append_log(
                LogEntry(
                    index=0,
                    kind="tool",
                    tool=f"tool_{i}",
                    arguments={"path": f"f{i}.py"},
                    success=True,
                    text="BODY" * 500,
                )
            )

        rendered = builder._render_log(state, chars=40_000)
        # The two newest keep their bodies; the older four are digests.
        assert rendered.count("BODY") > 0, "the recent entries lost their bodies"
        assert "tool_0" in rendered, "an old entry was dropped entirely"
        assert "chars elided" in rendered
        assert "earlier entries omitted" not in rendered

    def test_the_whole_action_history_survives_a_small_budget(self) -> None:
        """The behaviour this buys: with bodies squeezed instead of digested,
        a small budget drops the oldest entries and the model forgets what it
        did. With digests, every call is still named."""
        builder = _builder(keep=1)
        state = _state()
        for i in range(40):
            state.append_log(
                LogEntry(
                    index=0,
                    kind="tool",
                    tool=f"tool_{i}",
                    arguments={"path": f"f{i}.py"},
                    success=True,
                    text="y" * 3_000,
                )
            )

        rendered = builder._render_log(state, chars=6_000)
        for i in range(40):
            assert f"tool_{i}" in rendered, f"the record of tool_{i} was lost"

    def test_a_non_tool_entry_falls_back_to_a_short_render(self) -> None:
        """A note has no head line to keep, so it degrades to a truncated
        render rather than a digest -- still bounded, just differently."""
        entry = LogEntry(index=1, kind="note", text="z" * 5_000)
        assert len(entry.digest()) < 500


def _builder(*, keep: int) -> ContextBuilder:
    spec = ModelSpec(provider="mock", model="mock-react")
    settings = Settings(home=Path("/tmp/uaa-ctx-home"), workspace=Path("/tmp/uaa-ctx-ws"))
    settings.agent = AgentConfig(keep_recent_observations=keep)
    return ContextBuilder(
        settings=settings, model=MockModel(spec), tool_catalog="", summarizer=None
    )
