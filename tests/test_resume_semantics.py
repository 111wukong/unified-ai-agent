"""Resume semantics.

The claim being tested: a task can be interrupted at any point, including
in the middle of a tool call, and resuming does the right thing. "The right
thing" differs by tool, and getting it wrong is silent data corruption:

* an idempotent read can just be re-run;
* a non-idempotent shell command cannot -- re-running `git commit` is a
  double commit, and skipping it may lose the only copy of the work.

These tests simulate the crash by inserting the write-ahead ledger row that
a real crash would have left behind, then calling resume().
"""

from __future__ import annotations

import pytest

from unified_agent.agent.state import TaskStatus, replay
from unified_agent.observability.events import EventType
from tests.conftest import ScriptedModels


from unified_agent.types import idempotency_key


async def _start_task(agent, session_id: str, goal: str = "do the thing"):  # noqa: ANN001
    """Create a task row without running the loop."""
    return agent.store.create_task(session_id=session_id, goal=goal, budgets={"model": "scripted"})


class TestUnfinishedCalls:
    async def test_idempotent_call_is_rerun_on_resume(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted([{"content": "done"}])
        task_id = await _start_task(agent, session_id)

        key = idempotency_key(task_id, "step_1", "read_file", {"path": "app.py"})
        call_id = agent.store.begin_tool_call(
            task_id=task_id,
            step_id="step_1",
            attempt=1,
            name="read_file",
            arguments={"path": "app.py"},
            effect_class="read_only",
            idempotency_key=key,
        )
        # No finish_tool_call -> this is what a crash leaves behind.

        result = await agent.runtime.resume(task_id)

        assert result.status == TaskStatus.COMPLETED.value
        rows = agent.store.list_tool_calls(task_id)
        original = [r for r in rows if r.id == call_id][0]
        assert original.status == "retry"
        rerun = [r for r in rows if r.id != call_id]
        assert rerun and rerun[0].status == "succeeded"
        assert rerun[0].attempt == 2

        state = replay(agent.store.events(task_id), task_id=task_id)
        assert any("re-run" in e.text for e in state.log if e.kind == "system")
        assert any(e.tool == "read_file" and e.success for e in state.log)

    async def test_non_idempotent_call_is_marked_ambiguous(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted([{"content": "done"}])
        task_id = await _start_task(agent, session_id)

        key = idempotency_key(task_id, "step_1", "run_command", {"command": "git commit -m x"})
        call_id = agent.store.begin_tool_call(
            task_id=task_id,
            step_id="step_1",
            attempt=1,
            name="run_command",
            arguments={"command": "git commit -m x"},
            effect_class="execute_local",
            idempotency_key=key,
        )

        result = await agent.runtime.resume(task_id)

        rows = agent.store.list_tool_calls(task_id)
        record = [r for r in rows if r.id == call_id][0]
        assert record.status == "ambiguous", "a side-effecting call must not be silently re-run"
        assert len(rows) == 1, "no second execution may be attempted"

        events = agent.store.events(task_id)
        assert any(e.type is EventType.TOOL_AMBIGUOUS for e in events)

        state = replay(events, task_id=task_id)
        warnings = [e for e in state.log if e.kind == "system" and "UNKNOWN" in e.text]
        assert warnings, "the model must be told the outcome is unknown"
        assert "may have run twice" in warnings[0].text
        assert result.status == TaskStatus.COMPLETED.value

    async def test_completed_calls_are_not_reexecuted(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "done"},
            ]
        )
        first = await agent.runtime.run("read app.py", session_id=session_id)
        rows_before = agent.store.list_tool_calls(first.task_id)

        # Resuming a finished task is a no-op.
        second = await agent.runtime.resume(first.task_id)
        rows_after = agent.store.list_tool_calls(first.task_id)

        assert second.status == TaskStatus.COMPLETED.value
        assert len(rows_before) == len(rows_after) == 1


class TestReplayDeterminism:
    async def test_replay_is_deterministic(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "list_directory", "arguments": {"path": "."}}]},
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "all done"},
            ]
        )
        result = await agent.runtime.run("inspect", session_id=session_id)
        events = agent.store.events(result.task_id)

        first = replay(events, task_id=result.task_id).snapshot()
        second = replay(events, task_id=result.task_id).snapshot()
        assert first == second

        # The projection in SQLite must agree with the fold.
        task = agent.store.get_task(result.task_id)
        assert task["status"] == first["status"]

    async def test_replay_survives_a_partial_event_stream(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "done"},
            ]
        )
        result = await agent.runtime.run("read app.py", session_id=session_id)
        events = agent.store.events(result.task_id)

        # Truncating the stream (a crash between appends) must still yield a
        # usable, self-consistent state -- not an exception.
        for cut in range(1, len(events) + 1):
            partial = replay(events[:cut], task_id=result.task_id)
            assert partial.task_id == result.task_id
            assert partial.goal == "read app.py"


class TestHumanInTheLoop:
    async def test_confirmation_pauses_then_approval_executes_the_same_call(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "echo hello"}}]},
                {"content": "ran it"},
            ]
        )
        result = await agent.runtime.run("run echo", session_id=session_id)

        assert result.needs_approval
        assert result.pending_confirmation is not None
        assert result.pending_confirmation.tool == "run_command"
        assert agent.store.list_tool_calls(result.task_id) == []

        approved = await agent.runtime.approve(result.task_id)
        assert approved.status == TaskStatus.COMPLETED.value

        calls = agent.store.list_tool_calls(result.task_id)
        assert len(calls) == 1
        assert calls[0].name == "run_command"
        assert calls[0].arguments == {"command": "echo hello"}
        assert calls[0].status == "succeeded"
        assert "hello" in (calls[0].result or {}).get("output", "")

    async def test_refusal_is_recorded_and_the_model_is_told(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "echo hello"}}]},
                {"content": "I could not run it."},
            ]
        )
        result = await agent.runtime.run("run echo", session_id=session_id)
        assert result.needs_approval

        denied = await agent.runtime.deny(result.task_id, note="not on my machine")
        assert denied.status == TaskStatus.COMPLETED.value
        assert agent.store.list_tool_calls(result.task_id) == []

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        refusals = [e for e in state.log if e.kind == "system" and "REFUSED" in e.text]
        assert refusals, "the model must be told the user refused"
        assert "Do not attempt this call" in refusals[0].text
        assert not any(e.tool == "run_command" for e in state.log)

    async def test_pending_confirmation_round_trips_through_events(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "echo hi"}}]},
                {"content": "done"},
            ]
        )
        result = await agent.runtime.run("run echo", session_id=session_id)

        # A fresh process must be able to see the pending request.
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert state.status is TaskStatus.WAITING_CONFIRMATION
        assert state.pending_confirmation is not None
        assert state.pending_confirmation.request_id == result.pending_confirmation.request_id

        task = agent.store.get_task(result.task_id)
        assert task["pending_confirmation"] is not None

    async def test_approving_a_stale_request_id_is_rejected(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "echo hi"}}]},
                {"content": "done"},
            ]
        )
        result = await agent.runtime.run("run echo", session_id=session_id)
        with pytest.raises(ValueError):
            await agent.runtime.approve(result.task_id, request_id="req_does_not_exist")


class TestStepBudgetSurvivesResume:
    """The step counter must be derived from events, not from memory.

    If it resets on resume, the step budget can be reset indefinitely by
    resuming in a loop -- the budget stops being a budget.
    """

    async def test_steps_used_is_derived_from_events(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"tool_calls": [{"name": "list_directory", "arguments": {"path": "."}}]},
                {"content": "done"},
            ]
        )
        result = await agent.runtime.run("do three steps", session_id=session_id)
        assert result.status == TaskStatus.COMPLETED.value

        events = agent.store.events(result.task_id)
        step_events = [
            e for e in events if e.type is EventType.MODEL_RESPONSE and e.payload.get("phase") == "step"
        ]
        assert len(step_events) == 3

        state = replay(events, task_id=result.task_id)
        assert state.steps_used == 3
        assert state.model_calls >= 4, "the planner call is counted separately"

    async def test_counter_and_usage_survive_a_pause_and_resume(
        self, scripted, session_id: str
    ) -> None:
        """A budget-exceeded task is terminal, so the reachable case is a pause.

        A crash/pause/resume cycle is exactly where a reset counter would let
        the step budget be renewed indefinitely.
        """
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "list_directory", "arguments": {"path": "."}}]},
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "echo hi"}}]},
                {"content": "done"},
            ]
        )
        paused = await agent.runtime.run("two steps then ask", session_id=session_id)
        assert paused.needs_approval

        before = replay(agent.store.events(paused.task_id), task_id=paused.task_id)
        assert before.steps_used == 2, "the counter must survive a replay"
        assert before.usage.total_tokens > 0

        await agent.runtime.approve(paused.task_id)
        after = replay(agent.store.events(paused.task_id), task_id=paused.task_id)
        assert after.steps_used == 3, "the resumed run must continue, not restart"
        assert after.usage.total_tokens > before.usage.total_tokens

    async def test_budget_uses_the_replayed_counter(self, scripted, session_id: str) -> None:
        """max_steps is measured against the replayed total, not a fresh zero."""
        from tests.conftest import looping_script

        agent, _ = await scripted(looping_script(20))
        agent.settings.agent.max_steps = 2
        first = await agent.runtime.run("loop forever", session_id=session_id)
        assert first.status == TaskStatus.FAILED.value

        state = replay(agent.store.events(first.task_id), task_id=first.task_id)
        assert state.steps_used == 2
        # A terminal task is not resumable -- that is what keeps the budget
        # meaningful rather than renewable.
        assert state.status.terminal




class TestTaskWorkspace:
    """A task acts where it was created, not where the resuming process runs.

    Found by running the agent for real: a task created with
    `--workspace /tmp/demo` was approved from the project directory without
    that flag, and every subsequent tool call ran in the *project*. The model
    believed it was in the demo directory; the path fence, built from the
    resuming process's workspace, allowed the project. Two answers to "where
    is this task", and the wrong one won.
    """

    async def test_the_workspace_is_recorded_and_folded_back(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        agent, _ = await scripted([{"content": "done"}])
        result = await agent.runtime.run("do a thing", session_id=session_id)

        created = next(
            e for e in agent.store.events(result.task_id) if e.type is EventType.TASK_CREATED
        )
        assert created.payload["budgets"]["workspace"] == str(settings.workspace)

        # Folded from the event, so a resume reconstructs it without a join.
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert state.workspace == str(settings.workspace)

    async def test_resuming_from_a_different_directory_is_refused(
        self, scripted, session_id: str, settings, tmp_path  # noqa: ANN001
    ) -> None:
        """Silently relocating a task's side effects is worse than an error,
        and an error can name the flag to pass.

        The task has to be *unfinished* for this to matter -- a completed one
        returns before the check -- so it is paused on an approval, which is
        exactly the state `uaa task approve` is run from.
        """
        from unified_agent.agent.factory import build_agent

        agent, _ = await scripted(
            [{"tool_calls": [{"name": "run_command", "arguments": {"command": "ls"}}]}]
        )
        paused = await agent.runtime.run("run ls", session_id=session_id)
        assert paused.status == TaskStatus.WAITING_CONFIRMATION.value
        task_id = paused.task_id

        elsewhere = tmp_path / "somewhere-else"
        elsewhere.mkdir()
        other_settings = settings.model_copy(update={"workspace": elsewhere})
        other = await build_agent(
            settings=other_settings, models=ScriptedModels(agent.models.get())
        )
        try:
            with pytest.raises(ValueError, match="was created in"):
                await other.runtime.resume(task_id)
        finally:
            other.close()

    async def test_resuming_in_the_recorded_directory_is_fine(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted([{"content": "done"}])
        task_id = await _start_task(agent, session_id)
        result = await agent.runtime.resume(task_id)
        assert result.status == TaskStatus.COMPLETED.value

    async def test_an_older_task_without_a_recorded_workspace_still_resumes(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        """Tasks created before the field existed carry no workspace, and the
        check must not turn them into unresumable history."""
        from unified_agent.agent.state import replay as _replay

        agent, _ = await scripted([{"content": "done"}])
        task_id = agent.store.create_task(
            session_id=session_id, goal="legacy", budgets={"model": "scripted"}
        )
        state = _replay(agent.store.events(task_id), task_id=task_id)
        assert state.workspace == ""
        assert agent.store.task_workspace(task_id) == str(settings.workspace)
        # No recorded workspace on the task, so nothing to conflict with.
        result = await agent.runtime.resume(task_id)
        assert result.status == TaskStatus.COMPLETED.value


class TestCliWorkspaceAdoption:
    """`uaa task approve <id>` should not need `--workspace` to be correct."""

    def test_the_helper_adopts_the_recorded_workspace(
        self, settings, tmp_path  # noqa: ANN001
    ) -> None:
        from unified_agent.cli import _settings_for_task
        from unified_agent.storage.store import Store

        store = Store(settings.db_path)
        try:
            elsewhere = tmp_path / "recorded"
            elsewhere.mkdir()
            session = store.create_session(
                name="s", working_dir=str(elsewhere), model_alias="mock"
            )
            task_id = store.create_task(session_id=session, goal="g")
        finally:
            store.close()

        adopted = _settings_for_task(task_id, home=settings.home, workspace=None)
        assert adopted.workspace == elsewhere.resolve()

    def test_an_explicit_workspace_is_left_alone_so_the_runtime_can_refuse(
        self, settings, tmp_path  # noqa: ANN001
    ) -> None:
        """The helper must not paper over a conflict the user asked for: the
        runtime's refusal is what names the right flag."""
        from unified_agent.cli import _settings_for_task
        from unified_agent.storage.store import Store

        store = Store(settings.db_path)
        try:
            recorded = tmp_path / "recorded"
            recorded.mkdir()
            session = store.create_session(
                name="s", working_dir=str(recorded), model_alias="mock"
            )
            task_id = store.create_task(session_id=session, goal="g")
        finally:
            store.close()

        explicit = tmp_path / "explicit"
        explicit.mkdir()
        kept = _settings_for_task(task_id, home=settings.home, workspace=explicit)
        assert kept.workspace == explicit.resolve()
