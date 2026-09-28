"""Cancellation.

Two mechanisms, and they are not interchangeable: a running loop holds its own
copy of the state and never re-reads the event log, so it can only be stopped
by the marker file; a paused task has no loop to read the marker, so it has to
be cancelled by recording the terminal state directly.

Only the marker was implemented. Found by starting a task from the web
console, seeing the approval prompt, and clicking Cancel: the API answered
`"cancelling"` and the task sat in `waiting_confirmation` for ever. That is the
state a user is in *whenever* they are looking at an approval prompt.
"""

from __future__ import annotations


from unified_agent.agent.runtime import cancel_task
from unified_agent.agent.state import TaskStatus, replay
from unified_agent.observability.events import EventType
from unified_agent.storage.store import Store


def status_of(store: Store, task_id: str) -> str:
    return str(store.get_task(task_id)["status"])


async def paused_task(agent, session_id: str) -> str:  # noqa: ANN001
    """A task stopped at an approval prompt -- the state Cancel has to work in."""
    result = await agent.runtime.run("run ls", session_id=session_id)
    assert result.status == TaskStatus.WAITING_CONFIRMATION.value
    return result.task_id


class TestCancelDecision:
    async def test_a_paused_task_is_cancelled_outright(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        """The bug. Nothing is running, so nothing will read a marker."""
        agent, _ = await scripted(
            [{"tool_calls": [{"name": "run_command", "arguments": {"command": "ls"}}]}]
        )
        task_id = await paused_task(agent, session_id)

        outcome = agent.runtime.cancel(task_id)

        assert outcome == "cancelled"
        assert status_of(agent.store, task_id) == TaskStatus.CANCELLED.value
        assert not agent.runtime.is_cancelled(task_id), (
            "a marker would be misleading: no loop exists to consume it"
        )

    async def test_the_cancellation_is_in_the_event_log(
        self, scripted, session_id: str  # noqa: ANN001
    ) -> None:
        """Not a side table. `replay` has to agree, or a resume would
        resurrect a cancelled task."""
        agent, _ = await scripted(
            [{"tool_calls": [{"name": "run_command", "arguments": {"command": "ls"}}]}]
        )
        task_id = await paused_task(agent, session_id)
        agent.runtime.cancel(task_id)

        events = agent.store.events(task_id)
        assert events[-1].type is EventType.TASK_CANCELLED
        assert "waiting_confirmation" in events[-1].payload.get("reason", "")
        state = replay(events, task_id=task_id)
        assert state.status is TaskStatus.CANCELLED
        assert state.status.terminal

    async def test_a_running_task_is_asked_to_stop_not_overwritten(
        self, scripted, session_id: str  # noqa: ANN001
    ) -> None:
        """Appending a terminal event for a live loop would be undone by the
        loop's next write, so the marker is the only thing that works."""
        agent, _ = await scripted([{"content": "done"}])
        result = await agent.runtime.run("just answer", session_id=session_id)
        # Rewind the projection to a live status without touching the events,
        # which is the situation a running loop presents to `cancel`.
        agent.store.conn.execute(
            "UPDATE tasks SET status=? WHERE id=?", (TaskStatus.RUNNING.value, result.task_id)
        )
        agent.store.conn.commit()

        outcome = agent.runtime.cancel(result.task_id)

        assert outcome == "requested"
        assert agent.runtime.is_cancelled(result.task_id)
        assert status_of(agent.store, result.task_id) == TaskStatus.RUNNING.value, (
            "the loop owns the status until it stops"
        )

    async def test_cancelling_a_finished_task_changes_nothing(
        self, scripted, session_id: str  # noqa: ANN001
    ) -> None:
        agent, _ = await scripted([{"content": "done"}])
        result = await agent.runtime.run("just answer", session_id=session_id)
        before = len(agent.store.events(result.task_id))

        assert agent.runtime.cancel(result.task_id) == "already_terminal"
        assert len(agent.store.events(result.task_id)) == before, "no event was written"
        assert status_of(agent.store, result.task_id) == TaskStatus.COMPLETED.value

    def test_an_unknown_task_is_reported_as_such(self, settings) -> None:  # noqa: ANN001
        store = Store(settings.db_path)
        try:
            assert cancel_task(store, settings, "task_does_not_exist") == "not_found"
        finally:
            store.close()

    async def test_a_cancelled_task_leaves_no_approval_prompt(
        self, scripted, session_id: str  # noqa: ANN001
    ) -> None:
        """The event log keeps the request -- it happened. The *state* must not,
        or every reader renders a live approval prompt for a task that is over,
        and the console offers buttons that would then fail."""
        agent, _ = await scripted(
            [{"tool_calls": [{"name": "run_command", "arguments": {"command": "ls"}}]}]
        )
        task_id = await paused_task(agent, session_id)
        assert replay(agent.store.events(task_id), task_id=task_id).pending_confirmation

        agent.runtime.cancel(task_id)

        state = replay(agent.store.events(task_id), task_id=task_id)
        assert state.pending_confirmation is None
        assert agent.store.get_task(task_id)["pending_confirmation"] is None
        # The request is still in the record, which is the point of an
        # append-only log: cancelling does not erase that it was asked.
        assert any(
            e.type is EventType.CONFIRMATION_REQUESTED for e in agent.store.events(task_id)
        )

    async def test_a_completed_task_also_has_nothing_pending(
        self, scripted, session_id: str  # noqa: ANN001
    ) -> None:
        """The same invariant, reached a different way -- set in one place so a
        future terminal event cannot forget it."""
        agent, _ = await scripted([{"content": "done"}])
        result = await agent.runtime.run("just answer", session_id=session_id)
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert state.pending_confirmation is None

    async def test_cancelling_twice_is_idempotent(
        self, scripted, session_id: str  # noqa: ANN001
    ) -> None:
        agent, _ = await scripted(
            [{"tool_calls": [{"name": "run_command", "arguments": {"command": "ls"}}]}]
        )
        task_id = await paused_task(agent, session_id)

        assert agent.runtime.cancel(task_id) == "cancelled"
        assert agent.runtime.cancel(task_id) == "already_terminal"

    async def test_a_cancelled_task_cannot_be_resumed(
        self, scripted, session_id: str  # noqa: ANN001
    ) -> None:
        """Otherwise Cancel is a suggestion."""
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "ls"}}]},
                {"content": "carried on anyway"},
            ]
        )
        task_id = await paused_task(agent, session_id)
        agent.runtime.cancel(task_id)

        result = await agent.runtime.resume(task_id)
        assert result.status == TaskStatus.CANCELLED.value
        assert not any(
            e.type is EventType.CONFIRMATION_GRANTED for e in agent.store.events(task_id)
        ), "a cancelled task must not run its pending call"
