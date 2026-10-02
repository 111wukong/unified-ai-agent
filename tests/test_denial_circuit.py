"""The refusal circuit breaker.

One refusal might be a mistake. A run of them is a human saying no, and a model
that has been told no will often come back with a slightly different phrasing of
the same thing -- each attempt costing a model call and another interruption.
After `permissions.confirmations.denial_limit` consecutive refusals the task
stops and says why.

The count is folded from events rather than kept in memory, so resuming in the
middle of a streak cannot reset it -- otherwise crashing would be a way to ask
for ever.
"""

from __future__ import annotations

from wukong.agent.state import TaskStatus, replay
from wukong.observability.events import EventType


def command(text: str) -> dict:
    return {"tool_calls": [{"name": "run_command", "arguments": {"command": text}}]}


async def drive_to_limit(agent, session_id: str, *, denials: int):  # noqa: ANN001, ANN201
    """Run a task that keeps asking, refusing it `denials` times."""
    result = await agent.runtime.run("keep asking", session_id=session_id)
    for _ in range(denials):
        assert result.needs_approval, f"expected another request, got {result.status}"
        result = await agent.runtime.deny(result.task_id)
    return result


class TestRefusalCircuitBreaker:
    async def test_the_task_stops_after_the_limit(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        settings.permissions.confirmations.denial_limit = 3
        agent, _ = await scripted(
            [
                command("echo one"),
                command("echo two"),
                command("echo three"),
                {"content": "should never be reached"},
            ]
        )
        result = await drive_to_limit(agent, session_id, denials=3)

        assert result.status == TaskStatus.FAILED.value
        assert "consecutive refusals" in (result.error or "")
        assert "limit 3" in (result.error or ""), "the error should name the limit"

    async def test_nothing_is_executed_on_the_way(self, scripted, session_id: str, settings) -> None:  # noqa: ANN001
        """The whole point: refusing is refusing, however many times asked."""
        settings.permissions.confirmations.denial_limit = 2
        agent, _ = await scripted([command("echo one"), command("echo two")])
        result = await drive_to_limit(agent, session_id, denials=2)

        assert result.status == TaskStatus.FAILED.value
        assert agent.store.list_tool_calls(result.task_id) == []

    async def test_the_streak_is_folded_from_events(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        """Folded, not counted in memory -- otherwise a resume resets it and
        the agent can ask for ever by crashing between refusals."""
        settings.permissions.confirmations.denial_limit = 5
        agent, _ = await scripted([command("echo one"), command("echo two")])
        result = await agent.runtime.run("ask", session_id=session_id)
        result = await agent.runtime.deny(result.task_id)

        # Rebuild the state from the event log alone, as a resume would.
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert state.denied_streak == 1

    async def test_an_approval_resets_the_streak(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        """A human saying yes means this is no longer a run of refusals."""
        settings.permissions.confirmations.denial_limit = 2
        agent, _ = await scripted(
            [
                command("echo one"),
                command("echo two"),
                {"content": "done"},
            ]
        )
        result = await agent.runtime.run("ask", session_id=session_id)
        assert result.needs_approval

        # Refuse, then approve: with the streak reset, the second refusal below
        # must not be enough to trip a limit of two.
        result = await agent.runtime.deny(result.task_id)
        assert result.needs_approval
        result = await agent.runtime.approve(result.task_id)

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert state.denied_streak == 0

    async def test_a_successful_tool_call_resets_the_streak(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        """Progress is progress. An agent that got something done and then
        needs one more permission should not inherit an old count.

        Built so it can tell the difference: two refusals separated by a
        successful call must *not* trip a limit of two. Without the reset the
        second refusal would make the streak two and the task would stop.
        """
        settings.permissions.confirmations.denial_limit = 2
        agent, _ = await scripted(
            [
                command("echo one"),
                {"tool_calls": [{"name": "list_directory", "arguments": {"path": "."}}]},
                command("echo two"),
                {"content": "finished"},
            ]
        )
        result = await agent.runtime.run("ask twice", session_id=session_id)
        assert result.needs_approval

        # Refuse, let it make progress, then refuse again.
        result = await agent.runtime.deny(result.task_id)
        assert result.needs_approval, "the second request should still be asked"
        result = await agent.runtime.deny(result.task_id)

        assert result.status == TaskStatus.COMPLETED.value, (
            "the successful call in between should have cleared the count"
        )
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert state.denied_streak == 1, "one refusal since the last success"

    async def test_a_zero_limit_disables_the_stop(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        """Off is a legitimate setting, and it has to actually be off."""
        settings.permissions.confirmations.denial_limit = 0
        agent, _ = await scripted(
            [command("echo one"), command("echo two"), {"content": "gave up"}]
        )
        result = await drive_to_limit(agent, session_id, denials=2)

        assert result.status != TaskStatus.FAILED.value
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert state.denied_streak == 2

    async def test_the_countdown_is_only_shown_when_it_is_close(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        """Telling the model 'refusal 1 of 10' on the first refusal is noise.
        Telling it at 9 of 10 is the information it needs to stop."""
        settings.permissions.confirmations.denial_limit = 3
        agent, _ = await scripted([command("echo one"), command("echo two")])
        result = await agent.runtime.run("ask", session_id=session_id)
        result = await agent.runtime.deny(result.task_id)

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        first = [e for e in state.log if e.kind == "system" and "REFUSED" in e.text][-1]
        assert "refusal 1 of 3" not in first.text, "too early to warn"

        result = await agent.runtime.deny(result.task_id)
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        second = [e for e in state.log if e.kind == "system" and "REFUSED" in e.text][-1]
        assert "refusal 2 of 3" in second.text, "one short of the limit is worth saying"

    async def test_each_refusal_is_still_recorded(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        """Stopping early must not lose the audit trail of what was asked."""
        settings.permissions.confirmations.denial_limit = 2
        agent, _ = await scripted([command("echo one"), command("echo two")])
        result = await drive_to_limit(agent, session_id, denials=2)

        denied = [e for e in agent.store.events(result.task_id) if e.type is EventType.CONFIRMATION_DENIED]
        assert len(denied) == 2
        assert [e.payload["tool"] for e in denied] == ["run_command", "run_command"]
