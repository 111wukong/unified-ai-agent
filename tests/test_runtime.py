"""Agent loop behaviour: budgets, refusals, plan revision, error recovery."""

from __future__ import annotations


from wukong.agent.context import continue_prompt, plan_is_finished
from wukong.agent.planner import apply_plan_update
from wukong.agent.state import AgentState, PlanStep, StepStatus, TaskStatus, replay
from wukong.config import ModelSpec
from wukong.errors import ModelError
from wukong.observability.events import EventType
from wukong.types import EffectClass

from tests.conftest import ScriptedModels, looping_script



class TestHappyPath:
    async def test_plan_then_tools_then_answer(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "list_directory", "arguments": {"path": "."}}]},
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "The workspace contains app.py and notes.md."},
            ]
        )
        result = await agent.runtime.run("summarise the project", session_id=session_id)

        assert result.status == TaskStatus.COMPLETED.value
        assert "app.py" in (result.answer or "")
        assert result.steps == 3
        assert result.plan, "a plan must have been recorded"

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert [e.tool for e in state.log if e.tool] == ["list_directory", "read_file"]
        assert all(e.success for e in state.log if e.kind == "tool")

    async def test_events_are_gapless_and_ordered(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "done"},
            ]
        )
        result = await agent.runtime.run("read", session_id=session_id)
        events = agent.store.events(result.task_id)
        assert [e.seq for e in events] == list(range(1, len(events) + 1))
        types = {e.type for e in events}
        assert EventType.TASK_CREATED in types
        assert EventType.PLAN_CREATED in types
        assert EventType.TOOL_COMPLETED in types
        assert EventType.TASK_COMPLETED in types

    async def test_jsonl_log_is_written(self, scripted, session_id: str) -> None:
        agent, _ = await scripted([{"content": "done"}])
        await agent.runtime.run("noop", session_id=session_id)
        agent.sink.close()
        files = list(agent.settings.log_dir.glob("events-*.jsonl"))
        assert files, "an event log must exist"
        assert files[0].read_text(encoding="utf-8").strip()


class TestBudgets:
    async def test_max_steps_stops_the_loop(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(looping_script(20))
        agent.settings.agent.max_steps = 3
        result = await agent.runtime.run("list the files and summarise", session_id=session_id)

        assert result.status == TaskStatus.FAILED.value
        assert "steps budget exhausted" in (result.error or "")
        events = agent.store.events(result.task_id)
        exceeded = [e for e in events if e.type is EventType.BUDGET_EXCEEDED]
        assert exceeded and exceeded[0].payload["kind"] == "steps"

    async def test_token_budget_stops_the_loop(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "list_directory", "arguments": {"path": "."}}]},
                {"tool_calls": [{"name": "list_directory", "arguments": {"path": "sub"}}]},
                {"content": "done"},
            ]
        )
        agent.settings.agent.max_tokens = 100
        result = await agent.runtime.run("list files", session_id=session_id)

        assert result.status == TaskStatus.FAILED.value
        assert "tokens budget exhausted" in (result.error or "")

    async def test_cost_budget_is_enforced_not_just_reported(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "list_directory", "arguments": {"path": "."}}]},
                {"tool_calls": [{"name": "list_directory", "arguments": {"path": "sub"}}]},
                {"content": "done"},
            ]
        )
        spec = agent.settings.models["scripted"]
        spec.price_in = 1_000_000.0  # $1 per token
        spec.price_out = 1_000_000.0
        agent.settings.agent.max_cost_usd = 0.01

        result = await agent.runtime.run("list files", session_id=session_id)
        assert result.status == TaskStatus.FAILED.value
        assert "cost_usd budget exhausted" in (result.error or "")

    async def test_cancellation_marker_is_honoured(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(looping_script(20))
        agent.settings.agent.max_steps = 20

        original = agent.runtime._check_budget

        def cancel_after_first(state, max_steps):  # noqa: ANN001, ANN202
            if state.steps_used >= 1:
                agent.runtime.cancel(state.task_id)
            return original(state, max_steps)

        agent.runtime._check_budget = cancel_after_first  # type: ignore[assignment]
        result = await agent.runtime.run("list the files and summarise", session_id=session_id)
        assert result.status == TaskStatus.CANCELLED.value


class TestToolFailures:
    async def test_unknown_tool_becomes_an_observation(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "no_such_tool", "arguments": {}}]},
                {"content": "that tool does not exist"},
            ]
        )
        result = await agent.runtime.run("try a bad tool", session_id=session_id)

        assert result.status == TaskStatus.COMPLETED.value
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        failure = [e for e in state.log if e.tool == "no_such_tool"][0]
        assert not failure.success
        assert "unknown tool" in failure.text

    async def test_invalid_arguments_produce_a_repair_hint(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"wrong": "key"}}]},
                {"content": "fixed"},
            ]
        )
        result = await agent.runtime.run("read a file", session_id=session_id)

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        failure = [e for e in state.log if e.tool == "read_file"][0]
        assert not failure.success
        assert "missing required argument" in failure.text
        assert "Accepted keys" in failure.text, "the hint must tell the model what to send"

    async def test_denied_command_is_a_hard_stop(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "rm -rf /"}}]},
                {"content": "I was not allowed to do that."},
            ]
        )
        result = await agent.runtime.run("clean up", session_id=session_id)

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        entry = [e for e in state.log if e.tool == "run_command"][0]
        assert not entry.success
        assert "PERMISSION DENIED" in entry.text
        assert "hard stop" in entry.text
        # A denied call must never reach the ledger.
        assert agent.store.list_tool_calls(result.task_id) == []

    async def test_system_admin_is_denied(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {
                    "tool_calls": [
                        {"name": "run_command", "arguments": {"command": "sudo rm -rf /"}}
                    ]
                },
                {"content": "denied"},
            ]
        )
        result = await agent.runtime.run("escalate", session_id=session_id)
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert not [e for e in state.log if e.tool == "run_command"][0].success


class TestPlanAndFinish:
    async def test_update_plan_revises_the_checklist(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {
                    "tool_calls": [
                        {
                            "name": "update_plan",
                            "arguments": {
                                "reason": "the original plan was wrong",
                                "steps": [
                                    {"description": "New first step", "status": "completed"},
                                    {"description": "New second step", "status": "pending"},
                                ],
                            },
                        }
                    ]
                },
                {"content": "done"},
            ]
        )
        result = await agent.runtime.run("plan again", session_id=session_id)

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert [s.description for s in state.plan] == ["New first step", "New second step"]
        assert state.current_step == 1, "the cursor must follow the checklist, not a counter"
        events = agent.store.events(result.task_id)
        assert any(e.type is EventType.PLAN_REVISED for e in events)

    async def test_finish_tool_ends_the_task(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [{"tool_calls": [{"name": "finish", "arguments": {"answer": "the answer is 42"}}]}]
        )
        result = await agent.runtime.run("answer", session_id=session_id)
        assert result.status == TaskStatus.COMPLETED.value
        assert result.answer == "the answer is 42"

    async def test_blocked_finish_is_reported_as_failed(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {
                    "tool_calls": [
                        {
                            "name": "finish",
                            "arguments": {
                                "answer": "I need the API key to continue.",
                                "status": "blocked",
                            },
                        }
                    ]
                }
            ]
        )
        result = await agent.runtime.run("do something blocked", session_id=session_id)
        assert result.status == TaskStatus.FAILED.value
        assert result.answer == "I need the API key to continue."
        assert "blocked" in (result.error or "")

    async def test_malformed_finish_does_not_end_the_task(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "finish", "arguments": {}}]},
                {"content": "recovered and answered properly"},
            ]
        )
        result = await agent.runtime.run("answer", session_id=session_id)
        assert result.status == TaskStatus.COMPLETED.value
        assert "recovered" in (result.answer or "")


class TestModelErrors:
    async def test_retryable_errors_are_retried(self, scripted, session_id: str) -> None:
        agent, model = await scripted([{"content": "ok after retry"}])
        calls = {"n": 0}
        original = model._chat

        async def flaky(messages, **kwargs):  # noqa: ANN001, ANN202
            calls["n"] += 1
            if calls["n"] == 2:  # the loop call (1 is the planner)
                raise ModelError("429 rate limited", retryable=True)
            return await original(messages, **kwargs)

        model._chat = flaky  # type: ignore[assignment]
        agent.settings.agent.model_retry_attempts = 3

        result = await agent.runtime.run("say ok", session_id=session_id)
        assert result.status == TaskStatus.COMPLETED.value
        assert calls["n"] >= 3

    async def test_non_retryable_errors_fail_fast(self, scripted, session_id: str) -> None:
        agent, model = await scripted([{"content": "never reached"}])

        async def boom(messages, **kwargs):  # noqa: ANN001, ANN202
            raise ModelError("401 invalid api key", retryable=False)

        model._chat = boom  # type: ignore[assignment]
        result = await agent.runtime.run("say ok", session_id=session_id)
        assert result.status == TaskStatus.FAILED.value
        assert "401" in (result.error or "")


class TestSessionIsolation:
    async def test_tasks_are_scoped_to_their_session(self, scripted, session_id: str) -> None:
        agent, _ = await scripted([{"content": "a"}, {"content": "b"}])
        first = await agent.runtime.run("first", session_id=session_id)
        second = await agent.runtime.run("second", session_id=session_id)

        assert first.task_id != second.task_id
        assert first.session_id == second.session_id == session_id
        assert len(agent.store.list_tasks(session_id=session_id)) == 2

    async def test_ledger_rows_never_leak_across_tasks(self, scripted, session_id: str) -> None:
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "one"},
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "notes.md"}}]},
                {"content": "two"},
            ]
        )
        first = await agent.runtime.run("read app", session_id=session_id)
        second = await agent.runtime.run("read notes", session_id=session_id)

        assert len(agent.store.list_tool_calls(first.task_id)) == 1
        assert len(agent.store.list_tool_calls(second.task_id)) == 1
        assert (
            agent.store.list_tool_calls(second.task_id)[0].arguments["path"] == "notes.md"
        )


class TestScriptedModelSanity:
    async def test_model_spec_provider_is_mock(self, settings) -> None:  # noqa: ANN001
        assert isinstance(settings.models["scripted"], ModelSpec)
        assert settings.models["scripted"].provider == "mock"

    async def test_effect_enum_covers_the_spec_table(self) -> None:
        assert {e.value for e in EffectClass} == {
            "read_only",
            "write_local",
            "execute_local",
            "network",
            "external_side_effect",
            "system_admin",
        }


class TestEmptyResponse:
    """An empty response is a failure, not an empty answer.

    Reporting `completed` when the model said nothing is how a task that did
    nothing looks like a task that succeeded -- and for a reasoning model it
    is the ordinary shape of "the output budget ran out before any content
    was emitted", which is a configuration problem the user can fix once it
    is named.
    """

    async def test_an_empty_response_fails_rather_than_completing(
        self, settings, session_id: str
    ) -> None:
        from wukong.agent.factory import build_agent
        from wukong.models.mock import MockModel
        from wukong.types import ModelResponse

        class Silent(MockModel):
            async def _chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
                if kwargs.get("response_format"):
                    return self._structured(messages, kwargs["response_format"])
                return ModelResponse(content="", finish_reason="stop")

        agent = await build_agent(settings=settings, models=ScriptedModels(Silent(settings.models["scripted"])))
        try:
            result = await agent.runtime.run("do something", session_id=session_id)
        finally:
            agent.close()

        assert result.status == TaskStatus.FAILED.value, (
            "an empty response must not be reported as success"
        )
        assert "no content and no tool calls" in (result.error or "")

    async def test_a_truncated_response_names_the_setting_to_raise(
        self, settings, session_id: str
    ) -> None:
        """`finish_reason=length` is the actionable case: reasoning models
        spend output tokens on their reasoning, so a budget that looks
        generous for prose can be gone before any content appears."""
        from wukong.agent.factory import build_agent
        from wukong.models.mock import MockModel
        from wukong.types import ModelResponse

        class Truncated(MockModel):
            async def _chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
                if kwargs.get("response_format"):
                    return self._structured(messages, kwargs["response_format"])
                return ModelResponse(content="", finish_reason="length")

        agent = await build_agent(settings=settings, models=ScriptedModels(Truncated(settings.models["scripted"])))
        try:
            result = await agent.runtime.run("do something", session_id=session_id)
        finally:
            agent.close()

        assert result.status == TaskStatus.FAILED.value
        assert "output budget" in (result.error or "")
        assert "max_output_tokens" in (result.error or "")

    async def test_whitespace_only_content_is_also_not_an_answer(
        self, settings, session_id: str
    ) -> None:
        from wukong.agent.factory import build_agent
        from wukong.models.mock import MockModel
        from wukong.types import ModelResponse

        class Blank(MockModel):
            async def _chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
                if kwargs.get("response_format"):
                    return self._structured(messages, kwargs["response_format"])
                return ModelResponse(content="   \n  ", finish_reason="stop")

        agent = await build_agent(settings=settings, models=ScriptedModels(Blank(settings.models["scripted"])))
        try:
            result = await agent.runtime.run("do something", session_id=session_id)
        finally:
            agent.close()

        assert result.status == TaskStatus.FAILED.value


class TestAFinishedPlanIsNotTheWork:
    """A checklist with nothing left on it must not become the treadmill.

    From a real run: five steps, every one ticked off, and then the same
    all-complete plan re-sent thirteen times instead of an answer. The
    document the user asked for was already on disk. The task still ended
    `failed` on an exhausted step budget, 229k tokens later.

    Three things had to be true at once for that, so three things are pinned
    here: the tool list, the closing nudge, and what a rewrite does to a step
    that was already settled.
    """

    @staticmethod
    def _state(session_id: str, *statuses: StepStatus) -> AgentState:
        state = AgentState(task_id="task_x", session_id=session_id, goal="写一首古体诗")
        state.plan = [
            PlanStep(id=f"step_{i + 1}", description=f"第 {i + 1} 步", status=s)
            for i, s in enumerate(statuses)
        ]
        return state

    async def test_update_plan_is_withdrawn_once_nothing_is_outstanding(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted([])
        state = self._state(session_id, StepStatus.COMPLETED, StepStatus.COMPLETED)

        names = {spec["function"]["name"] for spec in agent.runtime._tool_specs(state)}

        assert "update_plan" not in names
        assert names, "the other tools must still be offered"

    async def test_update_plan_stays_while_a_step_is_outstanding(
        self, scripted, session_id: str
    ) -> None:
        agent, _ = await scripted([])
        state = self._state(session_id, StepStatus.COMPLETED, StepStatus.PENDING)

        names = {spec["function"]["name"] for spec in agent.runtime._tool_specs(state)}

        assert "update_plan" in names

    def test_a_skipped_step_counts_as_settled(self, session_id: str) -> None:
        assert plan_is_finished(self._state(session_id, StepStatus.SKIPPED))

    def test_the_nudge_stops_asking_for_tools_once_the_plan_is_done(
        self, session_id: str
    ) -> None:
        done = continue_prompt(self._state(session_id, StepStatus.COMPLETED))
        running = continue_prompt(self._state(session_id, StepStatus.PENDING))

        assert "update_plan" in done
        assert "final answer" in done
        assert running.startswith("Continue.")

    def test_a_revision_keeps_a_settled_step_settled(self, session_id: str) -> None:
        """The model re-sends the whole list; omitting `status` is not a reset."""
        state = self._state(session_id, StepStatus.COMPLETED, StepStatus.PENDING)

        apply_plan_update(
            state,
            [
                {"description": "第 1 步"},
                {"description": "第 2 步", "status": "running"},
            ],
        )

        assert state.plan[0].status is StepStatus.COMPLETED
        assert state.plan[1].status is StepStatus.RUNNING
