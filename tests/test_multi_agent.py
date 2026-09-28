"""Multi-agent delegation: the gates, and what the orchestrator gets back.

The interesting assertions here are not "does it run". They are the four
things the published orchestrator-worker guidance says make it work:

* the fan-out is capped and the cap is enforced, not merely described;
* every brief carries a goal, an output format and boundaries;
* sub-agent bodies land on disk and only a bounded summary comes back;
* sub-agents cannot write, so they cannot trample each other.

Plus the one property that makes this worth having at all: a sub-agent is a
real child task, so `uaa task events <child_id>` shows what it did and its
budget is its own.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import ScriptedModel

from unified_agent.agent.factory import build_agent
from unified_agent.agent.runtime import AgentRuntime
from unified_agent.config import ModelSpec
from unified_agent.errors import ModelError
from unified_agent.models.mock import MockModel
from unified_agent.observability.events import EventType
from unified_agent.orchestration.multi_agent import (
    AgentDeps,
    MultiAgentRunner,
    SubAgentTask,
    build_tasks,
    summarize,
)
from unified_agent.storage.store import Store
from unified_agent.tools.base import ToolContext
from unified_agent.tools.multi_agent import DelegateTool, build_multi_agent_tools, strongest_effect
from unified_agent.types import EffectClass


def task(**overrides: Any) -> SubAgentTask:
    base = {
        "goal": "Find every place the retry budget is read.",
        "output_format": "A list of file:line entries, one per site.",
        "boundaries": "Only src/. Ignore tests and docs.",
    }
    base.update(overrides)
    return SubAgentTask(**base)


class _ExplodingModel(MockModel):
    """Deterministic, except that a brief mentioning EXPLODE fails outright."""

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
        text = " ".join(m.content or "" for m in messages)
        if "EXPLODE" in text:
            raise ModelError("sub-agent blew up", retryable=False)
        return self._final(messages)


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


class TestGates:
    """The description asks; these refuse."""

    @staticmethod
    def _runner(settings) -> MultiAgentRunner:  # noqa: ANN001
        store = Store(settings.db_path)
        deps = AgentDeps(
            settings=settings, store=store, registry=None, engine=None, models=None
        )
        return MultiAgentRunner(deps=deps, parent_task_id="t_parent", session_id="s")

    def test_delegation_is_off_by_default(self, settings) -> None:  # noqa: ANN001
        assert settings.multi_agent.enabled is False
        runner = self._runner(settings)
        refusal = runner.check([task()])
        assert refusal and "disabled" in refusal
        runner.store.close()

    def test_the_fan_out_is_capped(self, settings) -> None:  # noqa: ANN001
        """The documented failure is fifty sub-agents for a small question."""
        settings.multi_agent.enabled = True
        settings.multi_agent.max_agents = 3
        runner = self._runner(settings)
        try:
            assert runner.check([task() for _ in range(3)]) is None
            refusal = runner.check([task() for _ in range(4)])
        finally:
            runner.store.close()

        assert refusal and "max_agents is 3" in refusal
        assert "fan-out for a small question" in refusal

    @pytest.mark.parametrize("field", ["goal", "output_format", "boundaries"])
    def test_every_part_of_the_brief_is_required(self, settings, field: str) -> None:  # noqa: ANN001
        """A one-line brief is the documented cause of two sub-agents doing
        the identical search."""
        settings.multi_agent.enabled = True
        runner = self._runner(settings)
        try:
            refusal = runner.check([task(**{field: "   "})])
        finally:
            runner.store.close()

        assert refusal and f"tasks[0].{field} is empty" in refusal

    def test_system_admin_cannot_be_granted_by_a_config_file(self, settings) -> None:  # noqa: ANN001
        """Same rule as the workflow approval layer: a TOML is not a policy
        authority, and sub-agents run unattended."""
        settings.multi_agent.enabled = True
        settings.multi_agent.allowed_effects = [EffectClass.SYSTEM_ADMIN]
        runner = self._runner(settings)
        try:
            refusal = runner.check([task()])
        finally:
            runner.store.close()

        assert refusal and "may not include system_admin" in refusal

    def test_an_empty_delegation_is_not_a_failure(self, settings) -> None:  # noqa: ANN001
        settings.multi_agent.enabled = True
        runner = self._runner(settings)
        try:
            assert runner.check([]) is None
        finally:
            runner.store.close()


class TestToolSurface:
    def test_the_tool_is_absent_when_delegation_is_off(self, settings) -> None:  # noqa: ANN001
        """Absent rather than refusing.

        A tool listed in the prompt but always failing costs a model call to
        discover and teaches the model that the catalogue lies.
        """
        deps = AgentDeps(settings=settings, store=None, registry=None, engine=None, models=None)
        assert build_multi_agent_tools(deps) == []

        settings.multi_agent.enabled = True
        tools = build_multi_agent_tools(deps)
        assert [t.spec.name for t in tools] == ["delegate"]

    def test_the_description_carries_the_scale_rules(self, settings) -> None:  # noqa: ANN001
        """The model cannot judge effort, so the sizing has to be in the text
        it reads -- the hard cap alone only stops the worst case."""
        settings.multi_agent.enabled = True
        deps = AgentDeps(settings=settings, store=None, registry=None, engine=None, models=None)
        description = build_multi_agent_tools(deps)[0].spec.description

        assert "do not delegate" in description
        assert "2 to 4 sub-agents" in description
        assert "15x the tokens" in description

    @pytest.mark.parametrize(
        ("allowed", "expected"),
        [
            ([], EffectClass.READ_ONLY),
            ([EffectClass.READ_ONLY], EffectClass.READ_ONLY),
            ([EffectClass.READ_ONLY, EffectClass.EXECUTE_LOCAL], EffectClass.EXECUTE_LOCAL),
            ([EffectClass.EXECUTE_LOCAL, EffectClass.NETWORK], EffectClass.NETWORK),
        ],
    )
    def test_the_tool_declares_the_strongest_effect_it_can_grant(
        self, settings, allowed, expected  # noqa: ANN001
    ) -> None:
        """Otherwise a tool could quietly hand out more than it declares, and
        the effect class would be decorative."""
        settings.multi_agent.enabled = True
        settings.multi_agent.allowed_effects = list(allowed)
        tool = DelegateTool(
            AgentDeps(settings=settings, store=None, registry=None, engine=None, models=None)
        )
        assert strongest_effect(list(allowed)) is expected
        assert tool.spec.effect_class is expected

    def test_the_brief_schema_requires_the_contract(self, settings) -> None:  # noqa: ANN001
        settings.multi_agent.enabled = True
        deps = AgentDeps(settings=settings, store=None, registry=None, engine=None, models=None)
        schema = build_multi_agent_tools(deps)[0].spec.parameters
        item = schema["properties"]["tasks"]["items"]

        assert set(item["required"]) == {"goal", "output_format", "boundaries"}
        assert schema["properties"]["tasks"]["maxItems"] == settings.multi_agent.max_agents


class TestBriefRendering:
    def test_the_brief_carries_all_four_parts(self) -> None:
        rendered = task(guidance="Start from src/unified_agent/agent/").render()

        assert "Find every place the retry budget is read." in rendered
        assert "## Output format" in rendered
        assert "## Where to look" in rendered
        assert "## Boundaries" in rendered

    def test_an_omitted_guidance_section_is_not_rendered_empty(self) -> None:
        rendered = task().render()
        assert "## Where to look" not in rendered
        assert "## Boundaries" in rendered

    def test_the_brief_tells_the_sub_agent_to_put_substance_in_the_answer(self) -> None:
        """Because only a clipped version of the answer comes back, a
        sub-agent that says "I would look at X" returns nothing usable."""
        assert "rather than describing what you would do" in task().render()


class TestSummary:
    def test_the_summary_names_the_report_path_not_the_body(self) -> None:
        from unified_agent.orchestration.multi_agent import SubAgentOutcome

        outcome = SubAgentOutcome(
            index=0,
            goal="Survey the retry paths",
            task_id="task_child",
            status="completed",
            summary="Found four call sites.",
            report_path="/tmp/reports/00-survey.md",
            steps=5,
        )
        text = summarize([outcome])

        assert "/tmp/reports/00-survey.md" in text
        assert "1/1 sub-agent(s) completed" in text
        assert "Do not delegate the same question again" in text

    def test_a_blocked_sub_agent_says_which_task_to_approve(self) -> None:
        from unified_agent.orchestration.multi_agent import SubAgentOutcome

        outcome = SubAgentOutcome(
            index=0,
            goal="Write the fix",
            task_id="task_child",
            status="waiting_confirmation",
            pending_tool="write_file",
        )
        text = summarize([outcome])

        assert "BLOCKED awaiting approval for `write_file`" in text
        assert "task_child" in text


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_settings(settings):  # noqa: ANN001
    """Orchestrator scripted, sub-agents on the stateless mock.

    Splitting the aliases is not just for the test: it is the cost shape the
    research describes (a strong lead, cheap workers), and it is the only way
    to get a deterministic assertion out of a concurrent fan-out.
    """
    settings.models["worker"] = ModelSpec(provider="mock", model="mock-react")
    settings.multi_agent.enabled = True
    settings.multi_agent.model = "worker"
    settings.multi_agent.parallelism = 1
    return settings


class PerAliasModels:
    """A registry that hands out a *different* model per alias.

    `ScriptedModels` returns one shared instance for every alias, which is
    fine for a single-threaded loop and wrong here: sub-agents running
    concurrently would share one script cursor and consume each other's
    steps, which is precisely the sharing this feature exists to avoid.
    """

    def __init__(self, *, orchestrator: Any, worker: Any) -> None:
        self._models = {"scripted": orchestrator, "worker": worker}

    def aliases(self) -> list[str]:
        return sorted(self._models)

    def get(self, alias: str | None = None) -> Any:
        return self._models[alias or "scripted"]

    def try_get(self, alias: str) -> Any:
        return self._models.get(alias)

    def summarizer(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


@pytest.fixture
def multi(multi_settings):  # noqa: ANN001
    """Factory: `await multi(script, worker=...)` -> agent."""

    created: list = []

    async def _make(script: list[dict[str, Any]], worker: Any = None):  # noqa: ANN202
        agent = await build_agent(
            settings=multi_settings,
            models=PerAliasModels(
                orchestrator=ScriptedModel(multi_settings.models["scripted"], script),
                worker=worker or MockModel(multi_settings.models["worker"]),
            ),
        )
        created.append(agent)
        return agent

    yield _make

    for agent in created:
        agent.close()


def delegate_script(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"tool_calls": [{"name": "delegate", "arguments": {"tasks": tasks}}]},
        {"content": "synthesised"},
    ]


class TestExecution:
    async def test_a_sub_agent_is_a_real_child_task(self, multi, session_id) -> None:  # noqa: ANN001
        """This is the whole reason the feature needs no new kernel: budgets,
        permissions, the event log and resume all apply because a sub-agent
        is an ordinary task."""
        agent = await multi(
            delegate_script(
                [
                    {
                        "goal": "Locate the retry budget reads.",
                        "output_format": "file:line list",
                        "boundaries": "src only",
                    },
                    {
                        "goal": "Locate the compaction triggers.",
                        "output_format": "file:line list",
                        "boundaries": "src only",
                    },
                ]
            )
        )
        result = await agent.runtime.run("survey the runtime", session_id=session_id)
        children = agent.store.list_children(result.task_id)
        events = agent.store.events(result.task_id)

        assert result.status == "completed", result.error
        assert len(children) == 2
        assert all(child["parent_task_id"] == result.task_id for child in children)

        kinds = [e.type for e in events]
        assert kinds.count(EventType.SUBAGENT_STARTED) == 2
        assert kinds.count(EventType.SUBAGENT_COMPLETED) == 2

    async def test_the_orchestrator_gets_paths_and_summaries_not_bodies(
        self, multi, multi_settings, session_id  # noqa: ANN001
    ) -> None:
        """Pasting full reports back into the parent's context is the
        "telephone game" that makes a fan-out worse than a single agent."""
        multi_settings.multi_agent.max_summary_chars = 80
        agent = await multi(
            delegate_script(
                [
                    {
                        "goal": "Inspect the sandbox probe path.",
                        "output_format": "a paragraph",
                        "boundaries": "sandbox/ only",
                    }
                ]
            )
        )
        result = await agent.runtime.run("survey", session_id=session_id)
        calls = agent.store.list_tool_calls(result.task_id)

        delegate_call = next(c for c in calls if c.name == "delegate")
        payload = delegate_call.result or {}
        result_payload = payload.get("output", "")
        reports = payload.get("metadata", {}).get("reports") or []

        assert "full report:" in result_payload, "the orchestrator was not given a path"
        # Read the path from the metadata, not by parsing the rendered output:
        # the output goes through tool-output truncation like any other.
        assert len(reports) == 1
        report = Path(reports[0])
        assert report.is_file(), f"report was not written: {report}"

        body = report.read_text(encoding="utf-8")
        assert "## Report" in body
        assert "## Output format requested" in body
        assert "## Boundaries" in body
        # The full report is on disk; only a clipped summary is in the result.
        assert len(result_payload) < len(body) + 2_000

    async def test_a_terse_output_format_is_not_treated_as_a_bad_brief(
        self, multi, session_id  # noqa: ANN001
    ) -> None:
        """"a list" is a legitimate output format.

        An earlier version put a `minLength` floor on these fields. That is
        not a specificity check -- it rejects short-but-clear briefs, and a
        model told "string is 6 chars, min 8" pads the string instead of
        improving the brief. The requirement is "not empty".
        """
        agent = await multi(
            delegate_script(
                [
                    {
                        "goal": "List the tools.",
                        "output_format": "a list",
                        "boundaries": "none",
                    }
                ]
            )
        )
        result = await agent.runtime.run("survey", session_id=session_id)
        calls = agent.store.list_tool_calls(result.task_id)

        delegate_call = next(c for c in calls if c.name == "delegate")
        assert delegate_call.status == "succeeded", delegate_call.error
        assert (delegate_call.result or {}).get("metadata", {}).get("completed") == 1

    async def test_a_failing_sub_agent_does_not_take_its_siblings_down(
        self, multi, session_id  # noqa: ANN001
    ) -> None:
        """`asyncio.gather(return_exceptions=True)` is the point: one broken
        worker must not cancel the others, and its failure must still be
        reported rather than swallowed."""
        agent = await multi(
            delegate_script(
                [
                    {
                        "goal": "EXPLODE on purpose.",
                        "output_format": "nothing",
                        "boundaries": "none",
                    },
                    {
                        "goal": "Describe the permission engine.",
                        "output_format": "a paragraph",
                        "boundaries": "permissions.py only",
                    },
                ]
            ),
            worker=_ExplodingModel(ModelSpec(provider="mock", model="mock-react")),
        )
        result = await agent.runtime.run("survey", session_id=session_id)
        calls = agent.store.list_tool_calls(result.task_id)

        delegate_call = next(c for c in calls if c.name == "delegate")
        payload = delegate_call.result or {}
        # One completed, one failed, and the tool itself still succeeded.
        assert payload.get("metadata", {}).get("completed") == 1
        assert delegate_call.status == "succeeded"
        assert "1/2 sub-agent(s) completed" in payload.get("output", "")

    async def test_parallelism_bounds_the_fan_out(
        self, multi, multi_settings, session_id, monkeypatch  # noqa: ANN001
    ) -> None:
        """The 3-5 range is where the speedup is measured; going wider by
        accident is how a local model endpoint gets hammered."""
        multi_settings.multi_agent.parallelism = 2
        briefs = [
            {
                "goal": f"Survey subsystem number {i}.",
                "output_format": "a list",
                "boundaries": "that subsystem only",
            }
            for i in range(4)
        ]

        active = 0
        peak = 0
        original = AgentRuntime.run

        async def tracked(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN001
            nonlocal active, peak
            if not kwargs.get("parent_task_id"):
                return await original(self, *args, **kwargs)
            active += 1
            peak = max(peak, active)
            try:
                # Long enough that a wider fan-out would overlap.
                await asyncio.sleep(0.02)
                return await original(self, *args, **kwargs)
            finally:
                active -= 1

        monkeypatch.setattr(AgentRuntime, "run", tracked)
        agent = await multi(delegate_script(briefs))
        result = await agent.runtime.run("survey", session_id=session_id)
        children = agent.store.list_children(result.task_id)

        assert len(children) == 4
        assert peak == 2, f"expected at most 2 concurrent sub-agents, saw {peak}"

    async def test_sub_agents_are_read_only_by_default(self, multi, session_id) -> None:  # noqa: ANN001
        """The orchestrator is the only writer of shared state.

        A sub-agent that could write would race its siblings; the grant is
        visible on the child task's own record, which is where to check it.
        """
        agent = await multi(
            delegate_script(
                [
                    {
                        "goal": "Read the agent loop.",
                        "output_format": "a summary",
                        "boundaries": "runtime.py only",
                    }
                ]
            )
        )
        result = await agent.runtime.run("survey", session_id=session_id)
        children = agent.store.list_children(result.task_id)
        child_events = agent.store.events(children[0]["id"])

        created = next(e for e in child_events if e.type is EventType.TASK_CREATED)
        assert created.payload["budgets"]["approved_effects"] == ["read_only"]

    async def test_a_sub_agent_budget_is_its_own(
        self, multi, multi_settings, session_id  # noqa: ANN001
    ) -> None:
        """One runaway worker must not be able to spend the parent's budget."""
        multi_settings.multi_agent.max_steps_per_agent = 3
        agent = await multi(
            delegate_script(
                [
                    {
                        "goal": "Keep listing the directory forever.",
                        "output_format": "a list",
                        "boundaries": "none",
                    }
                ]
            )
        )
        result = await agent.runtime.run("survey", session_id=session_id)
        children = agent.store.list_children(result.task_id)

        assert len(children) == 1
        # The parent finished normally; only the worker ran out of steps.
        assert result.status == "completed"
        assert children[0]["steps_used"] <= 3


class TestTaskParsing:
    def test_unknown_keys_are_ignored_and_missing_ones_default(self) -> None:
        tasks = build_tasks(
            [
                {"goal": "g", "output_format": "o", "boundaries": "b", "nonsense": 1},
                {"goal": "g2", "output_format": "o2", "boundaries": "b2", "guidance": "start here"},
            ]
        )
        assert len(tasks) == 2
        assert tasks[0].guidance == ""
        assert tasks[1].guidance == "start here"

    def test_non_dict_entries_are_dropped(self) -> None:
        assert build_tasks(["nope", 42, None]) == []


class TestToolContext:
    async def test_a_dry_run_does_not_spawn_anything(self, multi_settings) -> None:  # noqa: ANN001
        tool = DelegateTool(
            AgentDeps(
                settings=multi_settings, store=None, registry=None, engine=None, models=None
            )
        )
        ctx = ToolContext(
            task_id="t",
            session_id="s",
            step_id="step_1",
            workspace=Path("."),
            home=Path("."),
            artifact_dir=Path("."),
            dry_run=True,
        )
        result = await tool.run(
            {"tasks": [{"goal": "g", "output_format": "o", "boundaries": "b"}]}, ctx
        )
        assert result.success
        assert result.metadata.get("dry_run") is True
