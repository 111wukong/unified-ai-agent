"""Planner validation, and the repair loop that consumes its errors.

The planner has a fixed number of repair attempts, so a rejection message is
not just diagnostics -- it is the instruction the model has to act on. A
message that states a constraint without stating the constraint is one the
model can fail twice, and then the whole task dies before it starts.

Found by running a real task: the model returned a plan, the description was
under the threshold, the message said "too short" without saying how short,
and three attempts later the task ended with "planning failed".
"""

from __future__ import annotations

import json

import pytest

from unified_agent.agent.planner import Planner
from unified_agent.config import ModelSpec
from unified_agent.models.mock import MockModel

TOOLS = ["read_file", "run_tests", "apply_patch"]


@pytest.fixture
def planner(settings) -> Planner:  # noqa: ANN001
    return Planner(model=MockModel(ModelSpec(provider="mock", model="mock-react")), settings=settings)


def plan_of(*descriptions: str) -> str:
    return json.dumps({"steps": [{"description": d} for d in descriptions]})


class TestValidationMessages:
    def test_an_empty_response_is_not_reported_as_a_malformed_plan(
        self, planner: Planner
    ) -> None:
        """For a reasoning model, an empty response is the ordinary shape of
        "the output budget went on thinking" -- the same failure the main loop
        reports explicitly. Calling it a bad plan sends the repair loop after
        the wrong problem."""
        with pytest.raises(ValueError) as caught:
            planner._parse("", TOOLS)
        message = str(caught.value)
        assert "no content" in message
        assert "max_output_tokens" in message, "it must name the thing to change"

    def test_a_short_description_states_the_length_and_the_minimum(
        self, planner: Planner
    ) -> None:
        with pytest.raises(ValueError) as caught:
            planner._parse(plan_of("go"), TOOLS)
        message = str(caught.value)
        assert "steps[0].description" in message
        assert "2 character(s)" in message, "say what was received"
        assert "at least 4" in message, "and what is required"

    def test_a_missing_steps_key_shows_the_expected_shape(self, planner: Planner) -> None:
        with pytest.raises(ValueError) as caught:
            planner._parse(json.dumps({"plan": []}), TOOLS)
        assert '"steps"' in str(caught.value)
        assert "non-empty array" in str(caught.value)

    def test_a_non_object_step_names_the_type_it_got(self, planner: Planner) -> None:
        with pytest.raises(ValueError) as caught:
            planner._parse(json.dumps({"steps": ["run the tests"]}), TOOLS)
        message = str(caught.value)
        assert "steps[0] must be an object" in message
        assert "str" in message

    def test_an_unknown_tool_lists_what_is_allowed(self, planner: Planner) -> None:
        """Already the shape the others were brought up to: name the set."""
        payload = json.dumps(
            {"steps": [{"description": "run the tests", "expected_tools": ["nope"]}]}
        )
        with pytest.raises(ValueError) as caught:
            planner._parse(payload, TOOLS)
        message = str(caught.value)
        assert "nope" in message
        for tool in TOOLS:
            assert tool in message

    def test_a_valid_plan_parses(self, planner: Planner) -> None:
        steps = planner._parse(
            json.dumps(
                {
                    "steps": [
                        {"description": "run the test suite", "expected_tools": ["run_tests"]},
                        {"description": "fix the failing module", "expected_tools": ["apply_patch"]},
                    ]
                }
            ),
            TOOLS,
        )
        assert [s.id for s in steps] == ["step_1", "step_2"]
        assert steps[0].expected_tools == ["run_tests"]


class TestRepairLoop:
    async def test_the_rejection_reason_is_what_the_model_sees_next(
        self, settings, session_id: str  # noqa: ANN001
    ) -> None:
        """The loop is only as good as the message it feeds back."""
        from unified_agent.agent.state import AgentState
        from unified_agent.types import ModelResponse

        seen: list[str] = []

        class Recording(MockModel):
            def __init__(self, spec, replies: list[str]) -> None:  # noqa: ANN001
                super().__init__(spec)
                self.replies = list(replies)

            async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
                seen.append("\n".join(m.content or "" for m in messages))
                return ModelResponse(content=self.replies.pop(0), model=self.spec.model)

        planner = Planner(
            model=Recording(
                ModelSpec(provider="mock", model="mock-react"),
                [plan_of("go"), plan_of("run the test suite")],
            ),
            settings=settings,
        )
        state = AgentState(task_id="t", session_id=session_id, goal="fix the tests")
        steps = await planner.plan(state, tool_names=TOOLS)

        assert len(steps) == 1
        assert len(seen) == 2
        # The second request has to carry the reason, or the model is guessing.
        assert "at least 4" in seen[1]
        assert "That plan was rejected" in seen[1]

    async def test_giving_up_reports_the_last_reason(
        self, settings, session_id: str  # noqa: ANN001
    ) -> None:
        from unified_agent.agent.state import AgentState
        from unified_agent.errors import ModelError
        from unified_agent.types import ModelResponse

        class AlwaysBad(MockModel):
            async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
                return ModelResponse(content=plan_of("go"), model=self.spec.model)

        settings.agent.planner_repair_attempts = 1
        planner = Planner(
            model=AlwaysBad(ModelSpec(provider="mock", model="mock-react")), settings=settings
        )
        state = AgentState(task_id="t", session_id=session_id, goal="fix the tests")

        with pytest.raises(ModelError) as caught:
            await planner.plan(state, tool_names=TOOLS)
        message = str(caught.value)
        assert "2 attempt(s)" in message
        assert "at least 4" in message, "the final error must carry the reason, not just a count"
