"""Workflow DSL and runner.

The validation tests carry most of the weight here. The reason to choose
explicit `{{#node.field#}}` references over shared mutable state is that every
reference can be checked before anything runs, so a validator that misses a
case gives up the only advantage the design has.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from unified_agent.orchestration import (
    ExpressionError,
    NodeType,
    Problem,
    WorkflowError,
    WorkflowRunner,
    evaluate,
    parse_workflow,
    resolve,
)

MINIMAL = """
version: "1"
name: minimal
graph:
  nodes:
    - id: start
      type: start
      data:
        variables:
          - variable: goal
    - id: done
      type: end
      data:
        outputs:
          echo: "{{#start.goal#}}"
  edges:
    - source: start
      target: done
"""


def load(text: str):  # noqa: ANN201
    return parse_workflow(text)


def problems_of(text: str) -> list[Problem]:
    with pytest.raises(WorkflowError) as info:
        parse_workflow(text)
    return info.value.problems


def messages(problems: list[Problem]) -> str:
    """Message *and* hint.

    Most of what makes a load-time error useful is the hint -- the list of
    known node types, the fields a node does provide, the effects that exist.
    Joining only the messages tested half the report.
    """
    return " | ".join(f"{p.message} {p.hint}" for p in problems)


# ---------------------------------------------------------------------------


class TestParsing:
    def test_a_minimal_workflow_loads(self) -> None:
        workflow = load(MINIMAL)
        assert workflow.name == "minimal"
        assert set(workflow.nodes) == {"start", "done"}
        assert workflow.start.id == "start"
        assert [n.id for n in workflow.ends] == ["done"]

    def test_edges_accept_dify_snake_case_handles(self) -> None:
        workflow = load(
            MINIMAL.replace(
                "    - source: start\n      target: done",
                "    - source: start\n      source_handle: source\n"
                "      target: done\n      target_handle: target",
            )
        )
        assert workflow.edges[0].source_handle == "source"

    def test_edges_also_accept_camel_case(self) -> None:
        workflow = load(
            MINIMAL.replace(
                "    - source: start\n      target: done",
                "    - source: start\n      sourceHandle: source\n"
                "      target: done\n      targetHandle: target",
            )
        )
        assert workflow.edges[0].target_handle == "target"

    def test_invalid_yaml_is_reported_not_raised_raw(self) -> None:
        problems = problems_of("name: [unclosed")
        assert "invalid YAML" in messages(problems)

    def test_a_non_mapping_document_is_rejected(self) -> None:
        problems = problems_of("- just\n- a\n- list")
        assert "must be a mapping" in messages(problems)

    def test_name_and_version_are_required(self) -> None:
        problems = problems_of("graph:\n  nodes: []\n  edges: []\n")
        text = messages(problems)
        assert "`name` is required" in text
        assert "`version` is required" in text

    def test_graph_is_required(self) -> None:
        problems = problems_of('name: x\nversion: "1"\n')
        assert "`graph` is required" in messages(problems)


class TestNodeValidation:
    def test_unknown_node_type_lists_the_known_ones(self) -> None:
        problems = problems_of(
            MINIMAL.replace("type: start", "type: quantum")
        )
        text = messages(problems)
        assert "unknown node type 'quantum'" in text
        assert "agent" in text and "iteration" in text

    def test_duplicate_node_ids_are_rejected(self) -> None:
        problems = problems_of(
            MINIMAL.replace(
                "    - id: done\n      type: end",
                "    - id: start\n      type: end",
            )
        )
        assert "duplicate node id 'start'" in messages(problems)

    def test_missing_required_field_names_the_node_type(self) -> None:
        problems = problems_of(
            MINIMAL.replace(
                "    - id: done\n      type: end\n      data:\n        outputs:",
                "    - id: work\n      type: agent\n      data:\n        title: nope\n"
                "    - id: done\n      type: end\n      data:\n        outputs:",
            ).replace(
                "    - source: start\n      target: done",
                "    - source: start\n      target: work\n"
                "    - source: work\n      target: done",
            )
        )
        assert "`agent` nodes require `data.goal`" in messages(problems)

    def test_node_id_must_be_a_string(self) -> None:
        problems = problems_of(MINIMAL.replace("id: start", "id: 42"))
        assert "`id` is required" in messages(problems)


class TestEdgeValidation:
    def test_edge_to_unknown_node(self) -> None:
        problems = problems_of(
            MINIMAL.replace("      target: done", "      target: nowhere")
        )
        assert "target node 'nowhere' does not exist" in messages(problems)

    def test_self_loop_is_rejected(self) -> None:
        problems = problems_of(
            MINIMAL.replace("      target: done", "      target: start")
        )
        assert "cannot point at itself" in messages(problems)

    def test_duplicate_edges_are_rejected(self) -> None:
        problems = problems_of(
            MINIMAL.replace(
                "    - source: start\n      target: done",
                "    - source: start\n      target: done\n"
                "    - source: start\n      target: done",
            )
        )
        assert "duplicate edge" in messages(problems)


class TestReferenceValidation:
    def test_reference_to_unknown_node(self) -> None:
        problems = problems_of(MINIMAL.replace("{{#start.goal#}}", "{{#nope.goal#}}"))
        assert "reference to unknown node 'nope'" in messages(problems)

    def test_reference_to_unknown_field_lists_what_exists(self) -> None:
        problems = problems_of(MINIMAL.replace("{{#start.goal#}}", "{{#start.nope#}}"))
        text = messages(problems)
        assert "has no field 'nope'" in text
        assert "goal" in text, "the error should say what the node does provide"

    def test_agent_field_names_are_known(self) -> None:
        """`answer` is the one people will actually write."""
        workflow = load(
            MINIMAL.replace(
                "    - id: done\n      type: end\n      data:\n        outputs:\n"
                '          echo: "{{#start.goal#}}"',
                "    - id: work\n      type: agent\n      data:\n"
                '        goal: "{{#start.goal#}}"\n'
                "    - id: done\n      type: end\n      data:\n        outputs:\n"
                '          echo: "{{#work.answer#}}"',
            ).replace(
                "    - source: start\n      target: done",
                "    - source: start\n      target: work\n"
                "    - source: work\n      target: done",
            )
        )
        assert workflow.nodes["work"].outputs >= {"answer", "steps", "tokens"}

    def test_forward_reference_is_rejected(self) -> None:
        """A reference must point at a node that runs earlier.

        A topological sort could technically satisfy it, but a forward
        reference means the author is describing data flow that does not
        exist -- and catching that at load time is the point of the syntax.
        """
        problems = problems_of(
            MINIMAL.replace(
                '          echo: "{{#start.goal#}}"',
                '          echo: "{{#done.echo#}}"',
            )
        )
        assert "is not upstream of" in messages(problems)


class TestGraphValidation:
    def test_no_start_node(self) -> None:
        problems = problems_of(MINIMAL.replace("type: start", "type: end"))
        assert "no `start` node" in messages(problems)

    def test_two_start_nodes(self) -> None:
        problems = problems_of(
            MINIMAL.replace(
                "    - id: done\n      type: end",
                "    - id: second\n      type: start",
            )
        )
        assert "`start` nodes" in messages(problems)

    def test_no_end_node(self) -> None:
        problems = problems_of(MINIMAL.replace("type: end", "type: start"))
        assert "no `end` node" in messages(problems)

    def test_unreachable_node_is_reported(self) -> None:
        """An orphan never runs and says nothing about it."""
        problems = problems_of(
            MINIMAL.replace(
                "    - id: done\n      type: end",
                "    - id: orphan\n      type: agent\n      data:\n        goal: x\n"
                "    - id: done\n      type: end",
            )
        )
        assert "unreachable node(s): orphan" in messages(problems)

    def test_cycle_is_reported_with_the_nodes_involved(self) -> None:
        text = MINIMAL.replace(
            "    - source: start\n      target: done",
            "    - source: start\n      target: a\n"
            "    - source: a\n      target: b\n"
            "    - source: b\n      target: a",
        ).replace(
            "    - id: done\n      type: end\n      data:\n        outputs:\n"
            '          echo: "{{#start.goal#}}"',
            "    - id: a\n      type: tool\n      data:\n        tool: file_info\n"
            "    - id: b\n      type: tool\n      data:\n        tool: file_info",
        )
        problems = problems_of(text)
        assert "cycle involving" in messages(problems)

    def test_branch_targets_count_as_edges_for_ordering(self) -> None:
        """The bug this guards: only `graph.edges` counted, so a valid
        workflow was reported as having dangling references."""
        workflow = load(
            """
version: "1"
name: branchy
graph:
  nodes:
    - id: start
      type: start
      data:
        variables:
          - variable: mode
    - id: decide
      type: ifelse
      data:
        branches:
          - when: "{{#start.mode#}} == fast"
            target: fast
        else: slow
    - id: fast
      type: end
      data:
        outputs:
          picked: fast
    - id: slow
      type: end
      data:
        outputs:
          picked: slow
  edges:
    - source: start
      target: decide
"""
        )
        order = workflow.topological_order()
        assert order.index("decide") < order.index("fast")
        assert order.index("decide") < order.index("slow")

    def test_a_cycle_through_a_branch_is_detected(self) -> None:
        problems = problems_of(
            """
version: "1"
name: loopy
graph:
  nodes:
    - id: start
      type: start
    - id: decide
      type: ifelse
      data:
        branches:
          - when: "true"
            target: decide
        else: done
    - id: done
      type: end
  edges:
    - source: start
      target: decide
"""
        )
        assert "cycle involving" in messages(problems)

    def test_a_yaml_boolean_node_id_is_explained(self) -> None:
        """`id: yes` arrives as True, which the author cannot see in the file."""
        problems = problems_of(
            MINIMAL.replace("    - id: done\n      type: end",
                            "    - id: yes\n      type: end")
        )
        text = messages(problems)
        assert "parsed as the boolean" in text
        assert 'id: "yes"' in text, "the error should show the fix"


class TestBranchValidation:
    def _branchy(self, else_line: str, when: str = "{{#start.mode#}} == fast") -> str:
        return f"""
version: "1"
name: b
graph:
  nodes:
    - id: start
      type: start
      data:
        variables:
          - variable: mode
            default: fast
    - id: decide
      type: ifelse
      data:
        branches:
          - when: "{when}"
            target: done
{else_line}
    - id: done
      type: end
  edges:
    - source: start
      target: decide
"""

    def test_else_is_required_without_a_catch_all(self) -> None:
        """Without one, a false condition ends the run silently."""
        problems = problems_of(self._branchy(""))
        assert "no `else` and no catch-all branch" in messages(problems)

    def test_an_explicit_else_satisfies_the_rule(self) -> None:
        workflow = load(self._branchy("        else: done"))
        assert workflow.nodes["decide"].data["else"] == "done"

    def test_a_catch_all_branch_satisfies_the_rule(self) -> None:
        """`when: true` is already a catch-all, so an `else` is redundant."""
        text = self._branchy("").replace(
            "            target: done\n",
            "            target: done\n"
            '          - when: "true"\n            target: done\n',
            1,
        )
        assert load(text) is not None

    def test_branch_to_unknown_node(self) -> None:
        problems = problems_of(self._branchy("        else: nowhere"))
        assert "unknown node 'nowhere'" in messages(problems)

    def test_branch_without_when(self) -> None:
        # Drop the `when` line, keeping the branch. An assert on the
        # replacement prevents the test passing vacuously if the fixture's
        # condition ever changes -- which is exactly what happened here.
        base = self._branchy("        else: done")
        needle = '          - when: "{{#start.mode#}} == fast"\n            target: done'
        assert needle in base, "the fixture changed; update this test"
        text = base.replace(needle, "          - target: done")

        problems = problems_of(text)
        assert "`when` is required" in messages(problems)


class TestApprovalValidation:
    def _with_approvals(self, line: str) -> str:
        return MINIMAL.replace('version: "1"', f'version: "1"\n{line}')

    def test_known_effects_are_accepted(self) -> None:
        workflow = load(self._with_approvals("approvals:\n  - execute_local"))
        assert workflow.approvals == ["execute_local"]

    def test_unknown_effect_lists_the_known_ones(self) -> None:
        problems = problems_of(self._with_approvals("approvals:\n  - do_everything"))
        text = messages(problems)
        assert "unknown effect class 'do_everything'" in text
        assert "read_only" in text

    def test_system_admin_cannot_be_granted_by_a_workflow(self) -> None:
        """Same reason the HTTP layer refuses: a data file is not a policy
        authority, and this is the one effect that escapes the fence."""
        problems = problems_of(self._with_approvals("approvals:\n  - system_admin"))
        assert "cannot be granted by a workflow" in messages(problems)

    def test_per_node_approvals_are_validated_too(self) -> None:
        problems = problems_of(
            MINIMAL.replace(
                "    - id: done\n      type: end",
                "    - id: work\n      type: agent\n      data:\n"
                "        goal: x\n        approve:\n          - system_admin\n"
                "    - id: done\n      type: end",
            ).replace(
                "    - source: start\n      target: done",
                "    - source: start\n      target: work\n"
                "    - source: work\n      target: done",
            )
        )
        assert "cannot be granted by a workflow" in messages(problems)


class TestAllProblemsAtOnce:
    def test_every_problem_is_reported_not_just_the_first(self) -> None:
        """A load-time error you fix one at a time is one you stop using."""
        problems = problems_of(
            """
version: "1"
name: broken
graph:
  nodes:
    - id: start
      type: start
    - id: a
      type: agent
      data:
        goal: "{{#ghost.answer#}}"
    - id: b
      type: quantum
    - id: orphan
      type: end
  edges:
    - source: start
      target: a
    - source: a
      target: missing
"""
        )
        text = messages(problems)
        assert "unknown node type 'quantum'" in text
        assert "reference to unknown node 'ghost'" in text
        assert "target node 'missing' does not exist" in text
        assert "unreachable node(s)" in text
        assert len(problems) >= 4


# ---------------------------------------------------------------------------


class TestResolution:
    def test_a_single_reference_preserves_the_type(self) -> None:
        context = {"start": {"count": 7}}
        assert resolve("{{#start.count#}}", context) == 7

    def test_interpolation_produces_a_string(self) -> None:
        context = {"start": {"count": 7}}
        assert resolve("count is {{#start.count#}}", context) == "count is 7"

    def test_unresolved_reference_raises(self) -> None:
        with pytest.raises(ExpressionError, match="has no value yet"):
            resolve("{{#ghost.answer#}}", {})

    def test_unknown_field_lists_what_is_available(self) -> None:
        with pytest.raises(ExpressionError, match="it provides"):
            resolve("{{#start.nope#}}", {"start": {"goal": "x"}})

    def test_nested_indexing(self) -> None:
        context = {"n": {"payload": {"items": [{"name": "first"}]}}}
        assert resolve("{{#n.payload.items.0.name#}}", context) == "first"

    def test_a_single_reference_keeps_none_and_bool_as_they_are(self) -> None:
        """Type preservation is the point: `{{#a.count#}}` must stay an int so
        it can be compared numerically without a cast."""
        assert resolve("{{#s.x#}}", {"s": {"x": None}}) is None
        assert resolve("{{#s.x#}}", {"s": {"x": True}}) is True

    def test_interpolated_none_and_bool_render_readably(self) -> None:
        assert resolve("v={{#s.x#}}", {"s": {"x": None}}) == "v="
        assert resolve("v={{#s.x#}}", {"s": {"x": True}}) == "v=true"


class TestConditions:
    def test_equality_on_strings(self) -> None:
        assert evaluate("{{#a.v#}} == ok", {"a": {"v": "ok"}})
        assert not evaluate("{{#a.v#}} == bad", {"a": {"v": "ok"}})

    def test_quoted_literals(self) -> None:
        assert evaluate("{{#a.v#}} == 'hello world'", {"a": {"v": "hello world"}})

    def test_numeric_comparison(self) -> None:
        assert evaluate("{{#a.n#}} > 3", {"a": {"n": 5}})
        assert not evaluate("{{#a.n#}} > 9", {"a": {"n": 5}})

    def test_numeric_equality_across_types(self) -> None:
        """`"3" == 3` should hold; the author should not need to know which
        side arrived as a string."""
        assert evaluate("{{#a.n#}} == 3", {"a": {"n": "3"}})

    def test_ordered_comparison_of_versions_falls_back_to_strings(self) -> None:
        assert evaluate("{{#a.v#}} >= 1.10", {"a": {"v": "1.9"}})

    def test_contains_and_friends(self) -> None:
        context = {"a": {"v": "the tests passed"}}
        assert evaluate("{{#a.v#}} contains passed", context)
        assert evaluate("{{#a.v#}} startswith the", context)
        assert evaluate("{{#a.v#}} endswith passed", context)
        assert evaluate("{{#a.v#}} matches t.sts", context)

    def test_and_or(self) -> None:
        context = {"a": {"x": 5, "y": "ok"}}
        assert evaluate("{{#a.x#}} > 1 and {{#a.y#}} == ok", context)
        assert evaluate("{{#a.x#}} > 99 or {{#a.y#}} == ok", context)
        assert not evaluate("{{#a.x#}} > 99 and {{#a.y#}} == ok", context)

    def test_bare_true_and_false(self) -> None:
        assert evaluate("true", {})
        assert not evaluate("false", {})

    def test_truthiness_of_a_reference(self) -> None:
        assert evaluate("{{#a.v#}}", {"a": {"v": "something"}})
        assert not evaluate("{{#a.v#}}", {"a": {"v": ""}})
        assert not evaluate("{{#a.v#}}", {"a": {"v": "false"}})

    def test_an_unresolvable_reference_raises_rather_than_being_false(self) -> None:
        """The rule that matters: a branch that quietly never fires is
        indistinguishable from a condition that is genuinely false, so the
        workflow appears to work while skipping the path that mattered."""
        with pytest.raises(ExpressionError):
            evaluate("{{#ghost.v#}} == ok", {})

    def test_empty_condition_is_an_error(self) -> None:
        with pytest.raises(ExpressionError, match="empty condition"):
            evaluate("   ", {})

    def test_bad_regex_is_reported(self) -> None:
        with pytest.raises(ExpressionError, match="invalid pattern"):
            evaluate("{{#a.v#}} matches [unclosed", {"a": {"v": "x"}})


# ---------------------------------------------------------------------------


@pytest.fixture
def runner(settings):  # noqa: ANN001
    """A runner over a real agent, with a scripted model."""
    from tests.conftest import ScriptedModel, ScriptedModels
    from unified_agent.agent.factory import build_agent

    created: list = []

    def _make(script=None):  # noqa: ANN001
        model = ScriptedModel(settings.models["scripted"], script)
        agent = asyncio.run(build_agent(settings=settings, models=ScriptedModels(model)))
        created.append(agent)
        return WorkflowRunner(agent=agent), agent

    yield _make
    for agent in created:
        agent.close()


#: Two branches, one of which runs. Kept as a whole document: building it by
#: string surgery on LINEAR produced malformed YAML, which is a fragile way to
#: write a test and an easy way to test the wrong thing.
BRANCHY = """
version: "1"
name: branchy
approvals:
  - execute_local
  - write_local
graph:
  nodes:
    - id: start
      type: start
      data:
        variables:
          - variable: goal
            default: "list the python files"
    - id: work
      type: agent
      data:
        goal: "{{#start.goal#}}"
    - id: decide
      type: ifelse
      data:
        branches:
          - when: "{{#work.answer#}} contains good"
            target: "yes"
        else: "no"
    # Quoted: unquoted `yes`/`no` are YAML booleans, not strings.
    - id: "yes"
      type: end
      data:
        outputs:
          picked: "yes"
    - id: "no"
      type: end
      data:
        outputs:
          picked: "no"
  edges:
    - source: start
      target: work
    - source: work
      target: decide
"""

LINEAR = """
version: "1"
name: linear
approvals:
  - execute_local
  - write_local
graph:
  nodes:
    - id: start
      type: start
      data:
        variables:
          - variable: goal
            default: "list the python files"
    - id: work
      type: agent
      data:
        goal: "{{#start.goal#}}"
    - id: done
      type: end
      data:
        outputs:
          answer: "{{#work.answer#}}"
          steps: "{{#work.steps#}}"
  edges:
    - source: start
      target: work
    - source: work
      target: done
"""


class TestExecution:
    def test_a_linear_workflow_runs_and_collects_outputs(self, runner) -> None:  # noqa: ANN001
        engine, agent = runner([{"content": "all good"}])
        result = asyncio.run(engine.run(load(LINEAR)))
        assert result.completed, result.error
        assert result.outputs["answer"] == "all good"
        assert set(result.nodes) == {"start", "work", "done"}

    def test_the_run_is_a_task_with_its_own_event_stream(self, runner) -> None:  # noqa: ANN001
        engine, agent = runner([{"content": "done"}])
        result = asyncio.run(engine.run(load(LINEAR)))
        kinds = [e.type.value for e in agent.store.events(result.task_id)]
        assert kinds[0] == "task_created"
        assert "workflow_started" in kinds
        assert "node_started" in kinds
        assert "workflow_completed" in kinds

    def test_the_projection_matches_replay(self, runner) -> None:  # noqa: ANN001
        """`uaa task list` reads the table; `uaa task show` folds the events.
        Two sources of truth that disagree is worse than either alone."""
        from unified_agent.agent.state import replay

        engine, agent = runner([{"content": "done"}])
        result = asyncio.run(engine.run(load(LINEAR)))
        task = agent.store.get_task(result.task_id)
        state = replay(
            agent.store.events(result.task_id),
            task_id=result.task_id,
            session_id=task["session_id"],
        )
        assert task["status"] == state.status.value == "completed"

    def test_an_agent_node_creates_a_child_task(self, runner) -> None:  # noqa: ANN001
        engine, agent = runner([{"content": "done"}])
        result = asyncio.run(engine.run(load(LINEAR)))
        children = agent.store.list_children(result.task_id)
        assert len(children) == 1
        assert children[0]["status"] == "completed"

    def test_ifelse_picks_the_matching_branch(self, runner) -> None:  # noqa: ANN001
        engine, agent = runner([{"content": "all good"}])
        result = asyncio.run(engine.run(load(BRANCHY)))
        assert result.completed, result.error
        assert result.outputs["picked"] == "yes"

    def test_the_untaken_branch_does_not_run(self, runner) -> None:  # noqa: ANN001
        engine, agent = runner([{"content": "nothing to report"}])
        result = asyncio.run(engine.run(load(BRANCHY)))
        assert result.completed, result.error
        assert result.outputs["picked"] == "no"
        assert "yes" not in result.nodes, "the untaken branch ran"

    def test_a_tool_node_runs_without_a_model(self, runner) -> None:  # noqa: ANN001
        workflow = load(
            """
version: "1"
name: tools
graph:
  nodes:
    - id: start
      type: start
    - id: look
      type: tool
      data:
        tool: list_directory
        arguments:
          path: "."
    - id: done
      type: end
      data:
        outputs:
          listing: "{{#look.output#}}"
  edges:
    - source: start
      target: look
    - source: look
      target: done
"""
        )
        engine, agent = runner([])
        result = asyncio.run(engine.run(workflow))
        assert result.completed, result.error
        assert result.nodes["look"].outputs["success"] is True

    def test_an_unknown_tool_names_the_available_ones(self, runner) -> None:  # noqa: ANN001
        workflow = load(
            """
version: "1"
name: badtool
graph:
  nodes:
    - id: start
      type: start
    - id: look
      type: tool
      data:
        tool: teleport
    - id: done
      type: end
  edges:
    - source: start
      target: look
    - source: look
      target: done
"""
        )
        engine, agent = runner([])
        result = asyncio.run(engine.run(workflow))
        assert not result.completed
        assert "unknown tool 'teleport'" in result.nodes["look"].error

    def test_retry_happens_before_giving_up(self, runner) -> None:  # noqa: ANN001
        workflow = load(
            """
version: "1"
name: retries
graph:
  nodes:
    - id: start
      type: start
    - id: look
      type: tool
      data:
        tool: teleport
        error_strategy:
          retry:
            max_retries: 2
    - id: done
      type: end
  edges:
    - source: start
      target: look
    - source: look
      target: done
"""
        )
        engine, agent = runner([])
        result = asyncio.run(engine.run(workflow))
        assert result.nodes["look"].attempts == 3, "one attempt plus two retries"

    def test_on_error_continue_lets_the_rest_run(self, runner) -> None:  # noqa: ANN001
        workflow = load(
            """
version: "1"
name: forgiving
graph:
  nodes:
    - id: start
      type: start
    - id: bad
      type: tool
      data:
        tool: teleport
        error_strategy:
          on_error: continue
    - id: good
      type: tool
      data:
        tool: list_directory
        arguments:
          path: "."
    - id: done
      type: end
  edges:
    - source: start
      target: bad
    - source: bad
      target: good
    - source: good
      target: done
"""
        )
        engine, agent = runner([])
        result = asyncio.run(engine.run(workflow))
        assert result.completed, result.error
        assert result.nodes["bad"].status == "failed"
        assert result.nodes["good"].status == "completed"

    def test_iteration_runs_the_body_per_item(self, runner) -> None:  # noqa: ANN001
        workflow = load(
            """
version: "1"
name: loops
graph:
  nodes:
    - id: start
      type: start
      data:
        variables:
          - variable: items
            default: ["a", "b", "c"]
    - id: each
      type: iteration
      data:
        over: "{{#start.items#}}"
        body: body
    - id: body
      type: tool
      data:
        tool: file_info
        arguments:
          path: "."
    - id: done
      type: end
      data:
        outputs:
          count: "{{#each.count#}}"
  edges:
    - source: start
      target: each
    - source: each
      target: done
"""
        )
        engine, agent = runner([])
        result = asyncio.run(engine.run(workflow))
        assert result.completed, result.error
        assert result.outputs["count"] == 3

    def test_iteration_refuses_an_oversized_list(self, runner) -> None:  # noqa: ANN001
        workflow = load(
            """
version: "1"
name: toobig
graph:
  nodes:
    - id: start
      type: start
      data:
        variables:
          - variable: items
            default: [1, 2, 3, 4, 5, 6]
    - id: each
      type: iteration
      data:
        over: "{{#start.items#}}"
        body: body
        max_items: 3
    - id: body
      type: tool
      data:
        tool: file_info
        arguments:
          path: "."
    - id: done
      type: end
  edges:
    - source: start
      target: each
    - source: each
      target: done
"""
        )
        engine, agent = runner([])
        result = asyncio.run(engine.run(workflow))
        assert not result.completed
        assert "above max_items" in result.nodes["each"].error

    def test_a_pause_names_the_effect_and_where_to_allow_it(self, runner) -> None:  # noqa: ANN001
        """'ended waiting_confirmation' with no reason is a dead end."""
        workflow = load(
            LINEAR.replace("approvals:\n  - execute_local\n  - write_local\n", "")
        )
        engine, agent = runner(
            [{"tool_calls": [{"name": "run_tests", "arguments": {}}]}, {"content": "done"}]
        )
        result = asyncio.run(engine.run(workflow))
        assert not result.completed
        error = result.nodes["work"].error
        assert "needs approval" in error
        assert "approve:" in error, "the error should say how to fix it"

    def test_a_declared_approval_lets_it_through(self, runner) -> None:  # noqa: ANN001
        engine, agent = runner(
            [{"tool_calls": [{"name": "run_tests", "arguments": {}}]}, {"content": "done"}]
        )
        result = asyncio.run(engine.run(load(LINEAR)))
        assert result.completed, result.error

    def test_a_failed_run_explains_itself(self, runner) -> None:  # noqa: ANN001
        """A summary reading '1 node(s) completed' with status failed is worse
        than no summary: it looks like a partial success."""
        workflow = load(
            """
version: "1"
name: fails
graph:
  nodes:
    - id: start
      type: start
    - id: bad
      type: tool
      data:
        tool: teleport
    - id: done
      type: end
  edges:
    - source: start
      target: bad
    - source: bad
      target: done
"""
        )
        engine, agent = runner([])
        result = asyncio.run(engine.run(workflow))
        assert not result.completed
        assert "bad" in result.summary()
        assert result.failures()[0].node_id == "bad"

    def test_the_code_node_goes_through_the_shell_not_in_process(self, runner) -> None:  # noqa: ANN001
        """A workflow file is data; it must not be able to reach the runtime's
        memory. Running through `run_command` means the sandbox and the
        command guard apply."""
        workflow = load(
            """
version: "1"
name: code
approvals:
  - execute_local
graph:
  nodes:
    - id: start
      type: start
    - id: run
      type: code
      data:
        language: python
        code: "python3 -c 'print(6*7)'"
    - id: done
      type: end
      data:
        outputs:
          out: "{{#run.output#}}"
  edges:
    - source: start
      target: run
    - source: run
      target: done
"""
        )
        engine, agent = runner([])
        result = asyncio.run(engine.run(workflow))
        assert result.completed, result.error
        assert "42" in str(result.outputs["out"])

    def test_a_cycle_cannot_be_constructed_at_load_time(self) -> None:
        """Belt and braces: the runner assumes a DAG, and validation is what
        makes that safe."""
        problems = problems_of(
            MINIMAL.replace(
                "    - source: start\n      target: done",
                "    - source: start\n      target: done\n"
                "    - source: done\n      target: start",
            )
        )
        assert problems, "a cycle must not load"


class TestSampleWorkflow:
    def test_the_shipped_example_is_valid(self) -> None:
        path = Path(__file__).resolve().parents[1] / "workflows" / "triage-and-fix.yaml"
        assert path.is_file(), "the sample workflow should ship with the repo"
        workflow = parse_workflow(path.read_text(encoding="utf-8"), source_path=path)
        assert workflow.name == "triage-and-fix"
        assert {n.type for n in workflow.nodes.values()} >= {
            NodeType.START,
            NodeType.AGENT,
            NodeType.IFELSE,
            NodeType.END,
        }
        assert "execute_local" in workflow.approvals
