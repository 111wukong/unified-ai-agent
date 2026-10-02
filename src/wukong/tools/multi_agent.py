"""The `delegate` tool: the orchestrator's fan-out primitive.

Two decisions are baked into this file rather than left to the model.

**The scale rules are in the description.** A model cannot judge how much
effort a task deserves, and the documented failure mode is a large fan-out
for a small question. So the counts are stated where the model reads them,
and the cap is enforced separately in `MultiAgentRunner.check` -- the
description is a request, the check is the guarantee.

**The tool's own effect class is the maximum its sub-agents may use.** If
`multi_agent.allowed_effects` includes `execute_local`, then `delegate`
declares `execute_local` and the permission engine gates the delegation
itself. A tool that could quietly grant more than it declares would make
the effect class decorative.
"""

from __future__ import annotations

from typing import Any

from wukong.orchestration.multi_agent import (
    AgentDeps,
    MultiAgentRunner,
    build_tasks,
    summarize,
)
from wukong.tools.base import Tool, ToolContext, ToolSpec
from wukong.types import EffectClass, ToolResult

_DESCRIPTION = """\
Split a large, genuinely parallel piece of work across sub-agents, each with
its own context window. Each sub-agent returns a short summary plus a report
file; the full report text is written to disk and never pasted back here.

Use it when the work is BOTH parallel and wider than one context window --
surveying many files, comparing many independent options, gathering from many
separate sources. Do NOT use it for work you can finish yourself in a few
steps. It costs roughly 15x the tokens of answering directly, and for most
coding work it is slower and no better.

Size the fan-out to the task, not to how impressive it would look:
  - one fact to look up                -> do not delegate
  - a direct comparison of two things  -> 2 to 4 sub-agents
  - broad research across many sources -> up to the configured cap

Each sub-agent gets its own budget, so a sub-agent that wanders cannot spend
the rest of this task's budget. Sub-agents cannot write files or run
commands by default: they read and report, and you are the only writer of
shared state. If a sub-agent needs an effect it was not granted it will stop
and ask, and you should report that to the user rather than working around it.

Every brief needs all three required fields. A one-line brief is the
documented cause of several sub-agents performing the identical search.
"""


def _tool_spec(*, effect_class: EffectClass, max_agents: int) -> ToolSpec:
    # No length floors on the three required fields, deliberately. A
    # `minLength` is not a specificity check -- it rejects "a list", which is
    # a perfectly good output format, and a model that is told "string is 6
    # chars, min 8" pads the string rather than improving the brief. The
    # honest requirement is "not empty", which `check()` enforces on the
    # whitespace-only case that a schema cannot see.
    return ToolSpec(
        name="delegate",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max_agents,
                    "description": "One entry per sub-agent. Keep it to the fewest that cover the work.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "minLength": 1,
                                "description": (
                                    "What this sub-agent must find out or produce. Specific and "
                                    "self-contained: it cannot see this conversation."
                                ),
                            },
                            "output_format": {
                                "type": "string",
                                "minLength": 1,
                                "description": (
                                    "The exact shape wanted back, e.g. 'a markdown table with "
                                    "columns X, Y, Z' or 'a list of findings each with a file:line'."
                                ),
                            },
                            "boundaries": {
                                "type": "string",
                                "minLength": 1,
                                "description": (
                                    "What is in scope and what is explicitly not. Naming what to "
                                    "leave out is what stops two sub-agents covering the same ground."
                                ),
                            },
                            "guidance": {
                                "type": "string",
                                "description": (
                                    "Optional. Which files, directories, tools or sources to "
                                    "start from."
                                ),
                            },
                        },
                        "required": ["goal", "output_format", "boundaries"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
        # The delegation itself carries the strongest effect it can grant.
        effect_class=effect_class,
        # Sub-agents are separate tasks with their own budgets and their own
        # event streams. Re-running the fan-out after a crash would duplicate
        # all of that work and every report it produced.
        idempotent=False,
    )


class DelegateTool(Tool):
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        config = deps.settings.multi_agent
        self.spec = _tool_spec(
            effect_class=strongest_effect(config.allowed_effects),
            max_agents=max(1, config.max_agents),
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        tasks = build_tasks(args.get("tasks") or [])
        runner = MultiAgentRunner(
            deps=self.deps, parent_task_id=ctx.task_id, session_id=ctx.session_id
        )
        refusal = runner.check(tasks)
        if refusal:
            return ToolResult(success=False, error=refusal)

        outcomes = await runner.run(tasks)
        ok = sum(1 for o in outcomes if o.ok)
        return ToolResult(
            success=ok > 0,
            output=summarize(outcomes),
            error=None if ok else "every sub-agent failed",
            metadata={
                "requested": len(tasks),
                "completed": ok,
                "children": [o.task_id for o in outcomes if o.task_id],
                "reports": [o.report_path for o in outcomes if o.report_path],
                "tokens": sum(o.usage.total_tokens for o in outcomes),
            },
        )


def strongest_effect(effects: list[EffectClass]) -> EffectClass:
    """The highest-ranked effect in `effects`, or READ_ONLY for an empty list.

    Used so `delegate` cannot declare a weaker effect than the sub-agents it
    is allowed to spawn.
    """
    if not effects:
        return EffectClass.READ_ONLY
    return max(effects, key=lambda effect: effect.rank)


def build_multi_agent_tools(deps: AgentDeps) -> list[Tool]:
    """Empty unless delegation is switched on.

    The tool is *absent*, not merely refusing, when disabled: a capability
    listed in the prompt but always failing costs a model call to discover
    and teaches the model that the tool catalogue lies.
    """
    if not deps.settings.multi_agent.enabled:
        return []
    return [DelegateTool(deps)]
