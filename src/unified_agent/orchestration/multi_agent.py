"""Orchestrator-worker delegation.

The shape follows Anthropic's published orchestrator-worker findings, and
so do the four rules that make it work rather than merely run:

1. **Scale rules live in the prompt.** A model cannot judge how much effort
   a task deserves. The classic early failure is fifty sub-agents for a
   question that needed one, so the counts are stated in the tool
   description the model reads.
2. **Delegation is a contract, not a sentence.** Each sub-agent gets a goal,
   an output format, tool/source guidance and explicit boundaries. A
   one-line brief produces several sub-agents doing the same search.
3. **Sub-agent output goes to disk, not into the conversation.** The
   orchestrator receives a path plus a bounded summary. Pasting every full
   report into the parent's context is the "telephone game" that makes a
   multi-agent run worse than a single agent.
4. **The orchestrator is the only writer of shared state.** Sub-agents get
   read-only effects by default, so they cannot trample each other's edits
   or the orchestrator's plan.

A sub-agent is a full `AgentRuntime.run()` with `parent_task_id` set, which
is where the real benefit comes from: **context isolation for free**. It is
also why this needs no new execution kernel -- budgets, permissions, the
event log and resume all apply to a sub-agent because it is an ordinary
task. `uaa task events <child_id>` shows what any one of them did.

Kept synchronous-by-default with a bounded fan-out. The interface is
`async`, so raising `parallelism` is the only change needed to go wider.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from unified_agent.agent.state import TaskStatus
from unified_agent.observability.events import EventType
from unified_agent.types import EffectClass, TokenUsage


@dataclass
class SubAgentTask:
    """One delegation. All four fields are the contract; `guidance` is the
    one that is optional, because a sub-agent with clear boundaries and an
    output format rarely needs its sources spelled out."""

    goal: str
    output_format: str
    boundaries: str
    guidance: str = ""

    def render(self) -> str:
        """The brief the sub-agent actually receives as its goal."""
        lines = [self.goal.strip(), "", "## Output format", self.output_format.strip()]
        if self.guidance.strip():
            lines += ["", "## Where to look", self.guidance.strip()]
        lines += [
            "",
            "## Boundaries",
            self.boundaries.strip(),
            "",
            "Write your findings to your final answer. The orchestrator will save",
            "the full text to a file and read only a summary of it, so put the",
            "substance in the answer rather than describing what you would do.",
        ]
        return "\n".join(lines)


@dataclass
class SubAgentOutcome:
    index: int
    goal: str
    task_id: str
    status: str
    summary: str = ""
    report_path: str | None = None
    steps: int = 0
    model_calls: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: str | None = None
    pending_tool: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == TaskStatus.COMPLETED.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "goal": self.goal,
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
            "report_path": self.report_path,
            "steps": self.steps,
            "model_calls": self.model_calls,
            "tokens": self.usage.total_tokens,
            "error": self.error,
            "pending_tool": self.pending_tool,
        }


@dataclass
class AgentDeps:
    """The pieces a sub-agent needs, without the parent's runtime.

    A fresh `AgentRuntime` is built per sub-agent because the runtime is
    stateful -- it holds the current model and the context builder, so two
    runs sharing one instance would overwrite each other's context. What is
    shared is the *infrastructure*: one store, one tool registry, one
    permission engine, one model registry. That is what makes a sub-agent
    cheap to start and what keeps its permissions identical to the parent's.
    """

    settings: Any
    store: Any
    registry: Any
    engine: Any
    models: Any
    skills: Any = None
    bus: Any = None
    memory: Any = None


class MultiAgentRunner:
    """Fans a delegation out over child tasks and collects the reports."""

    def __init__(self, *, deps: AgentDeps, parent_task_id: str, session_id: str) -> None:
        self.deps = deps
        self.parent_task_id = parent_task_id
        self.session_id = session_id
        self.settings = deps.settings
        self.config = deps.settings.multi_agent
        self.store = deps.store

    # -- entry point ------------------------------------------------------
    async def run(
        self, tasks: list[SubAgentTask], *, model_alias: str | None = None
    ) -> list[SubAgentOutcome]:
        """Run every task, at most `parallelism` at a time.

        Returns in the caller's order, not in completion order: the
        orchestrator wrote the briefs in an order that means something, and
        a result list shuffled by whichever sub-agent finished first is a
        result list the model then has to re-sort from the task ids.
        """
        if not tasks:
            return []
        width = max(1, min(self.config.parallelism, len(tasks)))
        semaphore = asyncio.Semaphore(width)

        async def one(index: int, task: SubAgentTask) -> SubAgentOutcome:
            async with semaphore:
                return await self._run_one(index, task, model_alias=model_alias)
        outcomes = await asyncio.gather(
            *(one(i, t) for i, t in enumerate(tasks)), return_exceptions=True
        )

        results: list[SubAgentOutcome] = []
        for index, outcome in enumerate(outcomes):
            if isinstance(outcome, BaseException):
                # gather(return_exceptions=True) keeps one broken sub-agent
                # from cancelling its siblings. The failure still has to be
                # reported as a result, not swallowed.
                results.append(
                    SubAgentOutcome(
                        index=index,
                        goal=tasks[index].goal,
                        task_id="",
                        status=TaskStatus.FAILED.value,
                        error=f"{type(outcome).__name__}: {outcome}"[:400],
                    )
                )
            else:
                results.append(outcome)
        return results

    # -- one sub-agent ----------------------------------------------------
    async def _run_one(
        self, index: int, task: SubAgentTask, *, model_alias: str | None
    ) -> SubAgentOutcome:
        from unified_agent.agent.runtime import AgentRuntime

        runtime = AgentRuntime(
            settings=self.settings,
            store=self.store,
            registry=self.deps.registry,
            engine=self.deps.engine,
            models=self.deps.models,
            skills=self.deps.skills,
            # Deliberately no `on_progress` and no stream callback: a
            # sub-agent's output would interleave with the orchestrator's in
            # a single terminal or SSE stream and be unattributable. Its
            # events are on its own task_id, which is where to read them.
            bus=self.deps.bus,
            memory=self.deps.memory,
        )
        # `multi_agent.model` wins over the orchestrator's alias, so the
        # planner can be a strong model while the readers are cheap ones.
        alias = self.config.model or model_alias
        approved = list(self.config.allowed_effects)
        self.store.append(
            self.parent_task_id,
            EventType.SUBAGENT_STARTED,
            {"index": index, "goal": task.goal[:300], "approved_effects": [e.value for e in approved]},
        )

        try:
            result = await runtime.run(
                task.render(),
                session_id=self.session_id,
                model_alias=alias,
                approved_effects=approved,
                max_steps=self.config.max_steps_per_agent,
                parent_task_id=self.parent_task_id,
                # No reflection per sub-agent: five sub-agents would pay five
                # extra extraction calls, and the orchestrator reflects once
                # over the combined result anyway.
                reflect=False,
            )
        except Exception as exc:  # noqa: BLE001 - a sub-agent failure is data
            outcome = SubAgentOutcome(
                index=index,
                goal=task.goal,
                task_id="",
                status=TaskStatus.FAILED.value,
                error=f"{type(exc).__name__}: {exc}"[:400],
            )
            self.store.append(
                self.parent_task_id,
                EventType.SUBAGENT_FAILED,
                outcome.as_dict(),
            )
            return outcome

        report_path = self._write_report(index, task, result)
        outcome = SubAgentOutcome(
            index=index,
            goal=task.goal,
            task_id=result.task_id,
            status=result.status,
            summary=_clip(result.answer or result.error or "", self.config.max_summary_chars),
            report_path=report_path,
            steps=result.steps,
            model_calls=result.model_calls,
            usage=result.usage,
            error=result.error,
            pending_tool=(
                result.pending_confirmation.tool if result.pending_confirmation else None
            ),
        )
        self.store.append(
            self.parent_task_id,
            EventType.SUBAGENT_COMPLETED if outcome.ok else EventType.SUBAGENT_FAILED,
            outcome.as_dict(),
        )
        return outcome

    # -- report on disk ---------------------------------------------------
    def _write_report(self, index: int, task: SubAgentTask, result: Any) -> str | None:
        """Persist the full body. The orchestrator gets the path, not the text.

        Written by the runner rather than by the sub-agent because sub-agents
        are read-only by default -- and because a sub-agent that could write
        to the workspace is a sub-agent that can collide with its siblings.
        """
        body = result.answer or ""
        if not body and not result.error:
            return None
        directory = self.settings.artifact_dir / "subagents" / self.parent_task_id
        directory.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", task.goal.lower()).strip("-")[:40] or "task"
        path = directory / f"{index:02d}-{slug}.md"
        lines = [
            f"# Sub-agent {index}: {task.goal}",
            "",
            f"- status: {result.status}",
            f"- task: {result.task_id}",
            f"- steps: {result.steps}  model_calls: {result.model_calls}  "
            f"tokens: {result.usage.total_tokens}",
            "",
            "## Output format requested",
            "",
            task.output_format,
            "",
            "## Boundaries",
            "",
            task.boundaries,
            "",
            "## Report",
            "",
            body or f"(no answer; error: {result.error})",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)

    # -- policy -----------------------------------------------------------
    def check(self, tasks: list[SubAgentTask]) -> str | None:
        """Return a refusal message, or None if the delegation is allowed."""
        if not self.config.enabled:
            return (
                "delegation is disabled. Enable it with "
                "`uaa config set multi_agent.enabled true`."
            )
        if EffectClass.SYSTEM_ADMIN in self.config.allowed_effects:
            # Same reasoning as the workflow approval layer and the HTTP
            # layer: a data file is not a policy authority. `system_admin`
            # is a deliberate act, not a line in a TOML.
            return (
                "multi_agent.allowed_effects may not include system_admin. "
                "A configuration file is not a policy authority, and sub-agents "
                "run unattended."
            )
        if len(tasks) > self.config.max_agents:
            return (
                f"{len(tasks)} sub-agents requested but multi_agent.max_agents is "
                f"{self.config.max_agents}. Split the work into fewer, larger tasks "
                "or raise the cap deliberately -- a large fan-out for a small "
                "question is the standard way this goes wrong."
            )
        for i, task in enumerate(tasks):
            for name, value in (
                ("goal", task.goal),
                ("output_format", task.output_format),
                ("boundaries", task.boundaries),
            ):
                if not str(value or "").strip():
                    return (
                        f"tasks[{i}].{name} is empty. Every sub-agent needs a goal, "
                        "an output format and boundaries -- a vague brief produces "
                        "several sub-agents doing the same work."
                    )
        return None


def summarize(outcomes: list[SubAgentOutcome]) -> str:
    """The block the orchestrator reads. Paths and summaries, not bodies."""
    if not outcomes:
        return "no sub-agents ran"
    lines: list[str] = []
    ok = sum(1 for o in outcomes if o.ok)
    lines.append(f"{ok}/{len(outcomes)} sub-agent(s) completed.")
    for outcome in outcomes:
        lines.append("")
        lines.append(f"### [{outcome.index}] {outcome.goal[:200]}")
        lines.append(f"- status: {outcome.status}  steps: {outcome.steps}")
        if outcome.report_path:
            lines.append(f"- full report: {outcome.report_path}")
        if outcome.pending_tool:
            lines.append(
                f"- BLOCKED awaiting approval for `{outcome.pending_tool}`; the user "
                f"must approve task {outcome.task_id} before it can continue"
            )
        if outcome.error and not outcome.summary:
            lines.append(f"- error: {outcome.error}")
        if outcome.summary:
            lines.append("")
            lines.append(outcome.summary)
    lines.append("")
    lines.append(
        "Read the report files above with `read_file` if a summary is not enough. "
        "Do not delegate the same question again."
    )
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[... truncated at {limit} chars; see the report file]"


def build_tasks(raw: list[dict[str, Any]]) -> list[SubAgentTask]:
    return [
        SubAgentTask(
            goal=str(item.get("goal") or ""),
            output_format=str(item.get("output_format") or ""),
            boundaries=str(item.get("boundaries") or ""),
            guidance=str(item.get("guidance") or ""),
        )
        for item in raw
        if isinstance(item, dict)
    ]


__all__ = [
    "AgentDeps",
    "MultiAgentRunner",
    "SubAgentOutcome",
    "SubAgentTask",
    "build_tasks",
    "summarize",
]
