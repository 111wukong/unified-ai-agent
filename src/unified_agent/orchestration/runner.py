"""Workflow execution.

The runner is a *driver over the existing runtime*, not a parallel system.
That is the design decision everything else follows from:

* A workflow run **is a task**, so it inherits the event log, budgets,
  cancellation and the A2A task mapping without any of them being
  reimplemented.
* An `agent` node is a full `AgentRuntime.run()` in the same session, with
  `parent_task_id` set. So it gets the same permission engine, the same
  context budgeting, the same resume semantics -- and `uaa task show` can
  walk from the workflow down into each node's run.
* A `code` node goes through the **shell tool**, which means the sandbox and
  the command guard apply. Dify runs its code node in-process; here a
  workflow file is data, and data should not be able to reach into the
  runtime's memory.

Node execution is once-per-run. A DAG reached by two paths does not re-run
the shared node, which matches what a reader of the graph expects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from unified_agent.errors import ToolError
from unified_agent.observability.events import EventType
from unified_agent.orchestration.expressions import (
    ExpressionError,
    evaluate,
    resolve,
    resolve_deep,
)
from unified_agent.orchestration.workflow import Node, NodeType, Workflow
from unified_agent.storage.store import new_id
from unified_agent.tools.base import ToolContext
from unified_agent.types import EffectClass

ProgressHook = Callable[[str, dict[str, Any]], None]


@dataclass
class NodeResult:
    node_id: str
    node_type: str
    status: str = "pending"  # completed | failed | skipped
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    attempts: int = 1
    duration_ms: int = 0
    child_task_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node_id,
            "type": self.node_type,
            "status": self.status,
            "error": self.error,
            "attempts": self.attempts,
            "durationMs": self.duration_ms,
            "childTaskId": self.child_task_id,
        }


@dataclass
class WorkflowResult:
    workflow: str
    task_id: str
    session_id: str
    status: str = "running"
    outputs: dict[str, Any] = field(default_factory=dict)
    nodes: dict[str, NodeResult] = field(default_factory=dict)
    error: str = ""
    duration_s: float = 0.0

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def summary(self) -> str:
        """One line that always says *why* when the status is not completed.

        A failed run whose summary reads "1 node(s) completed" is worse than
        no summary: it looks like a partial success and gives the reader
        nothing to act on.
        """
        done = sum(1 for r in self.nodes.values() if r.status == "completed")
        failed = [r for r in self.nodes.values() if r.status == "failed"]
        parts = [self.status, f"{done} node(s) completed"]
        if self.error:
            parts.append(self.error[:200])
        elif failed:
            first = failed[0]
            parts.append(f"{first.node_id} failed: {first.error[:160]}")
        return "  ".join(parts)

    def failures(self) -> list[NodeResult]:
        return [r for r in self.nodes.values() if r.status == "failed"]


class WorkflowRunner:
    def __init__(
        self,
        *,
        agent: Any,
        on_progress: ProgressHook | None = None,
    ) -> None:
        self.agent = agent
        self.on_progress = on_progress

    # -- progress ---------------------------------------------------------
    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(kind, payload)
            except Exception:  # noqa: BLE001 - a UI callback must not break a run
                pass

    # -- entry point ------------------------------------------------------
    async def run(
        self,
        workflow: Workflow,
        *,
        inputs: dict[str, Any] | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        model_alias: str | None = None,
    ) -> WorkflowResult:
        store = self.agent.store
        settings = self.agent.settings
        session = session_id or store.ensure_session(
            name=f"workflow:{workflow.name}",
            working_dir=str(settings.workspace),
            model_alias=model_alias or settings.default_model,
        )
        run_task = store.create_task(
            session_id=session,
            goal=f"workflow:{workflow.name}",
            task_id=task_id,
        )
        started = time.monotonic()
        result = WorkflowResult(workflow=workflow.name, task_id=run_task, session_id=session)

        # A workflow run is a task, so it uses the same status vocabulary and
        # the same projection rule: emit the transition, then let `replay`
        # fold it. Writing `tasks.status` directly would make the projection
        # and the event stream two sources of truth, and `uaa task show` reads
        # the event stream.
        store.append(
            run_task, EventType.STATE_TRANSITION, {"from": "pending", "to": "running"}
        )
        store.append(
            run_task,
            EventType.WORKFLOW_STARTED,
            {
                "workflow": workflow.name,
                "version": workflow.version,
                "nodes": list(workflow.nodes),
                "inputs": inputs or {},
            },
        )
        self._emit("workflow_started", {"workflow": workflow.name, "task_id": run_task})

        context: dict[str, dict[str, Any]] = {}
        pending: list[str] = [workflow.start.id]
        declared_inputs = {
            str(item.get("name")): (inputs or {}).get(str(item.get("name")), item.get("default"))
            for item in workflow.inputs
            if item.get("name")
        }
        # A `start` node's own `variables` are the other way to declare inputs.
        for variable in workflow.start.data.get("variables") or []:
            if isinstance(variable, dict) and variable.get("variable"):
                name = str(variable["variable"])
                declared_inputs.setdefault(name, variable.get("default"))

        try:
            while pending:
                node_id = pending.pop(0)
                if node_id in result.nodes:
                    continue  # once per run
                node = workflow.nodes.get(node_id)
                if node is None:  # pragma: no cover - validation rejects this
                    continue

                if node.type is NodeType.START:
                    outputs = {**declared_inputs}
                else:
                    outputs = {}

                node_result = await self._execute_node(
                    node,
                    workflow=workflow,
                    context=context,
                    inputs=declared_inputs,
                    run_task=run_task,
                    session_id=session,
                    model_alias=model_alias,
                )
                result.nodes[node_id] = node_result
                if node_result.status == "completed":
                    context[node_id] = {**outputs, **node_result.outputs}
                else:
                    context[node_id] = {"status": node_result.status, "error": node_result.error}

                store.append(
                    run_task,
                    EventType.NODE_COMPLETED
                    if node_result.status == "completed"
                    else EventType.NODE_FAILED,
                    node_result.as_dict(),
                )
                self._emit(
                    "node",
                    {
                        "node": node_id,
                        "type": node.type.value,
                        "status": node_result.status,
                        "error": node_result.error,
                        "duration_ms": node_result.duration_ms,
                    },
                )

                if node_result.status == "failed":
                    on_error = node.error_strategy.on_error
                    if on_error == "continue":
                        # Carry on along the normal edges. Returning early here
                        # would mean the successor never gets queued, so
                        # "continue" would behave like "stop" -- and the node
                        # after a tolerated failure is exactly the one the
                        # author wanted to reach.
                        pending.extend(self._next_nodes(workflow, node, context))
                        continue
                    if on_error == "fail":
                        result.status = "failed"
                        result.error = f"node {node_id} failed: {node_result.error}"
                        break
                    if on_error in workflow.nodes:
                        pending.append(on_error)
                        continue
                    result.status = "failed"
                    result.error = (
                        f"node {node_id} failed and on_error points at unknown "
                        f"node {on_error!r}"
                    )
                    break

                pending.extend(self._next_nodes(workflow, node, context))

            if result.status == "running":
                result.status = "completed"
                result.outputs = self._collect_outputs(
                    workflow, context, set(result.nodes)
                )
        except ExpressionError as exc:
            result.status = "failed"
            result.error = f"expression error: {exc}"
        except Exception as exc:  # noqa: BLE001 - a run must always record its end
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"

        result.duration_s = time.monotonic() - started
        store.append(
            run_task, EventType.STATE_TRANSITION, {"from": "running", "to": result.status}
        )
        store.append(
            run_task,
            EventType.WORKFLOW_COMPLETED if result.completed else EventType.WORKFLOW_FAILED,
            {
                "status": result.status,
                "error": result.error,
                "outputs": result.outputs,
                "nodes": {k: v.as_dict() for k, v in result.nodes.items()},
            },
        )
        self._persist_projection(run_task, session, result)
        self._emit(
            "workflow_finished",
            {"workflow": workflow.name, "status": result.status, "task_id": run_task},
        )
        return result

    def _persist_projection(self, run_task: str, session_id: str, result: WorkflowResult) -> None:
        """Write the list-view projection by folding the events.

        Derived, not assembled separately: `uaa task list` reads this table,
        and a projection computed by a second code path is a projection that
        can disagree with the event log.
        """
        from unified_agent.agent.state import replay

        store = self.agent.store
        state = replay(store.events(run_task), task_id=run_task, session_id=session_id)
        store.save_projection(
            run_task,
            status=state.status.value,
            state={"workflow": result.workflow, "outputs": result.outputs},
            steps_used=len(result.nodes),
            tokens_in=state.usage.prompt_tokens,
            tokens_out=state.usage.completion_tokens,
            cost_usd=state.usage.cost_usd,
            result={"outputs": result.outputs, "nodes": len(result.nodes)},
            error=result.error or None,
        )

    # -- graph walking ----------------------------------------------------
    def _next_nodes(
        self, workflow: Workflow, node: Node, context: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Where to go after `node`.

        `context` is the *whole run's* outputs, not this node's: a branch
        condition references upstream nodes (`{{#analyse.answer#}}`), and
        passing the node's own outputs here makes every such reference
        unresolvable.
        """
        if node.type is NodeType.END:
            return []
        if node.type is NodeType.IFELSE:
            return [self._pick_branch(node, context)]
        return [edge.target for edge in workflow.outgoing(node.id)]

    def _pick_branch(self, node: Node, context: dict[str, dict[str, Any]]) -> str:
        """First matching branch wins; `else` is the fallback.

        Validation guarantees an `else` or a catch-all exists, so this cannot
        fall through to "nowhere" -- a workflow that silently stops is
        indistinguishable from one that finished.
        """
        for branch in node.data.get("branches") or []:
            if not isinstance(branch, dict):
                continue
            condition = str(branch.get("when") or "")
            if condition.strip().lower() == "true" or evaluate(condition, context):
                return str(branch["target"])
        fallback = node.data.get("else")
        if fallback:
            return str(fallback)
        raise ExpressionError(
            f"no branch matched in {node.id} and there is no else; "
            "validation should have rejected this"
        )

    def _collect_outputs(
        self,
        workflow: Workflow,
        context: dict[str, dict[str, Any]],
        executed: set[str],
    ) -> dict[str, Any]:
        """Merge the outputs of the end nodes that actually ran.

        Only those: a literal in an unexecuted `end` node -- `picked: "no"` --
        is a valid template that resolves without touching the context, so
        merging every end node lets a branch that never ran overwrite the
        result of the one that did. The workflow would then report the wrong
        answer while every node reported success.
        """
        merged: dict[str, Any] = {}
        for end in workflow.ends:
            if end.id not in executed:
                continue
            declared = end.data.get("outputs")
            if not isinstance(declared, dict):
                continue
            for name, template in declared.items():
                if isinstance(template, str):
                    merged[str(name)] = resolve(template, context)
                else:
                    merged[str(name)] = resolve_deep(template, context)
        return merged

    # -- node execution ---------------------------------------------------
    async def _execute_node(
        self,
        node: Node,
        *,
        workflow: Workflow,
        context: dict[str, dict[str, Any]],
        inputs: dict[str, Any],
        run_task: str,
        session_id: str,
        model_alias: str | None,
    ) -> NodeResult:
        result = NodeResult(node_id=node.id, node_type=node.type.value)
        started = time.monotonic()
        attempts = max(1, node.error_strategy.max_retries + 1)

        for attempt in range(1, attempts + 1):
            result.attempts = attempt
            try:
                outputs = await self._run_once(
                    node,
                    workflow=workflow,
                    context=context,
                    inputs=inputs,
                    run_task=run_task,
                    session_id=session_id,
                    model_alias=model_alias,
                    result=result,
                )
            except (ExpressionError, ToolError, ValueError, RuntimeError) as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts:
                    if node.error_strategy.retry_delay_s:
                        import asyncio

                        await asyncio.sleep(node.error_strategy.retry_delay_s)
                    continue
                result.status = "failed"
                result.duration_ms = int((time.monotonic() - started) * 1000)
                return result

            result.status = "completed"
            result.outputs = outputs
            result.error = ""
            result.duration_ms = int((time.monotonic() - started) * 1000)
            return result

        result.status = "failed"  # pragma: no cover - the loop always returns
        return result

    async def _run_once(
        self,
        node: Node,
        *,
        workflow: Workflow,
        context: dict[str, dict[str, Any]],
        inputs: dict[str, Any],
        run_task: str,
        session_id: str,
        model_alias: str | None,
        result: NodeResult,
    ) -> dict[str, Any]:
        store = self.agent.store
        store.append(run_task, EventType.NODE_STARTED, {"node": node.id, "type": node.type.value})

        if node.type is NodeType.START:
            return dict(inputs)

        if node.type is NodeType.END:
            return {}

        if node.type is NodeType.IFELSE:
            branch = self._pick_branch(node, context)
            return {"branch": branch}

        if node.type is NodeType.AGENT:
            return await self._run_agent_node(
                node,
                workflow=workflow,
                context=context,
                run_task=run_task,
                session_id=session_id,
                model_alias=model_alias,
                result=result,
            )

        if node.type is NodeType.TOOL:
            return await self._run_tool_node(node, context=context, run_task=run_task)

        if node.type is NodeType.CODE:
            return await self._run_code_node(node, context=context, run_task=run_task)

        if node.type is NodeType.ITERATION:
            return await self._run_iteration_node(
                node,
                workflow=workflow,
                context=context,
                run_task=run_task,
                session_id=session_id,
                model_alias=model_alias,
            )

        raise ValueError(f"unsupported node type {node.type.value}")  # pragma: no cover

    async def _run_agent_node(
        self,
        node: Node,
        *,
        workflow: Workflow,
        context: dict[str, dict[str, Any]],
        run_task: str,
        session_id: str,
        model_alias: str | None,
        result: NodeResult,
    ) -> dict[str, Any]:
        node_approvals = workflow.approvals_for(node)
        goal = resolve(str(node.data["goal"]), context)
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("the resolved goal is empty")
        # Resolve first, *then* fall back. A node that writes
        # `model: "{{#start.model#}}"` where the input defaults to "" must
        # fall through to the run-level model; falling back before resolving
        # means the template string itself counts as "specified" and the
        # empty result silently wins.
        declared_model = node.data.get("model")
        alias = resolve(declared_model, context) if isinstance(declared_model, str) else declared_model
        if not (isinstance(alias, str) and alias.strip()):
            alias = model_alias
        max_steps = node.data.get("max_steps")

        # The runtime owns task creation. Creating the row here and then
        # passing the id in would insert it twice -- `run` creates it too --
        # so the id is generated up front and `parent_task_id` carries the
        # linkage. A workflow run's children are then findable with
        # `store.list_children(run_task)`.
        child_id = new_id("task")
        result.child_task_id = child_id
        outcome = await self.agent.runtime.run(
            goal,
            session_id=session_id,
            model_alias=alias if isinstance(alias, str) and alias else None,
            max_steps=int(max_steps) if max_steps else None,
            task_id=child_id,
            parent_task_id=run_task,
            approved_effects=[EffectClass(a) for a in node_approvals],
        )
        if outcome.status == "waiting_confirmation":
            # Say which effect and where to allow it. "ended
            # waiting_confirmation" with an empty reason is a dead end for the
            # author -- they cannot tell whether to add an approval or to run
            # the workflow interactively.
            pending = outcome.pending_confirmation
            effect = pending.effect if pending else "unknown"
            raise RuntimeError(
                f"needs approval for {effect} (tool {pending.tool if pending else '?'}); "
                f"add `approve: [{effect}]` to this node or `approvals: [{effect}]` "
                f"to the workflow"
            )
        if outcome.status != "completed":
            raise RuntimeError(f"agent node ended {outcome.status}: {outcome.error or ''}")
        return {
            "answer": outcome.answer or "",
            "status": outcome.status,
            "steps": outcome.steps,
            "tokens": outcome.usage.total_tokens,
            "cost_usd": outcome.usage.cost_usd,
            "task_id": child_id,
        }

    async def _run_tool_node(
        self, node: Node, *, context: dict[str, dict[str, Any]], run_task: str
    ) -> dict[str, Any]:
        name = str(node.data["tool"])
        tool = self.agent.registry.maybe_get(name)
        if tool is None:
            raise ValueError(
                f"unknown tool {name!r}; available: {', '.join(sorted(self.agent.registry.names()))}"
            )
        arguments = resolve_deep(node.data.get("arguments") or {}, context)
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must resolve to a mapping")

        ctx = self._tool_context(run_task, node)
        outcome = await tool.run(arguments, ctx)
        if not outcome.success:
            raise ToolError(outcome.error or f"{name} failed")
        return {
            "output": outcome.output,
            "success": outcome.success,
            "error": outcome.error or "",
        }

    async def _run_code_node(
        self, node: Node, *, context: dict[str, dict[str, Any]], run_task: str
    ) -> dict[str, Any]:
        """Run a snippet through the shell tool, not in-process.

        Dify executes its code node inside the server. Doing that here would
        make a workflow file a way to reach the runtime's memory, and a
        workflow file is data. Going through `run_command` means the sandbox
        and the command guard apply exactly as they do for the agent.
        """
        snippet = str(node.data["code"])
        language = str(node.data.get("language") or "python").lower()
        if language not in {"python", "python3", "sh", "bash"}:
            raise ValueError(f"unsupported code language {language!r}: python | sh")

        declared = node.data.get("variables")
        if isinstance(declared, dict):
            scoped = context.setdefault(node.id, {})
            for name, value in declared.items():
                scoped[str(name)] = resolve_deep(value, context)

        rendered = resolve(snippet, context)
        runner = self.agent.registry.maybe_get("run_command")
        if runner is None:
            raise ValueError("the code node needs the run_command tool, which is not registered")
        ctx = self._tool_context(run_task, node)
        outcome = await runner.run({"command": str(rendered)}, ctx)
        if not outcome.success:
            raise ToolError(outcome.error or "the snippet exited non-zero")
        return {
            "output": outcome.output,
            "success": outcome.success,
            "error": outcome.error or "",
            "exit_code": 0,
        }

    async def _run_iteration_node(
        self,
        node: Node,
        *,
        workflow: Workflow,
        context: dict[str, dict[str, Any]],
        run_task: str,
        session_id: str,
        model_alias: str | None,
    ) -> dict[str, Any]:
        over = resolve_deep(node.data["over"], context)
        if not isinstance(over, list):
            raise ValueError(f"`over` must resolve to a list, got {type(over).__name__}")
        body_id = node.data.get("body") or node.data.get("on_each")
        if not body_id or body_id not in workflow.nodes:
            raise ValueError("an iteration node needs `body: <node id>`")
        body = workflow.nodes[str(body_id)]

        max_items = int(node.data.get("max_items") or 50)
        if len(over) > max_items:
            raise ValueError(
                f"`over` has {len(over)} items, above max_items={max_items}; "
                "raise max_items deliberately, not by accident"
            )

        item_name = str(node.data.get("item") or "item")
        results: list[Any] = []
        for index, item in enumerate(over):
            scoped = dict(context)
            scoped[node.id] = {**scoped.get(node.id, {}), item_name: item, "index": index}
            child = await self._execute_node(
                body,
                workflow=workflow,
                context=scoped,
                inputs={},
                run_task=run_task,
                session_id=session_id,
                model_alias=model_alias,
            )
            if child.status != "completed":
                raise RuntimeError(
                    f"iteration item {index} failed in {body.id}: {child.error}"
                )
            results.append(child.outputs)
        return {
            "results": results,
            "count": len(results),
            "joined": "\n".join(
                str(r.get("answer") or r.get("output") or r) for r in results
            ),
        }

    def _tool_context(self, run_task: str, node: Node) -> ToolContext:
        settings = self.agent.settings
        return ToolContext(
            task_id=run_task,
            session_id=f"{run_task}:{node.id}",
            step_id=node.id,
            workspace=Path(settings.workspace),
            home=Path(settings.home),
            artifact_dir=Path(settings.artifact_dir),
        )


__all__ = ["NodeResult", "WorkflowResult", "WorkflowRunner"]
