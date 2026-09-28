"""Workflow orchestration.

A declarative, statically-checkable way to compose agent runs. The shape is
Dify's DSL, chosen for one property: nodes reference each other's outputs
explicitly (`{{#node.field#}}`) rather than sharing mutable state, which means
every reference can be verified before anything runs.

The runner is a driver over the existing runtime, not a parallel system. A
workflow run is a task; an `agent` node is a real `AgentRuntime.run()` with
`parent_task_id` set. So budgets, permissions, the event log, resume and the
A2A task mapping all apply without being reimplemented.
"""

from unified_agent.orchestration.expressions import (
    ExpressionError,
    evaluate,
    resolve,
    resolve_deep,
)
from unified_agent.orchestration.multi_agent import (
    AgentDeps,
    MultiAgentRunner,
    SubAgentOutcome,
    SubAgentTask,
)
from unified_agent.orchestration.runner import (
    NodeResult,
    WorkflowResult,
    WorkflowRunner,
)
from unified_agent.orchestration.workflow import (
    Edge,
    ErrorStrategy,
    Node,
    NodeType,
    Problem,
    Workflow,
    WorkflowError,
    discover,
    load_workflow,
    parse_workflow,
    validate,
)

__all__ = [
    "AgentDeps",
    "Edge",
    "ErrorStrategy",
    "ExpressionError",
    "MultiAgentRunner",
    "Node",
    "NodeResult",
    "NodeType",
    "Problem",
    "SubAgentOutcome",
    "SubAgentTask",
    "Workflow",
    "WorkflowError",
    "WorkflowResult",
    "WorkflowRunner",
    "discover",
    "evaluate",
    "load_workflow",
    "parse_workflow",
    "resolve",
    "resolve_deep",
    "validate",
]
