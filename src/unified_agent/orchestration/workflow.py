"""Workflow DSL: schema, parsing, and static validation.

Shape borrowed from Dify, because its DSL has the one property that matters
here: **nodes do not share mutable state**. A node's inputs are explicit
references to another node's outputs (`{{#node.field#}}`), which means every
reference can be checked *before* anything runs.

That is the whole point of this module. LangGraph's shared typed state is
more flexible, but a reference into it is a runtime `KeyError`; here it is a
load-time error with a line number. A workflow that parses is a workflow
whose references cannot dangle.

Three deliberate departures from Dify:

* **Seven node types, not forty.** Dify has 40+ because it is a visual
  builder and each palette entry needs a node. This is a text format.
* **No in-process `code` node.** A `code` node in Dify executes inside the
  server process. Here it runs as a subprocess through the same sandbox and
  permission rules as any other command: a workflow file is *data*, and data
  should not be able to reach into the runtime's memory.
* **`error_strategy` is per node, not global.** A flaky network call and a
  deterministic parse step want different policies, and making that a global
  setting forces the wrong one on one of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """One validation failure, with enough context to fix it."""

    location: str
    message: str
    hint: str = ""

    def render(self) -> str:
        text = f"{self.location}: {self.message}"
        if self.hint:
            text += f"\n    hint: {self.hint}"
        return text


class WorkflowError(Exception):
    """Raised when a workflow cannot be loaded."""

    def __init__(self, problems: Iterable[Problem]) -> None:
        self.problems = list(problems)
        super().__init__(
            f"{len(self.problems)} problem(s): "
            + "; ".join(f"{p.location}: {p.message}" for p in self.problems[:3])
        )


# ---------------------------------------------------------------------------
# node types
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    START = "start"
    AGENT = "agent"
    TOOL = "tool"
    CODE = "code"
    IFELSE = "ifelse"
    ITERATION = "iteration"
    END = "end"


#: Fields each node type produces, and therefore what `{{#id.field#}}` may
#: reference. Anything not listed here is a load-time error rather than a
#: missing key at runtime.
OUTPUT_FIELDS: dict[NodeType, set[str]] = {
    NodeType.START: set(),  # start produces its declared inputs
    NodeType.AGENT: {"answer", "status", "steps", "tokens", "cost_usd", "task_id"},
    NodeType.TOOL: {"output", "success", "error"},
    NodeType.CODE: {"output", "success", "error", "exit_code"},
    NodeType.IFELSE: {"branch"},
    NodeType.ITERATION: {"results", "count", "joined"},
    NodeType.END: set(),
}

#: Fields every node produces, whatever its type.
COMMON_FIELDS = {"id", "type", "title"}


@dataclass
class ErrorStrategy:
    max_retries: int = 0
    retry_delay_s: float = 0.0
    on_error: str = "fail"  # "fail" | "continue" | a node id to branch to

    @classmethod
    def parse(cls, raw: Any, location: str, problems: list[Problem]) -> "ErrorStrategy":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            problems.append(Problem(location, "error_strategy must be a mapping"))
            return cls()
        strategy = cls()
        retry = raw.get("retry")
        if isinstance(retry, dict):
            strategy.max_retries = _as_int(retry.get("max_retries", 0), location, problems)
            strategy.retry_delay_s = _as_float(
                retry.get("delay_seconds", 0.0), location, problems
            )
        handling = raw.get("on_error", "fail")
        if not isinstance(handling, str) or not handling:
            problems.append(Problem(location, "error_strategy.on_error must be a string"))
        else:
            strategy.on_error = handling
        if strategy.max_retries < 0:
            problems.append(Problem(location, "max_retries cannot be negative"))
            strategy.max_retries = 0
        return strategy


@dataclass
class Node:
    id: str
    type: NodeType
    title: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error_strategy: ErrorStrategy = field(default_factory=ErrorStrategy)

    @property
    def outputs(self) -> set[str]:
        if self.type is NodeType.START:
            declared = self.data.get("variables") or []
            return {
                str(v.get("variable"))
                for v in declared
                if isinstance(v, dict) and v.get("variable")
            }
        return OUTPUT_FIELDS.get(self.type, set())

    def references(self) -> Iterator[tuple[str, str, str]]:
        """Every `{{#node.field#}}` this node reads: (node_id, field, where)."""
        for path, value in _walk(self.data):
            if not isinstance(value, str):
                continue
            for match in REFERENCE.finditer(value):
                yield match.group("node"), match.group("field"), f"{self.id}.{path}"


@dataclass
class Edge:
    source: str
    target: str
    source_handle: str = "source"
    target_handle: str = "target"
    label: str = ""


@dataclass
class Workflow:
    name: str
    version: str
    nodes: dict[str, Node]
    edges: list[Edge]
    description: str = ""
    inputs: list[dict[str, Any]] = field(default_factory=list)
    #: Effect classes this workflow is allowed to perform without prompting.
    #: Declared in the file, so it is reviewable, diffable, and does not
    #: require a human to be present when the workflow runs.
    approvals: list[str] = field(default_factory=list)
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def start(self) -> Node:
        return next(n for n in self.nodes.values() if n.type is NodeType.START)

    @property
    def ends(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.type is NodeType.END]

    def outgoing(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]

    def successors(self, node_id: str) -> list[str]:
        """Where control can go next, *including* branches and loop bodies.

        An `ifelse` routes through `data.branches[].target`, not through
        `graph.edges`; an `iteration` runs `data.body`. Treating only the
        declared edges as edges makes those routes invisible to the ordering,
        the cycle check and the reachability check -- so a perfectly valid
        workflow is reported as having dangling references, and a genuine
        cycle hidden inside a branch goes unnoticed.
        """
        targets = [e.target for e in self.outgoing(node_id)]
        node = self.nodes.get(node_id)
        if node is None:
            return targets
        if node.type is NodeType.IFELSE:
            for branch in node.data.get("branches") or []:
                if isinstance(branch, dict) and branch.get("target"):
                    targets.append(str(branch["target"]))
            if node.data.get("else"):
                targets.append(str(node.data["else"]))
        if node.type is NodeType.ITERATION:
            for key in ("body", "on_each"):
                if node.data.get(key):
                    targets.append(str(node.data[key]))
        seen: set[str] = set()
        return [t for t in targets if not (t in seen or seen.add(t))]

    def edge_pairs(self) -> list[tuple[str, str]]:
        """Every (source, target) pair, deduplicated.

        Declared edges plus the implicit ones from branches and loop bodies.

        Deduplicated on purpose, and it matters: `when: X -> done` together
        with `else: done` is two routes to the same target, which is one edge
        for the purposes of ordering and cycle detection. Counting the
        indegree twice and decrementing once -- which happens if the two
        passes use different edge sets -- leaves the indegree stuck above zero
        and reports a perfectly ordinary diamond as a cycle.
        """
        pairs: list[tuple[str, str]] = []
        for edge in self.edges:
            pairs.append((edge.source, edge.target))
        for node_id, node in self.nodes.items():
            pairs.extend((node_id, target) for target in self._implicit_targets(node))
        seen: set[tuple[str, str]] = set()
        return [p for p in pairs if not (p in seen or seen.add(p))]

    def _implicit_targets(self, node: Node) -> list[str]:
        """Routes this node takes that are not written in `graph.edges`."""
        targets: list[str] = []
        if node.type is NodeType.IFELSE:
            targets = [
                str(b.get("target"))
                for b in (node.data.get("branches") or [])
                if isinstance(b, dict) and b.get("target")
            ]
            if node.data.get("else"):
                targets.append(str(node.data["else"]))
        elif node.type is NodeType.ITERATION:
            targets = [str(node.data[key]) for key in ("body", "on_each") if node.data.get(key)]
        return [t for t in targets if t in self.nodes]

    def effective_edges(self) -> list[Edge]:
        """`edge_pairs` as Edge objects, for display and description."""
        return [Edge(source=s, target=t) for s, t in self.edge_pairs()]

    def topological_order(self) -> list[str]:
        """Kahn's algorithm.

        Tolerates dangling edges on purpose: the reference check needs an
        order to work out what is upstream, and that check runs *before* the
        graph is known to be well-formed. A `KeyError` here would replace a
        useful "target node does not exist" with a traceback.
        """
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        indegree = {node_id: 0 for node_id in self.nodes}
        for source, target in self.edge_pairs():
            if source not in adjacency or target not in indegree:
                continue
            adjacency[source].append(target)
            indegree[target] += 1
        queue = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for target in adjacency[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
                    queue.sort()
        return order

    def approvals_for(self, node: Node) -> list[str]:
        """Workflow-level approvals plus the node's own, deduplicated."""
        merged = [*self.approvals, *(node.data.get("approve") or [])]
        seen: set[str] = set()
        return [a for a in (str(x) for x in merged) if not (a in seen or seen.add(a))]

    def describe(self) -> str:
        implicit = len(self.effective_edges()) - len(self.edges)
        extra = f", {implicit} implicit" if implicit else ""
        lines = [f"{self.name}  ({len(self.nodes)} nodes, {len(self.edges)} edges{extra})"]
        if self.description:
            lines.append(f"  {self.description}")
        for node_id in self.topological_order():
            node = self.nodes[node_id]
            label = f"  {node_id:<18} {node.type.value:<10}"
            if node.title:
                label += f" {node.title}"
            lines.append(label)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# reference syntax
# ---------------------------------------------------------------------------

#: `{{#node.field#}}` -- Dify's syntax, chosen over `${node.field}` because
#: the delimiters cannot appear accidentally in prose the way a bare `$` can.
REFERENCE = re.compile(r"\{\{#\s*(?P<node>[A-Za-z_][\w-]*)\s*\.\s*(?P<field>[\w.-]+)\s*#\}\}")


def _walk(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _as_int(value: Any, location: str, problems: list[Problem]) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        problems.append(Problem(location, f"expected an integer, got {value!r}"))
        return 0


def _as_float(value: Any, location: str, problems: list[Problem]) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        problems.append(Problem(location, f"expected a number, got {value!r}"))
        return 0.0


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_REQUIRED_NODE_FIELDS: dict[NodeType, tuple[str, ...]] = {
    NodeType.START: (),
    NodeType.AGENT: ("goal",),
    NodeType.TOOL: ("tool",),
    NodeType.CODE: ("code",),
    NodeType.IFELSE: ("branches",),
    NodeType.ITERATION: ("over",),
    NodeType.END: (),
}


def parse_workflow(text: str, *, source_path: Path | None = None) -> Workflow:
    """Parse and validate. Raises `WorkflowError` listing every problem."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise WorkflowError([Problem(str(source_path or "<text>"), f"invalid YAML: {exc}")]) from exc

    problems: list[Problem] = []
    if not isinstance(raw, dict):
        raise WorkflowError([Problem("<root>", "the document must be a mapping")])

    name = str(raw.get("name") or "").strip()
    if not name:
        problems.append(Problem("<root>", "`name` is required"))
    version = str(raw.get("version") or "").strip()
    if not version:
        problems.append(
            Problem("<root>", "`version` is required", "the DSL is versioned; use \"1\"")
        )

    graph = raw.get("graph")
    if not isinstance(graph, dict):
        raise WorkflowError(
            [Problem("<root>", "`graph` is required and must be a mapping with nodes and edges")]
        )

    nodes = _parse_nodes(graph.get("nodes"), problems)
    edges = _parse_edges(graph.get("edges"), problems)

    workflow = Workflow(
        name=name,
        version=version,
        nodes=nodes,
        edges=edges,
        description=str(raw.get("description") or ""),
        inputs=_parse_inputs(raw.get("inputs"), problems),
        approvals=[str(a) for a in (raw.get("approvals") or [])],
        source_path=source_path,
        raw=raw,
    )

    problems.extend(validate(workflow))
    if problems:
        raise WorkflowError(problems)
    return workflow


def _parse_nodes(raw: Any, problems: list[Problem]) -> dict[str, Node]:
    nodes: dict[str, Node] = {}
    if raw is None:
        problems.append(Problem("graph", "`nodes` is required"))
        return nodes
    if not isinstance(raw, list):
        problems.append(Problem("graph.nodes", "must be a list"))
        return nodes

    for index, item in enumerate(raw):
        location = f"graph.nodes[{index}]"
        if not isinstance(item, dict):
            problems.append(Problem(location, "each node must be a mapping"))
            continue
        node_id = item.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            # YAML 1.1 reads `yes`, `no`, `on`, `off` as booleans, so a node
            # named `yes` arrives as True. That is a trap the author cannot
            # see in the file, so the message has to name it.
            if isinstance(node_id, bool):
                problems.append(
                    Problem(
                        location,
                        f"`id` was parsed as the boolean {node_id!r}",
                        'YAML reads yes/no/on/off as booleans -- quote it: '
                        'id: "yes"',
                    )
                )
            else:
                problems.append(
                    Problem(
                        location,
                        f"`id` is required and must be a non-empty string, got {node_id!r}",
                    )
                )
            continue
        node_id = node_id.strip()
        location = f"graph.nodes[{index}] ({node_id})"
        if node_id in nodes:
            problems.append(
                Problem(location, f"duplicate node id {node_id!r}", "ids must be unique")
            )
            continue

        raw_type = item.get("type")
        try:
            node_type = NodeType(str(raw_type))
        except ValueError:
            problems.append(
                Problem(
                    location,
                    f"unknown node type {raw_type!r}",
                    "known types: " + ", ".join(t.value for t in NodeType),
                )
            )
            continue

        data = item.get("data")
        if data is None:
            data = {}
        if not isinstance(data, dict):
            problems.append(Problem(location, "`data` must be a mapping"))
            continue

        for required in _REQUIRED_NODE_FIELDS[node_type]:
            if not data.get(required):
                problems.append(
                    Problem(location, f"`{node_type.value}` nodes require `data.{required}`")
                )

        nodes[node_id] = Node(
            id=node_id,
            type=node_type,
            title=str(data.get("title") or ""),
            data=data,
            error_strategy=ErrorStrategy.parse(
                data.get("error_strategy"), f"{location}.error_strategy", problems
            ),
        )
    return nodes


def _parse_edges(raw: Any, problems: list[Problem]) -> list[Edge]:
    edges: list[Edge] = []
    if raw is None:
        return edges
    if not isinstance(raw, list):
        problems.append(Problem("graph.edges", "must be a list"))
        return edges
    for index, item in enumerate(raw):
        location = f"graph.edges[{index}]"
        if not isinstance(item, dict):
            problems.append(Problem(location, "each edge must be a mapping"))
            continue
        source = item.get("source")
        target = item.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            problems.append(Problem(location, "`source` and `target` are required strings"))
            continue
        edges.append(
            Edge(
                source=source,
                target=target,
                # Dify uses snake_case handles; accept both so a DSL copied
                # from there loads unchanged.
                source_handle=str(item.get("source_handle") or item.get("sourceHandle") or "source"),
                target_handle=str(item.get("target_handle") or item.get("targetHandle") or "target"),
                label=str(item.get("label") or ""),
            )
        )
    return edges


def _parse_inputs(raw: Any, problems: list[Problem]) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        problems.append(Problem("inputs", "must be a list"))
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        location = f"inputs[{index}]"
        if not isinstance(item, dict) or not item.get("name"):
            problems.append(Problem(location, "each input needs a `name`"))
            continue
        out.append(dict(item))
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate(workflow: Workflow) -> list[Problem]:
    """Everything that can be checked without running anything."""
    problems: list[Problem] = []

    starts = [n for n in workflow.nodes.values() if n.type is NodeType.START]
    if not starts:
        problems.append(Problem("graph.nodes", "no `start` node", "a workflow needs exactly one"))
    elif len(starts) > 1:
        problems.append(
            Problem(
                "graph.nodes",
                f"{len(starts)} `start` nodes: {', '.join(n.id for n in starts)}",
                "there must be exactly one entry point",
            )
        )
    if not workflow.ends:
        problems.append(Problem("graph.nodes", "no `end` node", "a workflow needs at least one"))

    problems.extend(_validate_approvals(workflow))
    problems.extend(_validate_edges(workflow))
    problems.extend(_validate_references(workflow))
    problems.extend(_validate_branches(workflow))
    problems.extend(_validate_reachability(workflow))
    problems.extend(_validate_cycles(workflow))
    return problems


def _validate_approvals(workflow: Workflow) -> list[Problem]:
    """Approvals must name real effects, and never SYSTEM_ADMIN.

    A workflow file is data that may come from anywhere. Letting it grant
    itself the one effect class the HTTP layer also refuses to grant would
    make the file the way around that rule.
    """
    from unified_agent.types import EffectClass

    known = {e.value for e in EffectClass}
    forbidden = {EffectClass.SYSTEM_ADMIN.value}
    problems: list[Problem] = []

    declared: list[tuple[str, str]] = [("approvals", a) for a in workflow.approvals]
    for node in workflow.nodes.values():
        raw = node.data.get("approve")
        if raw is None:
            continue
        if not isinstance(raw, list):
            problems.append(Problem(f"graph.nodes ({node.id}).approve", "must be a list"))
            continue
        declared.extend((f"graph.nodes ({node.id}).approve", str(a)) for a in raw)

    for location, name in declared:
        if name not in known:
            problems.append(
                Problem(
                    location,
                    f"unknown effect class {name!r}",
                    "known: " + ", ".join(sorted(known - forbidden)),
                )
            )
        elif name in forbidden:
            problems.append(
                Problem(
                    location,
                    f"{name} cannot be granted by a workflow",
                    "a workflow file is data; this effect is refused over HTTP for the same reason",
                )
            )
    return problems


def _validate_edges(workflow: Workflow) -> list[Problem]:
    problems: list[Problem] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in workflow.edges:
        for role, node_id in (("source", edge.source), ("target", edge.target)):
            if node_id not in workflow.nodes:
                problems.append(
                    Problem(
                        f"graph.edges ({edge.source} -> {edge.target})",
                        f"{role} node {node_id!r} does not exist",
                        "known ids: " + ", ".join(sorted(workflow.nodes)),
                    )
                )
        key = (edge.source, edge.target, edge.label)
        if key in seen:
            problems.append(
                Problem(
                    f"graph.edges ({edge.source} -> {edge.target})",
                    "duplicate edge",
                )
            )
        seen.add(key)
        if edge.source == edge.target:
            problems.append(
                Problem(f"graph.edges ({edge.source})", "a node cannot point at itself")
            )
    return problems


def _validate_references(workflow: Workflow) -> list[Problem]:
    """Every `{{#node.field#}}` must resolve, and must point backwards.

    Forward references are rejected even though a topological sort could
    technically satisfy them: they mean the author is describing data flow
    that does not exist, and catching that at load time is the entire reason
    for choosing explicit references over shared state.
    """
    problems: list[Problem] = []
    order = workflow.topological_order()
    position = {node_id: index for index, node_id in enumerate(order)}

    for node in workflow.nodes.values():
        for source_id, field_name, where in node.references():
            location = f"graph.nodes ({node.id}).{where.split('.', 1)[-1]}"
            source = workflow.nodes.get(source_id)
            if source is None:
                problems.append(
                    Problem(
                        location,
                        f"reference to unknown node {source_id!r}",
                        "known ids: " + ", ".join(sorted(workflow.nodes)),
                    )
                )
                continue
            available = source.outputs | COMMON_FIELDS
            if field_name not in available:
                problems.append(
                    Problem(
                        location,
                        f"node {source_id!r} ({source.type.value}) has no field {field_name!r}",
                        "it provides: " + ", ".join(sorted(available)),
                    )
                )
            if (
                source_id in position
                and node.id in position
                and position[source_id] >= position[node.id]
            ):
                problems.append(
                    Problem(
                        location,
                        f"reference to {source_id!r} is not upstream of {node.id!r}",
                        "references must point at a node that runs earlier",
                    )
                )
    return problems


def _validate_branches(workflow: Workflow) -> list[Problem]:
    problems: list[Problem] = []
    for node in workflow.nodes.values():
        if node.type is not NodeType.IFELSE:
            continue
        branches = node.data.get("branches")
        if not isinstance(branches, list) or not branches:
            problems.append(Problem(f"graph.nodes ({node.id})", "`branches` must be a non-empty list"))
            continue
        for index, branch in enumerate(branches):
            location = f"graph.nodes ({node.id}).branches[{index}]"
            if not isinstance(branch, dict):
                problems.append(Problem(location, "each branch must be a mapping"))
                continue
            if not branch.get("when"):
                problems.append(Problem(location, "`when` is required"))
            target = branch.get("target")
            if not target:
                problems.append(Problem(location, "`target` is required"))
            elif target not in workflow.nodes:
                problems.append(Problem(location, f"unknown target node {target!r}"))
        fallback = node.data.get("else")
        if fallback and fallback not in workflow.nodes:
            problems.append(Problem(f"graph.nodes ({node.id}).else", f"unknown node {fallback!r}"))
        # A branch whose condition is literally `true` is already a catch-all,
        # so an `else` is redundant rather than required.
        has_catch_all = any(
            str(b.get("when") or "").strip().lower() == "true"
            for b in branches
            if isinstance(b, dict)
        )
        if not fallback and not has_catch_all:
            # Without an `else` and without a catch-all, a false condition
            # ends the run silently, which looks like a bug in the engine.
            problems.append(
                Problem(
                    f"graph.nodes ({node.id})",
                    "no `else` and no catch-all branch",
                    "add `else: <node>` so a false condition has a destination",
                )
            )
    return problems


def _validate_reachability(workflow: Workflow) -> list[Problem]:
    """Unreachable nodes never run, and say nothing about it."""
    if not workflow.nodes:
        return []
    starts = [n.id for n in workflow.nodes.values() if n.type is NodeType.START]
    if not starts:
        return []
    reachable: set[str] = set()
    queue = list(starts)
    while queue:
        node_id = queue.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        # `successors` already folds in branches and loop bodies, so there is
        # exactly one definition of "where control can go next".
        queue.extend(workflow.successors(node_id))

    orphans = sorted(set(workflow.nodes) - reachable)
    if orphans:
        return [
            Problem(
                "graph.nodes",
                f"unreachable node(s): {', '.join(orphans)}",
                "nothing can run them; connect them or delete them",
            )
        ]
    return []


def _validate_cycles(workflow: Workflow) -> list[Problem]:
    """Iteration bodies may loop; the node graph may not."""
    # One edge set for both the indegree and the decrement, deduplicated --
    # see `edge_pairs` for why mixing the two reports diamonds as cycles.
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in workflow.nodes}
    indegree = {node_id: 0 for node_id in workflow.nodes}
    for source, target in workflow.edge_pairs():
        if source not in adjacency or target not in indegree:
            continue  # a dangling edge is reported by _validate_edges
        adjacency[source].append(target)
        indegree[target] += 1
    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node_id = queue.pop()
        visited += 1
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(workflow.nodes):
        stuck = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        return [
            Problem(
                "graph.edges",
                f"cycle involving: {', '.join(stuck)}",
                "use an `iteration` node for repetition; the graph itself must be acyclic",
            )
        ]
    return []


def load_workflow(path: Path) -> Workflow:
    return parse_workflow(path.read_text(encoding="utf-8"), source_path=path)


def discover(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*.y*ml") if p.is_file() and p.name != "config.toml"
    )


__all__ = [
    "COMMON_FIELDS",
    "OUTPUT_FIELDS",
    "REFERENCE",
    "Edge",
    "ErrorStrategy",
    "Node",
    "NodeType",
    "Problem",
    "Workflow",
    "WorkflowError",
    "discover",
    "load_workflow",
    "parse_workflow",
    "validate",
]
