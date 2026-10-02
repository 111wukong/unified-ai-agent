"""A2A v1.0 wire objects.

The five core objects (Task, Message, Part, Artifact, Agent Card) and the
eight task states. Written against the published spec rather than a bespoke
`AgentMessage` shape, which is what the original design document proposed and
what this replaces.

The reason to adopt rather than invent: **MCP is how an agent reaches its own
tools; A2A is how an agent reaches another agent across a trust boundary.**
That boundary is where a self-designed protocol has to re-derive authentication,
streaming, cancellation and task identity -- and gets one of them wrong.

The most useful thing A2A gives this project is a name for a state it already
has: `INPUT_REQUIRED`. This runtime pauses a task when a tool needs human
approval, and that is exactly `INPUT_REQUIRED` -- so a human-in-the-loop pause
is expressible to another organisation without inventing a field for it.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from wukong.agent.state import TaskStatus


class TaskState(str, Enum):
    """The eight A2A states. Strictly ordered, and the order is the point.

    A client that only understands `SUBMITTED -> WORKING -> terminal` still
    behaves correctly against an agent that uses the interrupts, because the
    interrupts sit between WORKING and the terminal states rather than beside
    them.
    """

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    #: Accepted on the wire, never produced by this agent. That is deliberate:
    #: an enum that silently omits a spec value is worse than one carrying a
    #: value we do not generate, because a peer may send it. Producing it
    #: would mean "this task needs the client to supply credentials", and
    #: nothing here does that -- a missing credential surfaces as a tool
    #: failure, or as INPUT_REQUIRED when a human has to approve, and the HTTP
    #: layer answers an unauthenticated peer with 401 before A2A is reached.
    #: `tests/test_a2a.py` asserts that `state_for` never returns it.
    AUTH_REQUIRED = "auth-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    #: Likewise accepted but not produced: this runtime accepts a request and
    #: then fails, which tells a peer more than an outright refusal would.
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
            TaskState.REJECTED,
        }

    @property
    def interrupted(self) -> bool:
        """Waiting on the *client*, not on us. The run is not over."""
        return self in {TaskState.INPUT_REQUIRED, TaskState.AUTH_REQUIRED}


#: Internal status -> A2A state. Both `pending` and `planning` are `submitted`
#: and `working` respectively because A2A has no vocabulary for "deciding what
#: to do"; the distinction is ours and does not cross the boundary.
_STATUS_TO_STATE: dict[TaskStatus, TaskState] = {
    TaskStatus.PENDING: TaskState.SUBMITTED,
    TaskStatus.PLANNING: TaskState.WORKING,
    TaskStatus.RUNNING: TaskState.WORKING,
    TaskStatus.WAITING_CONFIRMATION: TaskState.INPUT_REQUIRED,
    TaskStatus.COMPLETED: TaskState.COMPLETED,
    TaskStatus.FAILED: TaskState.FAILED,
    TaskStatus.CANCELLED: TaskState.CANCELED,
}


def state_for(status: TaskStatus | str) -> TaskState:
    """Map an internal status onto the wire vocabulary.

    An unknown status becomes FAILED rather than raising: a peer is better
    served by "this did not succeed" than by a 500, and the alternative --
    leaking an internal status string the peer cannot parse -- is worse.
    """
    if isinstance(status, str):
        try:
            status = TaskStatus(status)
        except ValueError:
            return TaskState.FAILED
    return _STATUS_TO_STATE.get(status, TaskState.FAILED)


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str


class DataPart(BaseModel):
    kind: Literal["data"] = "data"
    data: dict[str, Any] = Field(default_factory=dict)


class FilePart(BaseModel):
    """Present so a peer's `file` part can be *parsed* and then refused.

    Accepting it would mean either fetching a URI the caller supplies -- an
    SSRF vector, and the same one the spec names for webhooks -- or accepting
    inline bytes into a runtime that has no attachment store and would drop
    them. Refusing with a clear message is the honest option, and parsing it
    is what lets the refusal say which part was the problem.
    """

    kind: Literal["file"] = "file"
    file: dict[str, Any] = Field(default_factory=dict)


Part = TextPart | DataPart | FilePart


def parse_part(raw: Any) -> Part | None:
    """Parse one part, or None if it is not a shape we recognise."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind == "text" and isinstance(raw.get("text"), str):
        return TextPart(text=raw["text"])
    if kind == "data" and isinstance(raw.get("data"), dict):
        return DataPart(data=raw["data"])
    if kind == "file":
        return FilePart(file=raw.get("file") or {})
    return None


def parts_to_text(parts: list[Part]) -> str:
    """Flatten parts into the single prompt string this runtime consumes.

    `data` parts are serialised rather than dropped: a peer that sends a
    structured payload and gets silence back would conclude the agent ignored
    it, and it did -- but silently, which is the failure mode this project
    spends most of its effort avoiding.
    """
    import json

    chunks: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            chunks.append(part.text)
        elif isinstance(part, DataPart):
            chunks.append(json.dumps(part.data, ensure_ascii=False, sort_keys=True))
    return "\n\n".join(c for c in chunks if c.strip())


# ---------------------------------------------------------------------------
# message / task / artifact
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: list[Part] = Field(default_factory=list)
    messageId: str = ""  # noqa: N815 - wire field names are camelCase
    taskId: str | None = None  # noqa: N815
    contextId: str | None = None  # noqa: N815
    metadata: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    artifactId: str = ""  # noqa: N815
    name: str = ""
    parts: list[Part] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatusObject(BaseModel):
    """A2A calls this `TaskStatus`; renamed to avoid colliding with the
    runtime's own status enum in the same import."""

    state: TaskState
    timestamp: str = ""
    message: Message | None = None


class Task(BaseModel):
    id: str
    contextId: str = ""  # noqa: N815
    status: TaskStatusObject
    artifacts: list[Artifact] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "Artifact",
    "DataPart",
    "FilePart",
    "Message",
    "Part",
    "Task",
    "TaskState",
    "TaskStatusObject",
    "TextPart",
    "now_iso",
    "parse_part",
    "parts_to_text",
    "state_for",
]
