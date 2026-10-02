"""A2A v1.0: agent-to-agent, both directions.

    types.py     the five core objects and the eight task states
    card.py      what this agent publishes about itself
    security.py  SSRF guard, untrusted-card vetting
    server.py    the JSON-RPC binding
    client.py    calling another agent

Why adopt the spec instead of a bespoke message shape: MCP is how an agent
reaches its own tools, A2A is how an agent reaches another agent across a
trust boundary. That boundary is where a hand-rolled protocol has to
re-derive authentication, streaming, cancellation and task identity -- and
gets one of them subtly wrong.

The concrete payoff for this project is a name for a state it already had.
`waiting_confirmation` is `INPUT_REQUIRED`, so a human-in-the-loop pause is
expressible to another organisation without inventing a field for it.
"""

from wukong.a2a.card import AgentCard, AgentSkill, build_card, describe_capability_gap
from wukong.a2a.client import A2AClient, A2AClientError, RemoteCall
from wukong.a2a.security import (
    SSRFBlocked,
    SecurityReport,
    check_parts,
    check_url,
    validate_remote_card,
)
from wukong.a2a.server import A2AServer, JsonRpcError
from wukong.a2a.types import (
    Artifact,
    Message,
    Part,
    Task,
    TaskState,
    TaskStatusObject,
    TextPart,
    parse_part,
    parts_to_text,
    state_for,
)

__all__ = [
    "A2AClient",
    "A2AClientError",
    "A2AServer",
    "AgentCard",
    "AgentSkill",
    "Artifact",
    "JsonRpcError",
    "Message",
    "Part",
    "RemoteCall",
    "SSRFBlocked",
    "SecurityReport",
    "Task",
    "TaskState",
    "TaskStatusObject",
    "TextPart",
    "build_card",
    "check_parts",
    "check_url",
    "describe_capability_gap",
    "parse_part",
    "parts_to_text",
    "state_for",
    "validate_remote_card",
]
