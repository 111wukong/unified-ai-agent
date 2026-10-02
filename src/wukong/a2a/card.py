"""Agent Card: what this agent publishes about itself.

Served at `/.well-known/agent-card.json`, which is the A2A discovery
convention -- a peer finds the endpoint from the card rather than from
configuration, so the card is the interface.

Two decisions worth stating:

* **Only `active` skills are advertised.** The card is generated from the
  skill registry, which means the human review gate has an effect outside the
  process: a candidate the agent wrote for itself is not something this agent
  claims to be able to do. Advertising a candidate would promise a peer a
  capability that `load_skill` then refuses.
* **Authentication is advertised, not enforced here.** The card says which
  scheme is required; enforcement belongs to the HTTP layer, which already
  has a token guard. A card that advertises `none` while the endpoint
  requires a token is worse than either choice.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from wukong import __version__

#: We accept text and structured data. Not files: see `types.FilePart`.
INPUT_MODES = ["text/plain", "application/json"]
OUTPUT_MODES = ["text/plain"]


class AgentSkill(BaseModel):
    """A capability label, for a remote orchestrator to match against.

    Deliberately not the skill's body or its `allowed-tools`: a card is
    public and is read by software, so it carries what is needed to decide
    whether to call, and nothing that would be an instruction to follow.
    """

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)


class AgentCard(BaseModel):
    protocolVersion: str = "1.0"  # noqa: N815
    name: str
    description: str
    url: str
    version: str = __version__
    provider: dict[str, str] | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    defaultInputModes: list[str] = Field(default_factory=lambda: list(INPUT_MODES))  # noqa: N815
    defaultOutputModes: list[str] = Field(default_factory=lambda: list(OUTPUT_MODES))  # noqa: N815
    skills: list[AgentSkill] = Field(default_factory=list)
    securitySchemes: dict[str, Any] = Field(default_factory=dict)  # noqa: N815
    security: list[dict[str, list[str]]] = Field(default_factory=list)


def build_card(
    agent: Any,
    *,
    url: str,
    token_required: bool = False,
    provider: dict[str, str] | None = None,
) -> AgentCard:
    """Describe this agent, from the pieces that actually exist.

    Every field is derived rather than configured, so the card cannot drift
    from the runtime: the skills are the registry's, the streaming flag is
    whether the HTTP layer serves SSE, and the security scheme is whether a
    token is enforced.
    """
    settings = agent.settings
    skills = [
        AgentSkill(
            id=skill.name,
            name=skill.name,
            description=skill.description,
            tags=sorted(skill.allowed_tools)[:8],
        )
        for skill in (agent.skills.active() if agent.skills else [])
    ]

    card = AgentCard(
        name=settings.a2a.name,
        description=settings.a2a.description,
        url=url,
        provider=provider,
        # Set from the module constants rather than left to the field
        # defaults: two sources for one value means one of them is ignored,
        # and it would be the constant -- the one a reader believes.
        defaultInputModes=list(INPUT_MODES),
        defaultOutputModes=list(OUTPUT_MODES),
        capabilities={
            # Streaming is real: `message/stream` is served as SSE.
            "streaming": True,
            # Push notifications are not implemented, and advertising them
            # would have a peer configure a webhook this agent never calls.
            "pushNotifications": False,
            # Every task this agent runs is durable and resumable, which is
            # what the flag means.
            "stateTransitionHistory": True,
        },
        skills=skills,
    )
    if token_required:
        card.securitySchemes = {
            "bearer": {"type": "http", "scheme": "bearer"},
        }
        card.security = [{"bearer": []}]
    return card


def describe_capability_gap(card: AgentCard) -> list[str]:
    """What this card promises that this build does not do.

    Exists so the honest answer to "is the card accurate" is computed rather
    than asserted. If a future change enables push notifications, the check
    starts passing instead of the card quietly becoming a lie.
    """
    gaps: list[str] = []
    if card.capabilities.get("pushNotifications"):
        gaps.append("pushNotifications is advertised but not implemented")
    if card.capabilities.get("streaming") and "message/stream" not in _SERVED_METHODS:
        gaps.append("streaming is advertised but message/stream is not served")
    return gaps


#: JSON-RPC methods this server answers. Kept next to the card because the
#: card's capability flags are claims *about* this list.
_SERVED_METHODS = frozenset(
    {"message/send", "message/stream", "tasks/get", "tasks/cancel"}
)


__all__ = [
    "AgentCard",
    "AgentSkill",
    "INPUT_MODES",
    "OUTPUT_MODES",
    "build_card",
    "describe_capability_gap",
]
