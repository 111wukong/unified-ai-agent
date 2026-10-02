"""Security for agent-to-agent traffic.

The A2A specification names five risks. Three of them apply to an
implementation that only *calls* other agents, and this module covers those:

1. **Webhook SSRF** -- an outbound request to a URL a peer supplies. This
   agent does not send webhooks (`capabilities.pushNotifications` is false),
   but *fetching a remote Agent Card* is the same shape: the URL comes from
   the caller, and the response goes into our process. Same guard.
2. **Card tampering** -- a card is untrusted input. It is validated against
   the schema before use, and every field that can reach a prompt is run
   through the injection detector the skill review already uses.
3. **Context poisoning** -- the sharpest one, and the reason (2) is not
   enough on its own. A card's `description` is *text an attacker writes*
   that a model may then read while deciding whether to call the agent. The
   spec's own advice is to treat card fields as untrusted; this project
   already has a prompt-injection tripwire, so it is reused rather than
   reinvented.

Not covered here, because the spec's mitigations are deployment concerns
rather than library ones: Signed Agent Cards (JWS) and replay protection.
Both are stated in the README's security-boundary section as things a
deployment must add, rather than silently absent.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from wukong.a2a.card import AgentCard
from wukong.a2a.types import parse_part
from wukong.skills.loader import _INJECTION_PATTERNS  # noqa: PLC2701 - shared tripwire

_ALLOWED_SCHEMES = {"http", "https"}


@dataclass
class SecurityReport:
    """What was wrong with an inbound object, in the order it matters."""

    subject: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"{self.subject}: {'ok' if self.ok else 'rejected'}"]
        lines += [f"  error: {e}" for e in self.errors]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SSRF
# ---------------------------------------------------------------------------


class SSRFBlocked(Exception):
    """The outbound URL is not one this agent will fetch."""


def check_url(
    url: str,
    *,
    allow_hosts: list[str],
    allow_private: bool = False,
    resolve: bool = True,
) -> str:
    """Return the host if the URL may be fetched, else raise `SSRFBlocked`.

    Three checks, in the order that fails fastest:

    1. scheme is http/https -- `file://` and `gopher://` are how an SSRF
       becomes a local file read;
    2. host is on the allowlist -- the allowlist is what makes this safe to
       turn on at all, so an empty one blocks everything;
    3. the resolved addresses are public -- an allowlisted name that resolves
       to `127.0.0.1` or `169.254.169.254` is the classic bypass, and it is
       the check people skip.

    `resolve=False` skips (3), for tests that must not touch DNS.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFBlocked(
            f"scheme {parsed.scheme or '(none)'!r} is not allowed; "
            f"expected one of {sorted(_ALLOWED_SCHEMES)}"
        )
    host = parsed.hostname
    if not host:
        raise SSRFBlocked(f"{url!r} has no host")

    if not allow_hosts:
        raise SSRFBlocked(
            "a2a.allow_hosts is empty, so no remote agent may be called. "
            "Add the host you mean to reach."
        )
    if not _host_allowed(host, allow_hosts):
        raise SSRFBlocked(
            f"host {host!r} is not in a2a.allow_hosts {allow_hosts}"
        )

    if resolve and not allow_private:
        addresses = _resolve(host, parsed.port)
        if not addresses:
            raise SSRFBlocked(f"could not resolve {host!r}")
        for address in addresses:
            if not _is_public(address):
                raise SSRFBlocked(
                    f"{host!r} resolves to {address}, which is a private, "
                    "loopback, link-local or reserved address. Set "
                    "a2a.allow_private_networks to reach it deliberately."
                )
    return host


def _host_allowed(host: str, allow_hosts: list[str]) -> bool:
    """Exact host, or a dot-boundary suffix match.

    A plain `endswith` would accept `evil-example.com` for an allowlist entry
    of `example.com`, which is the whole bypass in one line.
    """
    host = host.lower().rstrip(".")
    for entry in allow_hosts:
        entry = entry.lower().strip().rstrip(".")
        if not entry:
            continue
        if host == entry or host.endswith("." + entry):
            return True
    return False


def _resolve(host: str, port: int | None) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return []
    return sorted({info[4][0] for info in infos})


def _is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


# ---------------------------------------------------------------------------
# untrusted cards
# ---------------------------------------------------------------------------


def validate_remote_card(raw: Any, *, subject: str = "agent card") -> tuple[AgentCard | None, SecurityReport]:
    """Parse and vet a card a peer published.

    Two independent checks, because they catch different things:

    * **Schema** -- a card that does not validate is not a card, and using a
      half-parsed one is how a missing field becomes a `None` that reaches
      `str()` somewhere and renders as "None" in a prompt.
    * **Injection tripwire** -- a schema-valid card can still carry
      `description: "Ignore all previous instructions..."`, and that field is
      exactly what a model reads when deciding whether to call the agent.
      This is the spec's "treat card fields as untrusted" made concrete.

    A warning is not a rejection: a card whose *prose* looks suspicious is
    reported so a human can look, while a card that is malformed is refused
    outright. Rejecting on prose would make the tripwire a denial-of-service
    against any peer whose description happens to say "you are now able to".
    """
    report = SecurityReport(subject=subject)
    if not isinstance(raw, dict):
        report.errors.append("card is not a JSON object")
        return None, report

    try:
        card = AgentCard.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - any validation error is a refusal
        report.errors.append(f"card does not match the schema: {_first_line(exc)}")
        return None, report

    if card.protocolVersion.split(".")[0] != "1":
        report.errors.append(
            f"protocolVersion {card.protocolVersion!r} is not 1.x"
        )
    if not card.url:
        report.errors.append("card has no url")

    for where, text in _card_prose(card):
        for pattern, label in _INJECTION_PATTERNS:
            import re

            if match := re.search(pattern, text):
                report.warnings.append(
                    f"{where} contains a prompt-injection shape ({label}): "
                    f"{match.group(0)[:60]!r}"
                )
    return (card if report.ok else None), report


def _card_prose(card: AgentCard) -> list[tuple[str, str]]:
    """Every card field that could end up inside a prompt.

    Enumerated rather than "all string fields": a new field added to the card
    should be a deliberate decision about whether a model may read it.
    """
    found = [("description", card.description), ("name", card.name)]
    for skill in card.skills:
        found.append((f"skill[{skill.id}].description", skill.description))
        found.append((f"skill[{skill.id}].name", skill.name))
    return found


def _first_line(exc: Exception) -> str:
    text = str(exc).splitlines()
    return text[0] if text else type(exc).__name__


def check_parts(parts: list[Any], *, subject: str = "message") -> tuple[list[Any], SecurityReport]:
    """Parse a message's parts, refusing the shapes this agent will not take.

    `file` parts are refused with a message naming the part, not silently
    dropped: a peer that sends a file and gets a normal-looking answer back
    would believe the file was read.
    """
    report = SecurityReport(subject=subject)
    parsed = []
    for index, raw in enumerate(parts):
        part = parse_part(raw)
        if part is None:
            report.errors.append(
                f"parts[{index}] is not a recognised part "
                "(expected kind: text, data or file)"
            )
            continue
        if part.kind == "file":
            report.errors.append(
                f"parts[{index}] is a file part, which this agent does not accept. "
                "Send the content as a text or data part, or make the file "
                "reachable to the agent another way."
            )
            continue
        parsed.append(part)
    return parsed, report


__all__ = [
    "SSRFBlocked",
    "SecurityReport",
    "check_parts",
    "check_url",
    "validate_remote_card",
]
