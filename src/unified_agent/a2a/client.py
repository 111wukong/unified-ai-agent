"""A2A client: calling another agent.

Every outbound step here is a place where a peer-supplied value becomes a
request this process makes, so each one is guarded:

* the card URL is checked against `a2a.allow_hosts` and against the private
  address ranges, and **re-checked on every redirect hop** -- following a
  redirect transparently is how an allowlisted host becomes a proxy for an
  internal one, which is the same bug the HTTP tools already document;
* the card that comes back is treated as untrusted input and validated before
  any field of it is used;
* the call itself goes to the card's `url`, which is *also* peer-supplied, so
  it goes through the same check rather than being trusted because it arrived
  inside a validated card.

That last point is the one worth stating: a valid card is not a trusted card.
The schema says the shape is right, not that the host is one we may reach.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from unified_agent.a2a.card import AgentCard
from unified_agent.a2a.security import SecurityReport, SSRFBlocked, check_url, validate_remote_card
from unified_agent.a2a.types import TaskState

CARD_PATH = "/.well-known/agent-card.json"
_MAX_HOPS = 5


class A2AClientError(Exception):
    """A remote call failed, with a message worth showing a user."""


@dataclass
class RemoteCall:
    """What came back, and what was suspicious about it."""

    task: dict[str, Any]
    card: AgentCard
    report: SecurityReport

    @property
    def state(self) -> TaskState:
        return TaskState(self.task.get("status", {}).get("state", "failed"))

    @property
    def answer(self) -> str:
        """The agent's text, from the artifacts the spec defines."""
        chunks: list[str] = []
        for artifact in self.task.get("artifacts") or []:
            for part in artifact.get("parts") or []:
                if part.get("kind") == "text" and part.get("text"):
                    chunks.append(str(part["text"]))
        return "\n\n".join(chunks)


class A2AClient:
    def __init__(self, settings: Any) -> None:
        self.config = settings.a2a

    # -- discovery --------------------------------------------------------
    async def fetch_card(self, base_url: str) -> tuple[AgentCard, SecurityReport]:
        """Fetch and vet a peer's card.

        The card is *untrusted input*: `description` is text an attacker
        writes, and it is exactly the text a model reads when deciding
        whether to call the agent. So it is schema-validated and run through
        the injection tripwire before it is used for anything.
        """
        url = base_url.rstrip("/") + CARD_PATH
        raw = await self._get_json(url)
        card, report = validate_remote_card(raw, subject=f"card at {url}")
        if card is None:
            raise A2AClientError(
                f"{url} is not a usable agent card: " + "; ".join(report.errors)
            )
        # The card's own `url` is where calls go, and it is peer-supplied.
        # Validate it too rather than trusting it because the card parsed.
        check_url(
            card.url,
            allow_hosts=self.config.allow_hosts,
            allow_private=self.config.allow_private_networks,
        )
        return card, report

    # -- calls ------------------------------------------------------------
    async def send(
        self,
        base_url: str,
        text: str,
        *,
        context_id: str | None = None,
        peer: str | None = None,
    ) -> RemoteCall:
        card, report = await self.fetch_card(base_url)
        task = await self._rpc(
            card.url,
            "message/send",
            {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                    "messageId": f"msg-{abs(hash((text, context_id))) % 10**12}",
                    "contextId": context_id or "",
                },
                "metadata": {"peer": peer or self.config.name},
            },
        )
        if "error" in task:
            raise A2AClientError(_rpc_error(task))
        return RemoteCall(task=task.get("result") or {}, card=card, report=report)

    # -- transport --------------------------------------------------------
    async def _get_json(self, url: str) -> Any:
        body = await self._request("GET", url)
        try:
            return json.loads(body)
        except ValueError as exc:
            raise A2AClientError(f"{url} did not return JSON: {body[:200]!r}") from exc

    async def _rpc(self, url: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": f"{method}-1", "method": method, "params": params}
        body = await self._request("POST", url, json_body=payload)
        try:
            parsed = json.loads(body)
        except ValueError as exc:
            raise A2AClientError(f"{url} did not return JSON: {body[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise A2AClientError(f"{url} returned {type(parsed).__name__}, expected an object")
        return parsed

    async def _request(
        self, method: str, url: str, *, json_body: dict[str, Any] | None = None
    ) -> str:
        """One guarded request, following redirects manually.

        Redirects are followed by hand and re-checked at every hop. Letting
        `httpx` follow them would mean an allowlisted host can redirect to
        anything, and the allowlist would only be checking the first URL.
        """
        current = url
        hops: list[str] = []
        for _ in range(_MAX_HOPS):
            try:
                check_url(
                    current,
                    allow_hosts=self.config.allow_hosts,
                    allow_private=self.config.allow_private_networks,
                )
            except SSRFBlocked as exc:
                raise A2AClientError(f"refusing to call {current}: {exc}") from exc
            hops.append(current)
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.timeout_s,
                    follow_redirects=False,
                    headers={"User-Agent": f"{self.config.name}/a2a"},
                ) as client:
                    response = await client.request(method, current, json=json_body)
            except httpx.HTTPError as exc:
                raise A2AClientError(f"{type(exc).__name__}: {exc}") from exc

            if response.is_redirect and response.headers.get("location"):
                current = str(httpx.URL(current).join(response.headers["location"]))
                # A redirect turns a POST into a GET; re-POSTing the body to
                # a new host would leak the caller's message to it.
                method, json_body = "GET", None
                continue
            if response.status_code >= 400:
                raise A2AClientError(
                    f"{current} returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            raw = response.content[: self.config.max_response_bytes]
            return raw.decode(response.encoding or "utf-8", errors="replace")
        raise A2AClientError(
            f"too many redirects fetching {url} (hops: {', '.join(hops)})"
        )


def _rpc_error(response: dict[str, Any]) -> str:
    error = response.get("error") or {}
    detail = error.get("data")
    text = f"remote agent refused the call: {error.get('message')}"
    if isinstance(detail, dict) and detail.get("problems"):
        text += " (" + "; ".join(str(p) for p in detail["problems"]) + ")"
    return text


__all__ = ["A2AClient", "A2AClientError", "CARD_PATH", "RemoteCall"]
