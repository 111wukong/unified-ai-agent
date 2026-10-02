"""HTTP tools.

Domain allowlist is enforced by the PermissionEngine before the request is
built, but the redirect chain is the leak: an allowlisted host that
302s to `evil.example` would otherwise be followed transparently. So
redirects are followed manually, re-checking every hop.
"""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from wukong.tools.base import Tool, ToolContext, ToolSpec
from wukong.types import EffectClass, ToolResult

# Fallback only. The effective ceiling is `permissions.network.max_response_bytes`;
# kept as a module constant so the tools stay usable when constructed directly.
DEFAULT_MAX_BYTES = 2_000_000
_ALLOWED_METHODS = {"GET", "POST"}


class _HttpBase(Tool):
    effect_class = EffectClass.NETWORK

    def __init__(
        self,
        *,
        allow_check: Callable[[str], bool] | None = None,
        timeout_s: float = 60.0,
        user_agent: str = "wukong/0.1",
        max_response_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.allow_check = allow_check or (lambda url: True)
        self.timeout_s = timeout_s
        self.user_agent = user_agent
        self.max_response_bytes = max_response_bytes

    async def _request(
        self, method: str, url: str, *, body: dict[str, Any] | None, headers: dict[str, str]
    ) -> ToolResult:
        hops: list[str] = []
        current = url
        for _ in range(6):
            host = urlparse(current).hostname or ""
            if not self.allow_check(current):
                return ToolResult(
                    success=False,
                    error=f"redirect chain left the allowlist at {host!r} ({current})",
                    metadata={"hops": hops},
                )
            hops.append(current)
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_s,
                    follow_redirects=False,
                    headers={"User-Agent": self.user_agent, **headers},
                ) as client:
                    resp = await client.request(method, current, json=body)
            except httpx.HTTPError as exc:
                return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
            if resp.is_redirect and resp.headers.get("location"):
                current = str(httpx.URL(current).join(resp.headers["location"]))
                method, body = "GET", None
                continue
            raw = resp.content[: self.max_response_bytes]
            text = raw.decode(resp.encoding or "utf-8", errors="replace")
            if resp.headers.get("content-type", "").startswith("application/json"):
                try:
                    text = json.dumps(resp.json(), indent=2, ensure_ascii=False)
                except ValueError:
                    pass
            truncated = len(resp.content) > self.max_response_bytes
            return ToolResult(
                success=resp.is_success,
                output=f"HTTP {resp.status_code} {current}\n\n{text}",
                error=None if resp.is_success else f"HTTP {resp.status_code}",
                truncated=truncated,
                metadata={
                    "status": resp.status_code,
                    "bytes": len(resp.content),
                    "hops": hops,
                },
            )
        return ToolResult(success=False, error="too many redirects", metadata={"hops": hops})


class HttpGetTool(_HttpBase):
    spec = ToolSpec(
        name="http_get",
        description=(
            "HTTP GET a URL. The host must be on permissions.network.allow_domains. "
            "Redirects are re-checked against the allowlist."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "minLength": 8},
                "headers": {"type": "object", "description": "Extra request headers."},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.NETWORK,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        return await self._request(
            "GET", args["url"], body=None, headers=args.get("headers") or {}
        )


class HttpPostTool(_HttpBase):
    spec = ToolSpec(
        name="http_post",
        description=(
            "HTTP POST a JSON body to a URL. The host must be on the network "
            "allowlist. Treat this as a side-effecting call."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "minLength": 8},
                "body": {"type": "object"},
                "headers": {"type": "object"},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.EXTERNAL_SIDE_EFFECT,
        requires_confirmation=True,
        idempotent=False,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        return await self._request(
            "POST", args["url"], body=args.get("body"), headers=args.get("headers") or {}
        )


def build_net_tools(
    *,
    allow_check: Callable[[str], bool] | None = None,
    max_response_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Tool]:
    kwargs = {"allow_check": allow_check, "max_response_bytes": max_response_bytes}
    return [HttpGetTool(**kwargs), HttpPostTool(**kwargs)]
