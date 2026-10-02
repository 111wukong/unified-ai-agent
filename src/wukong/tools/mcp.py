"""MCP client (stdio), supporting both protocol eras.

The 2026-07-28 revision of MCP removed the `initialize`/`initialized`
handshake and the `Mcp-Session-Id` header: the protocol became stateless,
with protocol version / client identity / capabilities carried per request
in `_meta`, and a new `server/discover` RPC to declare what the server
supports. Compatibility is explicitly *not* guaranteed across that line.

So a client that wants to talk to the installed base has to do era
detection: try `server/discover` first, fall back to the legacy
`initialize` handshake. That is what this does.

The official SDK is the recommended path for production (the migration
guide says as much). This implementation exists so that the tool-provider
boundary is testable without a network, and so `wukong` has no
hard dependency on the SDK's release cadence.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from wukong.config import McpServerConfig
from wukong.errors import ToolError
from wukong.tools.base import Tool, ToolContext, ToolSpec
from wukong.tools.permissions import scrub_env
from wukong.types import ToolResult

CLIENT_INFO = {"name": "wukong", "version": "0.1.0"}
# Newest first: the version we prefer to advertise.
SUPPORTED_VERSIONS = ["2026-07-28", "2025-06-18", "2024-11-05"]


class McpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"MCP error {code}: {message}")
        self.code = code
        self.message = message


class McpStdioClient:
    """Newline-delimited JSON-RPC 2.0 over a child process's stdio."""

    def __init__(
        self,
        config: McpServerConfig,
        *,
        known_secrets: list[str] | None = None,
    ) -> None:
        self.config = config
        self.known_secrets = known_secrets or []
        self.proc: asyncio.subprocess.Process | None = None
        self.era: str | None = None
        self.server_info: dict[str, Any] = {}
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._stderr: list[str] = []

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        env = scrub_env(known_secrets=self.known_secrets)
        env.update({k: v for k, v in self.config.env.items()})
        try:
            self.proc = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            raise ToolError(
                f"MCP server {self.config.name!r}: command not found: {self.config.command}"
            ) from exc
        self._reader = asyncio.create_task(self._read_loop())
        asyncio.create_task(self._drain_stderr())

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self.proc and self.proc.returncode is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                self.proc.kill()

    async def __aenter__(self) -> "McpStdioClient":
        await self.start()
        await self.handshake()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                return
            self._stderr.append(line.decode("utf-8", errors="replace").rstrip())
            del self._stderr[:-40]

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            try:
                line = await self.proc.stdout.readline()
            except (asyncio.CancelledError, ValueError):
                return
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue
            if "id" in message and message["id"] in self._pending:
                future = self._pending.pop(message["id"])
                if not future.done():
                    future.set_result(message)

    # -- transport --------------------------------------------------------
    def _meta(self) -> dict[str, Any]:
        """Per-request metadata. Required in the stateless era; harmless before."""
        return {
            "protocolVersion": SUPPORTED_VERSIONS[0],
            "clientInfo": CLIENT_INFO,
            "capabilities": {},
        }

    async def _request(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        if self.proc is None or self.proc.stdin is None:
            raise ToolError(f"MCP server {self.config.name!r} is not running")
        self._next_id += 1
        request_id = self._next_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()
        try:
            message = await asyncio.wait_for(future, timeout=timeout or self.config.startup_timeout_s)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise ToolError(f"MCP {self.config.name}: {method} timed out") from None
        if "error" in message:
            raise McpError(
                int(message["error"].get("code", -1)), str(message["error"].get("message", ""))
            )
        return message.get("result") or {}

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()

    # -- handshake --------------------------------------------------------
    async def handshake(self) -> str:
        """Detect which protocol era the server speaks. Returns 'stateless' or 'legacy'."""
        try:
            result = await self._request(
                "server/discover", {"_meta": self._meta()}, timeout=min(10.0, self.config.startup_timeout_s)
            )
        except McpError:
            result = None
        except ToolError:
            result = None
        if result is not None:
            self.era = "stateless"
            self.server_info = result
            return self.era

        # Legacy: version negotiation then the initialized notification.
        try:
            result = await self._request(
                "initialize",
                {
                    "protocolVersion": SUPPORTED_VERSIONS[-1],
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
            )
        except McpError as exc:
            raise ToolError(
                f"MCP server {self.config.name!r}: neither server/discover nor initialize "
                f"succeeded ({exc}). stderr: {' | '.join(self._stderr[-3:])}"
            ) from exc
        self.era = "legacy"
        self.server_info = result
        await self._notify("notifications/initialized", {})
        return self.era

    # -- tools ------------------------------------------------------------
    async def list_tools(self) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if self.era == "stateless":
            params["_meta"] = self._meta()
        result = await self._request("tools/list", params)
        tools = result.get("tools") or []
        # `ttlMs` / `cacheScope` are required in the 2026-07-28 era; we do not
        # cache across runs yet, but surface them for the caller.
        self._list_meta = {
            "ttlMs": result.get("ttlMs"),
            "cacheScope": result.get("cacheScope"),
        }
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        params: dict[str, Any] = {"name": name, "arguments": arguments}
        if self.era == "stateless":
            params["_meta"] = self._meta()
        result = await self._request("tools/call", params)
        text = _flatten_content(result.get("content") or [])
        is_error = bool(result.get("isError"))
        # Stateless era: a long task may hand back a handle to poll.
        if handle := (result.get("_meta") or {}).get("taskHandle"):
            text += f"\n[server returned a task handle: {handle}]"
        return ToolResult(
            success=not is_error,
            output=text,
            error="the MCP server reported an error" if is_error else None,
            metadata={"mcp_server": self.config.name, "era": self.era},
        )


def _flatten_content(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text") or "")
        elif kind == "resource":
            resource = block.get("resource") or {}
            parts.append(f"[resource {resource.get('uri')}]\n{resource.get('text') or ''}")
        elif kind == "image":
            parts.append(f"[image {block.get('mimeType')}, {len(block.get('data') or '')} bytes]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False)[:2000])
    return "\n".join(parts) or "(no content)"


class McpTool(Tool):
    """An MCP tool projected into the internal Tool protocol.

    Effect class comes from the server's config, not from the server's own
    claims. A third-party MCP server describing its tool as "read only" is
    not evidence that it is.
    """

    def __init__(self, client: McpStdioClient, remote: dict[str, Any]) -> None:
        self.client = client
        self.remote_name = remote.get("name") or "unnamed"
        self.spec = ToolSpec(
            name=f"mcp__{client.config.name}__{self.remote_name}",
            description=(remote.get("description") or "")[:1024],
            parameters=remote.get("inputSchema") or {"type": "object", "properties": {}},
            effect_class=client.config.effect_class,
            requires_confirmation=client.config.requires_confirmation,
            # A remote tool's idempotency is unknown; assume the worst.
            idempotent=False,
            source=f"mcp:{client.config.name}",
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        return await self.client.call_tool(self.remote_name, args)


async def connect_servers(
    configs: list[McpServerConfig],
    *,
    known_secrets: list[str] | None = None,
) -> tuple[list[McpTool], list[str]]:
    """Start every enabled server and collect its tools. Never fatal."""
    tools: list[McpTool] = []
    problems: list[str] = []
    for config in configs:
        if not config.enabled:
            continue
        client = McpStdioClient(config, known_secrets=known_secrets)
        try:
            await client.start()
            await client.handshake()
            for remote in await client.list_tools():
                tools.append(McpTool(client, remote))
        except Exception as exc:  # noqa: BLE001 - one bad server must not kill the run
            problems.append(f"{config.name}: {type(exc).__name__}: {exc}")
            await client.close()
    return tools, problems


def mcp_env_preview(config: McpServerConfig) -> str:
    keys = sorted(config.env)
    return f"{config.command} {' '.join(config.args)} (env: {', '.join(keys) or 'none'})"


__all__ = ["McpStdioClient", "McpTool", "connect_servers", "McpError"]
