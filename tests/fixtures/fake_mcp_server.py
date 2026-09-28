"""A minimal MCP server over stdio, for testing both protocol eras.

Speaks the legacy (pre-2026-07-28) handshake by default; pass
`--era stateless` to refuse `initialize` and answer `server/discover`
instead, which is what the 2026-07-28 revision requires.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    era = "legacy"
    if "--era" in sys.argv:
        era = sys.argv[sys.argv.index("--era") + 1]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = message.get("method")
        request_id = message.get("id")

        if method == "server/discover":
            if era == "stateless":
                _reply(request_id, {"protocolVersion": "2026-07-28", "capabilities": {"tools": {}}})
            else:
                _error(request_id, -32601, "Method not found")
            continue

        if method == "initialize":
            if era == "stateless":
                _error(request_id, -32601, "initialize was removed in 2026-07-28")
            else:
                _reply(
                    request_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1.0"},
                    },
                )
            continue

        if method == "notifications/initialized":
            continue

        if method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo the given text back.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]
            }
            if era == "stateless":
                # Required fields in the 2026-07-28 revision.
                result["ttlMs"] = 60_000
                result["cacheScope"] = "public"
            _reply(request_id, result)
            continue

        if method == "tools/call":
            params = message.get("params") or {}
            args = params.get("arguments") or {}
            if params.get("name") != "echo":
                _error(request_id, -32602, f"unknown tool {params.get('name')!r}")
                continue
            if "fail" in args:
                _reply(
                    request_id,
                    {"content": [{"type": "text", "text": "boom"}], "isError": True},
                )
                continue
            _reply(
                request_id,
                {"content": [{"type": "text", "text": f"echo: {args.get('text', '')}"}]},
            )
            continue

        _error(request_id, -32601, f"Method not found: {method}")


def _reply(request_id: object, result: dict) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n")
    sys.stdout.flush()


def _error(request_id: object, code: int, message: str) -> None:
    sys.stdout.write(
        json.dumps({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})
        + "\n"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
