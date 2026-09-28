"""Example: adding your own tool.

The extension point is deliberately small. A tool is:

* a `ToolSpec` (name, description, JSON Schema, **effect class**, idempotency)
* a `run()` method that returns a `ToolResult`

Two fields carry most of the weight:

`effect_class` decides whether the call runs silently, prompts for approval,
or is refused. Getting this wrong in the *permissive* direction is how an
agent framework becomes dangerous; getting it wrong in the *strict*
direction makes it useless. `READ_ONLY` for inspection is what lets the
agent explore without training the user to approve blindly.

`idempotent` decides what happens when the process dies mid-call. `True`
means "safe to re-run on resume". Anything that touches the outside world
is `False`, and then the runtime refuses to guess -- it tells the model the
outcome is unknown.

Run it with:

    uaa tools | grep count_todos
    uaa run "how many TODOs are in this project"
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from unified_agent.agent.factory import build_agent
from unified_agent.agent.state import AgentState
from unified_agent.config import load_settings
from unified_agent.tools.base import Tool, ToolContext, ToolSpec
from unified_agent.types import EffectClass, ToolResult

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
_TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s]*(.*)")


class CountTodosTool(Tool):
    """Count TODO/FIXME markers. Pure inspection -> READ_ONLY, so no prompt."""

    spec = ToolSpec(
        name="count_todos",
        description=(
            "Count TODO / FIXME / XXX / HACK markers in a source tree, grouped "
            "by file. Read-only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory. Default '.'"},
                "glob": {"type": "string", "description": "Default '**/*'"},
            },
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
        idempotent=True,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        base = Path(args.get("path") or ".")
        if not base.is_absolute():
            base = ctx.workspace / base
        base = Path(os.path.realpath(base))
        if not base.is_dir():
            return ToolResult(success=False, error=f"not a directory: {base}")

        hits: dict[str, list[str]] = {}
        total = 0
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for name in files:
                full = Path(root) / name
                try:
                    text = full.read_text(encoding="utf-8", errors="strict")
                except (OSError, UnicodeDecodeError):
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    if match := _TODO_RE.search(line):
                        rel = str(full.relative_to(base))
                        hits.setdefault(rel, []).append(
                            f"  {lineno}: {match.group(1)} {match.group(2).strip()[:80]}"
                        )
                        total += 1

        if not hits:
            return ToolResult(success=True, output="(no TODO markers found)")

        lines = [f"{total} marker(s) in {len(hits)} file(s):", ""]
        for path in sorted(hits, key=lambda p: -len(hits[p])):
            lines.append(f"{path} ({len(hits[path])})")
            lines.extend(hits[path])
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"total": total, "files": len(hits)},
        )


async def main() -> None:
    settings = load_settings(create_if_missing=True)
    agent = await build_agent(settings=settings)
    try:
        # Register after construction. This is the whole extension mechanism:
        # anything in the registry is visible to the model by name.
        agent.registry.register(CountTodosTool())

        tool = agent.registry.get("count_todos")
        print("registered:")
        print(f"  {tool.spec.as_prompt_line()}")
        print(f"  effect={tool.spec.effect_class.value} idempotent={tool.spec.idempotent}")

        # The policy engine decides before anything runs. READ_ONLY means
        # this is allowed without prompting.
        verdict = agent.engine.decide(tool, {"path": "src"})
        print(f"  policy: {verdict.decision.value} ({verdict.reason or 'read-only, no prompt'})")

        # Build a context the way the runtime does, then call the tool.
        state = AgentState(task_id="example", session_id="example", goal="count todos")
        result = await tool.run({"path": "src"}, agent.runtime._tool_context(state))
        print("\nresult:")
        print("\n".join(result.output.splitlines()[:8]))
        if result.metadata:
            print(f"... metadata: {result.metadata}")
    finally:
        agent.close()


if __name__ == "__main__":
    asyncio.run(main())
