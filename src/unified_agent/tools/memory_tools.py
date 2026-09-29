"""Memory + skill + plan tools.

`load_skill` is the progressive-disclosure mechanism: only name/description
of every skill is in the system prompt, and the body (which can be
thousands of tokens) is pulled in by an explicit tool call. This is the
concrete answer to "skills instead of prompt accumulation".
"""

from __future__ import annotations

from typing import Any

from unified_agent.tools.base import Tool, ToolContext, ToolSpec
from unified_agent.types import EffectClass, ToolResult


class SaveMemoryTool(Tool):
    spec = ToolSpec(
        name="save_memory",
        description=(
            "Persist a durable fact or lesson for future sessions. Use for stable, "
            "reusable knowledge (project conventions, verified commands, decisions) — "
            "not for transient task state."
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "minLength": 8, "maxLength": 4000},
                "tags": {"type": "array", "items": {"type": "string"}},
                "scope": {
                    "type": "string",
                    "enum": ["project", "user"],
                    "description": "project = this repo only; user = everywhere.",
                },
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.WRITE_LOCAL,
    )

    def __init__(self, store, memory: Any = None) -> None:
        self.store = store
        self.memory = memory

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        content = args["content"]
        scope = args.get("scope") or "project"

        if self.memory is None:
            mid = self.store.add_memory(
                content=content,
                session_id=ctx.session_id,
                scope=scope,
                tags=args.get("tags") or [],
                importance=float(args.get("importance", 0.5)),
                source="agent",
            )
            return ToolResult(success=True, output=f"memory saved ({mid})", metadata={"id": mid})

        # Goes through the curator so a fact that contradicts a stored one
        # replaces it instead of sitting beside it. The result says which
        # happened, because "saved" and "replaced an older fact" are different
        # outcomes and the model should know which one it got.
        outcome = await self.memory.remember_text(
            content,
            scope=scope,
            session_id=ctx.session_id,
            source="agent",
            tags=args.get("tags") or [],
        )
        decision = outcome.decisions[0] if outcome.decisions else None
        verdict = decision.verdict.value if decision else "add"
        detail = f"memory {verdict}"
        if outcome.updated:
            old_id, new_id = outcome.updated[0]
            detail = f"memory replaced {old_id} with {new_id} ({decision.reason if decision else ''})"
        elif outcome.added:
            detail = f"memory saved ({outcome.added[0]})"
        elif outcome.duplicated:
            detail = "memory already known; nothing written"
        elif outcome.rejected:
            detail = "memory rejected as not worth keeping"
        return ToolResult(
            success=True,
            output=detail,
            metadata={
                "verdict": verdict,
                "added": outcome.added,
                "updated": [list(pair) for pair in outcome.updated],
            },
        )


class SearchMemoryTool(Tool):
    spec = ToolSpec(
        name="search_memory",
        description="Search previously saved memories. Call this before assuming you have no context.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "scope": {"type": "string", "enum": ["project", "user"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
    )

    def __init__(self, store, memory: Any = None) -> None:
        self.store = store
        self.memory = memory

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self.memory is not None:
            hits = await self.memory.recall(
                args["query"], scope=args.get("scope"), limit=int(args.get("limit") or 8)
            )
        else:
            hits = self.store.search_memories(
                args["query"], scope=args.get("scope"), limit=int(args.get("limit") or 8)
            )
        if not hits:
            return ToolResult(success=True, output="(no memories matched)", metadata={"count": 0})
        lines = [
            f"[{h['scope']}] {h['content']}  (tags: {h['tags'] or '-'}, {h['created_at'][:10]})"
            for h in hits
        ]
        return ToolResult(
            success=True, output="\n".join(lines), metadata={"count": len(hits)}
        )


class DeleteMemoryTool(Tool):
    spec = ToolSpec(
        name="delete_memory",
        description="Delete a memory by id. Use when a memory is wrong or obsolete.",
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.WRITE_LOCAL,
        requires_confirmation=True,
    )

    def __init__(self, store) -> None:
        self.store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        ok = self.store.delete_memory(args["id"])
        return ToolResult(
            success=ok, output="deleted" if ok else "", error=None if ok else "no such memory id"
        )


class LoadSkillTool(Tool):
    spec = ToolSpec(
        name="load_skill",
        description=(
            "Load the full instructions for a skill listed in the system prompt. "
            "Call this before doing work that matches a skill's description."
        ),
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
    )

    def __init__(self, skills) -> None:
        self.skills = skills

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = args["name"]
        skill = self.skills.get(name)
        if skill is None:
            return ToolResult(
                success=False,
                error=(
                    f"unknown skill {name!r}. Available: "
                    f"{', '.join(s.name for s in self.skills.list()) or '(none)'}"
                ),
            )
        if skill.status != "active":
            return ToolResult(
                success=False,
                error=f"skill {name!r} is {skill.status}, not active; it cannot be used",
            )
        return ToolResult(
            success=True,
            output=skill.render(),
            metadata={"skill": name, "allowed_tools": sorted(skill.allowed_tools)},
        )


class UpdatePlanTool(Tool):
    spec = ToolSpec(
        name="update_plan",
        description=(
            "Replace the current plan and tick steps off. Steps are *intentions* — do not "
            "pre-bind tool arguments here, decide them when you execute the step. "
            "Send the complete list every time, with the status of each step, so the "
            "checklist stays truthful."
        ),
        parameters={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "minLength": 4},
                            "expected_tools": {"type": "array", "items": {"type": "string"}},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "running", "completed", "failed", "skipped"],
                            },
                            "note": {"type": "string"},
                        },
                        "required": ["description"],
                        "additionalProperties": False,
                    },
                },
                "reason": {"type": "string", "description": "Why the plan changed."},
            },
            "required": ["steps"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        steps = args["steps"]
        marks = {"pending": " ", "running": "~", "completed": "x", "failed": "!", "skipped": "-"}
        body = "\n".join(
            f"[{marks.get(s.get('status', 'pending'), ' ')}] {i + 1}. {s['description']}"
            for i, s in enumerate(steps)
        )
        reason = args.get("reason")
        out = f"plan replaced with {len(steps)} step(s)"
        if reason:
            out += f" (reason: {reason})"
        return ToolResult(
            success=True, output=f"{out}\n{body}", metadata={"steps": steps, "reason": reason}
        )


class FinishTool(Tool):
    """Optional explicit completion.

    Not required: a plain assistant message with no tool calls also ends the
    task. Present because some models (especially via the text protocol)
    are much more reliable when there is an explicit terminal action.
    """

    spec = ToolSpec(
        name="finish",
        description="Finish the task and return the final answer to the user.",
        parameters={
            "type": "object",
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "status": {"type": "string", "enum": ["completed", "blocked"]},
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            output=args["answer"],
            metadata={"final": True, "status": args.get("status") or "completed"},
        )


class RecallTool(Tool):
    """Search this agent's own past tasks.

    Distinct from `search_memory`: memory holds facts the agent decided were
    worth keeping, while this searches what it actually *did* -- the goal it was
    given and the answer it produced. "Have I worked on this before" and "what
    did I conclude last time" are questions the memory store cannot answer,
    because nothing decided they were facts.
    """

    spec = ToolSpec(
        name="recall",
        description=(
            "Search past tasks by goal or by their final answer. Use this when "
            "the user refers to earlier work ('like last time', 'the thing we "
            "did before') or before starting something that sounds familiar. "
            "Returns matching task ids; read one with `uaa task show`."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
    )

    def __init__(self, store: Any) -> None:
        self.store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        rows = self.store.search_tasks(
            args["query"], limit=int(args.get("limit") or 8)
        )
        if not rows:
            return ToolResult(
                success=True,
                output=(
                    "no past task matches that. Either this is new work, or the "
                    "wording differs -- try the words from the goal itself."
                ),
                metadata={"matches": 0},
            )
        lines = []
        for row in rows:
            age = _ago(row.get("created_at") or "")
            lines.append(
                f"- {row['id']}  [{row['status']}]  {age}  {row['goal'][:160]}"
            )
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"matches": len(rows), "ids": [r["id"] for r in rows]},
        )


def _ago(created_at: str) -> str:
    """A rough age, so a list of tasks is ordered in the reader's head too."""
    from datetime import datetime, timezone

    try:
        when = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return "?"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - when
    seconds = int(delta.total_seconds())
    if seconds < 3600:
        return f"{max(seconds, 0) // 60}m ago"
    if seconds < 86_400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86_400}d ago"


def build_memory_tools(store, memory: Any = None) -> list[Tool]:
    return [
        SaveMemoryTool(store, memory),
        SearchMemoryTool(store, memory),
        RecallTool(store),
        DeleteMemoryTool(store),
    ]


def build_skill_tools(skills) -> list[Tool]:
    return [LoadSkillTool(skills)]
