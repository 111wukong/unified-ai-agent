"""Post-task reflection: memory extraction + skill candidacy.

Where the spec's "Reflector" differs from what is implemented here: there
is **no per-step reflection LLM call**. AutoGen's token cost blowup (the
comparison articles measure 5-6x LangGraph) comes precisely from adding a
model call per agent per round; a reflector that fires every step buys a
little and costs a lot.

In-loop reflection is free instead: a failed tool result is written into
the scratchpad, and the model reads it on its next turn and adapts. The
Reflector here runs **once per task**, at the end, for the things that
genuinely need a separate judgement call: what is worth remembering, and
whether this task revealed a reusable procedure.

Candidates never become active skills. `candidate -> validated -> approved
-> active` is enforced by state, so a skill the agent wrote for itself
cannot execute without a human moving it forward.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from unified_agent.agent.prompts import REFLECTOR_PROMPT
from unified_agent.agent.state import AgentState, TaskStatus
from unified_agent.config import Settings
from unified_agent.models.base import ChatModel
from unified_agent.observability.events import EventType
from unified_agent.types import Message

REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "minLength": 8, "maxLength": 1000},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        },
        "skill_candidate": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["name", "description"],
            "additionalProperties": False,
        },
    },
    "required": ["memories"],
    "additionalProperties": False,
}

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class Reflection:
    def __init__(self, memories: list[dict[str, Any]], skill_candidate: dict[str, Any] | None):
        self.memories = memories
        self.skill_candidate = skill_candidate


class Reflector:
    def __init__(
        self,
        *,
        model: ChatModel,
        settings: Settings,
        store: Any,
        memory: Any = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.store = store
        # When present, extracted facts go through the curator, which
        # reconciles them against what is already stored. Without it the
        # reflector appends, and an appended contradiction is exactly the
        # problem the curator exists to prevent.
        self.memory = memory

    def response_format(self) -> dict[str, Any] | None:
        caps = self.model.capabilities
        if caps.json_schema:
            return {
                "type": "json_schema",
                "json_schema": {"name": "reflection", "schema": REFLECTION_SCHEMA, "strict": True},
            }
        if caps.json_object:
            return {"type": "json_object"}
        return None

    def worth_running(self, state: AgentState) -> bool:
        """Reflection is only useful when the task actually did something.

        Skipping is the correct default: a task that failed, or that never
        called a tool, has nothing reusable to teach.
        """
        if state.status is not TaskStatus.COMPLETED:
            return False
        tool_entries = [e for e in state.log if e.kind == "tool"]
        return len(tool_entries) >= 2 and any(e.success for e in tool_entries)

    async def reflect(self, state: AgentState) -> Reflection:
        transcript = "\n\n".join(
            e.render(max_chars=1_200) for e in state.log if e.kind == "tool"
        )[:40_000]
        user = (
            f"# Goal\n\n{state.goal}\n\n"
            f"# What the agent did\n\n{transcript}\n\n"
            f"# Final answer\n\n{(state.answer or '')[:2_000]}\n"
        )
        resp = await self.model.chat(
            [
                Message(role="system", content=REFLECTOR_PROMPT),
                Message(role="user", content=user),
            ],
            temperature=0.0,
            response_format=self.response_format(),
            max_output_tokens=1_200,
        )
        state.usage = state.usage + resp.usage
        state.model_calls += 1
        # Persist the reflection call too: it costs tokens, and a cost budget
        # that only counts the visible loop is not a budget.
        self.store.append(
            state.task_id,
            EventType.MODEL_RESPONSE,
            {"phase": "reflect", "model": resp.model, "usage": resp.usage.model_dump()},
        )

        payload = _extract(resp.content)
        memories = [m for m in (payload.get("memories") or []) if isinstance(m, dict)]
        candidate = payload.get("skill_candidate")
        if not isinstance(candidate, dict) or not candidate.get("name"):
            candidate = None
        return Reflection(memories=memories, skill_candidate=candidate)

    async def run_and_persist(self, state: AgentState) -> Reflection:
        from unified_agent.memory.extract import Candidate

        reflection = await self.reflect(state)
        candidates = [
            Candidate(
                content=str(item.get("content") or "").strip(),
                tags=[str(t) for t in (item.get("tags") or [])],
                importance=float(item.get("importance", 0.5)),
            )
            for item in reflection.memories[:5]
            if len(str(item.get("content") or "").strip()) >= 8
        ]

        if self.memory is not None and candidates:
            outcome = await self.memory.remember(
                candidates, scope="project", session_id=state.session_id, source="reflection"
            )
            for memory_id in outcome.added:
                self.store.append(
                    state.task_id,
                    EventType.MEMORY_WRITTEN,
                    {"id": memory_id, "verdict": "add"},
                )
            for old_id, new_id in outcome.updated:
                self.store.append(
                    state.task_id,
                    EventType.MEMORY_WRITTEN,
                    {"id": new_id, "verdict": "update", "superseded": old_id},
                )
        else:
            for candidate in candidates:
                memory_id = self.store.add_memory(
                    content=candidate.content,
                    session_id=state.session_id,
                    scope="project",
                    tags=candidate.tags,
                    importance=candidate.importance,
                    source="reflection",
                )
                self.store.append(
                    state.task_id,
                    EventType.MEMORY_WRITTEN,
                    {"id": memory_id, "verdict": "add", "content": candidate.content},
                )
        if reflection.skill_candidate:
            path = self.write_candidate(reflection.skill_candidate, state)
            self.store.append(
                state.task_id,
                EventType.SKILL_CANDIDATE,
                {
                    "name": reflection.skill_candidate.get("name"),
                    "path": str(path) if path else None,
                    "reason": reflection.skill_candidate.get("reason"),
                },
            )
        return reflection

    def write_candidate(self, candidate: dict[str, Any], state: AgentState) -> Path | None:
        """Write a SKILL.md with status=candidate. Never auto-activates."""
        name = _slug(str(candidate.get("name") or ""))
        description = str(candidate.get("description") or "").strip()
        if not name or len(description) < 10:
            return None
        root = self.settings.home / "skills-candidates" / name
        root.mkdir(parents=True, exist_ok=True)
        steps = [
            f"{i + 1}. {s.description}"
            for i, s in enumerate(state.plan)
        ] or ["1. (no plan was recorded)"]
        body = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description[:1000]}\n"
            f"metadata:\n"
            f"  status: candidate\n"
            f"  generated_by: unified-ai-agent\n"
            f"  source_task: {state.task_id}\n"
            f"---\n\n"
            f"# {name}\n\n"
            f"{description}\n\n"
            f"## Procedure observed\n\n"
            + "\n".join(steps)
            + "\n\n## Review checklist\n\n"
            "- [ ] Is this reusable beyond the original task?\n"
            "- [ ] Are the steps correct, or did they merely happen to work once?\n"
            "- [ ] Does it need any tool the allowlist does not grant?\n\n"
            "_Candidate only. Move it to `approved` and then `active` before it can run._\n"
        )
        path = root / "SKILL.md"
        path.write_text(body, encoding="utf-8")
        return path


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:64]


def _extract(raw: str) -> dict[str, Any]:
    import json

    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}
