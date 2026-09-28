"""Structured planning with a repair loop.

Two decisions that differ from the spec:

1. **Plans are validated, not parsed-and-hoped.** A planner that returns
   free text and gets `json.loads`-ed is a planner that fails 5% of the
   time on trailing commas. Validation failure feeds the *exact* error back
   and retries, up to a small budget.
2. **Steps carry intent, not arguments.** See state.PlanStep.
"""

from __future__ import annotations

import json
import re
from typing import Any

from unified_agent.agent.prompts import PLANNER_PROMPT
from unified_agent.agent.state import AgentState, PlanStep, StepStatus
from unified_agent.config import Settings
from unified_agent.errors import ModelError
from unified_agent.models.base import ChatModel
from unified_agent.observability.events import EventType
from unified_agent.types import Message

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 600},
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "minLength": 4, "maxLength": 400},
                    "expected_tools": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["steps"],
    "additionalProperties": False,
}

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class Planner:
    def __init__(self, *, model: ChatModel, settings: Settings, store: Any = None) -> None:
        self.model = model
        self.settings = settings
        self.store = store

    def response_format(self) -> dict[str, Any] | None:
        caps = self.model.capabilities
        if caps.json_schema:
            return {
                "type": "json_schema",
                "json_schema": {"name": "plan", "schema": PLAN_SCHEMA, "strict": True},
            }
        if caps.json_object:
            return {"type": "json_object"}
        return None

    async def plan(self, state: AgentState, *, tool_names: list[str]) -> list[PlanStep]:
        user = (
            f"# Goal\n\n{state.goal}\n\n"
            f"# Available tools\n\n{', '.join(tool_names) or '(none)'}\n\n"
            f"# Constraints\n\n"
            f"- At most {self.settings.agent.max_plan_steps} steps.\n"
            "- Only use tools from the list above in `expected_tools`.\n"
        )
        messages = [
            Message(role="system", content=PLANNER_PROMPT),
            Message(role="user", content=user),
        ]
        response_format = self.response_format()
        last_error = ""
        attempts = self.settings.agent.planner_repair_attempts + 1

        for attempt in range(attempts):
            try:
                resp = await self.model.chat(
                    messages,
                    temperature=0.0,
                    response_format=response_format,
                    max_output_tokens=min(2_000, self.model.spec.max_output_tokens),
                )
            except ModelError:
                raise

            if self.store is not None:
                self.store.append(
                    state.task_id,
                    EventType.MODEL_RESPONSE,
                    {
                        "phase": "plan",
                        "model": resp.model,
                        "usage": resp.usage.model_dump(),
                        "content_preview": (resp.content or "")[:400],
                    },
                )
            state.usage = state.usage + resp.usage
            state.model_calls += 1

            try:
                steps = self._parse(resp.content, tool_names)
            except ValueError as exc:
                last_error = str(exc)
                if self.store is not None:
                    self.store.append(
                        state.task_id,
                        EventType.PLAN_INVALID,
                        {"attempt": attempt + 1, "error": last_error},
                    )
                if attempt == attempts - 1:
                    break
                messages = [
                    *messages,
                    Message(role="assistant", content=resp.content),
                    Message(
                        role="user",
                        content=(
                            f"That plan was rejected: {last_error}\n\n"
                            "Return corrected JSON only. Same schema, no prose."
                        ),
                    ),
                ]
                continue
            return steps

        raise ModelError(
            f"planner failed after {attempts} attempt(s): {last_error}", retryable=False
        )

    def _parse(self, raw: str, tool_names: list[str]) -> list[PlanStep]:
        payload = _extract_json(raw)
        if not isinstance(payload, dict):
            raise ValueError("response was not a JSON object")
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("`steps` must be a non-empty array")

        known = set(tool_names)
        steps: list[PlanStep] = []
        for i, item in enumerate(raw_steps[: self.settings.agent.max_plan_steps]):
            if not isinstance(item, dict):
                raise ValueError(f"steps[{i}] must be an object")
            description = str(item.get("description") or "").strip()
            if len(description) < 4:
                raise ValueError(f"steps[{i}].description is missing or too short")
            expected = [str(t) for t in (item.get("expected_tools") or [])]
            unknown = [t for t in expected if t not in known]
            if unknown:
                raise ValueError(
                    f"steps[{i}].expected_tools names unknown tools {unknown}; "
                    f"choose from: {', '.join(sorted(known))}"
                )
            steps.append(
                PlanStep(
                    id=f"step_{i + 1}",
                    description=description,
                    expected_tools=expected,
                    status=StepStatus.PENDING,
                )
            )
        if len(raw_steps) > self.settings.agent.max_plan_steps:
            steps[-1].note = (
                f"plan was trimmed from {len(raw_steps)} to "
                f"{self.settings.agent.max_plan_steps} steps"
            )
        return steps


def _extract_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise ValueError("model returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if match := _FENCE.search(text):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse JSON from the response: {exc}") from exc
    raise ValueError("response contained no JSON object")


def apply_plan_update(state: AgentState, steps_payload: list[dict[str, Any]]) -> None:
    """Replace the plan from an `update_plan` tool call."""
    new_steps: list[PlanStep] = []
    for i, item in enumerate(steps_payload[:12]):
        status = str(item.get("status") or "pending")
        if status not in {s.value for s in StepStatus}:
            status = "pending"
        new_steps.append(
            PlanStep(
                id=f"step_{i + 1}",
                description=str(item.get("description") or "").strip() or f"step {i + 1}",
                expected_tools=[str(t) for t in (item.get("expected_tools") or [])],
                status=StepStatus(status),
                note=str(item.get("note") or ""),
            )
        )
    state.plan = new_steps
    state.sync_current_step()
