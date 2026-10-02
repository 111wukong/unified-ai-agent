"""Offline deterministic model.

Exists for two reasons, both of which matter more than they look:

1. The whole test suite can exercise the real agent loop -- planning,
   permissions, tool execution, resume -- with zero network and zero cost.
   A test suite that needs an API key does not get run.
2. `wukong run --model mock "..."` gives a working end-to-end demo on a plane.

It is *not* a language model. It is a scripted policy: either an explicit
script (tests) or a small heuristic that mimics a ReAct loop (demo).
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from wukong.models.base import ChatModel, ModelCapabilities
from wukong.types import Message, ModelResponse, TokenUsage, ToolCall

_FILE_RE = re.compile(r"([\w./\-]+\.(?:py|md|json|toml|txt|js|ts|tsx|yaml|yml|go|rs))")
# Matches the rendered progress log line "[3] read_file({...}) -> ok".
_LOG_CALL_RE = re.compile(r"^\[\d+\]\s+(\w+)\(", re.MULTILINE)
_GOAL_RE = re.compile(r"#\s*Goal\s*\n+(.*?)(?:\n\n#|\Z)", re.DOTALL)

_KEYWORDS = {
    "list": ("文件", "目录", "结构", "项目", "file", "directory", "structure", "project", "list"),
    "scan": ("python", "py", "代码", "code", "扫描", "scan", "todo", "找", "find"),
    "test": ("测试", "test", "pytest", "pytest 失败"),
    "danger": ("危险", "dangerous", "拒绝", "refuse", "denied"),
}


class MockModel(ChatModel):
    provider = "mock"

    def __init__(self, spec, *, script: Sequence[dict[str, Any]] | None = None) -> None:
        super().__init__(spec)
        self._script = list(script) if script else None
        self._script_index = 0

    def declared_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            native_tool_calling=True,
            parallel_tool_calls=False,
            json_schema=True,
            json_object=True,
            streaming=False,
            vision=False,
            prompt_cache=False,
            max_context_tokens=200_000,
            max_output_tokens=4_096,
        )

    async def _chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        response_format: dict[str, Any] | None,
        max_output_tokens: int,
        stream=None,
    ) -> ModelResponse:
        if self._script is not None and self._script_index < len(self._script):
            step = self._script[self._script_index]
            self._script_index += 1
            return self._from_script(step)

        if response_format:
            return self._structured(messages, response_format)

        if tools:
            heuristic = self._heuristic(messages, tools)
            if heuristic is not None:
                return heuristic
        return self._final(messages)

    # -- script mode ------------------------------------------------------
    def _from_script(self, step: dict[str, Any]) -> ModelResponse:
        calls = [
            ToolCall(
                id=tc.get("id") or f"mock_{self._script_index}_{i}",
                name=tc["name"],
                arguments=tc.get("arguments") or {},
            )
            for i, tc in enumerate(step.get("tool_calls") or [])
        ]
        content = step.get("content") or ""
        return ModelResponse(
            content=content,
            tool_calls=calls,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140),
            model=self.model_name,
        )

    # -- structured output ------------------------------------------------
    def _structured(self, messages: list[Message], response_format: dict[str, Any]) -> ModelResponse:
        schema = (
            (response_format.get("json_schema") or {}).get("schema")
            or response_format.get("schema")
            or {}
        )
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        goal = _goal(messages)
        payload: dict[str, Any]

        if "steps" in props:
            payload = {
                "reasoning": "Deterministic mock plan derived from the goal.",
                "steps": [
                    {
                        "description": f"Inspect the workspace to ground the task: {goal[:80]}",
                        "expected_tools": ["list_directory"],
                    },
                    {
                        "description": "Locate the relevant source files and read the key one.",
                        "expected_tools": ["search_files", "read_file"],
                    },
                    {
                        "description": "Summarise findings for the user.",
                        "expected_tools": [],
                    },
                ],
            }
        elif "assessment" in props:
            payload = {"assessment": "on_track", "note": "mock assessment"}
        else:
            payload = {name: _placeholder(spec) for name, spec in props.items()}
        return ModelResponse(
            content=json.dumps(payload, ensure_ascii=False),
            usage=TokenUsage(prompt_tokens=80, completion_tokens=60, total_tokens=140),
            model=self.model_name,
        )

    # -- heuristic ReAct --------------------------------------------------
    def _heuristic(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ModelResponse | None:
        available = {t["function"]["name"] for t in tools}
        goal = _goal(messages)
        # The context is rebuilt from state each turn, so prior tool calls
        # appear as rendered progress-log lines rather than as structured
        # `tool_calls` on assistant messages. Read them from the text.
        already = _called_tools(messages)
        observations = [m.content for m in messages if m.role == "user"]

        def make(name: str, args: dict[str, Any]) -> ModelResponse:
            return ModelResponse(
                tool_calls=[ToolCall(id=f"mock_{len(already)}_{name}", name=name, arguments=args)],
                usage=TokenUsage(prompt_tokens=120, completion_tokens=30, total_tokens=150),
                model=self.model_name,
            )

        wants_danger = _matches(goal, _KEYWORDS["danger"])
        if wants_danger and "run_command" in available and "run_command" not in already:
            return make("run_command", {"command": "rm -rf /tmp/wukong-danger-demo"})

        if _matches(goal, _KEYWORDS["list"]) and "list_directory" in available:
            if "list_directory" not in already:
                return make("list_directory", {"path": ".", "depth": 2})

        if _matches(goal, _KEYWORDS["test"]) and "run_tests" in available:
            if "run_tests" not in already:
                return make("run_tests", {})

        if _matches(goal, _KEYWORDS["scan"]) and "search_files" in available:
            if "search_files" not in already:
                return make("search_files", {"glob": "**/*.py", "max_results": 40})

        if "read_file" in available and "read_file" not in already and observations:
            candidate = _first_path(" ".join(observations))
            if candidate:
                return make("read_file", {"path": candidate, "limit": 120})

        return None

    def _final(self, messages: list[Message]) -> ModelResponse:
        calls = _called_tools(messages)
        goal = _goal(messages)
        log_text = ""
        for msg in messages:
            if msg.role == "user" and _LOG_CALL_RE.search(msg.content or ""):
                log_text = msg.content
        lines = [
            f"[mock model] goal: {goal}",
            "",
            f"Executed {len(calls)} tool call(s): "
            + (", ".join(calls) if calls else "(none — no matching heuristic)"),
        ]
        if log_text:
            excerpt = "\n".join(log_text.strip().splitlines()[-14:])
            lines += ["", "Most recent progress:", "```", excerpt, "```"]
        else:
            lines += ["", "No tools were needed; answering from context alone."]
        return ModelResponse(
            content="\n".join(lines),
            usage=TokenUsage(prompt_tokens=150, completion_tokens=90, total_tokens=240),
            model=self.model_name,
        )


def _goal(messages: list[Message]) -> str:
    """Pull just the goal out of the (deliberately verbose) prompt."""
    for msg in messages:
        if msg.role != "user" or not msg.content:
            continue
        if match := _GOAL_RE.search(msg.content):
            return match.group(1).strip().replace("\n", " ")[:200]
        if not msg.content.startswith("#"):
            return msg.content.strip().replace("\n", " ")[:200]
    return ""


def _called_tools(messages: list[Message]) -> list[str]:
    """Tools already invoked, recovered from the rendered progress log."""
    out: list[str] = []
    for msg in messages:
        for tc in msg.tool_calls:
            out.append(tc.name)
        if msg.role == "user" and msg.content:
            out.extend(_LOG_CALL_RE.findall(msg.content))
    return out


def _matches(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def _first_path(text: str) -> str | None:
    """Pick a file to read, preferring source over docs."""
    candidates: list[str] = []
    for match in _FILE_RE.finditer(text):
        candidate = match.group(1)
        if candidate.startswith((".", "/")) or ".." in candidate:
            continue
        candidates.append(candidate)
    for suffix in (".py", ".ts", ".tsx", ".js", ".go", ".rs"):
        for candidate in candidates:
            if candidate.endswith(suffix):
                return candidate
    return candidates[0] if candidates else None


def _placeholder(spec: Any) -> Any:
    if not isinstance(spec, dict):
        return None
    kind = spec.get("type")
    if "enum" in spec:
        return spec["enum"][0]
    return {
        "string": "",
        "number": 0,
        "integer": 0,
        "boolean": False,
        "array": [],
        "object": {},
    }.get(kind, None)
