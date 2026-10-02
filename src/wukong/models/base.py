"""Model abstraction.

The original spec's `ChatModel` Protocol is 6 lines and hides the one thing
that actually breaks in practice: providers disagree about *capabilities*.
OpenAI does strict JSON Schema, Anthropic does not; Ollama and most local
models have no native tool calling at all; only some support prompt
caching. A Protocol that pretends otherwise produces code that works on
one provider and silently degrades on the rest.

So the interface carries a capability matrix, and the runtime consults it:

* `native_tool_calling=False`  -> the TextToolProtocol wrapper takes over
* `json_schema=False`          -> structured planning falls back to
                                  "JSON object + repair loop"
* `max_context_tokens`         -> the context builder's budget
* `prompt_cache`               -> whether to bother marking stable prefixes
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from wukong.config import ModelSpec
from wukong.types import Message, ModelResponse, ToolCall

StreamCallback = Callable[[str], Awaitable[None] | None]


class ModelCapabilities(BaseModel):
    """What a provider can do, before user overrides.

    Every field here is read by something. That is a rule, not an
    observation: `vision` used to sit in this list and nothing consulted it,
    which made it a capability flag that could only mislead -- the runtime
    cannot put an image in a message, so a model marked vision-capable still
    could not be asked to look at one. It comes back with the feature.
    """

    native_tool_calling: bool = True
    #: Whether the provider accepts a request-side flag asking it to emit
    #: several tool calls in one turn. Adapters whose API has no such flag
    #: (Anthropic) simply do not read it; the runtime handles multiple calls
    #: either way, so this changes the request, never the loop.
    parallel_tool_calls: bool = False
    json_schema: bool = False
    json_object: bool = False
    streaming: bool = True
    prompt_cache: bool = False
    max_context_tokens: int = 128_000
    #: Hard ceiling on output tokens. `ModelSpec.max_output_tokens` is the
    #: per-request budget; this is what the model can physically emit, and
    #: the request is clamped to it.
    max_output_tokens: int = 8_192

    def merged(self, overrides: dict[str, Any]) -> "ModelCapabilities":
        unknown = set(overrides) - set(self.model_fields)
        if unknown:
            raise ValueError(f"unknown capability overrides: {sorted(unknown)}")
        return self.model_copy(update=overrides)


class ChatModel(ABC):
    provider: str = "unknown"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.model_name = spec.model
        self.capabilities = self.declared_capabilities().merged(spec.capabilities)
        #: Built lazily by adapters that use more than one key for an alias.
        self._pool: Any = None
        #: The credential the in-flight request is using, so a refusal can be
        #: attributed to the key that caused it rather than to the alias.
        self._in_use: Any = None

    @abstractmethod
    def declared_capabilities(self) -> ModelCapabilities:
        """What this provider/model can do before user overrides."""

    @abstractmethod
    async def _chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        response_format: dict[str, Any] | None,
        max_output_tokens: int,
        stream: StreamCallback | None,
    ) -> ModelResponse:
        raise NotImplementedError

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        stream: StreamCallback | None = None,
    ) -> ModelResponse:
        resp = await self._chat(
            messages,
            tools=tools,
            temperature=self.spec.temperature if temperature is None else temperature,
            response_format=response_format,
            # Clamped to the model's own ceiling. Asking a provider for more
            # output tokens than it can emit is a config mistake, and the
            # failure is a 400 in the middle of a task rather than a sentence
            # at startup -- so the capability is the ceiling and the spec is
            # the budget.
            max_output_tokens=min(
                max_output_tokens or self.spec.max_output_tokens,
                self.capabilities.max_output_tokens,
            ),
            stream=stream if self.capabilities.streaming else None,
        )
        resp.provider = self.provider
        resp.model = resp.model or self.model_name
        price_in, price_out = self.spec.resolved_prices()
        if price_in or price_out:
            usage = resp.usage
            usage.cost_usd = round(
                usage.prompt_tokens / 1_000_000 * price_in
                + usage.completion_tokens / 1_000_000 * price_out,
                8,
            )
        return resp

    async def aclose(self) -> None:
        return None

    # -- token accounting -------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def count_messages(self, messages: list[Message]) -> int:
        total = 0
        for msg in messages:
            total += estimate_tokens(msg.content) + 4
            for tc in msg.tool_calls:
                total += estimate_tokens(tc.canonical_arguments()) + estimate_tokens(tc.name) + 6
        return total


# ---------------------------------------------------------------------------
# token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Heuristic token count, weighted for CJK.

    Latin text runs about 3.6 chars/token; CJK is close to 1 token per
    character for every mainstream tokenizer. Getting this wrong by 3x
    means the context budget is wrong by 3x, and the failure mode is a
    400 error mid-task rather than a warning.
    """
    if not text:
        return 0
    cjk = 0
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF  # CJK unified
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF  # kana
            or 0xAC00 <= code <= 0xD7AF  # hangul
            or 0xF900 <= code <= 0xFAFF
        ):
            cjk += 1
    other = len(text) - cjk
    return int(cjk * 1.05 + other / 3.6) + 1


# ---------------------------------------------------------------------------
# text protocol fallback
# ---------------------------------------------------------------------------

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

_TOOL_BLOCK = re.compile(
    rf"{re.escape(TOOL_CALL_OPEN)}\s*(.*?)\s*{re.escape(TOOL_CALL_CLOSE)}", re.DOTALL
)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

TEXT_PROTOCOL_INSTRUCTIONS = """\
# Tool calling protocol

You do not have a native tool-calling API. To call a tool, emit exactly one
block of this form and nothing else:

<tool_call>{{"name": "tool_name", "arguments": {{"arg": "value"}}}}</tool_call>

Rules:
- One tool call per reply. Stop immediately after the closing tag.
- `arguments` must be a JSON object matching the tool's parameters exactly.
- Do not wrap the block in prose or code fences.
- When you have the final answer, reply with plain text and no tool_call block.

# Available tools

{catalog}
"""


def render_text_protocol(tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for tool in tools:
        fn = tool.get("function", tool)
        params = fn.get("parameters", {}) or {}
        props = params.get("properties", {})
        required = set(params.get("required", []))
        rendered = ", ".join(
            f"{name}: {spec.get('type', 'any')}{'' if name in required else '?'}"
            for name, spec in props.items()
        )
        lines.append(f"- {fn.get('name')}({rendered})")
        lines.append(f"    {fn.get('description', '').strip()}")
        if props:
            lines.append(f"    schema: {json.dumps(params, ensure_ascii=False)}")
    return "\n".join(lines)


def parse_text_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract tool calls from free text. Returns (remaining_text, calls)."""
    calls: list[ToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    for index, match in enumerate(_TOOL_BLOCK.finditer(text)):
        payload = _loads(match.group(1))
        if payload:
            calls.append(_to_call(payload, index))
            consumed_spans.append(match.span())

    if not calls:
        # Tolerate a fenced JSON block that clearly looks like a tool call.
        for index, match in enumerate(_FENCE.finditer(text)):
            payload = _loads(match.group(1))
            if payload and "name" in payload and ("arguments" in payload or "parameters" in payload):
                calls.append(_to_call(payload, index))
                consumed_spans.append(match.span())

    if not calls:
        return text, []

    remaining = text
    for start, end in reversed(consumed_spans):
        remaining = remaining[:start] + remaining[end:]
    return remaining.strip(), calls


def _loads(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        # Common failure: trailing comma or single quotes.
        repaired = raw.replace("'", '"')
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _to_call(payload: dict[str, Any], index: int) -> ToolCall:
    name = str(payload.get("name") or payload.get("tool") or "").strip()
    args = payload.get("arguments") or payload.get("parameters") or payload.get("args") or {}
    if isinstance(args, str):
        args = _loads(args) or {}
    if not isinstance(args, dict):
        args = {}
    return ToolCall(id=f"text_{index}_{abs(hash(name)) % 100000}", name=name, arguments=args)


class TextProtocolModel(ChatModel):
    """Wraps a model that cannot call tools natively.

    This is what makes "model-agnostic" real: a Qwen2.5 running in LM Studio
    or Ollama gets the same agent loop, at the cost of one extra parsing
    layer and a slightly higher malformed-call rate.
    """

    def __init__(self, inner: ChatModel) -> None:
        self.inner = inner
        self.spec = inner.spec
        self.provider = inner.provider
        self.model_name = inner.model_name
        caps = inner.capabilities.model_copy(update={"native_tool_calling": False})
        self.capabilities = caps
        self._inner_caps = inner.capabilities

    def declared_capabilities(self) -> ModelCapabilities:  # pragma: no cover
        return self.capabilities

    async def _chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        response_format: dict[str, Any] | None,
        max_output_tokens: int,
        stream: StreamCallback | None,
    ) -> ModelResponse:
        # `chat` below is the real entry point for this class. This override
        # exists so the ABC can be satisfied and so a subclass that calls
        # `super()._chat(...)` still behaves sensibly.
        return await self.inner._chat(
            messages,
            tools=tools,
            temperature=temperature,
            response_format=response_format,
            max_output_tokens=max_output_tokens,
            stream=stream,
        )

    async def chat(self, messages, **kwargs):  # type: ignore[override]
        tools = kwargs.pop("tools", None)
        if tools:
            preamble = TEXT_PROTOCOL_INSTRUCTIONS.format(catalog=render_text_protocol(tools))
            messages = [Message(role="system", content=preamble, pinned=True), *messages]
        resp = await self.inner.chat(messages, tools=None, **kwargs)
        if tools and resp.content:
            content, calls = parse_text_tool_calls(resp.content)
            resp.content = content
            resp.tool_calls = calls
            resp.via_text_protocol = bool(calls)
        return resp

    async def aclose(self) -> None:
        await self.inner.aclose()

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)


__all__ = [
    "ChatModel",
    "ModelCapabilities",
    "TextProtocolModel",
    "estimate_tokens",
    "parse_text_tool_calls",
    "render_text_protocol",
    "StreamCallback",
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
]
