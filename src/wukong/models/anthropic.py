"""Anthropic Messages API adapter.

Kept native rather than routed through an OpenAI-compatible gateway because
the capability differences are real and load-bearing:

* no `response_format` — structured output is "please emit JSON" + repair
* `tool_use` / `tool_result` content blocks instead of `tool_calls`
* prompt caching via explicit `cache_control` breakpoints, which is a
  large cost lever on a 30-step agent run
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from wukong.errors import ModelError
from wukong.models.base import ChatModel, ModelCapabilities, StreamCallback
from wukong.types import Message, ModelResponse, TokenUsage, ToolCall

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class AnthropicModel(ChatModel):
    provider = "anthropic"

    def declared_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            native_tool_calling=True,
            parallel_tool_calls=True,
            json_schema=False,
            json_object=False,
            streaming=True,
            vision=True,
            prompt_cache=True,
            max_context_tokens=200_000,
            max_output_tokens=8_192,
        )

    @property
    def endpoint(self) -> str:
        base = (self.spec.base_url or "https://api.anthropic.com").rstrip("/")
        return f"{base}/v1/messages"

    def _headers(self) -> dict[str, str]:
        key = self.spec.api_key()
        if not key:
            raise ModelError(
                f"no API key: set {self.spec.key_env()} in the environment", retryable=False
            )
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

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
        system_text, converted = _convert(messages)
        if response_format:
            system_text += (
                "\n\nRespond with a single JSON object and nothing else. "
                "No prose, no code fences."
            )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": converted,
        }
        if system_text:
            block: dict[str, Any] = {"type": "text", "text": system_text}
            if self.capabilities.prompt_cache:
                # Cache the (stable) system prefix; it is re-sent every step.
                block["cache_control"] = {"type": "ephemeral"}
            payload["system"] = [block]
        if tools:
            payload["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]

        timeout = httpx.Timeout(self.spec.timeout_s, connect=15.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if stream:
                    return await self._stream(client, payload, stream)
                resp = await client.post(self.endpoint, headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise ModelError(f"anthropic timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ModelError(f"transport error: {exc}", retryable=True) from exc

        if resp.status_code >= 400:
            raise ModelError(
                f"HTTP {resp.status_code}: {resp.text[:400]}",
                retryable=resp.status_code in _RETRYABLE_STATUS,
            )
        return self._parse(resp.json())

    async def _stream(
        self, client: httpx.AsyncClient, payload: dict[str, Any], on_delta: StreamCallback
    ) -> ModelResponse:
        payload = {**payload, "stream": True}
        content_parts: list[str] = []
        blocks: dict[int, dict[str, Any]] = {}
        usage = TokenUsage()

        async with client.stream(
            "POST", self.endpoint, headers=self._headers(), json=payload
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                raise ModelError(
                    f"HTTP {resp.status_code}: {body[:400]}",
                    retryable=resp.status_code in _RETRYABLE_STATUS,
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "message_start":
                    usage.prompt_tokens = int(
                        (event.get("message", {}).get("usage") or {}).get("input_tokens") or 0
                    )
                elif kind == "content_block_start":
                    block = event.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        blocks[int(event["index"])] = {
                            "id": block.get("id") or "",
                            "name": block.get("name") or "",
                            "json": "",
                        }
                elif kind == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text") or ""
                        content_parts.append(text)
                        result = on_delta(text)
                        if hasattr(result, "__await__"):
                            await result  # type: ignore[func-returns-value]
                    elif delta.get("type") == "input_json_delta":
                        slot = blocks.setdefault(
                            int(event["index"]), {"id": "", "name": "", "json": ""}
                        )
                        slot["json"] += delta.get("partial_json") or ""
                elif kind == "message_delta":
                    usage.completion_tokens = int(
                        (event.get("usage") or {}).get("output_tokens") or 0
                    )

        calls = [
            ToolCall(id=slot["id"] or f"call_{i}", name=slot["name"], arguments=_loads(slot["json"]))
            for i, slot in sorted(blocks.items())
            if slot["name"]
        ]
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return ModelResponse(
            content="".join(content_parts),
            tool_calls=calls,
            usage=usage,
            model=self.model_name,
        )

    def _parse(self, data: dict[str, Any]) -> ModelResponse:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.get("id") or f"call_{len(calls)}",
                        name=block.get("name") or "",
                        arguments=block.get("input") or {},
                    )
                )
        raw_usage = data.get("usage") or {}
        prompt = int(raw_usage.get("input_tokens") or 0)
        completion = int(raw_usage.get("output_tokens") or 0)
        return ModelResponse(
            content="".join(text_parts),
            tool_calls=[c for c in calls if c.name],
            usage=TokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_tokens=int(raw_usage.get("cache_read_input_tokens") or 0),
                total_tokens=prompt + completion,
            ),
            model=data.get("model") or self.model_name,
            finish_reason=data.get("stop_reason"),
        )


def _convert(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Split system out, and translate tool messages into user tool_result blocks."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
            continue
        if msg.role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content[:100_000],
                        }
                    ],
                }
            )
            continue
        if msg.role == "assistant" and msg.tool_calls:
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
                for tc in msg.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": msg.role, "content": msg.content})
    return "\n\n".join(p for p in system_parts if p), out


def _loads(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw[:2000]}
    return value if isinstance(value, dict) else {"__value__": value}
