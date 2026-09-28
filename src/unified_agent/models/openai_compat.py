"""OpenAI-compatible adapter.

One adapter covers OpenAI, DeepSeek, 通义千问 (dashscope compatible mode),
Moonshot, OpenRouter, vLLM, LM Studio, Ollama's `/v1` endpoint and most
gateways. That is the single highest-leverage piece of "model-agnostic":
not an abstraction layer, just one wire format that everybody already
speaks.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from unified_agent.errors import ModelError
from unified_agent.models.base import ChatModel, ModelCapabilities, StreamCallback
from unified_agent.types import Message, ModelResponse, TokenUsage, ToolCall

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class OpenAICompatModel(ChatModel):
    provider = "openai_compat"

    def declared_capabilities(self) -> ModelCapabilities:
        # Only real OpenAI is assumed to support strict json_schema. Gateways
        # and local servers vary; the alias can override per model.
        is_openai = not self.spec.base_url
        return ModelCapabilities(
            native_tool_calling=True,
            parallel_tool_calls=is_openai,
            json_schema=is_openai,
            json_object=True,
            streaming=True,
            vision=is_openai,
            prompt_cache=is_openai,
            max_context_tokens=128_000,
            max_output_tokens=8_192,
        )

    @property
    def endpoint(self) -> str:
        base = (self.spec.base_url or "https://api.openai.com/v1").rstrip("/")
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.spec.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

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
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [m.to_provider_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        timeout = httpx.Timeout(self.spec.timeout_s, connect=15.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if stream:
                    return await self._stream(client, payload, stream)
                resp = await client.post(self.endpoint, headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise ModelError(f"provider timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ModelError(f"transport error: {exc}", retryable=True) from exc

        if resp.status_code >= 400:
            raise ModelError(
                f"HTTP {resp.status_code}: {_error_text(resp)}",
                retryable=resp.status_code in _RETRYABLE_STATUS,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise ModelError(f"provider returned non-JSON: {resp.text[:400]}") from exc
        return self._parse(data)

    # -- streaming --------------------------------------------------------
    async def _stream(
        self, client: httpx.AsyncClient, payload: dict[str, Any], on_delta: StreamCallback
    ) -> ModelResponse:
        content_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        usage = TokenUsage()
        finish_reason: str | None = None
        model_name = self.model_name

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
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                model_name = event.get("model") or model_name
                if raw_usage := event.get("usage"):
                    usage = _parse_usage(raw_usage)
                for choice in event.get("choices") or []:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    if text := delta.get("content"):
                        content_parts.append(text)
                        result = on_delta(text)
                        if hasattr(result, "__await__"):
                            await result  # type: ignore[func-returns-value]
                    for tc in delta.get("tool_calls") or []:
                        index = int(tc.get("index") or 0)
                        slot = tool_acc.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]

        calls = [
            ToolCall(id=slot["id"] or f"call_{i}", name=slot["name"], arguments=_loads(slot["arguments"]))
            for i, slot in sorted(tool_acc.items())
            if slot["name"]
        ]
        if not usage.completion_tokens:
            usage.completion_tokens = self.count_tokens("".join(content_parts))
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return ModelResponse(
            content="".join(content_parts),
            tool_calls=calls,
            usage=usage,
            model=model_name,
            finish_reason=finish_reason,
        )

    # -- parsing ----------------------------------------------------------
    def _parse(self, data: dict[str, Any]) -> ModelResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"provider returned no choices: {json.dumps(data)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        calls: list[ToolCall] = []
        for i, raw in enumerate(raw_calls):
            fn = raw.get("function") or {}
            calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{i}",
                    name=fn.get("name") or "",
                    arguments=_loads(fn.get("arguments")),
                )
            )
        content = message.get("content") or ""
        if isinstance(content, list):  # some gateways return content blocks
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        usage = _parse_usage(data.get("usage") or {})
        return ModelResponse(
            content=content,
            tool_calls=[c for c in calls if c.name],
            usage=usage,
            model=data.get("model") or self.model_name,
            finish_reason=choice.get("finish_reason"),
        )


def _parse_usage(raw: dict[str, Any]) -> TokenUsage:
    details = raw.get("prompt_tokens_details") or {}
    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=int(details.get("cached_tokens") or 0),
        total_tokens=int(raw.get("total_tokens") or (prompt + completion)),
    )


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"__raw__": str(raw)[:2000]}
    return value if isinstance(value, dict) else {"__value__": value}


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:400]
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)[:400]
    return str(err or data)[:400]
