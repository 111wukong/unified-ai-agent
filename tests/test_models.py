"""Model layer: capability matrix, text-protocol fallback, cost accounting."""

from __future__ import annotations

import json

import pytest

from unified_agent.config import ModelSpec
from unified_agent.errors import ConfigError, ModelError
from unified_agent.models.base import (
    ModelCapabilities,
    TextProtocolModel,
    estimate_tokens,
    parse_text_tool_calls,
    render_text_protocol,
)
from unified_agent.models.mock import MockModel
from unified_agent.models.registry import build_model
from unified_agent.types import Message, ModelResponse, TokenUsage, ToolCall


class TestCapabilities:
    def test_unknown_override_is_rejected(self) -> None:
        caps = ModelCapabilities()
        with pytest.raises(ValueError, match="unknown capability"):
            caps.merged({"native_tool_calling": True, "typo_here": 1})

    def test_overrides_apply(self) -> None:
        caps = ModelCapabilities(native_tool_calling=False).merged({"native_tool_calling": True})
        assert caps.native_tool_calling is True

    def test_openai_compat_defaults_differ_for_gateways(self) -> None:
        """A custom base_url is not assumed to support strict json_schema."""
        real = build_model("openai", ModelSpec(provider="openai_compat", model="gpt-4.1"))
        gateway = build_model(
            "local",
            ModelSpec(
                provider="openai_compat", model="qwen2.5", base_url="http://127.0.0.1:1234/v1"
            ),
        )
        assert real.capabilities.json_schema is True
        assert gateway.capabilities.json_schema is False
        assert gateway.capabilities.json_object is True

    def test_anthropic_declares_no_json_schema_but_has_cache(self) -> None:
        model = build_model("claude", ModelSpec(provider="anthropic", model="claude-sonnet-4-5"))
        assert model.capabilities.json_schema is False
        assert model.capabilities.prompt_cache is True
        assert model.capabilities.max_context_tokens == 200_000

    def test_unknown_provider_is_a_config_error(self) -> None:
        # `provider` is a Literal, so pydantic normally blocks this. Use
        # model_construct to reach the defensive branch in build_model.
        bogus = ModelSpec.model_construct(provider="not_a_provider", model="x")
        with pytest.raises(ConfigError, match="unknown provider"):
            build_model("x", bogus)


class TestTextProtocol:
    def test_parses_a_well_formed_block(self) -> None:
        text = 'Sure, let me look.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
        remaining, calls = parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "a.py"}
        assert "<tool_call>" not in remaining
        assert "let me look" in remaining

    def test_parses_a_fenced_json_block(self) -> None:
        text = 'I will read it.\n```json\n{"name": "read_file", "arguments": {"path": "b.py"}}\n```'
        _, calls = parse_text_tool_calls(text)
        assert calls and calls[0].name == "read_file"

    def test_repairs_common_json_damage(self) -> None:
        text = "<tool_call>{'name': 'read_file', 'arguments': {'path': 'a.py',},}</tool_call>"
        _, calls = parse_text_tool_calls(text)
        assert calls and calls[0].arguments == {"path": "a.py"}

    def test_plain_prose_yields_no_calls(self) -> None:
        remaining, calls = parse_text_tool_calls("Here is the answer: 42.")
        assert calls == []
        assert remaining == "Here is the answer: 42."

    def test_multiple_blocks(self) -> None:
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        _, calls = parse_text_tool_calls(text)
        assert [c.name for c in calls] == ["a", "b"]

    def test_catalog_renders_required_and_optional_args(self) -> None:
        catalog = render_text_protocol(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file.",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        )
        assert "read_file(path: string, limit: integer?)" in catalog
        assert "Read a file." in catalog

    async def test_wrapper_injects_the_catalog_and_parses_calls(self) -> None:
        spec = ModelSpec(provider="mock", model="mock-react")
        inner = MockModel(spec)
        wrapper = TextProtocolModel(inner)
        assert wrapper.capabilities.native_tool_calling is False

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        captured: dict = {}

        async def fake_chat(messages, **kwargs):  # noqa: ANN001, ANN202
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return ModelResponse(
                content='<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
            )

        inner.chat = fake_chat  # type: ignore[assignment]
        resp = await wrapper.chat([Message(role="user", content="read x.py")], tools=tools)

        assert captured["kwargs"].get("tools") is None, "the inner model must not get tools"
        assert "read_file(path: string)" in captured["messages"][0].content
        assert resp.tool_calls[0].name == "read_file"
        assert resp.via_text_protocol is True
        assert resp.content == ""


class TestCostAccounting:
    async def test_cost_is_computed_from_prices(self) -> None:
        spec = ModelSpec(provider="mock", model="mock-react", price_in=3.0, price_out=15.0)
        model = MockModel(spec)

        async def fake_chat(messages, **kwargs):  # noqa: ANN001, ANN202
            return ModelResponse(
                content="hi",
                usage=TokenUsage(
                    prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000
                ),
            )

        model._chat = fake_chat  # type: ignore[assignment]
        resp = await model.chat([Message(role="user", content="x")])
        assert resp.usage.cost_usd == pytest.approx(18.0)

    async def test_no_prices_means_no_cost(self) -> None:
        model = MockModel(ModelSpec(provider="mock", model="unknown-model"))
        resp = await model.chat([Message(role="user", content="x")])
        assert resp.usage.cost_usd == 0.0

    def test_known_model_has_a_default_price(self) -> None:
        spec = ModelSpec(provider="openai_compat", model="gpt-4.1")
        assert spec.resolved_prices() == (2.0, 8.0)


class TestMockModel:
    async def test_script_is_consumed_in_order(self) -> None:
        model = MockModel(
            ModelSpec(provider="mock", model="mock-react"),
            script=[
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]},
                {"content": "final answer"},
            ],
        )
        first = await model.chat([Message(role="user", content="go")])
        second = await model.chat([Message(role="user", content="go")])
        assert first.tool_calls[0].name == "read_file"
        assert second.content == "final answer"
        assert not second.tool_calls

    async def test_structured_output_is_schema_shaped(self) -> None:
        model = MockModel(ModelSpec(provider="mock", model="mock-react"))
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "plan",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"description": {"type": "string"}},
                            },
                        }
                    },
                    "required": ["steps"],
                },
            },
        }
        resp = await model.chat(
            [Message(role="user", content="# Goal\n\nfind bugs\n")], response_format=schema
        )
        payload = json.loads(resp.content)
        assert payload["steps"] and all("description" in s for s in payload["steps"])

    async def test_heuristic_walks_a_small_project(self, workspace) -> None:
        model = MockModel(ModelSpec(provider="mock", model="mock-react"))
        tools = [
            {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
            for n in ("list_directory", "search_files", "read_file", "run_tests", "run_command")
        ]
        messages = [
            Message(
                role="user",
                content="# Goal\n\n查看当前目录下的 Python 文件并总结\n\n# Available tools\n\nx",
            )
        ]
        first = await model.chat(messages, tools=tools)
        assert first.tool_calls[0].name == "list_directory"

        messages.append(
            Message(
                role="user",
                content="# Progress so far\n\n[0] list_directory({}) -> ok\n  app.py  (10 B)",
            )
        )
        second = await model.chat(messages, tools=tools)
        assert second.tool_calls[0].name == "search_files"

    async def test_danger_keyword_triggers_a_refused_command(self) -> None:
        """The offline demo can exercise the permission path on purpose."""
        model = MockModel(ModelSpec(provider="mock", model="mock-react"))
        tools = [
            {"type": "function", "function": {"name": "run_command", "description": "", "parameters": {}}}
        ]
        resp = await model.chat(
            [Message(role="user", content="试一下危险命令看看会不会被拒绝")], tools=tools
        )
        assert resp.tool_calls[0].name == "run_command"
        assert "rm -rf" in resp.tool_calls[0].arguments["command"]


class TestRegistry:
    def test_missing_api_key_is_a_clear_error(self, settings) -> None:  # noqa: ANN001
        from unified_agent.models.registry import ModelRegistry

        settings.models["needs_key"] = ModelSpec(
            provider="openai_compat", model="gpt-4.1", api_key_env="UAA_TEST_MISSING_KEY"
        )
        registry = ModelRegistry(settings)
        with pytest.raises(ModelError, match="needs an API key"):
            registry.get("needs_key")

    def test_unknown_alias_lists_known_aliases(self, settings) -> None:  # noqa: ANN001
        from unified_agent.models.registry import ModelRegistry

        registry = ModelRegistry(settings)
        with pytest.raises(ConfigError, match="not configured"):
            registry.get("nope")


class TestTokenEstimation:
    def test_latin_and_cjk_scale_differently(self) -> None:
        assert estimate_tokens("hello world") < estimate_tokens("你好世界你好世界")
        assert estimate_tokens("a" * 360) == pytest.approx(101, abs=2)

    def test_counts_tool_calls_in_messages(self) -> None:
        model = MockModel(ModelSpec(provider="mock", model="mock-react"))
        messages = [
            Message(role="assistant", tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "a"})])
        ]
        assert model.count_messages(messages) > 5
