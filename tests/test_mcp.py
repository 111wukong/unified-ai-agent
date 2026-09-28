"""MCP client, both protocol eras.

The 2026-07-28 revision removed the `initialize` handshake and made the
protocol stateless. Compatibility across that line is explicitly not
guaranteed, so the client has to detect which era a server speaks. These
tests run a real child process over stdio and assert both paths work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from unified_agent.config import McpServerConfig
from unified_agent.tools.base import ToolContext
from unified_agent.tools.mcp import McpStdioClient, McpTool, connect_servers
from unified_agent.tools.permissions import PermissionEngine
from unified_agent.tools.registry import ToolRegistry
from unified_agent.types import EffectClass

FIXTURE = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def server(era: str, **kwargs) -> McpServerConfig:
    return McpServerConfig(
        name="fake",
        command=sys.executable,
        args=[str(FIXTURE), "--era", era],
        **kwargs,
    )


@pytest.fixture
def ctx(workspace, tmp_path):  # noqa: ANN001
    return ToolContext(
        task_id="t",
        session_id="s",
        step_id="step_1",
        workspace=workspace,
        home=tmp_path / "home",
        artifact_dir=tmp_path / "artifacts",
    )


class TestEraDetection:
    async def test_legacy_server_falls_back_to_initialize(self) -> None:
        client = McpStdioClient(server("legacy"))
        await client.start()
        try:
            era = await client.handshake()
            assert era == "legacy"
            assert client.server_info["serverInfo"]["name"] == "fake"
        finally:
            await client.close()

    async def test_stateless_server_uses_discover(self) -> None:
        client = McpStdioClient(server("stateless"))
        await client.start()
        try:
            era = await client.handshake()
            assert era == "stateless"
            assert client.server_info["protocolVersion"] == "2026-07-28"
        finally:
            await client.close()

    async def test_both_eras_expose_the_same_tools(self) -> None:
        for era in ("legacy", "stateless"):
            async with McpStdioClient(server(era)) as client:
                tools = await client.list_tools()
                assert [t["name"] for t in tools] == ["echo"]


class TestToolProjection:
    async def test_schema_and_description_are_carried_over(self) -> None:
        async with McpStdioClient(server("stateless")) as client:
            remote = (await client.list_tools())[0]
            tool = McpTool(client, remote)

        assert tool.spec.name == "mcp__fake__echo"
        assert tool.spec.parameters["required"] == ["text"]
        assert tool.spec.source == "mcp:fake"

    async def test_calls_round_trip(self, ctx: ToolContext) -> None:
        async with McpStdioClient(server("stateless")) as client:
            remote = (await client.list_tools())[0]
            result = await McpTool(client, remote).run({"text": "hello"}, ctx)
        assert result.success
        assert result.output == "echo: hello"

    async def test_server_side_error_becomes_a_failed_result(self, ctx: ToolContext) -> None:
        async with McpStdioClient(server("legacy")) as client:
            remote = (await client.list_tools())[0]
            result = await McpTool(client, remote).run({"text": "x", "fail": "1"}, ctx)
        assert not result.success
        assert result.output == "boom"

    async def test_effect_class_comes_from_local_config_not_the_server(self) -> None:
        """A third-party server calling itself read-only is not evidence."""
        config = server("legacy", effect_class=EffectClass.READ_ONLY, requires_confirmation=False)
        async with McpStdioClient(config) as client:
            tool = McpTool(client, (await client.list_tools())[0])
        assert tool.spec.effect_class is EffectClass.READ_ONLY

        config2 = server("legacy")  # default is the cautious end
        async with McpStdioClient(config2) as client:
            tool2 = McpTool(client, (await client.list_tools())[0])
        assert tool2.spec.effect_class is EffectClass.EXECUTE_LOCAL
        assert tool2.spec.requires_confirmation is True
        assert tool2.spec.idempotent is False, "remote idempotency is unknown; assume the worst"


class TestRegistryIntegration:
    async def test_tools_are_registered_and_policy_applies(
        self, settings, workspace, tmp_path
    ) -> None:  # noqa: ANN001
        tools, problems = await connect_servers([server("stateless")])
        assert not problems
        assert len(tools) == 1

        registry = ToolRegistry(tools)
        engine = PermissionEngine(
            settings.permissions, workspace=workspace, home=tmp_path / "home"
        )
        verdict = engine.decide(registry.get("mcp__fake__echo"), {"text": "hi"})
        assert verdict.needs_confirmation, "MCP tools need approval by default"

        # ...and the CLI pre-approval releases it.
        engine2 = PermissionEngine(
            settings.permissions,
            workspace=workspace,
            home=tmp_path / "home",
            cli_approvals=[EffectClass.EXECUTE_LOCAL],
        )
        assert engine2.decide(registry.get("mcp__fake__echo"), {"text": "hi"}).allowed

    async def test_a_broken_server_does_not_kill_the_run(self) -> None:
        broken = McpServerConfig(name="broken", command="/nonexistent/binary", args=[])
        tools, problems = await connect_servers([broken, server("stateless")])
        assert len(tools) == 1, "the healthy server must still be usable"
        assert problems and "broken" in problems[0]

    async def test_disabled_servers_are_skipped(self) -> None:
        tools, problems = await connect_servers([server("legacy", enabled=False)])
        assert tools == [] and problems == []

    async def test_connect_servers_handles_empty_config(self) -> None:
        tools, problems = await connect_servers([])
        assert tools == [] and problems == []


class TestRobustness:
    async def test_missing_command_reports_clearly(self) -> None:
        client = McpStdioClient(McpServerConfig(name="x", command="/nope/missing", args=[]))
        from unified_agent.errors import ToolError

        with pytest.raises(ToolError):
            await client.start()

    async def test_client_survives_a_server_that_never_answers(self) -> None:
        config = McpServerConfig(
            name="mute",
            command=sys.executable,
            args=["-c", "import time; time.sleep(30)"],
            startup_timeout_s=1.0,
        )
        client = McpStdioClient(config)
        await client.start()
        try:
            with pytest.raises(Exception):  # noqa: B017 - ToolError after the timeout
                await client.handshake()
        finally:
            await client.close()
