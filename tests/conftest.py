"""Shared fixtures.

Everything runs offline: `ScriptedModel` delegates structured-output calls
(planning) to the deterministic mock and consumes an explicit script for the
agent loop. No network, no API key, no cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from unified_agent.agent.factory import build_agent
from unified_agent.config import (
    AgentConfig,
    ModelSpec,
    NetworkPolicy,
    PermissionConfig,
    Settings,
)
from unified_agent.models.mock import MockModel
from unified_agent.types import Decision, EffectClass


class ScriptedModel(MockModel):
    """A MockModel whose loop turns come from a fixed script."""

    def __init__(self, spec: ModelSpec, script: Sequence[dict[str, Any]] | None = None) -> None:
        super().__init__(spec)
        self.script = list(script or [])
        self.cursor = 0

    async def _chat(
        self,
        messages,
        *,
        tools,
        temperature,
        response_format,
        max_output_tokens,
        stream=None,
    ):
        if response_format:
            # Planning / reflection: let the deterministic mock answer.
            return self._structured(messages, response_format)
        if self.cursor < len(self.script):
            step = self.script[self.cursor]
            self.cursor += 1
            return self._from_script(step)
        return self._final(messages)


class ScriptedModels:
    """Stands in for ModelRegistry."""

    def __init__(self, model: ScriptedModel) -> None:
        self.model = model

    def aliases(self) -> list[str]:
        return ["scripted"]

    def get(self, alias: str | None = None):  # noqa: ANN201
        return self.model

    def try_get(self, alias: str):  # noqa: ANN201
        return self.model

    def summarizer(self):  # noqa: ANN201
        return None

    async def aclose(self) -> None:
        return None


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "app.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef broken():\n    return 1 / 0\n",
        encoding="utf-8",
    )
    (root / "notes.md").write_text("# notes\n\nhello\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "util.py").write_text("VALUE = 42\n", encoding="utf-8")
    return root


@pytest.fixture
def settings(tmp_path: Path, workspace: Path) -> Settings:
    """Permissive enough to exercise the loop; policies tested separately."""
    permissions = PermissionConfig()
    permissions.defaults = {
        EffectClass.READ_ONLY: Decision.ALLOW,
        EffectClass.WRITE_LOCAL: Decision.ALLOW,
        EffectClass.EXECUTE_LOCAL: Decision.CONFIRM,
        EffectClass.NETWORK: Decision.CONFIRM,
        EffectClass.EXTERNAL_SIDE_EFFECT: Decision.CONFIRM,
        EffectClass.SYSTEM_ADMIN: Decision.DENY,
    }
    permissions.network = NetworkPolicy(allow_domains=["example.com"])
    return Settings(
        home=tmp_path / "home",
        workspace=workspace,
        default_model="scripted",
        models={"scripted": ModelSpec(provider="mock", model="mock-react")},
        permissions=permissions,
        agent=AgentConfig(max_steps=8, tool_output_chars=400),
    )


@pytest.fixture
def scripted(settings: Settings):
    """Factory: make_agent(script) -> (agent, model)."""

    created: list = []

    async def _make(script: Sequence[dict[str, Any]] | None = None):
        model = ScriptedModel(settings.models["scripted"], script)
        agent = await build_agent(settings=settings, models=ScriptedModels(model))
        created.append(agent)
        return agent, model

    yield _make

    for agent in created:
        agent.close()


def looping_script(times: int, tool: str = "list_directory", **arguments: Any) -> list[dict]:
    """A script that keeps calling tools, so a budget is the only way out."""
    return [
        {"tool_calls": [{"name": tool, "arguments": arguments or {"path": "."}}]}
        for _ in range(times)
    ]


@pytest.fixture
async def session_id(settings: Settings, scripted):  # noqa: ANN001
    agent, _ = await scripted([])
    return agent.store.ensure_session(
        name="test", working_dir=str(settings.workspace), model_alias="scripted"
    )
