"""Wiring. One place that knows how the pieces fit together."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unified_agent.agent.runtime import AgentRuntime, ProgressHook
from unified_agent.config import Settings
from unified_agent.models.registry import ModelRegistry
from unified_agent.observability.jsonl import JsonlSink
from unified_agent.observability.bus import BusSink, EventBus
from unified_agent.observability.redact import Redactor
from unified_agent.sandbox import build_sandbox
from unified_agent.skills.registry import SkillRegistry
from unified_agent.storage.store import Store
from unified_agent.tools.fs import FS_TOOLS
from unified_agent.tools.git import build_git_tools
from unified_agent.tools.mcp import connect_servers
from unified_agent.tools.memory_tools import (
    FinishTool,
    UpdatePlanTool,
    build_memory_tools,
    build_skill_tools,
)
from unified_agent.tools.net import build_net_tools
from unified_agent.tools.permissions import PermissionEngine
from unified_agent.tools.registry import ToolRegistry
from unified_agent.tools.shell import build_shell_tools
from unified_agent.types import EffectClass


@dataclass
class Agent:
    settings: Settings
    store: Store
    registry: ToolRegistry
    engine: PermissionEngine
    skills: SkillRegistry
    models: ModelRegistry
    runtime: AgentRuntime
    redactor: Redactor
    sink: JsonlSink
    mcp_problems: list[str] = field(default_factory=list)
    bus: EventBus | None = None
    sandbox: object | None = None
    sandbox_selection: object | None = None

    def close(self) -> None:
        self.sink.close()
        self.store.close()

    def skill_dirs(self) -> list[Path]:
        return list(self.skills.dirs)


def default_skill_dirs(settings: Settings) -> list[Path]:
    """Project skills first, then user-level skills.

    Project wins on a name collision because a repository's conventions
    should beat a global preference.
    """
    dirs = [settings.workspace / d for d in settings.skill_dirs]
    dirs.append(settings.home / "skills")
    return dirs


async def build_agent(
    *,
    settings: Settings,
    store: Store | None = None,
    on_progress: ProgressHook | None = None,
    cli_approvals: tuple[EffectClass, ...] = (),
    connect_mcp: bool = True,
    skill_dirs: list[Path] | None = None,
    models: ModelRegistry | None = None,
) -> Agent:
    settings.ensure_dirs()
    redactor = Redactor(settings.secret_values())
    bus = EventBus()
    sink = BusSink(JsonlSink(settings.log_dir, redactor=redactor), bus)
    store = store or Store(settings.db_path, redactor=redactor, sink=sink)

    engine = PermissionEngine(
        settings.permissions,
        workspace=settings.workspace,
        home=settings.home,
        cli_approvals=cli_approvals,
    )

    registry = ToolRegistry()
    for tool in FS_TOOLS:
        registry.register(tool)

    secrets = settings.secret_values()
    selection = build_sandbox(
        settings.sandbox.backend,
        home=settings.home,
        extra_write_dirs=settings.sandbox.extra_write_dirs,
        docker_image=settings.sandbox.docker_image,
        docker_network=settings.sandbox.docker_network,
        docker_mounts=settings.sandbox.docker_mounts,
    )
    sandbox = selection.sandbox
    for tool in build_shell_tools(
        known_secrets=secrets,
        timeout_s=settings.permissions.shell.default_timeout_s,
        sandbox=sandbox,
        sandbox_mode=settings.sandbox.mode,
    ):
        registry.register(tool)
    for tool in build_net_tools(allow_check=lambda url: engine.check_url(url).allowed):
        registry.register(tool)
    for tool in build_memory_tools(store):
        registry.register(tool)
    for tool in build_git_tools(
        known_secrets=secrets, sandbox=sandbox, sandbox_mode=settings.sandbox.mode
    ):
        registry.register(tool)
    registry.register(UpdatePlanTool())
    registry.register(FinishTool())

    skills = SkillRegistry(skill_dirs or default_skill_dirs(settings))
    for tool in build_skill_tools(skills):
        registry.register(tool)

    mcp_problems: list[str] = []
    if connect_mcp and settings.mcp_servers:
        mcp_tools, mcp_problems = await connect_servers(
            settings.mcp_servers, known_secrets=secrets
        )
        for tool in mcp_tools:
            if registry.has(tool.spec.name):
                continue
            registry.register(tool)

    # Discovery happens after the builtin tools are known so the security
    # review can flag a skill that asks for a tool that does not exist.
    load_result = skills.discover(known_tools=set(registry.names()))
    store_skills(store, load_result.loaded)

    models = models or ModelRegistry(settings)
    runtime = AgentRuntime(
        settings=settings,
        store=store,
        registry=registry,
        engine=engine,
        models=models,
        skills=skills,
        on_progress=on_progress,
        bus=bus,
    )
    return Agent(
        settings=settings,
        store=store,
        registry=registry,
        engine=engine,
        skills=skills,
        models=models,
        runtime=runtime,
        redactor=redactor,
        sink=sink,
        mcp_problems=mcp_problems,
        bus=bus,
        sandbox=sandbox,
        sandbox_selection=selection,
    )


def store_skills(store: Store, skills: list[Any]) -> None:
    for skill in skills:
        store.upsert_skill(
            name=skill.name,
            path=str(skill.path),
            description=skill.description,
            source=skill.source,
            status=skill.status,
            allowed_tools=" ".join(sorted(skill.allowed_tools)),
            sha256=skill.sha256,
        )
