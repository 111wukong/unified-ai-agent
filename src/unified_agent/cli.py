"""`uaa` command line interface."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from datetime import datetime, timezone

from unified_agent import __version__
from unified_agent.agent.factory import build_agent, default_skill_dirs
from unified_agent.agent.reflector import Reflector
from unified_agent.agent.state import TaskStatus
from unified_agent.config import (
    ConfigEditor,
    Settings,
    config_to_toml,
    default_home,
    load_settings,
    write_default_config,
)
from unified_agent.errors import ConfigError, ModelError
from unified_agent.skills.loader import SkillError, parse_skill_file, review_skill
from unified_agent.tools.permissions import PermissionEngine
from unified_agent.types import EffectClass

app = typer.Typer(
    name="uaa",
    help="unified-ai-agent — a local-first agent runtime with durable execution.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Read and write configuration.", no_args_is_help=True)
task_app = typer.Typer(help="Inspect and resume tasks.", no_args_is_help=True)
memory_app = typer.Typer(help="Inspect and search memories.", no_args_is_help=True)
workflow_app = typer.Typer(
    help="Compose agent runs declaratively. Validated before anything runs.",
    no_args_is_help=True,
)
app.add_typer(workflow_app, name="workflow")

skill_app = typer.Typer(help="Inspect and validate skills.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(task_app, name="task")
app.add_typer(memory_app, name="memory")
app.add_typer(skill_app, name="skill")

console = Console()
err_console = Console(stderr=True, style="bold red")

STATUS_STYLE = {
    "completed": "bold green",
    "failed": "bold red",
    "cancelled": "yellow",
    "running": "cyan",
    "planning": "cyan",
    "waiting_confirmation": "bold yellow",
    "pending": "dim",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _settings(
    home: Optional[Path], workspace: Optional[Path], *, create: bool = False
) -> Settings:
    try:
        return load_settings(home=home, workspace=workspace, create_if_missing=create)
    except ConfigError as exc:
        err_console.print(f"config error: {escape(str(exc))}")
        raise typer.Exit(2) from exc


def _parse_effects(raw: list[str]) -> tuple[EffectClass, ...]:
    out: list[EffectClass] = []
    for item in raw:
        for chunk in item.replace(",", " ").split():
            try:
                out.append(EffectClass(chunk.strip().lower()))
            except ValueError:
                err_console.print(
                    f"unknown effect {escape(repr(chunk))}. Known: "
                    + ", ".join(e.value for e in EffectClass)
                )
                raise typer.Exit(2) from None
    return tuple(dict.fromkeys(out))


def _progress_printer(quiet: bool):
    def hook(kind: str, payload: dict[str, Any]) -> None:
        if quiet:
            return
        if kind == "plan":
            console.print("[dim]plan[/dim]")
            for i, step in enumerate(payload["steps"], 1):
                console.print(f"  [dim]{i}.[/dim] {escape(str(step))}")
        elif kind == "plan_revised":
            console.print("[dim]plan revised[/dim]")
            for i, step in enumerate(payload["steps"], 1):
                console.print(f"  [dim]{i}.[/dim] {escape(str(step))}")
        elif kind == "tool":
            mark = "[green]ok[/green]" if payload["success"] else "[red]fail[/red]"
            console.print(f"  [dim]→[/dim] {escape(str(payload['name']))} {mark}")
        elif kind == "model_retry":
            console.print(
                f"  [yellow]retry[/yellow] model call #{payload['attempt']}: "
                f"{escape(str(payload['error'])[:120])}"
            )
        elif kind == "compacted":
            console.print("[dim]context compacted[/dim]")
        elif kind == "confirmation":
            console.print(
                f"  [yellow]needs approval[/yellow] {escape(str(payload['effect']))}: "
                f"{escape(str(payload['preview']))}"
            )
        elif kind == "resumed":
            console.print(f"[dim]resumed task {payload['task_id']}[/dim]")
        elif kind == "approval_granted":
            console.print(
                f"  [green]approved[/green] {escape(str(payload['tool']))} "
                f"({escape(str(payload['effect']))})"
            )

    return hook


def _warn_about_sandbox(agent, *, quiet: bool = False) -> None:
    """Say it out loud when isolation is weaker than configured.

    `sandbox.wrap()` deliberately no-ops when the backend is unavailable, so
    a user who configured Seatbelt would otherwise run unsandboxed without
    ever being told.
    """
    if quiet:
        return
    selection = getattr(agent, "sandbox_selection", None)
    warning = selection.warning() if selection is not None else None
    if warning:
        err_console.print(f"[yellow]sandbox[/yellow]: {escape(warning)}")


def _print_result(result, *, quiet: bool = False) -> None:
    if result.needs_approval:
        pending = result.pending_confirmation
        console.print()
        console.print(
            Panel(
                f"[bold]{escape(pending.preview)}[/bold]\n\n"
                f"effect: [yellow]{escape(pending.effect)}[/yellow]\n"
                f"reason: {escape(pending.reason)}\n"
                f"request: [dim]{pending.request_id}[/dim]",
                title="[bold yellow]Approval required[/bold yellow]",
                border_style="yellow",
            )
        )
        console.print(
            f"Approve with: [cyan]uaa task approve {result.task_id}[/cyan]   "
            f"Refuse with: [cyan]uaa task deny {result.task_id}[/cyan]"
        )
        return

    style = STATUS_STYLE.get(result.status, "white")
    if result.answer:
        console.print()
        # markup=False: the answer is arbitrary text (pytest output, regexes,
        # TOML sections). Rich would silently eat anything in [brackets].
        console.print(result.answer, markup=False)
    if not quiet:
        console.print()
        console.print(
            f"[{style}]{result.status}[/{style}]  "
            f"[dim]steps={result.steps} model_calls={result.model_calls} "
            f"tokens={result.usage.total_tokens} cost=${result.usage.cost_usd:.4f} "
            f"{result.duration_s:.1f}s task={result.task_id}[/dim]"
        )
    if result.error and result.status != TaskStatus.COMPLETED.value:
        err_console.print(f"error: {result.error}")


async def _maybe_reflect(agent, result, *, quiet: bool) -> None:
    """Post-task learning, best-effort. Never allowed to break the run."""
    if not result.answer:
        return
    from unified_agent.agent.state import replay

    try:
        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        reflector = Reflector(
            model=agent.models.get(),
            settings=agent.settings,
            store=agent.store,
            memory=getattr(agent, "memory", None),
        )
        if not reflector.worth_running(state):
            return
        reflection = await reflector.run_and_persist(state)
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            console.print(f"[dim]reflection skipped: {type(exc).__name__}[/dim]")
        return
    if quiet:
        return
    for memory in reflection.memories[:5]:
        console.print(f"[dim]remembered: {escape(str(memory.get('content'))[:100])}[/dim]")
    if reflection.skill_candidate:
        console.print(
            f"[dim]skill candidate written: "
            f"{escape(str(reflection.skill_candidate.get('name')))} "
            "(status=candidate, not usable until approved)[/dim]"
        )


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"unified-ai-agent {__version__}")


@app.command()
def init(
    home: Optional[Path] = typer.Option(None, "--home", help="Config directory (default ~/.uaa)."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
) -> None:
    """Create the config file and data directories."""
    target_home = (home or default_home()).expanduser()
    path = write_default_config(target_home / "config.toml", force=force)
    settings = load_settings(home=target_home, create_if_missing=True)
    settings.ensure_dirs()
    console.print(f"config:   [cyan]{path}[/cyan]")
    console.print(f"data dir: [cyan]{settings.home}[/cyan]")
    console.print(f"db:       [cyan]{settings.db_path}[/cyan]")
    console.print(f"logs:     [cyan]{settings.log_dir}[/cyan]")
    console.print(
        "\nNext: set a provider key in your environment "
        "(OPENAI_API_KEY / DEEPSEEK_API_KEY / ANTHROPIC_API_KEY), or run "
        "[cyan]uaa run --model mock \"...\"[/cyan] offline."
    )


@app.command()
def doctor(
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Check the environment and report anything that will bite."""
    settings = _settings(home, workspace)
    table = Table(title="environment", show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("result")

    table.add_row("python", sys.version.split()[0])
    table.add_row("config", str(settings.home / "config.toml"))
    table.add_row("workspace", str(settings.workspace))
    table.add_row("default model", settings.default_model)

    for alias in sorted(settings.models):
        spec = settings.models[alias]
        if spec.provider == "mock":
            table.add_row(f"model:{alias}", "[green]offline, always available[/green]")
        elif spec.api_key():
            table.add_row(f"model:{alias}", f"[green]key found ({spec.key_env()})[/green]")
        else:
            table.add_row(f"model:{alias}", f"[yellow]no key ({spec.key_env()})[/yellow]")

    engine = PermissionEngine(
        settings.permissions, workspace=settings.workspace, home=settings.home
    )
    table.add_row(
        "shell allowlist",
        f"{len(settings.permissions.shell.allow)} entries, "
        f"metacharacters {'allowed' if settings.permissions.shell.allow_metacharacters else 'blocked'}",
    )
    table.add_row(
        "network",
        "[red]allow_all[/red]"
        if settings.permissions.network.allow_all
        else f"{len(settings.permissions.network.allow_domains)} domain(s)",
    )
    from unified_agent.sandbox import build_sandbox

    selection = build_sandbox(
        settings.sandbox.backend,
        home=settings.home,
        extra_write_dirs=settings.sandbox.extra_write_dirs,
        docker_image=settings.sandbox.docker_image,
        docker_network=settings.sandbox.docker_network,
        docker_mounts=settings.sandbox.docker_mounts,
    )
    table.add_row(
        "sandbox",
        f"[green]{selection.sandbox.name}[/green] "
        f"({selection.sandbox.isolation}) mode={settings.sandbox.mode.value}",
    )
    if warning := selection.warning():
        table.add_row("sandbox warning", f"[yellow]{escape(warning)}[/yellow]")
    table.add_row(
        "sensitive paths",
        f"denies ~/.ssh, .env, *.pem — sample: {engine.paths.is_sensitive(settings.home / '.ssh' / 'id_rsa')}",
    )
    for d in default_skill_dirs(settings):
        table.add_row(f"skill dir {d.name}", "[green]exists[/green]" if d.is_dir() else "[dim]missing[/dim]")
    console.print(table)

    leaks = _check_shell_rc_for_keys()
    if leaks:
        console.print(
            "\n[yellow]note[/yellow]: API keys look like they are hard-coded in your shell "
            "config. The runtime reads keys from the environment, which is fine, but a key "
            "sitting in a plaintext dotfile is a different problem:"
        )
        for path, keys in leaks:
            console.print(f"  [dim]{path}[/dim]: {', '.join(keys)}")


def _check_shell_rc_for_keys() -> list[tuple[str, list[str]]]:
    import re

    found: list[tuple[str, list[str]]] = []
    pattern = re.compile(r"^\s*export\s+([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET))\s*=")
    for name in (".zshrc", ".bashrc", ".bash_profile", ".profile"):
        path = Path.home() / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        keys = sorted({m.group(1) for m in pattern.finditer(text)})
        if keys:
            found.append((str(path), keys))
    return found


@app.command()
def tools(
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
    show_schema: bool = typer.Option(False, "--schema", help="Print JSON schemas."),
) -> None:
    """List the registered tools and their permission class."""
    settings = _settings(home, workspace)

    async def _run() -> None:
        agent = await build_agent(settings=settings)
        try:
            table = Table(show_header=True, header_style="bold")
            table.add_column("tool")
            table.add_column("effect")
            table.add_column("confirm")
            table.add_column("idempotent")
            table.add_column("source")
            table.add_column("description", overflow="fold")
            for item in agent.registry.describe():
                decision = settings.permissions.defaults[EffectClass(item["effect"])]
                table.add_row(
                    item["name"],
                    item["effect"],
                    "[yellow]yes[/yellow]" if item["requires_confirmation"] else decision.value,
                    "yes" if item["idempotent"] else "[red]no[/red]",
                    item["source"],
                    escape(item["description"].splitlines()[0][:70]),
                )
            console.print(table)
            if show_schema:
                for spec in agent.registry.specs():
                    console.print_json(json.dumps(spec["function"]["parameters"]))
            for problem in agent.mcp_problems:
                err_console.print(f"mcp: {problem}")
        finally:
            agent.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@config_app.command("show")
def config_show(
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Print the effective configuration."""
    settings = _settings(home, workspace)
    console.print(config_to_toml(settings), markup=False)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="e.g. agent.max_steps, models.local.model, default_model"),
    value: str = typer.Argument(...),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """Set a configuration value (dotted path)."""
    settings = _settings(home, workspace=None, create=True)
    editor = ConfigEditor(settings)
    try:
        applied = editor.set(key, value)
    except ConfigError as exc:
        err_console.print(escape(str(exc)))
        raise typer.Exit(2) from exc
    path = editor.save()
    console.print(f"{key} = {applied!r}  [dim]({path})[/dim]")


# ---------------------------------------------------------------------------
# run / chat
# ---------------------------------------------------------------------------


@app.command()
def run(
    goal: str = typer.Argument(..., help="What the agent should do."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model alias."),
    session: Optional[str] = typer.Option(None, "--session", help="Reuse a named session."),
    approve: list[str] = typer.Option(
        [], "--approve", help="Pre-approve an effect class, e.g. --approve execute_local."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Pre-approve every effect except system_admin."
    ),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
    reflect: bool = typer.Option(
        False, "--reflect", help="Run post-task memory/skill extraction."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    as_json: bool = typer.Option(False, "--json"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Run a single task."""
    settings = _settings(home, workspace)
    effects = list(_parse_effects(approve))
    if yes:
        effects = [e for e in EffectClass if e is not EffectClass.SYSTEM_ADMIN]

    async def _run() -> int:
        agent = await build_agent(
            settings=settings,
            on_progress=None if as_json else _progress_printer(quiet),
            cli_approvals=tuple(effects),
        )
        try:
            _warn_about_sandbox(agent, quiet=quiet or as_json)
            session_id = agent.store.ensure_session(
                name=session or "default",
                working_dir=str(settings.workspace),
                model_alias=model or settings.default_model,
            )
            try:
                result = await agent.runtime.run(
                    goal, session_id=session_id, model_alias=model, max_steps=max_steps
                )
            except ModelError as exc:
                err_console.print(escape(str(exc)))
                return 3
            if reflect:
                await _maybe_reflect(agent, result, quiet=quiet)
            if as_json:
                console.print_json(result.model_dump_json())
            else:
                _print_result(result, quiet=quiet)
            return 0 if result.status in {"completed", "waiting_confirmation"} else 1
        finally:
            agent.close()

    raise typer.Exit(asyncio.run(_run()))


@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    session: Optional[str] = typer.Option("chat", "--session"),
    approve: list[str] = typer.Option([], "--approve"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Interactive session. Each line is a new task in the same session."""
    settings = _settings(home, workspace)
    effects = _parse_effects(approve)
    console.print(
        Panel(
            f"workspace: [cyan]{settings.workspace}[/cyan]\n"
            f"model:     [cyan]{model or settings.default_model}[/cyan]\n"
            "Type a task, or 'exit'.",
            title="uaa chat",
        )
    )

    async def _loop() -> None:
        agent = await build_agent(
            settings=settings,
            on_progress=_progress_printer(False),
            cli_approvals=effects,
        )
        try:
            _warn_about_sandbox(agent)
            session_id = agent.store.ensure_session(
                name=session,
                working_dir=str(settings.workspace),
                model_alias=model or settings.default_model,
            )
            while True:
                try:
                    line = console.input("[bold cyan]uaa>[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    return
                if line.lower() in {"exit", "quit", ":q"}:
                    return
                if not line:
                    continue
                try:
                    result = await agent.runtime.run(
                        line, session_id=session_id, model_alias=model
                    )
                except ModelError as exc:
                    err_console.print(escape(str(exc)))
                    continue
                _print_result(result)
        finally:
            agent.close()

    try:
        asyncio.run(_loop())
    except KeyboardInterrupt:
        console.print()


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


@task_app.command("list")
def task_list(
    limit: int = typer.Option(20, "--limit", "-n"),
    session: Optional[str] = typer.Option(None, "--session"),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """List recent tasks."""
    settings = _settings(home, None)
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        rows = store.list_tasks(session_id=session, limit=limit)
    finally:
        store.close()
    if not rows:
        console.print("[dim]no tasks[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    for column in ("id", "status", "steps", "tokens", "cost", "goal"):
        table.add_column(column, overflow="fold")
    for row in rows:
        style = STATUS_STYLE.get(row["status"], "white")
        table.add_row(
            row["id"],
            f"[{style}]{row['status']}[/{style}]",
            str(row["steps_used"]),
            str(row["tokens_in"] + row["tokens_out"]),
            f"${row['cost_usd']:.4f}",
            row["goal"][:60],
        )
    console.print(table)


@task_app.command("show")
def task_show(
    task_id: str,
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """Show a task's plan, log and ledger."""
    settings = _settings(home, None)
    from unified_agent.agent.state import replay
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        task = store.get_task(task_id)
        if task is None:
            err_console.print(f"unknown task {task_id}")
            raise typer.Exit(1)
        state = replay(store.events(task_id), task_id=task_id, session_id=task["session_id"])
        calls = store.list_tool_calls(task_id)
        artifacts = store.list_artifacts(task_id)
    finally:
        store.close()

    style = STATUS_STYLE.get(state.status.value, "white")
    console.print(
        Panel(
            f"goal:   {escape(state.goal)}\n"
            f"status: [{style}]{state.status.value}[/{style}]\n"
            f"steps:  {state.steps_used}   model calls: {state.model_calls}\n"
            f"tokens: {state.usage.total_tokens}   cost: ${state.usage.cost_usd:.4f}",
            title=f"task {task_id}",
        )
    )
    if state.plan:
        console.print("[bold]plan[/bold]")
        marks = {"pending": " ", "running": "~", "completed": "x", "failed": "!", "skipped": "-"}
        for i, step in enumerate(state.plan, 1):
            console.print(
                f"  [{marks[step.status.value]}] {i}. {escape(step.description)}"
            )

    if calls:
        console.print("\n[bold]tool ledger[/bold]")
        table = Table(show_header=True, header_style="bold")
        for column in ("tool", "status", "attempt", "ms", "args"):
            table.add_column(column, overflow="fold")
        for call in calls:
            colour = {"succeeded": "green", "failed": "red", "ambiguous": "bold yellow"}.get(
                call.status, "dim"
            )
            table.add_row(
                call.name,
                f"[{colour}]{call.status}[/{colour}]",
                str(call.attempt),
                "-" if call.duration_ms is None else str(call.duration_ms),
                escape(json.dumps(call.arguments, ensure_ascii=False)[:60]),
            )
        console.print(table)

    if artifacts:
        console.print("\n[bold]artifacts[/bold]")
        for artifact in artifacts:
            console.print(f"  {artifact['path']}  ({artifact['bytes']} bytes)")

    if state.answer:
        console.print("\n[bold]answer[/bold]")
        console.print(state.answer, markup=False)
    if state.error:
        err_console.print(f"error: {state.error}")


@task_app.command("events")
def task_events(
    task_id: str,
    limit: int = typer.Option(60, "--limit", "-n"),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """Print the event stream (the source of truth)."""
    settings = _settings(home, None)
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        events = store.events(task_id)
    finally:
        store.close()
    for event in events[-limit:]:
        console.print(f"[dim]{event.seq:>4}[/dim]  {escape(event.summary())}")


@task_app.command("resume")
def task_resume(
    task_id: str,
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Resume an interrupted task. Resolves any in-flight tool calls first."""
    settings = _settings(home, workspace)

    async def _run() -> int:
        agent = await build_agent(settings=settings, on_progress=_progress_printer(False))
        try:
            result = await agent.runtime.resume(task_id)
            _print_result(result)
            return 0 if result.status in {"completed", "waiting_confirmation"} else 1
        finally:
            agent.close()

    raise typer.Exit(asyncio.run(_run()))


@task_app.command("approve")
def task_approve(
    task_id: str,
    request: Optional[str] = typer.Option(None, "--request", help="Specific request id."),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Approve a pending tool call and continue the task."""
    settings = _settings(home, workspace)

    async def _run() -> int:
        agent = await build_agent(settings=settings, on_progress=_progress_printer(False))
        try:
            result = await agent.runtime.approve(task_id, request_id=request)
            _print_result(result)
            return 0 if result.status in {"completed", "waiting_confirmation"} else 1
        finally:
            agent.close()

    raise typer.Exit(asyncio.run(_run()))


@task_app.command("deny")
def task_deny(
    task_id: str,
    note: str = typer.Option("", "--note"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Refuse a pending tool call. The agent is told and must not work around it."""
    settings = _settings(home, workspace)

    async def _run() -> int:
        agent = await build_agent(settings=settings, on_progress=_progress_printer(False))
        try:
            result = await agent.runtime.deny(task_id, note=note)
            _print_result(result)
            return 0
        finally:
            agent.close()

    raise typer.Exit(asyncio.run(_run()))


@task_app.command("cancel")
def task_cancel(
    task_id: str,
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """Request cancellation. Works across processes; the running loop stops at its next step."""
    settings = _settings(home, None)
    settings.ensure_dirs()
    marker = settings.state_dir / f"{task_id}.cancel"
    marker.write_text("cancel", encoding="utf-8")
    console.print(f"cancellation requested for {task_id} [dim]({marker})[/dim]")


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


@memory_app.command("list")
def memory_list(
    scope: Optional[str] = typer.Option(None, "--scope"),
    limit: int = typer.Option(50, "--limit", "-n"),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """List stored memories."""
    settings = _settings(home, None)
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        rows = store.list_memories(scope=scope, limit=limit)
    finally:
        store.close()
    if not rows:
        console.print("[dim]no memories[/dim]")
        return
    for row in rows:
        console.print(
            f"[dim]{row['id']}[/dim] [{row['scope']}] {escape(row['content'][:120])}"
            + (f"  [dim]({escape(row['tags'])})[/dim]" if row["tags"] else "")
        )


@memory_app.command("search")
def memory_search(
    query: str,
    limit: int = typer.Option(8, "--limit", "-n"),
    mode: str = typer.Option("auto", "--mode", help="auto | fts | vector | hybrid"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Search memories.

    `hybrid` merges FTS5 and vector results by reciprocal rank fusion, which
    needs no weight tuning because it compares positions rather than scores.
    `auto` picks hybrid when an embedding provider is configured.
    """
    settings = _settings(home, workspace)
    from unified_agent.memory import build_memory_service
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        memory = build_memory_service(settings=settings, store=store)
        rows = asyncio.run(memory.recall(query, limit=limit, mode=mode))
        if mode == "auto" and not memory.embeddings.available:
            console.print(
                f"[dim]vectors: {escape(memory.embeddings.describe())} — set "
                "[cyan]memory.embedding_model[/cyan] for semantic search[/dim]"
            )
    finally:
        store.close()
    if not rows:
        console.print("[dim]no matches[/dim]")
        return
    for row in rows:
        score = f" [dim]{row['similarity']:.3f}[/dim]" if "similarity" in row else ""
        console.print(
            f"[dim]{row['id']}[/dim] [{row['scope']}]{score} {escape(row['content'][:160])}"
        )


@memory_app.command("stats")
def memory_stats(
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """What is stored, and whether vector search is actually available."""
    settings = _settings(home, workspace)
    from unified_agent.memory import build_memory_service
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        stats = build_memory_service(settings=settings, store=store).stats()
    finally:
        store.close()

    console.print(f"active:      [bold]{stats['active']}[/bold]")
    console.print(f"superseded:  {stats['superseded']}")
    if stats["by_source"]:
        console.print(
            "by source:   "
            + ", ".join(f"{k}={v}" for k, v in sorted(stats["by_source"].items()))
        )
    if stats["by_scope"]:
        console.print(
            "by scope:    "
            + ", ".join(f"{k}={v}" for k, v in sorted(stats["by_scope"].items()))
        )
    vectors = stats["vectors"]
    console.print()
    console.print(f"embeddings:  {escape(str(stats['embeddings']))}")
    console.print(f"semantic:    {'yes' if stats['semantic'] else '[yellow]no[/yellow]'}")
    console.print(f"accelerator: {escape(str(vectors['accelerator']))}")
    console.print(f"vectors:     {vectors['vectors']}")
    for entry in vectors["models"]:
        console.print(f"  {escape(str(entry['model']))} dim={entry['dim']} n={entry['count']}")
    if not stats["semantic"]:
        console.print()
        console.print(
            "[dim]The offline fallback hashes character n-grams: it finds "
            "near-duplicates, not meaning. Set [cyan]memory.embedding_model[/cyan] "
            "for real semantic search.[/dim]"
        )


@memory_app.command("reindex")
def memory_reindex(
    scope: Optional[str] = typer.Option(None, "--scope"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Embed memories that have no vector for the current embedding model."""
    settings = _settings(home, workspace)
    from unified_agent.memory import build_memory_service
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        result = asyncio.run(
            build_memory_service(settings=settings, store=store).reindex(scope=scope)
        )
    finally:
        store.close()

    console.print(f"provider: {escape(str(result['provider']))}")
    console.print(f"embedded: {result['embedded']} / {result['pending']}")
    if note := result.get("note"):
        console.print(f"[yellow]{escape(str(note))}[/yellow]")


@memory_app.command("history")
def memory_history(
    memory_id: str,
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Show what a memory replaced, and what replaced it.

    This is what invalidating instead of overwriting buys: the store can
    still answer "what did this used to be".
    """
    settings = _settings(home, workspace)
    from unified_agent.memory import build_memory_service
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        chain = build_memory_service(settings=settings, store=store).history(memory_id)
    finally:
        store.close()

    if not chain:
        err_console.print(f"unknown memory {memory_id}")
        raise typer.Exit(1)
    for index, entry in enumerate(chain):
        marker = "[green]current[/green]" if not entry["superseded_by"] else "[dim]superseded[/dim]"
        arrow = "  " if index == 0 else "← "
        console.print(f"{arrow}{marker} [dim]{entry['id']}[/dim] {escape(entry['content'][:150])}")
        if entry["replaced_reason"]:
            console.print(f"      [dim]reason: {escape(entry['replaced_reason'][:120])}[/dim]")


@memory_app.command("add")
def memory_add(
    content: str,
    scope: str = typer.Option("project", "--scope"),
    tags: str = typer.Option("", "--tags", help="Comma separated."),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """Add a memory by hand."""
    settings = _settings(home, None)
    settings.ensure_dirs()
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        mid = store.add_memory(
            content=content,
            scope=scope,
            tags=[t.strip() for t in tags.split(",") if t.strip()],
            source="user",
        )
    finally:
        store.close()
    console.print(f"saved [dim]{mid}[/dim]")


@memory_app.command("forget")
def memory_forget(
    memory_id: str,
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """Delete a memory."""
    settings = _settings(home, None)
    from unified_agent.storage.store import Store

    store = Store(settings.db_path)
    try:
        ok = store.delete_memory(memory_id)
    finally:
        store.close()
    console.print("deleted" if ok else "no such memory")


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


@skill_app.command("list")
def skill_list(
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """List discovered skills."""
    settings = _settings(home, workspace)
    from unified_agent.skills.registry import SkillRegistry

    registry = SkillRegistry(default_skill_dirs(settings))
    result = registry.discover()
    if not result.loaded:
        console.print(f"[dim]no skills in {', '.join(str(d) for d in registry.dirs)}[/dim]")
    for skill in result.loaded:
        console.print(
            f"[bold]{skill.name}[/bold]  [dim]{skill.source} · {skill.sha256}[/dim]\n"
            f"  {escape(skill.description[:140])}"
            + (f"\n  [dim]tools: {', '.join(sorted(skill.allowed_tools))}[/dim]" if skill.allowed_tools else "")
        )
    for path, problem in result.errors:
        err_console.print(f"{path}: {escape(problem)}")


@skill_app.command("validate")
def skill_validate(
    path: Path = typer.Argument(..., help="A skill directory or a SKILL.md path."),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Validate a skill against the Agent Skills spec and run the security review."""
    target = path / "SKILL.md" if path.is_dir() else path
    if not target.is_file():
        err_console.print(f"no SKILL.md at {target}")
        raise typer.Exit(1)
    try:
        skill = parse_skill_file(target)
    except SkillError as exc:
        err_console.print(escape(str(exc)))
        raise typer.Exit(1) from exc

    # Resolve the real tool set so `allowed-tools` referencing a tool that
    # does not exist is caught here rather than at runtime, mid-task.
    settings = _settings(home, workspace)

    async def _known_tools() -> set[str]:
        agent = await build_agent(settings=settings)
        try:
            return set(agent.registry.names())
        finally:
            agent.close()

    known = asyncio.run(_known_tools())
    report = review_skill(skill, known_tools=known)
    console.print(f"[green]spec: valid[/green]  {skill.name} ({skill.sha256})")
    console.print(f"[dim]checked against {len(known)} registered tools[/dim]")
    console.print(report.render(), markup=False)
    raise typer.Exit(1 if report.blocked else 0)


@skill_app.command("show")
def skill_show(
    name: str,
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Print a skill's full instructions."""
    settings = _settings(home, workspace)
    from unified_agent.skills.registry import SkillRegistry

    registry = SkillRegistry(default_skill_dirs(settings))
    registry.discover()
    skill = registry.get(name)
    if skill is None:
        err_console.print(f"unknown skill {name!r}")
        raise typer.Exit(1)
    # Skill bodies are Markdown with [links](url) -- markup=False keeps them.
    console.print(skill.render(), markup=False)


# ---------------------------------------------------------------------------
# sandbox / serve
# ---------------------------------------------------------------------------


@app.command()
def sandbox(
    backend: str = typer.Option("auto", "--backend", help="auto | seatbelt | docker | none"),
    report: bool = typer.Option(
        False,
        "--report",
        help="Also write the result to <home>/sandbox-verify.json.",
    ),
    report_to: Optional[Path] = typer.Option(
        None, "--report-to", help="Write the result to this path instead."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON."),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Report which process-isolation backend is actually active, and prove it.

    Run this from a normal terminal. Anywhere else -- a container, or an
    environment that sandboxes its child processes -- macOS refuses to apply a
    narrowing Seatbelt profile, so the probe fails there even though it would
    succeed for you. `--report` writes the result to disk so it can be read
    back afterwards instead of copy-pasted.
    """
    import json

    from unified_agent.sandbox import build_sandbox, seatbelt_probe

    settings = _settings(home, workspace)
    settings.ensure_dirs()

    probe = seatbelt_probe()
    selection = build_sandbox(
        backend,
        home=settings.home,
        extra_write_dirs=settings.sandbox.extra_write_dirs,
        docker_image=settings.sandbox.docker_image,
        docker_network=settings.sandbox.docker_network,
        docker_mounts=settings.sandbox.docker_mounts,
    )

    live: list[dict[str, Any]] = []
    if selection.sandbox.name == "seatbelt":
        live = _sandbox_live_check(
            selection.sandbox, settings.workspace, settings.sandbox.mode
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": sys.platform,
        "python": sys.executable,
        "probe": {"ok": probe.ok, "detail": probe.detail},
        "requested": selection.requested,
        "selected": selection.sandbox.name,
        "isolation": selection.sandbox.isolation,
        "fell_back": selection.fell_back,
        "notes": selection.notes,
        "caveats": selection.sandbox.caveats(),
        "live_check": live,
        "verdict": _sandbox_verdict(selection, probe, live),
    }

    target: Path | None = None
    if report_to is not None:
        target = report_to
    elif report or as_json:
        target = settings.home / "sandbox-verify.json"
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if as_json:
        console.print_json(json.dumps(payload, ensure_ascii=False))
        return

    console.print("[bold]probe[/bold]")
    console.print(f"  seatbelt: {'[green]ok[/green]' if probe.ok else '[red]unavailable[/red]'}")
    console.print(f"    {escape(probe.detail)}")
    console.print()
    console.print("[bold]selection[/bold]")
    for line in selection.summary().splitlines():
        console.print(f"  {escape(line)}")
    console.print()
    console.print("[bold]limits[/bold]  (read these before trusting it)")
    for caveat in selection.sandbox.caveats():
        console.print(f"  - {escape(caveat)}")

    if live:
        console.print()
        console.print("[bold]live check[/bold]  (writes outside the allowlist must fail)")
        for row in live:
            style = "green" if row["ok"] else "red"
            console.print(
                f"  {escape(row['label'])}: [{style}]{escape(row['result'])}[/{style}]"
                + (f"  [dim]{escape(row['detail'][:80])}[/dim]" if row["detail"] else "")
            )

    console.print()
    console.print(f"[bold]verdict[/bold]  {escape(payload['verdict'])}")
    if target is not None:
        console.print(f"[dim]written to {escape(str(target))}[/dim]")

    if not probe.ok and sys.platform == "darwin":
        # The one case where the user has to act, so give them the exact line
        # rather than a description of it. A command they have to assemble is
        # a command that does not get run.
        console.print()
        console.print(
            "[bold]seatbelt could not be probed here.[/bold] macOS refuses to install a "
            "narrowing profile from a process that is already sandboxed, so this check "
            "has to run outside one."
        )
        console.print("[dim]Open Terminal.app and run:[/dim]")
        # Quoted: project paths routinely contain spaces ("WorkBuddy AI"),
        # and an unquoted path is a command that fails when pasted.
        console.print(
            f"  [cyan]cd {_sh(settings.workspace)} && "
            f"{_sh(sys.executable)} -m unified_agent.cli sandbox --report[/cyan]"
        )


def _sh(value: Any) -> str:
    """Shell-quote a path for a copy-pasteable command."""
    import shlex

    return shlex.quote(str(value))


def _sandbox_verdict(selection: Any, probe: Any, live: list[dict[str, Any]]) -> str:
    """One sentence a reader can act on."""
    if selection.sandbox.name == "none":
        if probe.ok:
            return "seatbelt works here but a different backend was requested"
        return (
            "NO ISOLATION ACTIVE. If you are reading this from inside a container or a "
            "sandboxed parent process, run `uaa sandbox` again from a normal terminal."
        )
    if not live:
        return f"{selection.sandbox.name} is active; no live check was run"
    # `ok` on a write row means the isolation held. So a row with `ok` false
    # is an escape -- the one outcome that must never be reported as success.
    escapes = [row for row in live if row["label"].startswith("write") and not row["ok"]]
    reads_ok = [row for row in live if row["label"].startswith("read") and row["ok"]]
    if escapes:
        return (
            f"{selection.sandbox.name} is active but a write ESCAPED the allowlist -- "
            "treat the isolation as not working"
        )
    if reads_ok:
        return (
            f"{selection.sandbox.name} is active: it blocked a write outside the "
            "workspace and still allowed reads"
        )
    return (
        f"{selection.sandbox.name} is active but reads were blocked, which breaks the "
        "toolchain -- the agent cannot read the code it is meant to work on"
    )


def _sandbox_live_check(sandbox: object, workspace: Path, mode: object) -> list[dict[str, Any]]:
    """Actually attempt an escape. A report that never tries is worthless.

    Returns rows shaped for the JSON report, so the printed view and the file
    are the same data rather than two renderings that can drift.
    """
    import asyncio
    import tempfile

    from unified_agent.tools.base import ToolContext
    from unified_agent.tools.shell import RunCommandTool

    async def _run() -> list[dict[str, Any]]:
        outside = Path(tempfile.mkdtemp(prefix="uaa-sbx-"))
        target = outside / "escape.txt"
        ctx = ToolContext(
            task_id="probe",
            session_id="probe",
            step_id="step_1",
            workspace=workspace,
            home=sandbox.home,  # type: ignore[attr-defined]
            artifact_dir=outside,
        )
        tool = RunCommandTool(sandbox=sandbox, sandbox_mode=mode)  # type: ignore[arg-type]
        rows: list[dict[str, Any]] = []

        result = await tool.run({"command": f"touch {target}"}, ctx)
        # `ok` is "the isolation behaved as intended", which for a write is
        # the opposite of "the command succeeded".
        rows.append(
            {
                "label": "write outside the workspace",
                "ok": not target.exists(),
                "result": "ESCAPED" if target.exists() else "blocked",
                "detail": result.error or "",
            }
        )
        result = await tool.run({"command": "cat /etc/hosts"}, ctx)
        rows.append(
            {
                "label": "read a system file",
                "ok": result.success,
                "result": "allowed" if result.success else "blocked",
                "detail": result.error or "",
            }
        )
        return rows

    return asyncio.run(_run())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
    token: bool = typer.Option(
        False,
        "--token",
        help="Require a per-run session token on API calls (recommended).",
    ),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Start the HTTP/WebSocket service and the web console.

    `POST /agui` speaks the AG-UI protocol over SSE; the console at `/`
    consumes it. Bind to 127.0.0.1 by default: this service can execute
    commands, so exposing it is a deliberate decision, not a default.
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        err_console.print(
            "the API needs fastapi and uvicorn: pip install -e \".[api]\""
        )
        raise typer.Exit(2) from exc

    import secrets as _secrets

    from unified_agent.api.app import create_app
    from unified_agent.sandbox import build_sandbox

    settings = _settings(home, workspace)
    settings.ensure_dirs()
    session_token = _secrets.token_urlsafe(24) if token else None
    application = create_app(
        settings, token=session_token, allowed_hosts=[host, "127.0.0.1", "localhost", "::1"]
    )

    selection = build_sandbox(
        settings.sandbox.backend,
        home=settings.home,
        extra_write_dirs=settings.sandbox.extra_write_dirs,
        docker_image=settings.sandbox.docker_image,
        docker_network=settings.sandbox.docker_network,
        docker_mounts=settings.sandbox.docker_mounts,
    )
    console.print(f"sandbox: [cyan]{selection.sandbox.describe()}[/cyan]")
    if warning := selection.warning():
        console.print(f"[yellow]warning[/yellow]: {escape(warning)}")

    if session_token:
        console.print(
            f"console: [cyan]http://{host}:{port}/#{session_token}[/cyan]"
            "  [dim](the token is in the fragment; it never reaches the server)[/dim]"
        )
        console.print(
            "[dim]api calls need the header[/dim] "
            f"[cyan]X-UAA-Token: {session_token}[/cyan]"
        )
    else:
        console.print(f"console: [cyan]http://{host}:{port}/[/cyan]")
    console.print(f"api docs: [cyan]http://{host}:{port}/docs[/cyan]")
    console.print(f"ag-ui:   [cyan]POST http://{host}:{port}/agui[/cyan]  (SSE)")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            "[bold yellow]warning[/bold yellow]: bound to a non-loopback address. "
            "This service can execute commands on this machine."
        )
    uvicorn.run(application, host=host, port=port, reload=reload, log_level="info")


@app.command()
def desktop(
    port: int = typer.Option(0, "--port", help="0 picks a free port."),
    width: int = typer.Option(1180, "--width"),
    height: int = typer.Option(820, "--height"),
    bundle: bool = typer.Option(
        False, "--bundle", help="Build a macOS .app bundle instead of opening a window."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Where to put the .app (default ~/Applications)."
    ),
    debug: bool = typer.Option(False, "--debug", help="Open webview devtools."),
    no_token: bool = typer.Option(
        False, "--no-token", help="Disable the session token (weaker; loopback only)."
    ),
    check: bool = typer.Option(
        False, "--check", help="Start and stop the server without opening a window."
    ),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Open the desktop window, or build a double-clickable app bundle.

    The window is the operating system's own webview around the same server
    and the same console the browser uses, so there is no second UI to keep
    in sync. `--bundle` produces a macOS `.app`; `--check` exercises the
    server lifecycle without a display, which is what CI can verify.
    """
    import secrets as _secrets

    from unified_agent.desktop import (
        BundleError,
        DesktopUnavailable,
        build_app_bundle,
        desktop_available,
        run_desktop,
    )

    settings = _settings(home, workspace)
    settings.ensure_dirs()

    if bundle:
        target = output or (Path.home() / "Applications")
        target.mkdir(parents=True, exist_ok=True)
        try:
            result = build_app_bundle(target_dir=target, version=__version__)
        except BundleError as exc:
            err_console.print(escape(str(exc)))
            raise typer.Exit(2) from exc
        console.print(f"[green]built[/green] {escape(str(result.path))}")
        console.print(escape(result.summary()), markup=False)
        console.print(f"\nOpen it with: [cyan]open {escape(str(result.path))}[/cyan]")
        return

    if check:
        # The headless check exists for environments with no display, so it
        # must not gate on the webview being importable -- otherwise CI, which
        # is exactly that environment, can never run it.
        console.print("webview: [dim]skipped (--check verifies the server only)[/dim]")
    else:
        ok, detail = desktop_available()
        console.print(f"webview: [cyan]{escape(detail)}[/cyan]")
        if not ok:
            err_console.print(
                "\nInstall the desktop extra, or use [cyan]uaa serve[/cyan] and open "
                "the console in a browser:\n  pip install -e \".[desktop]\""
            )
            raise typer.Exit(2)

    token = None if no_token else _secrets.token_urlsafe(24)
    try:
        code = run_desktop(
            settings=settings,
            port=port,
            token=token,
            size=(width, height),
            debug=debug,
            headless_check=check,
        )
    except DesktopUnavailable as exc:
        err_console.print(escape(str(exc)))
        raise typer.Exit(3) from exc
    if check:
        console.print("[green]ok[/green] server lifecycle works headlessly")
    raise typer.Exit(code)


@workflow_app.command("list")
def workflow_list(
    directory: Optional[Path] = typer.Option(None, "--dir", help="Defaults to ./workflows"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """List workflow files and whether each one loads."""
    from unified_agent.orchestration import WorkflowError, discover, load_workflow

    settings = _settings(home, workspace)
    root = directory or (settings.workspace / "workflows")
    paths = discover(root)
    if not paths:
        console.print(f"[dim]no workflow files under {escape(str(root))}[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("workflow")
    table.add_column("nodes", justify="right")
    table.add_column("status")
    table.add_column("path", style="dim")
    for path in paths:
        try:
            workflow = load_workflow(path)
            status = "[green]ok[/green]"
            nodes = str(len(workflow.nodes))
            name = workflow.name
        except WorkflowError as exc:
            status = f"[red]{len(exc.problems)} problem(s)[/red]"
            nodes = "-"
            name = path.stem
        table.add_row(name, nodes, status, str(path.relative_to(root)))
    console.print(table)


@workflow_app.command("validate")
def workflow_validate(
    target: str = typer.Argument(..., help="A file path or a name under ./workflows"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Check a workflow without running it.

    Reports *every* problem at once rather than the first: a load-time error
    you have to fix one at a time is a load-time error you stop using.
    """
    from unified_agent.orchestration import WorkflowError, load_workflow

    settings = _settings(home, workspace)
    path = _workflow_path(target, settings)
    if path is None or not path.is_file():
        err_console.print(f"no workflow at {escape(target)}")
        raise typer.Exit(1)

    try:
        workflow = load_workflow(path)
    except WorkflowError as exc:
        err_console.print(f"[red]invalid[/red] {escape(str(path))}")
        for problem in exc.problems:
            console.print(f"  {escape(problem.render())}")
        raise typer.Exit(1) from exc

    console.print(f"[green]valid[/green] {escape(str(path))}")
    console.print(escape(workflow.describe()), markup=False)
    raise typer.Exit(0)


@workflow_app.command("show")
def workflow_show(
    target: str = typer.Argument(...),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Print a workflow's graph: nodes in execution order and their wiring."""
    from unified_agent.orchestration import WorkflowError, load_workflow

    settings = _settings(home, workspace)
    path = _workflow_path(target, settings)
    if path is None or not path.is_file():
        err_console.print(f"no workflow at {escape(target)}")
        raise typer.Exit(1)
    try:
        workflow = load_workflow(path)
    except WorkflowError as exc:
        for problem in exc.problems:
            console.print(f"  {escape(problem.render())}")
        raise typer.Exit(1) from exc

    console.print(f"[bold]{escape(workflow.name)}[/bold] [dim]{escape(str(path))}[/dim]")
    if workflow.description:
        console.print(escape(workflow.description.strip()), markup=False)
    console.print()

    for node_id in workflow.topological_order():
        node = workflow.nodes[node_id]
        console.print(f"[bold]{escape(node_id)}[/bold]  [dim]{node.type.value}[/dim]")
        if node.title:
            console.print(f"    {escape(node.title)}")
        for field_name in ("goal", "tool", "code", "over"):
            if node.data.get(field_name):
                preview = str(node.data[field_name]).strip().splitlines()[0][:90]
                console.print(f"    {field_name}: {escape(preview)}")
        if node.type.value == "ifelse":
            for branch in node.data.get("branches") or []:
                console.print(
                    f"    when {escape(str(branch.get('when')))} -> "
                    f"{escape(str(branch.get('target')))}"
                )
            if node.data.get("else"):
                console.print(f"    else -> {escape(str(node.data['else']))}")
        for successor in workflow.successors(node_id):
            console.print(f"    [dim]-> {escape(successor)}[/dim]")
        console.print()


@workflow_app.command("run")
def workflow_run(
    target: str = typer.Argument(...),
    inputs: Optional[list[str]] = typer.Option(
        None, "--input", "-i", help="key=value, repeatable"
    ),
    model: Optional[str] = typer.Option(None, "--model"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Run a workflow. A run is a task, so `uaa task events <id>` shows it."""
    from unified_agent.orchestration import (
        WorkflowError,
        WorkflowRunner,
        load_workflow,
    )

    settings = _settings(home, workspace)
    path = _workflow_path(target, settings)
    if path is None or not path.is_file():
        err_console.print(f"no workflow at {escape(target)}")
        raise typer.Exit(1)
    try:
        workflow = load_workflow(path)
    except WorkflowError as exc:
        err_console.print("[red]the workflow does not load:[/red]")
        for problem in exc.problems:
            console.print(f"  {escape(problem.render())}")
        raise typer.Exit(1) from exc

    resolved: dict[str, Any] = {}
    for item in inputs or []:
        key, _, value = item.partition("=")
        if not key:
            err_console.print(f"bad --input {escape(item)!r}; expected key=value")
            raise typer.Exit(2)
        resolved[key.strip()] = value

    missing = [
        str(spec["name"])
        for spec in workflow.inputs
        if spec.get("required") and not spec.get("default") and str(spec["name"]) not in resolved
    ]
    if missing:
        err_console.print(f"missing required input(s): {', '.join(missing)}")
        raise typer.Exit(2)

    async def _run() -> Any:
        agent = await build_agent(settings=settings, on_progress=_progress_printer(quiet=False))
        try:
            runner = WorkflowRunner(agent=agent, on_progress=_progress_printer(quiet=False))
            return await runner.run(workflow, inputs=resolved, model_alias=model)
        finally:
            agent.close()

    result = asyncio.run(_run())
    console.print()
    style = "green" if result.completed else "red"
    console.print(f"[bold {style}]{escape(result.status)}[/bold {style}]")
    console.print(escape(result.summary()), markup=False)
    console.print(f"[dim]task={result.task_id}  {result.duration_s:.1f}s[/dim]")

    if not result.completed:
        # Say exactly which node failed and why. A run that reports failure
        # without a reason is a run the user has to debug by reading the DB.
        for node in result.failures():
            err_console.print(f"[red]{escape(node.node_id)}[/red] {escape(node.error[:400])}")
        if result.error and not result.failures():
            err_console.print(escape(result.error[:400]))
    if result.outputs:
        console.print()
        console.print("[bold]outputs[/bold]")
        for key, value in result.outputs.items():
            console.print(f"  {escape(str(key))}: {escape(str(value)[:400])}")
    raise typer.Exit(0 if result.completed else 1)


def _workflow_path(target: str, settings: Any) -> Optional[Path]:
    """Accept either a path or a bare name resolved under ./workflows."""
    candidate = Path(target).expanduser()
    if candidate.is_file():
        return candidate
    root = Path(settings.workspace) / "workflows"
    for suffix in (".yaml", ".yml"):
        named = root / f"{target}{suffix}"
        if named.is_file():
            return named
    direct = root / target
    return direct if direct.is_file() else None


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


@app.command()
def demo(
    goal: str = typer.Argument("查看当前目录下的 Python 文件并总结"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """End-to-end run using the offline mock model. No API key needed."""
    settings = _settings(home, workspace)

    async def _run() -> int:
        agent = await build_agent(settings=settings, on_progress=_progress_printer(False))
        try:
            session_id = agent.store.ensure_session(
                name="demo", working_dir=str(settings.workspace), model_alias="mock"
            )
            result = await agent.runtime.run(goal, session_id=session_id, model_alias="mock")
            _print_result(result)
            console.print(
                f"\n[dim]inspect with: uaa task show {result.task_id} / "
                f"uaa task events {result.task_id}[/dim]"
            )
            return 0
        finally:
            agent.close()

    raise typer.Exit(asyncio.run(_run()))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
