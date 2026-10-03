"""`wukong` command line interface."""

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

from datetime import datetime, timedelta, timezone

from wukong import __version__
from wukong.agent.factory import build_agent, default_skill_dirs
from wukong.agent.state import TaskStatus
from wukong.observability.events import EventType
from wukong.config import (
    ConfigEditor,
    Settings,
    config_to_toml,
    default_home,
    load_settings,
    write_default_config,
)
from wukong.errors import ConfigError, ModelError
from wukong.skills.loader import SkillError, parse_skill_file, review_skill
from wukong.tools.permissions import PermissionEngine
from wukong.types import EffectClass

app = typer.Typer(
    name="wukong",
    help="wukong — 本地优先的 agent 运行时，支持持久化执行。",
    no_args_is_help=True,
    # Off, deliberately, and it is not laziness. Click's completion machinery
    # writes into the user's shell config on first run. In a sandboxed or
    # read-only home that does not fail -- it *hangs*, and the command had to
    # be killed with SIGTERM. An installer that edits `~/.zshrc` without
    # asking is also the wrong default shape: `opencode completion` prints a
    # script and lets the user decide, which is the one to copy.
    add_completion=False,
)
config_app = typer.Typer(help="读写配置。", no_args_is_help=True)
task_app = typer.Typer(help="查看与恢复任务。", no_args_is_help=True)
memory_app = typer.Typer(help="查看与检索记忆。", no_args_is_help=True)
workflow_app = typer.Typer(
    help="用声明式文件编排 agent 运行，跑之前先校验。",
    no_args_is_help=True,
)
app.add_typer(workflow_app, name="workflow")

skill_app = typer.Typer(help="查看与校验技能。", no_args_is_help=True)
a2a_app = typer.Typer(help="Agent 之间通信：发布自己、调用别人。", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(task_app, name="task")
app.add_typer(memory_app, name="memory")
app.add_typer(skill_app, name="skill")
app.add_typer(a2a_app, name="a2a")

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


def _read_store(settings: Settings):
    """Open the database for a command that only reads.

    Every listing command used to open a writable connection. A read path
    that *can* write is a read path whose bug corrupts the store, so the
    read-only commands go through here instead.
    """
    from wukong.storage.store import Store

    return Store.readonly(settings.db_path)


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
            f"Approve with: [cyan]wukong task approve {result.task_id}[/cyan]   "
            f"Refuse with: [cyan]wukong task deny {result.task_id}[/cyan]"
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


def _report_reflection(agent, result, *, quiet: bool) -> None:
    """Show what the run learned, by reading the events it left.

    The extraction itself belongs to the runtime -- it has to happen for
    every entry point, not just this one -- so the CLI only renders the
    result. Doing the work here too would mean an HTTP-started task learns
    nothing while a terminal-started one learns, which is what used to be
    the case.
    """
    if quiet:
        return
    for event in agent.store.events(result.task_id):
        if event.type is EventType.MEMORY_WRITTEN:
            preview = str(event.payload.get("preview") or "")
            if preview:
                console.print(f"[dim]remembered: {escape(preview[:100])}[/dim]")
        elif event.type is EventType.SKILL_CANDIDATE:
            name = event.payload.get("name")
            if name:
                console.print(
                    f"[dim]skill candidate written: {escape(str(name))} "
                    "(status=candidate, not usable until approved)[/dim]"
                )


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """打印版本。"""
    console.print(f"wukong {__version__}")


@app.command()
def init(
    home: Optional[Path] = typer.Option(None, "--home", help="Config directory (default ~/.wukong)."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
) -> None:
    """创建配置文件和数据目录。"""
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
        "[cyan]wukong run --model mock \"...\"[/cyan] offline."
    )


def _store_health(settings: Settings, *, sample: int = 20) -> list[tuple[str, str]]:
    """Compare the stored projection against the event fold.

    `tasks.state` is written on every step and read by nothing, while the
    project's central claim is "events are truth, the projection is a cache
    and is wrong by construction if the two disagree". That claim is only
    worth anything if something checks it, so this does.

    Sampling the most recent N tasks keeps `wukong doctor` fast on a large
    database while still catching a projection that has stopped being
    updated.
    """
    from wukong.agent.state import replay
    from wukong.storage.store import Store

    if not settings.db_path.exists():
        return [("store", "[dim]尚未创建[/dim]")]

    store = Store.readonly(settings.db_path)
    try:
        tasks = store.list_tasks(limit=sample)
        drift: list[str] = []
        for task in tasks:
            full = store.get_task(task["id"]) or {}
            state = replay(store.events(task["id"]), task_id=task["id"])
            if full.get("status") != state.status.value:
                drift.append(f"{task['id']} status")
            elif int(full.get("steps_used") or 0) != state.steps_used:
                drift.append(f"{task['id']} steps")
        ambiguous = sum(
            1
            for task in tasks
            for record in store.list_tool_calls(task["id"])
            if record.status == "ambiguous"
        )
    finally:
        store.close()

    rows: list[tuple[str, str]] = []
    if not tasks:
        rows.append(("store", "[dim]no tasks recorded[/dim]"))
    elif drift:
        rows.append(
            (
                "projection",
                f"[red]{len(drift)}/{len(tasks)} disagree with the event log[/red] "
                f"[dim]({', '.join(drift[:3])})[/dim]",
            )
        )
    else:
        rows.append(
            ("projection", f"[green]{len(tasks)} task(s) agree with the event log[/green]")
        )
    if ambiguous:
        rows.append(
            (
                "ambiguous calls",
                f"[yellow]{ambiguous}[/yellow] [dim](outcome unknown after an "
                "interrupt; may have run)[/dim]",
            )
        )
    return rows


@app.command()
def doctor(
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """检查环境，报告会咬人的地方。"""
    settings = _settings(home, workspace)
    table = Table(title="环境", show_header=True, header_style="bold")
    table.add_column("检查项")
    table.add_column("结果")

    table.add_row("python", sys.version.split()[0])
    table.add_row("config", str(settings.home / "config.toml"))
    table.add_row("workspace", str(settings.workspace))
    table.add_row("default model", settings.default_model)

    for alias in sorted(settings.models):
        spec = settings.models[alias]
        if spec.provider == "mock":
            table.add_row(f"model:{alias}", "[green]离线，始终可用[/green]")
        elif spec.api_key():
            table.add_row(f"model:{alias}", f"[green]已配置密钥（{spec.key_env()}）[/green]")
        else:
            table.add_row(f"model:{alias}", f"[yellow]未配置密钥（{spec.key_env()}）[/yellow]")

    engine = PermissionEngine(
        settings.permissions, workspace=settings.workspace, home=settings.home
    )
    table.add_row(
        "shell allowlist",
        f"白名单 {len(settings.permissions.shell.allow)} 条，"
        f"元字符 {'允许' if settings.permissions.shell.allow_metacharacters else '禁止'}",
    )
    table.add_row(
        "network",
        "[red]allow_all[/red]"
        if settings.permissions.network.allow_all
        else f"{len(settings.permissions.network.allow_domains)} 个域名",
    )
    from wukong.sandbox import build_sandbox

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
    for label, detail in _store_health(settings):
        table.add_row(label, detail)
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
    effect: Optional[str] = typer.Option(
        None, "--effect", help="Only tools in one effect class, e.g. execute_local."
    ),
) -> None:
    """列出已注册的工具及其权限等级。"""
    settings = _settings(home, workspace)
    wanted: Optional[EffectClass] = None
    if effect:
        try:
            wanted = EffectClass(effect)
        except ValueError as exc:
            err_console.print(
                f"unknown effect {effect!r}; expected one of "
                f"{', '.join(e.value for e in EffectClass)}"
            )
            raise typer.Exit(2) from exc

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
            for item in agent.registry.describe(effect=wanted):
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
# stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    home: Optional[Path] = typer.Option(None, help="配置目录（默认 ~/.wukong）。"),
    days: int = typer.Option(0, help="只统计最近 N 天的任务。0 表示全部。"),
) -> None:
    """这些任务花了多少，以及哪些在失败。

    读的是投影表，所以是一条查询而不是把每个事件折一遍 —— 这是概览视图，
    正是投影存在的理由。`task show` 才是必须精确折叠的那个。
    """
    settings = _settings(home, None)
    since = None
    if days > 0:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    store = _read_store(settings)
    try:
        summary = store.task_stats(since=since)
    finally:
        store.close()

    if not summary["tasks"]:
        console.print("[dim]还没有任务。[/dim]")
        return

    totals = Table(title="用量", show_header=True, header_style="bold")
    totals.add_column("指标")
    totals.add_column("值", justify="right")
    totals.add_row("任务数", str(summary["tasks"]))
    totals.add_row("总步数", str(summary["steps"]))
    totals.add_row("输入 token", f"{summary['tokens_in']:,}")
    totals.add_row("输出 token", f"{summary['tokens_out']:,}")
    totals.add_row("合计 token", f"{summary['tokens_in'] + summary['tokens_out']:,}")
    totals.add_row("成本", f"${summary['cost_usd']:.4f}")
    console.print(totals)

    by_status = Table(title="按状态", show_header=True, header_style="bold")
    by_status.add_column("状态")
    by_status.add_column("数量", justify="right")
    for name, count in sorted(summary["by_status"].items(), key=lambda kv: -kv[1]):
        by_status.add_row(name, str(count))
    console.print(by_status)

    if summary["most_expensive"]:
        dearest = Table(title="最贵的任务", show_header=True, header_style="bold")
        dearest.add_column("任务")
        dearest.add_column("状态")
        dearest.add_column("步数", justify="right")
        dearest.add_column("成本", justify="right")
        dearest.add_column("目标")
        for task in summary["most_expensive"]:
            dearest.add_row(
                task["id"],
                task["status"],
                str(task["steps_used"]),
                f"${task['cost_usd'] or 0:.4f}",
                escape((task["goal"] or "").splitlines()[0][:36]),
            )
        console.print(dearest)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@config_app.command("show")
def config_show(
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """打印生效的配置。"""
    settings = _settings(home, workspace)
    console.print(config_to_toml(settings), markup=False)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="e.g. agent.max_steps, models.local.model, default_model"),
    value: str = typer.Argument(...),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """设置一个配置项（点号路径）。"""
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
    multi_agent: bool = typer.Option(
        False,
        "--multi-agent",
        help="Expose the `delegate` tool so this task can fan out to sub-agents.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    as_json: bool = typer.Option(False, "--json"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """运行单个任务。"""
    settings = _settings(home, workspace)
    # The runtime owns reflection (so the HTTP layer gets it too); the flag
    # just turns it on for this run.
    settings.agent.reflect = reflect
    if multi_agent:
        settings.multi_agent.enabled = True
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
                _report_reflection(agent, result, quiet=quiet or as_json)
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
    """交互式会话。每一行是同一会话里的一个新任务。"""
    settings = _settings(home, workspace)
    effects = _parse_effects(approve)
    console.print(
        Panel(
            f"workspace: [cyan]{settings.workspace}[/cyan]\n"
            f"model:     [cyan]{model or settings.default_model}[/cyan]\n"
            "Type a task, or 'exit'.",
            title="wukong chat",
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
                    line = console.input("[bold cyan]wukong>[/bold cyan] ").strip()
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
    resumable: bool = typer.Option(
        False, "--resumable", help="Only tasks that can still be picked up."
    ),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """列出最近的任务。"""
    settings = _settings(home, None)
    from wukong.agent.state import RESUMABLE_STATUSES

    store = _read_store(settings)
    try:
        rows = store.list_tasks(
            session_id=session,
            statuses=[s.value for s in RESUMABLE_STATUSES] if resumable else None,
            limit=limit,
        )
    finally:
        store.close()
    if not rows:
        console.print(
            "[dim]no resumable tasks[/dim]"
            if resumable
            else "[dim]no tasks[/dim]"
        )
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
    """查看某个任务的计划、日志与工具台账。"""
    settings = _settings(home, None)
    from wukong.agent.state import replay

    store = _read_store(settings)
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
    """打印事件流（唯一真相）。"""
    settings = _settings(home, None)

    store = _read_store(settings)
    try:
        events = store.events(task_id)
    finally:
        store.close()
    for event in events[-limit:]:
        console.print(f"[dim]{event.seq:>4}[/dim]  {escape(event.summary())}")


@task_app.command("rewind")
def task_rewind(
    task_id: str,
    apply: bool = typer.Option(
        False, "--apply", help="Actually restore. Without this it is a preview."
    ),
    from_seq: int = typer.Option(
        0, "--from-seq", help="Only undo writes at or after this event sequence number."
    ),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """按检查点还原某个任务改过的文件。

    Previews by default. Restoring files is destructive in the one way that
    matters -- it can overwrite work done after the agent's -- so the
    destructive action has to be asked for explicitly rather than being the
    default of a command someone types to see what happened.
    """
    from wukong.agent.rewind import apply_rewind, plan_rewind

    settings = _settings(home, None)
    store = _read_store(settings)
    try:
        plan = plan_rewind(store, task_id, from_seq=from_seq)
    finally:
        store.close()

    console.print(plan.render(), markup=False)
    if not apply:
        if plan.actionable:
            console.print()
            console.print(
                f"[dim]preview only. Re-run with [cyan]--apply[/cyan] to "
                f"rewrite {len(plan.actionable)} file(s).[/dim]"
            )
        return
    if not plan.actionable:
        return
    for line in apply_rewind(plan):
        console.print(f"[green]{escape(line)}[/green]")
    if plan.blocked:
        err_console.print(
            f"{len(plan.blocked)} file(s) were left untouched; see the list above."
        )


@task_app.command("search")
def task_search(
    query: str,
    limit: int = typer.Option(8, "--limit", "-n"),
    all_tasks: bool = typer.Option(
        False, "--all", help="Include sub-agent and workflow-child tasks."
    ),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """按目标或最终答案搜索过去的任务。"""
    settings = _settings(home, None)
    store = _read_store(settings)
    try:
        rows = store.search_tasks(query, limit=limit, include_children=all_tasks)
    finally:
        store.close()

    if not rows:
        console.print(f"no task matches {query!r}")
        console.print(
            "[dim]Sub-agent tasks are excluded by default; pass --all to include them.[/dim]"
        )
        return
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("task", style="cyan", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("when", no_wrap=True)
    table.add_column("goal", overflow="fold")
    for row in rows:
        table.add_row(
            row["id"],
            row["status"],
            (row.get("created_at") or "")[:16].replace("T", " "),
            row["goal"][:120],
        )
    console.print(table)


@task_app.command("resume")
def task_resume(
    task_id: str,
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """恢复被中断的任务。先处理掉在途的工具调用。"""
    settings = _settings_for_task(task_id, home, workspace)

    async def _run() -> int:
        agent = await build_agent(settings=settings, on_progress=_progress_printer(False))
        try:
            result = await agent.runtime.resume(task_id)
            _print_result(result)
            return 0 if result.status in {"completed", "waiting_confirmation"} else 1
        finally:
            agent.close()

    raise typer.Exit(asyncio.run(_run()))


def _settings_for_task(
    task_id: str, home: Optional[Path], workspace: Optional[Path]
) -> Settings:
    """Settings for a command that resumes an existing task.

    A task acts where it was created, so `wukong task approve <id>` adopts the
    recorded workspace when `--workspace` was not given -- the user should not
    have to remember which directory a task was started in.

    When `--workspace` *was* given and disagrees, this leaves it alone and the
    runtime refuses: silently relocating a task's side effects is worse than an
    error, and an error can name the flag to pass.
    """
    settings = _settings(home, workspace)
    if workspace is not None or not settings.db_path.exists():
        return settings
    from wukong.storage.store import Store

    store = Store.readonly(settings.db_path)
    try:
        recorded = store.task_workspace(task_id)
    finally:
        store.close()
    if recorded and Path(recorded).resolve() != settings.workspace.resolve():
        settings.workspace = Path(recorded).resolve()
    return settings


@task_app.command("approve")
def task_approve(
    task_id: str,
    request: Optional[str] = typer.Option(None, "--request", help="Specific request id."),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """批准一个待定的工具调用，任务继续。"""
    settings = _settings_for_task(task_id, home, workspace)

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
    """拒绝一个待定的工具调用。会明确告诉模型，且它不得绕路。"""
    settings = _settings_for_task(task_id, home, workspace)

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
    """取消任务。已暂停的立即取消；运行中的在下一个步边界停下。"""
    from wukong.agent.runtime import cancel_task

    settings = _settings(home, None)
    settings.ensure_dirs()
    store = _read_store(settings)
    try:
        outcome = cancel_task(store, settings, task_id)
    finally:
        store.close()

    if outcome == "not_found":
        err_console.print(f"no such task: {task_id}")
        raise typer.Exit(1)
    if outcome == "already_terminal":
        console.print(f"{task_id} has already finished; nothing to cancel")
        return
    if outcome == "cancelled":
        console.print(f"cancelled {task_id}")
        return
    console.print(f"cancellation requested for {task_id} [dim](stops at the next step)[/dim]")


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


@memory_app.command("list")
def memory_list(
    scope: Optional[str] = typer.Option(None, "--scope"),
    limit: int = typer.Option(50, "--limit", "-n"),
    home: Optional[Path] = typer.Option(None, "--home"),
) -> None:
    """列出已存的记忆。"""
    settings = _settings(home, None)

    store = _read_store(settings)
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
    from wukong.memory import build_memory_service

    store = _read_store(settings)
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
    """存了什么，以及向量检索到底可不可用。"""
    settings = _settings(home, workspace)
    from wukong.memory import build_memory_service

    store = _read_store(settings)
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
    stale = int(stats.get("stale_vectors") or 0)
    if stale:
        console.print(
            f"stale:       [yellow]{stale}[/yellow] "
            "[dim](built with a different embedding model; invisible to search)[/dim]"
        )
        console.print("[dim]Run [cyan]wukong memory reindex[/cyan] to rebuild them.[/dim]")
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
    """给还没有向量的记忆补上当前嵌入模型的向量。"""
    settings = _settings(home, workspace)
    from wukong.memory import build_memory_service
    from wukong.storage.store import Store

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
    """查看一条记忆替换了什么，以及后来又被什么替换。

    This is what invalidating instead of overwriting buys: the store can
    still answer "what did this used to be".
    """
    settings = _settings(home, workspace)
    from wukong.memory import build_memory_service

    store = _read_store(settings)
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
    """手动添加一条记忆。"""
    settings = _settings(home, None)
    settings.ensure_dirs()
    from wukong.storage.store import Store

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
    """删除一条记忆。"""
    settings = _settings(home, None)
    from wukong.storage.store import Store

    store = Store(settings.db_path)
    try:
        ok = store.delete_memory(memory_id)
    finally:
        store.close()
    console.print("deleted" if ok else "no such memory")


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


_SKILL_STATUS_STYLE = {
    "active": "green",
    "validated": "cyan",
    "approved": "cyan",
    "candidate": "yellow",
    "deprecated": "dim",
}


def _skill_registry(settings: Any, *, write: bool = False):
    """A registry wired to the stored ladder, without building a whole agent.

    `build_agent` would connect MCP servers and load models to answer a
    question about a directory listing. What this needs from the store is
    only the status column.

    `write=True` only for `promote`: everything else about a skill listing is
    a read, and a read that opens a writable connection is a read whose bug
    can corrupt the store.
    """
    from wukong.agent.factory import candidate_skill_dirs, store_skills
    from wukong.skills.registry import SkillRegistry
    from wukong.storage.store import Store

    store = Store(settings.db_path) if write else Store.readonly(settings.db_path)
    registry = SkillRegistry(
        default_skill_dirs(settings),
        candidate_dirs=candidate_skill_dirs(settings),
        status_overrides=store.skill_statuses(),
        on_promote=lambda skill: store_skills(store, [skill]),
    )
    return registry, store


@skill_app.command("list")
def skill_list(
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """列出已发现的技能，包括等待审核的那些。"""
    settings = _settings(home, workspace)
    registry, store = _skill_registry(settings)
    try:
        result = registry.discover()
        counts = store.skill_run_counts()
    finally:
        store.close()

    roots = registry.dirs + registry.candidate_dirs
    if not result.loaded:
        console.print(f"[dim]no skills in {', '.join(str(d) for d in roots)}[/dim]")
    if result.loaded:
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("skill")
        table.add_column("status")
        table.add_column("source", style="dim")
        table.add_column("runs", justify="right", style="dim")
        for skill in result.loaded:
            runs = counts.get(skill.name) or {}
            total = sum(runs.values())
            run_text = f"{total} ({runs.get('completed', 0)} ok)" if total else "-"
            style = _SKILL_STATUS_STYLE.get(skill.status, "")
            table.add_row(
                skill.name,
                f"[{style}]{skill.status}[/{style}]" if style else skill.status,
                skill.source,
                run_text,
            )
        console.print(table)
    for path, problem in result.errors:
        err_console.print(f"{path}: {escape(problem)}")

    pending = registry.pending_review()
    if pending:
        console.print()
        console.print(
            "[yellow]"
            + escape(f"{len(pending)} skill(s) awaiting review: ")
            + "[/yellow]"
            + escape(", ".join(f"{s.name} ({s.status})" for s in pending))
        )
        console.print(
            "[dim]`wukong skill review <name>` to see what is missing, then "
            "`wukong skill promote <name> <status>`.[/dim]"
        )


@skill_app.command("promote")
def skill_promote(
    name: str,
    to: str = typer.Argument(..., help="One rung up: validated, approved, active, deprecated."),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """把技能沿 candidate -> validated -> approved -> active 往上推一级。

    Skipping rungs is refused: each one is a separate judgement (does it
    parse / is it correct / does a human accept it / may it run). `deprecated`
    is reachable from anywhere.
    """
    settings = _settings(home, workspace)
    registry, store = _skill_registry(settings, write=True)
    try:
        registry.discover()
        try:
            skill = registry.promote(name, to)
        except SkillError as exc:
            err_console.print(escape(str(exc)))
            raise typer.Exit(1) from exc
    finally:
        store.close()

    console.print(f"{skill.name}: [bold]{skill.status}[/bold]")
    if skill.status == "active":
        console.print("[dim]it will appear in the skill index on the next run[/dim]")
    else:
        from wukong.skills.registry import STATUS_ORDER

        nxt = STATUS_ORDER[STATUS_ORDER.index(skill.status) + 1]
        console.print(f"[dim]next rung: {nxt}[/dim]")


@skill_app.command("review")
def skill_review(
    name: str,
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """说明一个技能要能跑，还差什么。"""
    settings = _settings(home, workspace)
    registry, store = _skill_registry(settings)
    try:
        result = registry.discover()
        skill = registry.get(name)
        runs = store.list_skill_runs(skill_name=name)
    finally:
        store.close()

    if skill is None:
        err_console.print(f"unknown skill {name!r}")
        raise typer.Exit(1)

    report = next((r for r in result.reports if r.skill == name), None)
    console.print(f"[bold]{skill.name}[/bold]  {skill.status}  [dim]{skill.path}[/dim]")
    console.print()
    console.print(escape(skill.description), markup=False)
    console.print()
    console.print(f"[dim]allowed tools: {', '.join(sorted(skill.allowed_tools)) or '(none)'}[/dim]")
    console.print(f"[dim]sha256: {skill.sha256}[/dim]")
    console.print(f"[dim]runs recorded: {len(runs)}[/dim]")
    if report is not None:
        console.print()
        console.print(report.render(), markup=False)

    checklist = _review_checklist(skill.body)
    if checklist:
        console.print()
        console.print(checklist, markup=False)

    if skill.status != "active":
        console.print()
        console.print(
            "[yellow]not runnable[/yellow] [dim]-- promote it to `active` once the "
            "checklist above is answered.[/dim]"
        )


def _review_checklist(body: str) -> str:
    """Pull the `## Review checklist` section out of a generated candidate.

    The reflector writes one when it invents a skill; surfacing it here is
    the difference between "a human gate" and "a directory nobody opens".
    """
    marker = "## Review checklist"
    if marker not in body:
        return ""
    section = body.split(marker, 1)[1]
    lines = [ln for ln in section.splitlines() if ln.strip()]
    return f"{marker}\n" + "\n".join(lines[:12])


@skill_app.command("runs")
def skill_runs(
    name: Optional[str] = typer.Argument(None, help="Limit to one skill."),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """查看哪些任务加载过某个技能，以及结果如何。

    This is the only evidence behind a `deprecate` decision: a skill that is
    loaded often and never finishes a task is a skill to retire.
    """
    settings = _settings(home, workspace)

    store = _read_store(settings)
    try:
        rows = store.list_skill_runs(skill_name=name)
    finally:
        store.close()

    if not rows:
        console.print("[dim]no skill runs recorded yet[/dim]")
        return
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("skill")
    table.add_column("outcome")
    table.add_column("task", style="dim")
    table.add_column("when", style="dim")
    for row in rows:
        style = "green" if row["outcome"] == "completed" else "red"
        table.add_row(
            row["skill_name"],
            f"[{style}]{row['outcome']}[/{style}]",
            row["task_id"] or "-",
            (row["created_at"] or "")[:19],
        )
    console.print(table)


@skill_app.command("validate")
def skill_validate(
    path: Path = typer.Argument(..., help="A skill directory or a SKILL.md path."),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """按 Agent Skills 规范校验技能，并跑一遍安全审查。"""
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
    """打印一个技能的完整说明。"""
    settings = _settings(home, workspace)
    registry, store = _skill_registry(settings)
    try:
        registry.discover()
        skill = registry.get(name)
    finally:
        store.close()

    if skill is None:
        err_console.print(f"unknown skill {name!r}")
        raise typer.Exit(1)
    # Skill bodies are Markdown with [links](url) -- markup=False keeps them.
    console.print(f"[dim]status: {skill.status}[/dim]")
    console.print(skill.render(), markup=False)


# ---------------------------------------------------------------------------
# a2a
# ---------------------------------------------------------------------------


@a2a_app.command("card")
def a2a_card(
    url: Optional[str] = typer.Option(
        None, "--url", help="The address peers should use. Defaults to a local placeholder."
    ),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """打印这个 agent 会发布的 Agent Card。"""
    settings = _settings(home, workspace)
    settings.a2a.enabled = True  # building the card is not the same as serving it

    async def _run() -> None:
        from wukong.a2a import A2AServer, describe_capability_gap

        agent = await build_agent(settings=settings, connect_mcp=False)
        try:
            server = A2AServer(agent)
            card = server.card(url=(url or settings.a2a.public_url or "http://127.0.0.1:8765") + "/a2a")
            console.print_json(card.model_dump_json())
            for gap in describe_capability_gap(card):
                err_console.print(f"[yellow]card claims something this build does not do: {gap}[/yellow]")
        finally:
            agent.close()

    asyncio.run(_run())


@a2a_app.command("check")
def a2a_check(
    url: str = typer.Argument(..., help="Base URL of the remote agent."),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """取回并审查对端的 Agent Card，但不调用它。

    Worth doing before a call: the card is untrusted input, and this prints
    what the security review thought of it.
    """
    settings = _settings(home, workspace)

    async def _run() -> int:
        from wukong.a2a import A2AClient, A2AClientError

        client = A2AClient(settings)
        try:
            card, report = await client.fetch_card(url)
        except (A2AClientError, Exception) as exc:  # noqa: BLE001
            err_console.print(escape(str(exc)))
            return 1
        console.print(f"[bold]{card.name}[/bold]  [dim]{card.url}[/dim]")
        console.print(escape(card.description), markup=False)
        console.print(f"[dim]protocol {card.protocolVersion} · {len(card.skills)} skill(s)[/dim]")
        for skill in card.skills:
            console.print(f"  [bold]{skill.id}[/bold] [dim]{escape(skill.description[:100])}[/dim]")
        console.print(report.render(), markup=False)
        return 0 if report.ok else 1

    raise typer.Exit(asyncio.run(_run()))


@a2a_app.command("call")
def a2a_call(
    url: str = typer.Argument(..., help="Base URL of the remote agent."),
    message: str = typer.Argument(..., help="What to ask it."),
    context: Optional[str] = typer.Option(None, "--context", help="Group related calls."),
    as_json: bool = typer.Option(False, "--json"),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """调用一个远端 agent。

    The host must be on `a2a.allow_hosts`; an empty list means this agent
    calls nobody, which is the default.
    """
    settings = _settings(home, workspace)

    async def _run() -> int:
        from wukong.a2a import A2AClient, A2AClientError

        client = A2AClient(settings)
        try:
            call = await client.send(url, message, context_id=context)
        except A2AClientError as exc:
            err_console.print(escape(str(exc)))
            return 1
        if as_json:
            console.print_json(json.dumps(call.task, ensure_ascii=False))
        else:
            style = "green" if call.state.value == "completed" else "yellow"
            console.print(f"[{style}]{call.state.value}[/{style}]  [dim]task={call.task.get('id')}[/dim]")
            if call.answer:
                console.print(call.answer, markup=False)
            for problem in call.report.warnings:
                err_console.print(f"[yellow]card warning: {escape(problem)}[/yellow]")
        return 0 if call.state.value == "completed" else 1

    raise typer.Exit(asyncio.run(_run()))


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
    bisect: bool = typer.Option(
        False,
        "--bisect",
        help="Test which Seatbelt rule is refused, and whether the fallback works.",
    ),
    home: Optional[Path] = typer.Option(None, "--home"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """报告实际生效的进程隔离后端是哪个，并真的验证一次。

    Run this from a normal terminal. Anywhere else -- a container, or an
    environment that sandboxes its child processes -- macOS refuses to apply a
    narrowing Seatbelt profile, so the probe fails there even though it would
    succeed for you. `--report` writes the result to disk so it can be read
    back afterwards instead of copy-pasted.
    """
    import json

    from wukong.sandbox import (
        build_sandbox,
        builtin_profile_probe,
        environment_fingerprint,
        seatbelt_probe,
    )

    settings = _settings(home, workspace)
    settings.ensure_dirs()

    probe = seatbelt_probe()
    builtin = builtin_profile_probe()
    environment = environment_fingerprint()
    selection = build_sandbox(
        backend,
        home=settings.home,
        extra_write_dirs=settings.sandbox.extra_write_dirs,
        docker_image=settings.sandbox.docker_image,
        docker_network=settings.sandbox.docker_network,
        docker_mounts=settings.sandbox.docker_mounts,
    )

    if bisect:
        _run_bisect(settings, report_to=report_to, as_json=as_json)
        return

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
        # Both of these are needed to read the probe correctly: one says
        # whether the run was clean, the other whether the generated profile
        # was the problem or any restrictive profile is refused here.
        "environment": environment,
        "builtin_profile": {"ok": builtin.ok, "detail": builtin.detail},
        "requested": selection.requested,
        "selected": selection.sandbox.name,
        "isolation": selection.sandbox.isolation,
        "fell_back": selection.fell_back,
        "notes": selection.notes,
        "caveats": selection.sandbox.caveats(),
        "live_check": live,
        "verdict": _sandbox_verdict(selection, probe, live, environment, builtin),
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

    console.print("[bold]environment[/bold]")
    if environment.get("inside_parent_sandbox"):
        console.print(
            "  [yellow]this run is inside a parent sandbox[/yellow] "
            f"({len(environment.get('sandbox_markers', []))} marker(s))"
        )
        console.print(
            "  [dim]so a failing probe here does NOT mean Seatbelt is broken on "
            "your machine[/dim]"
        )
    else:
        console.print("  [green]clean[/green] (no sandbox markers in the environment)")
    if environment.get("term_program"):
        console.print(f"  terminal: {escape(environment['term_program'])}")
    console.print()
    console.print("[bold]probe[/bold]")
    console.print(f"  seatbelt: {'[green]ok[/green]' if probe.ok else '[red]unavailable[/red]'}")
    console.print(f"    {escape(probe.detail)}")
    console.print(
        f"  system profile (-n no-network): "
        f"{'[green]ok[/green]' if builtin.ok else '[red]refused[/red]'}"
    )
    console.print(f"    {escape(builtin.detail)}")
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

    if not probe.ok and sys.platform == "darwin" and environment.get("inside_parent_sandbox"):
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
            f"{_sh(sys.executable)} -m wukong.cli sandbox --report[/cyan]"
        )


def _run_bisect(settings: Any, *, report_to: Any = None, as_json: bool = False) -> None:
    """跑 profile 逐条排查，并打印结果表。

    Exists because the failure reproduces on the user's machine and not on
    ours: the experiment has to be shipped rather than run. It tests the
    candidate *fix* in the same pass, so isolating the cause does not cost a
    second round trip.
    """
    import json

    from wukong.sandbox import diagnose

    outcome = diagnose(Path(settings.workspace), Path(settings.home))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": sys.platform,
        "workspace": str(settings.workspace),
        **outcome,
    }

    target = report_to or (settings.home / "sandbox-bisect.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if as_json:
        console.print_json(json.dumps(payload, ensure_ascii=False))
        return

    console.print("[bold]profile bisect[/bold]")
    for row in payload["results"]:
        mark = "[green]applied[/green]" if row["ok"] else "[red]refused[/red]"
        console.print(f"  {mark}  {escape(row['key'])}")
        console.print(f"    [dim]{escape(row['question'])}[/dim]")
        if not row["ok"]:
            console.print(f"    {escape(row['detail'][:150])}")
        for live in row["live"]:
            live_mark = "[green]ok[/green]" if live["ok"] else "[red]WRONG[/red]"
            console.print(
                f"    live: {escape(live['label'])} -> {live['result']} "
                f"(expected {live['expected']}) {live_mark}"
            )
    console.print()
    console.print(f"[bold]conclusion[/bold]  {escape(payload['conclusion'])}")
    console.print(f"[dim]written to {escape(str(target))}[/dim]")


def _sh(value: Any) -> str:
    """Shell-quote a path for a copy-pasteable command."""
    import shlex

    return shlex.quote(str(value))


def _sandbox_verdict(
    selection: Any,
    probe: Any,
    live: list[dict[str, Any]],
    environment: dict[str, Any] | None = None,
    builtin: Any = None,
) -> str:
    """One sentence a reader can act on.

    The environment is part of the verdict, not a footnote. "Seatbelt does not
    work on this machine" and "Seatbelt cannot be tested from here" lead to
    opposite decisions, and a probe result alone cannot tell them apart.
    """
    environment = environment or {}
    if selection.sandbox.name == "none":
        if probe.ok:
            return "seatbelt works here but a different backend was requested"
        if environment.get("inside_parent_sandbox"):
            return (
                "NO ISOLATION ACTIVE -- and this run was itself inside a sandbox "
                f"({', '.join(environment.get('sandbox_markers', [])[:2])}...), so the probe "
                "result says nothing about this machine. Re-run from Terminal.app."
            )
        if builtin is not None and not builtin.ok:
            return (
                "NO ISOLATION ACTIVE. This machine refuses to apply any restrictive "
                "Seatbelt profile, including the system's own -- so Seatbelt is not a "
                "usable backend here. The path fence and command guard still apply."
            )
        return "NO ISOLATION ACTIVE. Run `wukong sandbox` again from a normal terminal."
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

    from wukong.tools.base import ToolContext
    from wukong.tools.shell import RunCommandTool

    async def _run() -> list[dict[str, Any]]:
        outside = Path(tempfile.mkdtemp(prefix="wukong-sbx-"))
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
    """启动 HTTP/WebSocket 服务和 Web 控制台。

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

    from wukong.api.app import create_app
    from wukong.sandbox import build_sandbox

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
            f"[cyan]X-WUKONG-Token: {session_token}[/cyan]"
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
    """打开桌面窗口，或打包成可双击的应用。

    The window is the operating system's own webview around the same server
    and the same console the browser uses, so there is no second UI to keep
    in sync. `--bundle` produces a macOS `.app`; `--check` exercises the
    server lifecycle without a display, which is what CI can verify.
    """
    import secrets as _secrets

    from wukong.desktop import (
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
                "\nInstall the desktop extra, or use [cyan]wukong serve[/cyan] and open "
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
    from wukong.orchestration import WorkflowError, discover, load_workflow

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
    from wukong.orchestration import WorkflowError, load_workflow

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
    from wukong.orchestration import WorkflowError, load_workflow

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
    """Run a workflow. A run is a task, so `wukong task events <id>` shows it."""
    from wukong.orchestration import (
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
    """用离线 mock 模型跑一遍端到端，不需要 API key。"""
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
                f"\n[dim]inspect with: wukong task show {result.task_id} / "
                f"wukong task events {result.task_id}[/dim]"
            )
            return 0
        finally:
            agent.close()

    raise typer.Exit(asyncio.run(_run()))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
