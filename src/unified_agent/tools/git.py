"""Git tools.

Worth having as first-class tools rather than telling the model to
`run_command("git diff")`, for one concrete reason: **effect class**.
Inspecting a repository is `READ_ONLY`, so it runs without a confirmation
prompt. Routing it through `run_command` makes it `EXECUTE_LOCAL` and puts
a y/n prompt in front of every `git status`, which trains the user to
approve without reading -- the worst possible outcome for a permission
system.

Committing is the opposite: `EXTERNAL_SIDE_EFFECT`, confirmation required,
not idempotent.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from unified_agent.tools.base import Tool, ToolContext, ToolSpec
from unified_agent.sandbox.base import Sandbox, SandboxMode
from unified_agent.tools.permissions import scrub_env
from unified_agent.types import EffectClass, ToolResult

_MAX_OUTPUT = 100_000


async def _git(
    argv: list[str], *, cwd: str, env: dict[str, str], timeout: float = 60.0
) -> tuple[int, str]:
    """`argv` is the full command line -- already sandbox-wrapped by the caller."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"{' '.join(argv)} timed out after {timeout:.0f}s"
    text = stdout.decode("utf-8", errors="replace")
    if len(text) > _MAX_OUTPUT:
        text = text[:_MAX_OUTPUT] + f"\n[... output cut at {_MAX_OUTPUT} chars]"
    return proc.returncode or 0, text


class _GitTool(Tool):
    def __init__(
        self,
        *,
        known_secrets: list[str] | None = None,
        sandbox: Sandbox | None = None,
        sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE,
    ) -> None:
        self.known_secrets = known_secrets or []
        self.sandbox = sandbox
        self.sandbox_mode = sandbox_mode

    def _env(self, ctx: ToolContext) -> dict[str, str]:
        env = scrub_env(ctx.env or None, known_secrets=self.known_secrets)
        # Keep git non-interactive: a credential prompt would hang the task
        # until the timeout, which looks like a bug rather than an auth issue.
        env.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_PAGER": "cat",
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
        return env

    async def _run(self, args: list[str], ctx: ToolContext, timeout: float = 60.0) -> ToolResult:
        if ctx.dry_run:
            return self.dry_run({"args": args}, ctx)
        env = self._env(ctx)
        argv = ["git", *args]
        if self.sandbox is not None:
            argv = self.sandbox.wrap(
                argv, workspace=ctx.workspace, mode=self.sandbox_mode, env=env
            )
        code, text = await _git(argv, cwd=str(ctx.workspace), env=env, timeout=timeout)
        if code != 0 and "not a git repository" in text:
            return ToolResult(
                success=False,
                error=f"{ctx.workspace} is not a git repository",
                output=text,
            )
        return ToolResult(
            success=code == 0,
            output=text.strip() or "(no output)",
            error=None if code == 0 else f"git exited {code}",
            metadata={"argv": ["git", *args], "exit_code": code},
        )


class GitStatusTool(_GitTool):
    spec = ToolSpec(
        name="git_status",
        description=(
            "Show the working tree status in short format, plus the current branch. "
            "Read-only: no confirmation needed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "include_branch": {"type": "boolean", "description": "Default true."}
            },
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
        tags=["git"],
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        status = await self._run(["status", "--short", "--branch"], ctx)
        if not status.success:
            return status
        return status


class GitDiffTool(_GitTool):
    spec = ToolSpec(
        name="git_diff",
        description=(
            "Show changes. By default the unstaged diff against the index; set "
            "staged=true for the index, or ref='HEAD~1' to compare against a "
            "revision. Read-only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Limit to this path."},
                "staged": {"type": "boolean"},
                "ref": {"type": "string", "description": "A revision, branch or commit."},
                "stat_only": {"type": "boolean", "description": "Summary instead of the full diff."},
                "max_lines": {"type": "integer", "minimum": 20, "maximum": 5000},
            },
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
        tags=["git"],
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        argv = ["diff"]
        if args.get("staged"):
            argv.append("--staged")
        if args.get("stat_only"):
            argv.append("--stat")
        if ref := args.get("ref"):
            argv.append(str(ref))
        argv.append("--")
        if path := args.get("path"):
            argv.append(str(path))

        result = await self._run(argv, ctx)
        if not result.success or "--stat" in argv:
            return result

        max_lines = int(args.get("max_lines") or 1200)
        lines = result.output.splitlines()
        if len(lines) > max_lines:
            result.output = "\n".join(lines[:max_lines]) + (
                f"\n[... {len(lines) - max_lines} more diff lines; "
                "narrow with path= or raise max_lines]"
            )
            result.truncated = True
        if not result.output.strip():
            result.output = "(no changes)"
        return result


class GitLogTool(_GitTool):
    spec = ToolSpec(
        name="git_log",
        description="Recent commit history, one line per commit. Read-only.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "path": {"type": "string"},
            },
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
        tags=["git"],
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        limit = int(args.get("limit") or 20)
        argv = ["log", f"-{limit}", "--pretty=format:%h %ad %an %s", "--date=short"]
        if path := args.get("path"):
            argv += ["--", str(path)]
        return await self._run(argv, ctx)


class GitCommitTool(_GitTool):
    spec = ToolSpec(
        name="git_commit",
        description=(
            "Stage and commit. Side-effecting and NOT idempotent: it needs "
            "explicit approval every time. Prefer letting the user commit."
        ),
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "minLength": 4, "maxLength": 2000},
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to stage. Omit to commit the index as-is.",
                },
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.EXTERNAL_SIDE_EFFECT,
        requires_confirmation=True,
        idempotent=False,
        tags=["git"],
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        if paths := args.get("paths"):
            staged = await self._run(["add", "--", *[str(p) for p in paths]], ctx)
            if not staged.success:
                return staged
        return await self._run(["commit", "-m", str(args["message"])], ctx, timeout=120)


def build_git_tools(
    *,
    known_secrets: list[str] | None = None,
    sandbox: Sandbox | None = None,
    sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE,
) -> list[Tool]:
    if not _git_available():
        return []
    kwargs = {"known_secrets": known_secrets, "sandbox": sandbox, "sandbox_mode": sandbox_mode}
    return [
        GitStatusTool(**kwargs),
        GitDiffTool(**kwargs),
        GitLogTool(**kwargs),
        GitCommitTool(**kwargs),
    ]


def _git_available() -> bool:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory and os.path.exists(os.path.join(directory, "git")):
            return True
    return os.path.exists("/usr/bin/git")
