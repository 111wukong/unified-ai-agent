"""Shell tools.

`shell=False` always. The command string is parsed with `shlex.split` and
executed as argv, so a metacharacter can never become a second process —
the CommandGuard refusing metacharacters is then about giving the model an
honest error instead of silently running something different from what it
asked for.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from pathlib import Path
from typing import Any

from unified_agent.errors import ToolError
from unified_agent.sandbox.base import Sandbox, SandboxMode
from unified_agent.tools.base import Tool, ToolContext, ToolSpec
from unified_agent.tools.permissions import scrub_env
from unified_agent.types import EffectClass, ToolResult

# Fallback only. The effective ceiling is `permissions.shell.max_output_bytes`;
# this constant exists so the tool is still usable when constructed directly
# (tests, embedding) without a settings object.
DEFAULT_MAX_CAPTURE = 200_000

_VENV_DIRS = (".venv", "venv", "env")


def find_venv_bin(workspace: Path) -> Path | None:
    """Locate the project's own virtualenv.

    Without this, `run_tests` is broken for every Python project that uses
    a venv -- which is most of them. `pytest` is not on PATH; it is in
    `<project>/.venv/bin/pytest`, and the scrubbed subprocess environment
    deliberately does not inherit the parent's PATH additions.
    """
    for name in _VENV_DIRS:
        candidate = workspace / name / "bin"
        if candidate.is_dir():
            return candidate
    return None


def _with_venv_on_path(env: dict[str, str], workspace: Path) -> dict[str, str]:
    venv_bin = find_venv_bin(workspace)
    if venv_bin is None:
        return env
    env = dict(env)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("VIRTUAL_ENV", str(venv_bin.parent))
    return env


async def _exec(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
    max_capture: int = DEFAULT_MAX_CAPTURE,
) -> tuple[int, str, float]:
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(
            f"command timed out after {timeout_s:.0f}s: {' '.join(argv)}"
        ) from None
    elapsed = time.monotonic() - started
    text = stdout.decode("utf-8", errors="replace")
    if len(text) > max_capture:
        text = text[:max_capture] + f"\n[... output cut at {max_capture} chars]"
    return proc.returncode or 0, text, elapsed


class RunCommandTool(Tool):
    spec = ToolSpec(
        name="run_command",
        description=(
            "Run a single command. No shell: pipes, redirects and && are rejected. "
            "Run one command per call and use its output to decide the next one. "
            "Output is stdout and stderr merged."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "cwd": {"type": "string", "description": "Working directory. Default: workspace root."},
                "timeout_s": {"type": "number", "minimum": 1, "maximum": 900},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.EXECUTE_LOCAL,
        # A command's effect is not recoverable: re-running `git commit`
        # after an unknown outcome is not safe.
        idempotent=False,
    )

    def __init__(
        self,
        *,
        known_secrets: list[str] | None = None,
        default_timeout_s: float = 120.0,
        sandbox: Sandbox | None = None,
        sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE,
        max_output_bytes: int = DEFAULT_MAX_CAPTURE,
    ):
        self.known_secrets = known_secrets or []
        self.default_timeout_s = default_timeout_s
        self.sandbox = sandbox
        self.sandbox_mode = sandbox_mode
        self.max_output_bytes = max_output_bytes

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args["command"]
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return ToolResult(success=False, error=f"could not parse command: {exc}")
        if not argv:
            return ToolResult(success=False, error="empty command")
        cwd = Path(args["cwd"]) if args.get("cwd") else ctx.workspace
        if not cwd.is_absolute():
            cwd = ctx.workspace / cwd
        cwd = Path(os.path.realpath(cwd))
        if not cwd.is_dir():
            return ToolResult(success=False, error=f"cwd is not a directory: {cwd}")
        timeout_s = float(args.get("timeout_s") or self.default_timeout_s)

        if ctx.dry_run:
            return self.dry_run(args, ctx)

        env = scrub_env(
            ctx.env or None,
            known_secrets=self.known_secrets,
            overrides={"PYTHONUNBUFFERED": "1"},
        )
        env = _with_venv_on_path(env, ctx.workspace)
        if self.sandbox is not None:
            # The sandbox rewrites the argv, it does not replace the policy
            # engine: path fence and command guard already ran.
            argv = self.sandbox.wrap(
                argv, workspace=ctx.workspace, mode=self.sandbox_mode, env=env
            )
        try:
            code, output, elapsed = await _exec(
                argv,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                max_capture=self.max_output_bytes,
            )
        except ToolError as exc:
            return ToolResult(success=False, error=str(exc), metadata={"argv": argv})
        except FileNotFoundError as exc:
            return ToolResult(
                success=False,
                error=f"command not found: {argv[0]!r} ({exc})",
                metadata={"argv": argv},
            )

        header = f"$ {command}\n(exit {code}, {elapsed:.2f}s, cwd {cwd})"
        return ToolResult(
            success=code == 0,
            output=f"{header}\n{output}".rstrip(),
            error=None if code == 0 else f"exit code {code}",
            metadata={"exit_code": code, "duration_s": round(elapsed, 3), "argv": argv},
        )


class RunTestsTool(Tool):
    spec = ToolSpec(
        name="run_tests",
        description=(
            "Run the project's test command and return a compact summary. "
            "Detects pytest, npm test or go test from the workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Optional path or test id to run."},
                "cwd": {"type": "string"},
                "timeout_s": {"type": "number", "minimum": 1, "maximum": 900},
            },
            "additionalProperties": False,
        },
        effect_class=EffectClass.EXECUTE_LOCAL,
        idempotent=False,
    )

    def __init__(
        self,
        *,
        known_secrets: list[str] | None = None,
        sandbox: Sandbox | None = None,
        sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE,
        max_output_bytes: int = DEFAULT_MAX_CAPTURE,
    ):
        self._runner = RunCommandTool(
            known_secrets=known_secrets,
            sandbox=sandbox,
            sandbox_mode=sandbox_mode,
            max_output_bytes=max_output_bytes,
        )

    def detect(self, workspace: Path) -> list[str]:
        venv_bin = find_venv_bin(workspace)
        if venv_bin and (venv_bin / "pytest").exists():
            pytest_cmd = str(venv_bin / "pytest")
        else:
            pytest_cmd = "pytest"
        if (workspace / "pytest.ini").exists() or (workspace / "pyproject.toml").exists():
            return [pytest_cmd, "-q", "--no-header"]
        if (workspace / "package.json").exists():
            return ["npm", "test", "--silent"]
        if (workspace / "go.mod").exists():
            return ["go", "test", "./..."]
        return [pytest_cmd, "-q", "--no-header"]

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cwd = Path(args["cwd"]) if args.get("cwd") else ctx.workspace
        if not cwd.is_absolute():
            cwd = ctx.workspace / cwd
        argv = self.detect(cwd)
        if target := args.get("target"):
            argv.append(target)
        return await self._runner.run(
            {
                "command": " ".join(shlex.quote(a) for a in argv),
                "cwd": str(cwd),
                "timeout_s": args.get("timeout_s") or 300,
            },
            ctx,
        )


class RunLinterTool(Tool):
    spec = ToolSpec(
        name="run_linter",
        description="Run the detected linter (ruff, eslint or go vet) over the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "cwd": {"type": "string"},
            },
            "additionalProperties": False,
        },
        effect_class=EffectClass.EXECUTE_LOCAL,
        idempotent=False,
    )

    def __init__(
        self,
        *,
        known_secrets: list[str] | None = None,
        sandbox: Sandbox | None = None,
        sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE,
        max_output_bytes: int = DEFAULT_MAX_CAPTURE,
    ):
        self._runner = RunCommandTool(
            known_secrets=known_secrets,
            sandbox=sandbox,
            sandbox_mode=sandbox_mode,
            max_output_bytes=max_output_bytes,
        )

    def detect(self, workspace: Path) -> list[str]:
        venv_bin = find_venv_bin(workspace)
        if venv_bin and (venv_bin / "ruff").exists():
            if (workspace / "ruff.toml").exists() or (workspace / "pyproject.toml").exists():
                return [str(venv_bin / "ruff"), "check", "."]
        if (workspace / "ruff.toml").exists() or (workspace / "pyproject.toml").exists():
            return ["ruff", "check", "."]
        if (workspace / "package.json").exists():
            return ["npm", "run", "lint", "--silent"]
        if (workspace / "go.mod").exists():
            return ["go", "vet", "./..."]
        return ["ruff", "check", "."]

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cwd = Path(args["cwd"]) if args.get("cwd") else ctx.workspace
        if not cwd.is_absolute():
            cwd = ctx.workspace / cwd
        argv = self.detect(cwd)
        if target := args.get("target"):
            argv.append(target)
        return await self._runner.run(
            {"command": " ".join(shlex.quote(a) for a in argv), "cwd": str(cwd)}, ctx
        )


def build_shell_tools(
    *,
    known_secrets: list[str] | None = None,
    timeout_s: float = 120.0,
    sandbox: Sandbox | None = None,
    sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE,
    max_output_bytes: int = DEFAULT_MAX_CAPTURE,
) -> list[Tool]:
    kwargs = {
        "sandbox": sandbox,
        "sandbox_mode": sandbox_mode,
        "max_output_bytes": max_output_bytes,
    }
    return [
        RunCommandTool(
            known_secrets=known_secrets, default_timeout_s=timeout_s, **kwargs
        ),
        RunTestsTool(known_secrets=known_secrets, **kwargs),
        RunLinterTool(known_secrets=known_secrets, **kwargs),
    ]
