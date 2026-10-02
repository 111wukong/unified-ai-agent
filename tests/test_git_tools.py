"""Git tools: read-only inspection must not require approval."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from wukong.tools.base import ToolContext
from wukong.tools.git import (
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    build_git_tools,
)
from wukong.tools.permissions import PermissionEngine
from wukong.tools.registry import ToolRegistry
from wukong.types import EffectClass


@pytest.fixture
def repo(tmp_path):  # noqa: ANN001
    root = tmp_path / "repo"
    root.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
    }
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.com"],
        ["config", "user.name", "t"],
    ):
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, env=env, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"], cwd=root, env=env, check=True, capture_output=True
    )
    (root / "a.py").write_text("x = 2\n", encoding="utf-8")
    return root


@pytest.fixture
def ctx(repo, tmp_path):  # noqa: ANN001
    return ToolContext(
        task_id="t",
        session_id="s",
        step_id="step_1",
        workspace=repo,
        home=tmp_path,
        artifact_dir=tmp_path / "artifacts",
    )


class TestReadOnlyTools:
    async def test_status_reports_the_dirty_file(self, ctx: ToolContext) -> None:
        result = await GitStatusTool().run({}, ctx)
        assert result.success
        assert "a.py" in result.output

    async def test_diff_shows_the_change(self, ctx: ToolContext) -> None:
        result = await GitDiffTool().run({}, ctx)
        assert result.success
        assert "-x = 1" in result.output
        assert "+x = 2" in result.output

    async def test_diff_stat_only(self, ctx: ToolContext) -> None:
        result = await GitDiffTool().run({"stat_only": True}, ctx)
        assert result.success
        assert "1 file changed" in result.output

    async def test_diff_against_a_revision(self, ctx: ToolContext) -> None:
        result = await GitDiffTool().run({"ref": "HEAD"}, ctx)
        assert result.success
        assert "+x = 2" in result.output

    async def test_diff_max_lines(self, ctx: ToolContext) -> None:
        result = await GitDiffTool().run({"max_lines": 20}, ctx)
        assert result.success

    async def test_log_lists_commits(self, ctx: ToolContext) -> None:
        result = await GitLogTool().run({"limit": 5}, ctx)
        assert result.success
        assert "initial" in result.output

    async def test_not_a_repository_is_reported_clearly(self, tmp_path) -> None:  # noqa: ANN001
        """The directory must be outside *any* repo, including the project's.

        Using the `workspace` fixture here is wrong when the test runner's
        temp directory lives inside a checkout: `git status` walks up, finds
        the project's own `.git`, and succeeds. The test then asserts the
        opposite of what it is actually exercising.
        """
        import tempfile

        outside = Path(tempfile.mkdtemp(prefix="wukong-not-a-repo-"))
        plain = ToolContext(
            task_id="t",
            session_id="s",
            step_id="step_1",
            workspace=outside,
            home=tmp_path,
            artifact_dir=tmp_path,
        )
        result = await GitStatusTool().run({}, plain)
        assert not result.success
        assert "not a git repository" in (result.error or "")


class TestEffectClasses:
    def test_inspection_is_read_only_so_it_needs_no_approval(
        self, settings, tmp_path, repo
    ) -> None:  # noqa: ANN001
        """The whole reason these are tools and not `run_command('git diff')`."""
        engine = PermissionEngine(
            settings.permissions, workspace=repo, home=tmp_path / "home"
        )
        registry = ToolRegistry(build_git_tools())
        for name in ("git_status", "git_diff", "git_log"):
            verdict = engine.decide(registry.get(name), {})
            assert verdict.allowed, f"{name} must not require a confirmation prompt"

    def test_commit_requires_confirmation_and_is_not_idempotent(
        self, settings, tmp_path, repo
    ) -> None:  # noqa: ANN001
        engine = PermissionEngine(
            settings.permissions, workspace=repo, home=tmp_path / "home"
        )
        tool = GitCommitTool()
        verdict = engine.decide(tool, {"message": "a real commit message"})
        assert verdict.needs_confirmation
        assert tool.spec.idempotent is False

    async def test_commit_is_blocked_in_dry_run(self, ctx: ToolContext) -> None:
        ctx.dry_run = True
        result = await GitCommitTool().run({"message": "should not happen"}, ctx)
        assert result.success
        assert result.metadata.get("dry_run")

    async def test_build_git_tools_returns_all_four(self) -> None:
        names = {t.spec.name for t in build_git_tools()}
        assert names == {"git_status", "git_diff", "git_log", "git_commit"}

    def test_git_environment_is_non_interactive(self, ctx: ToolContext) -> None:
        """A credential prompt would hang until the timeout and look like a bug."""
        env = GitStatusTool()._env(ctx)
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GIT_PAGER"] == "cat"


class TestEffectConsistency:
    def test_read_only_tools_are_idempotent(self) -> None:
        for tool in build_git_tools():
            if tool.spec.effect_class is EffectClass.READ_ONLY:
                assert tool.spec.idempotent is True
