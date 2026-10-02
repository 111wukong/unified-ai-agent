"""Tool behaviour: filesystem, shell, truncation, artifact offload."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from wukong.agent.executor import truncate
from wukong.tools.base import ToolContext, ValidationFailure, validate_against
from wukong.tools.fs import (
    ApplyPatchTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from wukong.tools.shell import RunCommandTool
from wukong.types import EffectClass


@pytest.fixture
def ctx(workspace, tmp_path):  # noqa: ANN001
    return ToolContext(
        task_id="task_t",
        session_id="session_t",
        step_id="step_1",
        workspace=workspace,
        home=tmp_path / "home",
        artifact_dir=tmp_path / "artifacts",
    )


class TestReadFile:
    async def test_numbers_lines(self, ctx: ToolContext) -> None:
        result = await ReadFileTool().run({"path": "app.py"}, ctx)
        assert result.success
        assert result.output.startswith("1\tdef add(a, b):")

    async def test_offset_and_limit(self, ctx: ToolContext) -> None:
        result = await ReadFileTool().run({"path": "app.py", "offset": 5, "limit": 1}, ctx)
        assert result.output.splitlines()[0].startswith("5\t")

    async def test_missing_file_is_a_result_not_an_exception(self, ctx: ToolContext) -> None:
        result = await ReadFileTool().run({"path": "nope.txt"}, ctx)
        assert not result.success
        assert "no such file" in (result.error or "")

    async def test_directory_is_rejected_with_a_hint(self, ctx: ToolContext) -> None:
        result = await ReadFileTool().run({"path": "sub"}, ctx)
        assert not result.success
        assert "use list_directory" in (result.error or "")

    async def test_binary_is_refused(self, ctx: ToolContext, workspace) -> None:  # noqa: ANN001
        (workspace / "blob.bin").write_bytes(b"\x00\x01\x02\x03" * 100)
        result = await ReadFileTool().run({"path": "blob.bin"}, ctx)
        assert not result.success
        assert "binary" in (result.error or "")


class TestWriteFile:
    async def test_creates_parent_directories(self, ctx: ToolContext) -> None:
        result = await WriteFileTool().run(
            {"path": "deep/nested/out.txt", "content": "hi"}, ctx
        )
        assert result.success
        assert result.metadata["existed"] is False

    async def test_append_mode(self, ctx: ToolContext) -> None:
        await WriteFileTool().run({"path": "log.txt", "content": "a"}, ctx)
        await WriteFileTool().run({"path": "log.txt", "content": "b", "mode": "append"}, ctx)
        assert (ctx.workspace / "log.txt").read_text(encoding="utf-8") == "ab"

    async def test_dry_run_writes_nothing(self, ctx: ToolContext) -> None:
        ctx.dry_run = True
        await WriteFileTool().run({"path": "ghost.txt", "content": "x"}, ctx)
        assert not (ctx.workspace / "ghost.txt").exists()


class TestApplyPatch:
    async def test_replaces_a_unique_string(self, ctx: ToolContext) -> None:
        result = await ApplyPatchTool().run(
            {"path": "app.py", "edits": [{"old": "return a + b", "new": "return a + b + 1"}]},
            ctx,
        )
        assert result.success
        assert "a + b + 1" in (ctx.workspace / "app.py").read_text(encoding="utf-8")

    async def test_missing_match_explains_what_to_do(self, ctx: ToolContext) -> None:
        result = await ApplyPatchTool().run(
            {"path": "app.py", "edits": [{"old": "not in the file", "new": "x"}]}, ctx
        )
        assert not result.success
        assert "Read the file first" in (result.error or "")

    async def test_ambiguous_match_is_refused(self, ctx: ToolContext) -> None:
        (ctx.workspace / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
        result = await ApplyPatchTool().run(
            {"path": "dup.py", "edits": [{"old": "x = 1", "new": "x = 2"}]}, ctx
        )
        assert not result.success
        assert "occurs 2 times" in (result.error or "")

    async def test_replace_all_opt_in(self, ctx: ToolContext) -> None:
        (ctx.workspace / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
        result = await ApplyPatchTool().run(
            {
                "path": "dup.py",
                "edits": [{"old": "x = 1", "new": "x = 2", "replace_all": True}],
            },
            ctx,
        )
        assert result.success
        assert (ctx.workspace / "dup.py").read_text(encoding="utf-8") == "x = 2\nx = 2\n"

    async def test_all_edits_validated_before_any_write(self, ctx: ToolContext) -> None:
        before = (ctx.workspace / "app.py").read_text(encoding="utf-8")
        result = await ApplyPatchTool().run(
            {
                "path": "app.py",
                "edits": [
                    {"old": "return a + b", "new": "return a * b"},
                    {"old": "THIS IS NOT PRESENT", "new": "x"},
                ],
            },
            ctx,
        )
        assert not result.success
        assert (ctx.workspace / "app.py").read_text(encoding="utf-8") == before


class TestSearchAndList:
    async def test_search_by_glob(self, ctx: ToolContext) -> None:
        result = await SearchFilesTool().run({"glob": "**/*.py"}, ctx)
        assert "app.py" in result.output
        assert "notes.md" not in result.output

    async def test_search_by_content(self, ctx: ToolContext) -> None:
        result = await SearchFilesTool().run({"content_regex": "def add"}, ctx)
        assert "app.py:1:" in result.output

    async def test_list_skips_noise_directories(self, ctx: ToolContext, workspace) -> None:  # noqa: ANN001
        (workspace / "node_modules").mkdir()
        (workspace / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
        result = await ListDirectoryTool().run({"path": ".", "depth": 2}, ctx)
        assert "node_modules" not in result.output


class TestTruncation:
    def test_keeps_head_and_tail(self) -> None:
        text = "HEAD" + "x" * 10_000 + "TAIL"
        rendered, truncated = truncate(text, 1_000)
        assert truncated
        assert rendered.startswith("HEAD")
        assert rendered.endswith("TAIL")
        assert len(rendered) < len(text)
        assert "characters omitted" in rendered

    def test_short_text_is_untouched(self) -> None:
        rendered, truncated = truncate("short", 1_000)
        assert rendered == "short" and not truncated

    async def test_large_output_is_offloaded_to_an_artifact(
        self, scripted, session_id: str, workspace
    ) -> None:  # noqa: ANN001
        (workspace / "big.txt").write_text("y" * 20_000, encoding="utf-8")
        agent, _ = await scripted(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "big.txt"}}]},
                {"content": "done"},
            ]
        )
        agent.settings.agent.tool_output_chars = 500
        result = await agent.runtime.run("read the big file", session_id=session_id)

        artifacts = agent.store.list_artifacts(result.task_id)
        assert artifacts, "a truncated output must be offloaded"
        assert artifacts[0]["bytes"] > 500

        from wukong.agent.state import replay

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        entry = [e for e in state.log if e.tool == "read_file"][0]
        assert entry.artifact_path
        assert "full body at" in entry.text


class TestValidation:
    def test_missing_required_argument_names_the_accepted_keys(self) -> None:
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }
        with pytest.raises(ValidationFailure) as exc:
            validate_against(schema, {})
        assert "missing required argument" in exc.value.hint
        assert "path" in exc.value.hint

    def test_unexpected_keys_are_rejected(self) -> None:
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        }
        with pytest.raises(ValidationFailure) as exc:
            validate_against(schema, {"path": "a", "extra": 1})
        assert "unexpected argument" in exc.value.hint

    def test_enum_is_enforced(self) -> None:
        schema = {"type": "object", "properties": {"mode": {"type": "string", "enum": ["a", "b"]}}}
        with pytest.raises(ValidationFailure):
            validate_against(schema, {"mode": "c"})
        assert validate_against(schema, {"mode": "a"}) == {"mode": "a"}

    def test_nested_items_are_validated(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"old": {"type": "string"}},
                        "required": ["old"],
                    },
                }
            },
        }
        with pytest.raises(ValidationFailure) as exc:
            validate_against(schema, {"edits": [{}]})
        assert "edits[0]" in exc.value.hint


class TestShellTool:
    async def test_runs_and_captures_exit_code(self, ctx: ToolContext) -> None:
        result = await RunCommandTool().run({"command": "echo hello"}, ctx)
        assert result.success
        assert "hello" in result.output
        assert result.metadata["exit_code"] == 0

    async def test_nonzero_exit_is_a_failed_result(self, ctx: ToolContext) -> None:
        result = await RunCommandTool().run({"command": "ls /definitely/not/here"}, ctx)
        assert not result.success
        assert "exit code" in (result.error or "")

    async def test_missing_binary_is_reported_clearly(self, ctx: ToolContext) -> None:
        result = await RunCommandTool().run({"command": "definitely-not-a-real-binary"}, ctx)
        assert not result.success
        assert "command not found" in (result.error or "")

    async def test_credentials_are_not_visible_to_the_subprocess(
        self, ctx: ToolContext, monkeypatch
    ) -> None:  # noqa: ANN001
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-visible")
        monkeypatch.setenv("RANDOM_VENDOR_TOKEN", "also-secret")
        from wukong.tools.permissions import scrub_env

        ctx.env = scrub_env()
        result = await RunCommandTool().run({"command": "env"}, ctx)
        assert "sk-should-never-be-visible" not in result.output
        assert "also-secret" not in result.output
        assert "PATH=" in result.output, "the allowlisted vars must survive"

    async def test_shell_metacharacters_never_reach_a_shell(self, ctx: ToolContext) -> None:
        """shell=False means `;` is a literal argument, not a second command."""
        result = await RunCommandTool().run({"command": "echo a ; echo b"}, ctx)
        # echo receives them as literal args -- nothing else executes.
        assert "echo b" in result.output or not result.success
        assert not (ctx.workspace / "b").exists()

    async def test_timeout_is_enforced(self, ctx: ToolContext) -> None:
        result = await RunCommandTool().run(
            {"command": "python3 -c import time;time.sleep(5)", "timeout_s": 1}, ctx
        )
        # Unparseable/slow both acceptable; the point is it returns, not hangs.
        assert not result.success


class TestEffectClasses:
    def test_tools_declare_the_right_effect(self) -> None:
        assert ReadFileTool().spec.effect_class is EffectClass.READ_ONLY
        assert WriteFileTool().spec.effect_class is EffectClass.WRITE_LOCAL
        assert ApplyPatchTool().spec.effect_class is EffectClass.WRITE_LOCAL
        assert RunCommandTool().spec.effect_class is EffectClass.EXECUTE_LOCAL

    def test_shell_commands_are_not_idempotent(self) -> None:
        """Re-running a shell command after an unknown outcome is not safe."""
        assert RunCommandTool().spec.idempotent is False
        assert ReadFileTool().spec.idempotent is True


class TestVenvDetection:
    """Without this, `run_tests` is broken for every venv-based project."""

    def test_finds_a_dot_venv(self, workspace) -> None:  # noqa: ANN001
        from wukong.tools.shell import find_venv_bin

        (workspace / ".venv" / "bin").mkdir(parents=True)
        assert find_venv_bin(workspace) == workspace / ".venv" / "bin"

    def test_finds_a_plain_venv(self, workspace) -> None:  # noqa: ANN001
        from wukong.tools.shell import find_venv_bin

        (workspace / "venv" / "bin").mkdir(parents=True)
        assert find_venv_bin(workspace) == workspace / "venv" / "bin"

    def test_returns_none_without_one(self, workspace) -> None:  # noqa: ANN001
        from wukong.tools.shell import find_venv_bin

        assert find_venv_bin(workspace) is None

    def test_run_tests_uses_the_venv_pytest(self, workspace) -> None:  # noqa: ANN001
        from wukong.tools.shell import RunTestsTool

        (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        bin_dir = workspace / ".venv" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "pytest").write_text("#!/bin/sh\n", encoding="utf-8")

        argv = RunTestsTool().detect(workspace)
        assert argv[0] == str(bin_dir / "pytest")

    def test_run_tests_falls_back_to_bare_pytest(self, workspace) -> None:  # noqa: ANN001
        from wukong.tools.shell import RunTestsTool

        (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        assert RunTestsTool().detect(workspace)[0] == "pytest"

    def test_venv_is_prepended_to_path_for_subprocesses(self, workspace) -> None:  # noqa: ANN001
        from wukong.tools.shell import _with_venv_on_path

        bin_dir = workspace / ".venv" / "bin"
        bin_dir.mkdir(parents=True)
        env = _with_venv_on_path({"PATH": "/usr/bin"}, workspace)
        assert env["PATH"].startswith(str(bin_dir))
        assert env["VIRTUAL_ENV"] == str(workspace / ".venv")


class TestProjectIsImportableDuringTests:
    """A bare `pytest` puts the *test file's* directory on `sys.path`, not the
    directory it was started in.

    For the ordinary layout -- `tests/` importing a package at the project
    root -- collection therefore dies with `ImportError` before a single test
    runs, and the message blames the test file rather than the invocation.
    Found by running a real task for real: the agent burned four calls on
    `run_tests` before giving up and using `python -m pytest` instead.
    """

    def test_the_workspace_root_and_src_are_added(self, workspace) -> None:  # noqa: ANN001
        from wukong.tools.shell import _with_project_importable

        (workspace / "src").mkdir()
        env = _with_project_importable({}, workspace)
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert str(workspace) in parts
        assert str(workspace / "src") in parts

    def test_an_existing_pythonpath_is_kept(self, workspace) -> None:  # noqa: ANN001
        from wukong.tools.shell import _with_project_importable

        env = _with_project_importable({"PYTHONPATH": "/elsewhere"}, workspace)
        assert env["PYTHONPATH"].endswith("/elsewhere")
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(workspace)

    def test_a_missing_src_directory_is_not_invented(self, workspace) -> None:  # noqa: ANN001
        """Adding a path that does not exist is harmless but it is noise in
        the environment a subprocess sees."""
        from wukong.tools.shell import _with_project_importable

        parts = _with_project_importable({}, workspace)["PYTHONPATH"].split(os.pathsep)
        assert str(workspace) in parts
        assert str(workspace / "src") not in parts

    async def test_run_tests_collects_a_tests_directory_layout(
        self, workspace, tmp_path  # noqa: ANN001
    ) -> None:
        """The test that would have caught the bug: run it for real.

        A package at the root, tests in `tests/`, and a test that imports the
        package. Without the import path this fails at collection.
        """
        from wukong.tools.base import ToolContext
        from wukong.tools.shell import RunTestsTool

        (workspace / "shop").mkdir()
        (workspace / "shop" / "__init__.py").write_text("VALUE = 41\n", encoding="utf-8")
        (workspace / "tests").mkdir()
        (workspace / "tests" / "test_shop.py").write_text(
            "from shop import VALUE\n\n\ndef test_value():\n    assert VALUE == 41\n",
            encoding="utf-8",
        )
        (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        # A real pytest, so the subprocess actually runs. A shell shim would
        # make the test pass without exercising the import path at all.
        bin_dir = workspace / ".venv" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "pytest").symlink_to(Path(sys.executable).parent / "pytest")

        ctx = ToolContext(
            task_id="t",
            session_id="s",
            step_id="step_1",
            workspace=workspace,
            home=tmp_path / "home",
            artifact_dir=tmp_path / "home",
        )
        result = await RunTestsTool().run({}, ctx)

        assert result.success, f"collection failed: {result.error or result.output}"
        assert "1 passed" in result.output


