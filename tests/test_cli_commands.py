"""Every command a user can type, run once.

Coverage had `cli.py` at 19%: the *rendering* was tested, the commands were
not. The CLI is the thinnest layer in the project and the one a user meets
first, so "does it run at all" is worth asserting even where the logic
underneath is covered elsewhere -- an import that moved, a renamed option or
a `Settings` field that no longer exists all fail here and nowhere else.

Every command runs against a throwaway `WUKONG_HOME`, so nothing here can
touch the real config or database.
"""

from __future__ import annotations

import pytest

pytest.importorskip("typer", reason="typer is a core dependency")

from typer.testing import CliRunner  # noqa: E402

from wukong.cli import _skill_verdict, app  # noqa: E402


@pytest.fixture
def cli(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
    """A runner pointed at a throwaway home, in a throwaway directory."""
    monkeypatch.setenv("WUKONG_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    return CliRunner()


def invoke(cli, *args: str):  # noqa: ANN001, ANN201
    return cli.invoke(app, list(args))


class TestEveryCommandRuns:
    """One assertion per command, and the assertion is "it did not crash".

    A CLI that raises a traceback on `wukong task list` against an empty
    database is broken in the way users actually meet, and no unit test of
    `Store.list_tasks` would notice."""

    def test_help_lists_the_commands(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "--help")
        assert result.exit_code == 0
        assert "run" in result.stdout

    def test_version(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "version")
        assert result.exit_code == 0
        assert result.stdout.strip()

    def test_init_creates_a_config(self, cli, tmp_path) -> None:  # noqa: ANN001
        result = invoke(cli, "init")
        assert result.exit_code == 0, result.output
        assert (tmp_path / "home" / "config.toml").exists()

    def test_doctor_reports_without_crashing(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "doctor")
        assert result.exit_code == 0, result.output

    def test_tools_lists_the_registry(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "tools")
        assert result.exit_code == 0, result.output
        assert "read_file" in result.stdout

    def test_config_show_on_a_fresh_home(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "config", "show")
        assert result.exit_code == 0, result.output

    def test_config_set_then_show_round_trips(self, cli) -> None:  # noqa: ANN001
        invoke(cli, "init")
        set_result = invoke(cli, "config", "set", "agent.max_steps", "7")
        assert set_result.exit_code == 0, set_result.output

        shown = invoke(cli, "config", "show")
        assert "7" in shown.stdout

    def test_task_list_on_an_empty_database(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "task", "list")
        assert result.exit_code == 0, result.output

    def test_skill_list(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "skill", "list")
        assert result.exit_code == 0, result.output

    def test_memory_search_on_an_empty_database(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "memory", "search", "anything")
        assert result.exit_code == 0, result.output

    def test_sandbox_reports(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "sandbox")
        assert result.exit_code == 0, result.output

    def test_workflow_list(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "workflow", "list")
        assert result.exit_code == 0, result.output

    def test_stats_on_an_empty_database(self, cli) -> None:  # noqa: ANN001
        """A fresh install runs this before it runs anything else."""
        result = invoke(cli, "stats")
        assert result.exit_code == 0, result.output
        assert "还没有任务" in result.stdout


class TestOfflineRun:
    """`run --model mock` is the one path that needs no key, no network and no
    cost -- which makes it the one path CI can actually exercise end to end."""

    def test_a_run_with_the_mock_model_completes(self, cli, tmp_path) -> None:  # noqa: ANN001
        (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        invoke(cli, "init")

        result = invoke(cli, "run", "--model", "mock", "列出当前目录的文件")

        assert result.exit_code == 0, result.output
        # The task has to be findable afterwards -- a run that reports success
        # but leaves no record is the failure this catches.
        #
        # Asserted on a short prefix: Rich wraps the goal column and draws a
        # column border inside the wrap, so the full string never appears
        # contiguously in the rendered table.
        listed = invoke(cli, "task", "list")
        assert "列出当前" in listed.stdout
        assert "completed" in listed.stdout


class TestSkillVerdict:
    """The judgement layer over the run counts.

    Worth testing on its own because it is the part that *refuses* to
    conclude: a verdict drawn from two runs is how a useful skill gets retired
    for bad luck, and a verdict drawn from zero runs is how an unused one stays
    `active` forever looking fine.
    """

    def test_an_unrun_active_skill_is_called_out(self) -> None:
        assert "从没跑过" in _skill_verdict(status="active", runs=0, ok=0)

    def test_a_small_sample_gets_no_verdict(self) -> None:
        assert "样本太少" in _skill_verdict(status="active", runs=2, ok=0)

    def test_a_low_success_rate_is_flagged(self) -> None:
        assert "该看它的说明了" in _skill_verdict(status="active", runs=10, ok=3)

    def test_a_middling_rate_is_called_middling(self) -> None:
        assert "时好时坏" in _skill_verdict(status="active", runs=10, ok=6)

    def test_a_high_success_rate_is_left_alone(self) -> None:
        assert "表现稳定" in _skill_verdict(status="active", runs=10, ok=9)

    def test_a_skill_still_on_the_ladder_is_not_judged_by_runs(self) -> None:
        """It has not been approved yet, so its run count is not the question.
        Saying "never run" about a candidate would be true and useless."""
        verdict = _skill_verdict(status="candidate", runs=0, ok=0)

        assert "还在阶梯上" in verdict
        assert "从没跑过" not in verdict


class TestSkillHealthCommand:
    def test_it_runs_on_an_empty_database(self, cli) -> None:  # noqa: ANN001
        result = invoke(cli, "skill", "health")
        assert result.exit_code == 0, result.output
