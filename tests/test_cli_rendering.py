"""CLI rendering: Rich must never eat the agent's own output.

This is a silent-failure class. `Console.print` interprets `[brackets]` as
style markup, so a regex like `\\brm\\s+(-[a-zA-Z]*\\s+)*` renders as
`\\brm\\s+(-*\\s+)*-*`, pytest's `[100%]` disappears, and a TOML `[models.x]`
header vanishes. Nothing errors, nothing logs -- the text is just wrong.

Every one of these was a real occurrence before the fix.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from unified_agent.agent.runtime import AgentResult
from unified_agent.agent.state import PendingConfirmation
from unified_agent import cli as cli_module


@pytest.fixture
def captured(monkeypatch):  # noqa: ANN001
    """Replace the CLI's console with a recording one."""
    buffer = io.StringIO()
    recorder = Console(file=buffer, width=400, force_terminal=False, no_color=True)
    monkeypatch.setattr(cli_module, "console", recorder)
    monkeypatch.setattr(cli_module, "err_console", recorder)
    return buffer


def _result(answer: str) -> AgentResult:
    return AgentResult(
        task_id="task_1",
        session_id="session_1",
        status="completed",
        answer=answer,
    )


class TestAnswerRendering:
    def test_regex_brackets_survive(self, captured) -> None:
        """The original symptom: a blocked-pattern message lost [a-zA-Z]."""
        answer = r"PERMISSION DENIED — command matches a blocked pattern (/\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]/)"
        cli_module._print_result(_result(answer), quiet=True)
        rendered = captured.getvalue()
        assert "[a-zA-Z]" in rendered, f"brackets were eaten: {rendered!r}"

    def test_pytest_summary_survives(self, captured) -> None:
        cli_module._print_result(
            _result("tests/test_x.py ................ [100%]\n190 passed"), quiet=True
        )
        assert "[100%]" in captured.getvalue()

    def test_toml_section_survives(self, captured) -> None:
        cli_module._print_result(_result("[models.default]\nmodel = \"gpt-4.1\""), quiet=True)
        assert "[models.default]" in captured.getvalue()

    def test_markdown_link_survives(self, captured) -> None:
        cli_module._print_result(_result("See [the docs](https://example.com/x)"), quiet=True)
        assert "[the docs](https://example.com/x)" in captured.getvalue()

    def test_log_levels_survive(self, captured) -> None:
        cli_module._print_result(_result("[INFO] started\n[WARN] retrying"), quiet=True)
        rendered = captured.getvalue()
        assert "[INFO]" in rendered and "[WARN]" in rendered


class TestApprovalPanelRendering:
    def test_tool_arguments_with_brackets_survive(self, captured) -> None:
        pending = PendingConfirmation(
            request_id="req_1",
            step_id="step_1",
            tool="run_command",
            arguments={"command": "grep -E '[0-9]+' file"},
            effect="execute_local",
            reason=r"command matches a blocked pattern (/\brm\s+(-[a-zA-Z]*\s+)*/)",
            preview="run_command({\"command\": \"grep -E '[0-9]+' file\"})",
        )
        result = AgentResult(
            task_id="task_1",
            session_id="session_1",
            status="waiting_confirmation",
            pending_confirmation=pending,
        )
        cli_module._print_result(result)
        rendered = captured.getvalue()
        assert "[0-9]+" in rendered
        assert "[a-zA-Z]" in rendered


class TestProgressHookRendering:
    def test_plan_step_brackets_survive(self, captured) -> None:
        hook = cli_module._progress_printer(quiet=False)
        hook("plan", {"steps": ["Run pytest and read [100%] output", "Fix [ERROR] handling"]})
        rendered = captured.getvalue()
        assert "[100%]" in rendered
        assert "[ERROR]" in rendered

    def test_tool_failure_message_survives(self, captured) -> None:
        hook = cli_module._progress_printer(quiet=False)
        hook("tool", {"name": "run_command", "success": False})
        assert "run_command" in captured.getvalue()

    def test_quiet_mode_prints_nothing(self, captured) -> None:
        hook = cli_module._progress_printer(quiet=True)
        hook("plan", {"steps": ["anything"]})
        hook("tool", {"name": "x", "success": True})
        assert captured.getvalue() == ""


class TestConfigRendering:
    def test_config_show_keeps_toml_sections(self, captured, settings, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setattr(
            cli_module, "_settings", lambda home, workspace, create=False: settings
        )
        cli_module.config_show(home=None, workspace=None)
        rendered = captured.getvalue()
        assert "[models.scripted]" in rendered
        assert "[permissions]" in rendered
        assert "[agent]" in rendered
