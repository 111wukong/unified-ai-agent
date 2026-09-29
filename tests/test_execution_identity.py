"""What was approved is what runs.

An approval gate creates a gap: between showing the request and acting on the
answer, the world can change. A command's `argv[0]` is resolved through PATH at
execution time, so the binary that runs need not be the one that was on PATH
when the request was displayed; a file about to be overwritten may have been
edited while the request sat waiting.

Both are refused rather than executed, because the approval was for what was
shown and no longer covers what would happen.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from unified_agent.agent.execution_identity import ExecutionIdentity, fingerprint, verify
from unified_agent.agent.state import TaskStatus, replay
from unified_agent.config import Decision
from unified_agent.types import EffectClass


def quoted(path: Path) -> str:
    """A path as it would appear in a real command.

    This project lives under a directory with a space in its name, so an
    unquoted path would be split in two by `shlex.split` -- which is exactly how
    the shell tool would split it.
    """
    return shlex.quote(str(path))


def script(path: Path, body: str = "#!/bin/sh\necho hi\n") -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def allowlisted(settings, target: Path) -> Path:  # noqa: ANN001
    """Put a temp script on the command allowlist.

    By *name*: the guard matches on the command's basename, and an allowlist
    entry is a command plus optional leading arguments -- so an absolute path
    containing a space cannot be expressed as one entry. These tests are about
    the approval gate, not the allowlist, so the name just has to be allowed
    for the gate to be the thing under test.
    """
    settings.permissions.shell.allow.append(target.name)
    return target


def command_call(target: Path) -> dict:
    return {"tool_calls": [{"name": "run_command", "arguments": {"command": quoted(target)}}]}


class TestFingerprint:
    def test_a_command_binds_to_the_executable_it_resolves_to(self, tmp_path: Path) -> None:
        target = script(tmp_path / "run.sh")
        identity = fingerprint(
            "run_command", {"command": f"{quoted(target)} --flag"}, workspace=tmp_path, env={}
        )
        assert identity is not None
        assert identity.kind == "executable"
        assert identity.target == str(Path(os.path.realpath(target)))

    def test_a_command_that_cannot_be_resolved_binds_to_nothing(self, tmp_path: Path) -> None:
        """Nothing to compare later, so there is nothing to check. Binding to
        an unresolvable name would produce a mismatch on every run."""
        assert (
            fingerprint(
                "run_command",
                {"command": "definitely-not-a-real-binary-xyz"},
                workspace=tmp_path,
                env={},
            )
            is None
        )

    def test_a_file_binds_to_its_contents(self, tmp_path: Path) -> None:
        target = tmp_path / "notes.md"
        target.write_text("original\n", encoding="utf-8")
        identity = fingerprint("write_file", {"path": "notes.md"}, workspace=tmp_path, env={})
        assert identity is not None and identity.kind == "file"

        target.write_text("changed\n", encoding="utf-8")
        assert verify(identity, workspace=tmp_path, env={}) is not None

    def test_creating_a_file_binds_to_nothing(self, tmp_path: Path) -> None:
        """A new file has no pre-image to protect, and the approval is for the
        content, which the model already fixed."""
        assert fingerprint("write_file", {"path": "new.md"}, workspace=tmp_path, env={}) is None

    def test_read_only_tools_bind_to_nothing(self, tmp_path: Path) -> None:
        assert fingerprint("read_file", {"path": "x"}, workspace=tmp_path, env={}) is None

    def test_a_round_trip_through_the_event_payload(self, tmp_path: Path) -> None:
        target = script(tmp_path / "run.sh")
        identity = fingerprint("run_command", {"command": quoted(target)}, workspace=tmp_path, env={})
        assert identity is not None
        assert ExecutionIdentity.from_dict(identity.as_dict()) == identity

    def test_an_empty_payload_is_not_an_identity(self) -> None:
        assert ExecutionIdentity.from_dict({}) is None


class TestDriftIsRefused:
    async def test_an_executable_that_changed_is_not_run(
        self, scripted, session_id: str, settings, tmp_path: Path  # noqa: ANN001
    ) -> None:
        """The failure this exists for: you approve `run.sh`, something
        replaces `run.sh`, and the thing that runs is not what you read."""
        target = allowlisted(settings, script(tmp_path / "run.sh", "#!/bin/sh\necho original\n"))
        agent, _ = await scripted([command_call(target)])
        result = await agent.runtime.run("run it", session_id=session_id)
        assert result.needs_approval
        assert result.pending_confirmation is not None
        assert result.pending_confirmation.identity, "the request should carry a fingerprint"

        # Something replaces the binary while the request waits for an answer.
        script(target, "#!/bin/sh\necho replaced\n")

        approved = await agent.runtime.approve(result.task_id)

        assert approved.status == TaskStatus.FAILED.value
        assert "changed since this was approved" in (approved.error or "")
        assert agent.store.list_tool_calls(result.task_id) == [], (
            "nothing may run when the approval no longer applies"
        )

    async def test_a_deleted_executable_is_not_run(
        self, scripted, session_id: str, settings, tmp_path: Path  # noqa: ANN001
    ) -> None:
        target = allowlisted(settings, script(tmp_path / "run.sh"))
        agent, _ = await scripted([command_call(target)])
        result = await agent.runtime.run("run it", session_id=session_id)
        assert result.needs_approval
        target.unlink()

        approved = await agent.runtime.approve(result.task_id)

        assert approved.status == TaskStatus.FAILED.value
        assert "is gone" in (approved.error or "")

    async def test_a_file_edited_while_waiting_is_not_overwritten(
        self, scripted, session_id: str, settings, workspace: Path  # noqa: ANN001
    ) -> None:
        """You approve replacing this content. Someone edits the file. Writing
        anyway would discard an edit nobody reviewed."""
        settings.permissions.defaults[EffectClass.WRITE_LOCAL] = Decision.CONFIRM
        # In the *workspace*: a file elsewhere is outside the fence and the
        # agent would be refused before any of this mattered.
        target = workspace / "notes.md"
        target.write_text("the version the model read\n", encoding="utf-8")

        agent, _ = await scripted(
            [
                {
                    "tool_calls": [
                        {
                            "name": "write_file",
                            "arguments": {"path": "notes.md", "content": "replacement\n"},
                        }
                    ]
                }
            ]
        )
        result = await agent.runtime.run("rewrite it", session_id=session_id)
        assert result.needs_approval

        target.write_text("an edit made while the request was waiting\n", encoding="utf-8")

        approved = await agent.runtime.approve(result.task_id)

        assert approved.status == TaskStatus.FAILED.value
        assert "was edited after this was approved" in (approved.error or "")
        assert target.read_text(encoding="utf-8") == "an edit made while the request was waiting\n"

    async def test_an_unchanged_target_is_still_approved(
        self, scripted, session_id: str, settings, tmp_path: Path  # noqa: ANN001
    ) -> None:
        """The check has to be silent when nothing moved, or every approval
        would fail and the gate would be useless."""
        target = allowlisted(settings, script(tmp_path / "run.sh", "#!/bin/sh\necho fine\n"))
        agent, _ = await scripted([command_call(target), {"content": "ran it"}])
        result = await agent.runtime.run("run it", session_id=session_id)
        assert result.needs_approval

        approved = await agent.runtime.approve(result.task_id)

        assert approved.status == TaskStatus.COMPLETED.value
        calls = agent.store.list_tool_calls(result.task_id)
        assert len(calls) == 1 and calls[0].status == "succeeded"
        assert "fine" in (calls[0].result or {}).get("output", "")

    async def test_the_fingerprint_survives_a_resume(
        self, scripted, session_id: str, settings, tmp_path: Path  # noqa: ANN001
    ) -> None:
        """It is in the event payload, so a resume in another process still
        knows what was approved."""
        target = allowlisted(settings, script(tmp_path / "run.sh"))
        agent, _ = await scripted([command_call(target)])
        result = await agent.runtime.run("run it", session_id=session_id)

        state = replay(agent.store.events(result.task_id), task_id=result.task_id)
        assert state.pending_confirmation is not None
        restored = ExecutionIdentity.from_dict(state.pending_confirmation.identity)
        assert restored is not None
        assert restored.target == str(Path(os.path.realpath(target)))
        assert verify(restored, workspace=tmp_path, env={}) is None

    async def test_the_drift_message_explains_it_is_the_approval_that_lapsed(
        self, scripted, session_id: str, settings, tmp_path: Path  # noqa: ANN001
    ) -> None:
        """Not 'permission denied' -- the human did approve. What changed is
        the thing they approved."""
        target = allowlisted(settings, script(tmp_path / "run.sh"))
        agent, _ = await scripted([command_call(target)])
        result = await agent.runtime.run("run it", session_id=session_id)
        script(target, "#!/bin/sh\necho different\n")

        approved = await agent.runtime.approve(result.task_id)

        error = approved.error or ""
        assert "does not cover this" in error
        assert "Ask again" in error
