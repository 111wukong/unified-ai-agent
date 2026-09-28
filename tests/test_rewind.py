"""Rewind: putting files back, from the checkpoints taken before each write.

The claim under test: a task's file changes can be undone from its own
record. The runtime could always replay its *state*; without checkpoints it
knew exactly which step broke a file and could not put the file back.

Two properties carry the weight:

* the pre-image is captured **before** the tool runs, so a crash mid-write
  still leaves the original;
* it is stored **verbatim**. A redacted copy restores a corrupted file, and
  the whole point is a byte-for-byte restore.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import ScriptedModel, ScriptedModels

from unified_agent.agent.factory import build_agent
from unified_agent.agent.rewind import apply_rewind, plan_rewind
from unified_agent.observability.events import EventType
from unified_agent.storage.store import Store


async def run_writes(settings, session_id: str, script: list[dict]) -> str:  # noqa: ANN001
    """Run a scripted task and return its task id."""
    agent = await build_agent(
        settings=settings, models=ScriptedModels(ScriptedModel(settings.models["scripted"], script))
    )
    try:
        result = await agent.runtime.run("edit some files", session_id=session_id)
        return result.task_id
    finally:
        agent.close()


def write(path: str, content: str) -> dict:
    return {"tool_calls": [{"name": "write_file", "arguments": {"path": path, "content": content}}]}


class TestCheckpointCapture:
    async def test_a_write_records_the_original_before_touching_it(
        self, scripted, session_id: str, workspace: Path
    ) -> None:
        agent, _ = await scripted([write("app.py", "def add(a, b):\n    return a - b\n"), {"content": "done"}])
        before = (workspace / "app.py").read_text()
        result = await agent.runtime.run("break app.py", session_id=session_id)

        checkpoints = [
            e for e in agent.store.events(result.task_id) if e.type is EventType.FILE_CHECKPOINT
        ]
        assert len(checkpoints) == 1
        payload = checkpoints[0].payload
        assert payload["existed"] is True
        assert payload["restorable"] is True
        assert payload["artifact"] is not None
        # The file on disk is the new content; the checkpoint holds the old.
        assert (workspace / "app.py").read_text() != before
        assert Path(payload["artifact"]).read_text() == before

    async def test_the_checkpoint_is_verbatim_not_redacted(
        self, scripted, session_id: str, workspace: Path, settings  # noqa: ANN001
    ) -> None:
        """A redacted pre-image restores a corrupted file.

        Checkpoints are the one artifact that must not go through the
        redactor: they exist to be written back byte-for-byte, and the
        content is a copy of a file already sitting in the workspace.
        """
        secret_line = 'API_KEY = "sk-this-looks-exactly-like-a-credential-000111"\n'
        original = f"# config\n{secret_line}DEBUG = True\n"
        (workspace / "conf.py").write_text(original, encoding="utf-8")

        agent, _ = await scripted([write("conf.py", "# replaced\n"), {"content": "done"}])
        result = await agent.runtime.run("blank the config", session_id=session_id)
        checkpoint = next(
            e for e in agent.store.events(result.task_id) if e.type is EventType.FILE_CHECKPOINT
        )

        stored = Path(checkpoint.payload["artifact"]).read_text()
        assert stored == original, "the checkpoint was altered on the way in"

    async def test_a_new_file_is_checkpointed_as_absent(
        self, scripted, session_id: str
    ) -> None:
        """Rewinding a created file means deleting it, and "it did not exist"
        is a fact to record rather than to infer from a missing artifact."""
        agent, _ = await scripted([write("brand_new.py", "x = 1\n"), {"content": "done"}])
        result = await agent.runtime.run("create a file", session_id=session_id)
        checkpoint = next(
            e for e in agent.store.events(result.task_id) if e.type is EventType.FILE_CHECKPOINT
        )
        assert checkpoint.payload["existed"] is False
        assert checkpoint.payload["artifact"] is None

    async def test_a_read_only_tool_records_nothing(
        self, scripted, session_id: str
    ) -> None:
        """Only tools that declare `snapshot_paths` are checkpointed, so a
        task that only reads leaves no snapshot debris."""
        agent, _ = await scripted(
            [{"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]}, {"content": "done"}]
        )
        result = await agent.runtime.run("read it", session_id=session_id)
        assert not [
            e for e in agent.store.events(result.task_id) if e.type is EventType.FILE_CHECKPOINT
        ]


class TestRewindPlan:
    async def test_rewinding_restores_the_original(
        self, scripted, session_id: str, workspace: Path, settings  # noqa: ANN001
    ) -> None:
        original = (workspace / "app.py").read_text()
        agent, _ = await scripted([write("app.py", "def add(a, b):\n    return a - b\n"), {"content": "done"}])
        result = await agent.runtime.run("break it", session_id=session_id)

        store = Store(settings.db_path)
        try:
            plan = plan_rewind(store, result.task_id)
        finally:
            store.close()

        assert [s.action for s in plan.steps] == ["restore"]
        assert apply_rewind(plan) == [f"restored {workspace / 'app.py'}"]
        assert (workspace / "app.py").read_text() == original

    async def test_rewinding_a_created_file_deletes_it(
        self, scripted, session_id: str, workspace: Path, settings  # noqa: ANN001
    ) -> None:
        agent, _ = await scripted([write("scratch.py", "x = 1\n"), {"content": "done"}])
        result = await agent.runtime.run("create it", session_id=session_id)
        assert (workspace / "scratch.py").exists()

        store = Store(settings.db_path)
        try:
            plan = plan_rewind(store, result.task_id)
        finally:
            store.close()

        assert [s.action for s in plan.steps] == ["delete"]
        apply_rewind(plan)
        assert not (workspace / "scratch.py").exists()

    async def test_multiple_writes_rewind_to_the_earliest_state(
        self, scripted, session_id: str, workspace: Path, settings  # noqa: ANN001
    ) -> None:
        """Three writes to one file: a full rewind must reach the state
        before the *first* one, not the state before the last."""
        original = (workspace / "app.py").read_text()
        agent, _ = await scripted(
            [
                write("app.py", "one\n"),
                write("app.py", "two\n"),
                write("app.py", "three\n"),
                {"content": "done"},
            ]
        )
        result = await agent.runtime.run("edit it thrice", session_id=session_id)
        assert (workspace / "app.py").read_text() == "three\n"

        store = Store(settings.db_path)
        try:
            plan = plan_rewind(store, result.task_id)
        finally:
            store.close()

        assert len(plan.steps) == 1, "one row per file, not one per write"
        apply_rewind(plan)
        assert (workspace / "app.py").read_text() == original

    async def test_a_second_rewind_is_a_no_op(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        """Rewinding twice must not rewrite identical bytes -- and must say
        so, because "nothing happened" and "it worked" look the same."""
        agent, _ = await scripted([write("app.py", "changed\n"), {"content": "done"}])
        result = await agent.runtime.run("edit it", session_id=session_id)

        store = Store(settings.db_path)
        try:
            apply_rewind(plan_rewind(store, result.task_id))
            second = plan_rewind(store, result.task_id)
        finally:
            store.close()

        assert [s.action for s in second.steps] == ["already"]
        assert second.actionable == []
        assert apply_rewind(second) == []

    async def test_from_seq_narrows_the_window(
        self, scripted, session_id: str, workspace: Path, settings  # noqa: ANN001
    ) -> None:
        """`--from-seq` undoes only the later writes: the state before the
        first write at or after that sequence number."""
        agent, _ = await scripted(
            [write("app.py", "first change\n"), write("app.py", "second change\n"), {"content": "done"}]
        )
        result = await agent.runtime.run("edit twice", session_id=session_id)
        events = agent.store.events(result.task_id)
        checkpoints = [e for e in events if e.type is EventType.FILE_CHECKPOINT]
        assert len(checkpoints) == 2

        store = Store(settings.db_path)
        try:
            plan = plan_rewind(store, result.task_id, from_seq=checkpoints[1].seq)
        finally:
            store.close()

        assert [s.action for s in plan.steps] == ["restore"]
        apply_rewind(plan)
        # Back to the state before the *second* write, not the original.
        assert (workspace / "app.py").read_text() == "first change\n"

    async def test_an_unrestorable_checkpoint_is_reported_not_skipped(
        self, scripted, session_id: str, workspace: Path, settings  # noqa: ANN001
    ) -> None:
        """A file too large to snapshot must be named. A partial restore that
        silently omits a file is worse than one that says what it could not do."""
        agent, _ = await scripted([write("app.py", "small change\n"), {"content": "done"}])
        result = await agent.runtime.run("edit it", session_id=session_id)

        store = Store(settings.db_path)
        try:
            # Simulate the over-limit case the way the executor records it.
            store.append(
                result.task_id,
                EventType.FILE_CHECKPOINT,
                {
                    "call_id": "call_x",
                    "path": str(workspace / "huge.bin"),
                    "existed": True,
                    "artifact": None,
                    "bytes": 5_000_000,
                    "restorable": False,
                    "reason": "the original is 5000000 bytes, over the limit",
                },
            )
            plan = plan_rewind(store, result.task_id)
        finally:
            store.close()

        assert [s.action for s in plan.steps] == ["restore", "blocked"]
        assert "over the limit" in plan.blocked[0].reason
        assert "cannot be restored" in plan.render()

    async def test_a_task_with_no_writes_rewinds_to_nothing(
        self, scripted, session_id: str, settings  # noqa: ANN001
    ) -> None:
        agent, _ = await scripted([{"content": "I did nothing"}])
        result = await agent.runtime.run("just answer", session_id=session_id)

        store = Store(settings.db_path)
        try:
            plan = plan_rewind(store, result.task_id)
        finally:
            store.close()

        assert plan.steps == []
        assert "nothing to rewind" in plan.render()


class TestApplyPatchIsCheckpointed:
    async def test_patching_a_file_also_records_a_pre_image(
        self, scripted, session_id: str, workspace: Path, settings  # noqa: ANN001
    ) -> None:
        """`apply_patch` declares `snapshot_paths` too, so a patch is as
        undoable as a whole-file write."""
        original = (workspace / "notes.md").read_text()
        agent, _ = await scripted(
            [
                {
                    "tool_calls": [
                        {
                            "name": "apply_patch",
                            "arguments": {
                                "path": "notes.md",
                                "edits": [{"old": "# notes", "new": "# NOTES"}],
                            },
                        }
                    ]
                },
                {"content": "done"},
            ]
        )
        result = await agent.runtime.run("retitle the notes", session_id=session_id)
        assert "# NOTES" in (workspace / "notes.md").read_text()

        store = Store(settings.db_path)
        try:
            plan = plan_rewind(store, result.task_id)
        finally:
            store.close()

        apply_rewind(plan)
        assert (workspace / "notes.md").read_text() == original
