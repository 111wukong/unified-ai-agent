"""Edit safety: order-independent patches, and a parse gate before writing.

Two failure modes, both of which produce a file that looks edited and is
wrong:

* **Order dependence.** Applying edits one after another to a mutating string
  looks equivalent to resolving them all against the original, and is not. A
  later `old` can match text an earlier `new` just inserted, so the edit lands
  somewhere the model never intended. Every edit here is resolved against the
  file as it is *now*, then applied by descending offset.
* **A patch that lands and leaves an unparseable file.** Without a parse gate
  the mistake surfaces several steps later as a confusing test failure, and a
  model reasoning from a broken premise usually "fixes" something else first.

The gate is deliberately narrow: Python, JSON, TOML and YAML, which are the
parsers already installed. Adding a dependency to catch an error the model
fixes in one turn is a bad trade, so an unknown extension is not guessed at.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wukong.tools.base import ToolContext
from wukong.tools.fs import ApplyPatchTool, WriteFileTool


def ctx_for(workspace: Path, home: Path) -> ToolContext:
    return ToolContext(
        task_id="t",
        session_id="s",
        step_id="step_1",
        workspace=workspace,
        home=home,
        artifact_dir=home,
    )


@pytest.fixture
def patch(tmp_path: Path):  # noqa: ANN201
    """A file and a context, so each test only states what it is about."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return ApplyPatchTool(), workspace, ctx_for(workspace, tmp_path / "home")


class TestOrderIndependence:
    async def test_an_edit_does_not_match_text_an_earlier_edit_inserted(
        self, patch  # noqa: ANN001
    ) -> None:
        """The concrete failure.

        Sequentially, edit #1 turns the file into `y = 2\\ny = 2\\n`, and edit
        #2 then replaces the *first* `y = 2` -- the one edit #1 just wrote --
        so the original `y = 2` survives untouched and the model's intent is
        silently inverted.
        """
        tool, workspace, ctx = patch
        target = workspace / "m.py"
        target.write_text("x = 1\ny = 2\n", encoding="utf-8")

        result = await tool.run(
            {
                "path": "m.py",
                "edits": [
                    {"old": "x = 1", "new": "y = 2"},
                    {"old": "y = 2", "new": "z = 3"},
                ],
            },
            ctx,
        )
        assert result.success, result.error
        assert target.read_text() == "y = 2\nz = 3\n"

    async def test_the_result_does_not_depend_on_the_order_given(
        self, patch  # noqa: ANN001
    ) -> None:
        tool, workspace, ctx = patch
        target = workspace / "m.py"
        original = "x = 1\ny = 2\n"
        edits = [
            {"old": "x = 1", "new": "y = 2"},
            {"old": "y = 2", "new": "z = 3"},
        ]

        target.write_text(original, encoding="utf-8")
        assert (await tool.run({"path": "m.py", "edits": edits}, ctx)).success
        forward = target.read_text()

        target.write_text(original, encoding="utf-8")
        assert (await tool.run({"path": "m.py", "edits": list(reversed(edits))}, ctx)).success
        backward = target.read_text()

        assert forward == backward

    async def test_overlapping_edits_are_refused(self, patch  # noqa: ANN001
    ) -> None:
        """Overlap reintroduces order dependence, so it is refused rather
        than resolved by a tie-break nobody can predict."""
        tool, workspace, ctx = patch
        (workspace / "m.py").write_text("abcdef = 1\n", encoding="utf-8")

        result = await tool.run(
            {
                "path": "m.py",
                "edits": [
                    {"old": "abcdef", "new": "x"},
                    {"old": "cde", "new": "y"},
                ],
            },
            ctx,
        )
        assert not result.success
        assert "overlap" in result.error
        assert (workspace / "m.py").read_text() == "abcdef = 1\n", "nothing may be written"

    async def test_replace_all_still_replaces_every_occurrence(self, patch  # noqa: ANN001
    ) -> None:
        tool, workspace, ctx = patch
        (workspace / "m.py").write_text("a = 1\nb = 2\na = 3\n", encoding="utf-8")

        result = await tool.run(
            {"path": "m.py", "edits": [{"old": "a", "new": "z", "replace_all": True}]},
            ctx,
        )
        assert result.success, result.error
        assert (workspace / "m.py").read_text() == "z = 1\nb = 2\nz = 3\n"

    async def test_a_non_unique_match_is_still_refused(self, patch  # noqa: ANN001
    ) -> None:
        tool, workspace, ctx = patch
        (workspace / "m.py").write_text("a = 1\nb = 2\na = 3\n", encoding="utf-8")
        result = await tool.run({"path": "m.py", "edits": [{"old": "a", "new": "z"}]}, ctx)
        assert not result.success
        assert "occurs 2 times" in result.error


class TestParseGate:
    async def test_a_patch_that_would_not_parse_is_refused(
        self, patch  # noqa: ANN001
    ) -> None:
        """The failure this prevents: the edit lands, and the next step fails
        for a reason that has nothing to do with what the model was doing."""
        tool, workspace, ctx = patch
        target = workspace / "m.py"
        target.write_text("def f():\n    return 1\n", encoding="utf-8")

        result = await tool.run(
            {"path": "m.py", "edits": [{"old": "return 1", "new": "return 1 1 1"}]},
            ctx,
        )
        assert not result.success
        assert "does not parse" in result.error
        assert "Nothing was written" in result.error
        assert target.read_text() == "def f():\n    return 1\n"

    async def test_the_escape_hatch_allows_a_deliberately_invalid_file(
        self, patch  # noqa: ANN001
    ) -> None:
        """A parser fixture is a real thing to write, so there is a way --
        but it has to be asked for."""
        tool, workspace, ctx = patch
        target = workspace / "m.py"
        target.write_text("x = 1\n", encoding="utf-8")
        result = await tool.run(
            {
                "path": "m.py",
                "edits": [{"old": "x = 1", "new": "x = = ="}],
                "allow_syntax_errors": True,
            },
            ctx,
        )
        assert result.success, result.error
        assert target.read_text() == "x = = =\n"

    @pytest.mark.parametrize(
        ("name", "valid", "broken"),
        [
            ("m.py", "x = 1\n", "def f(:\n"),
            ("m.json", '{"a": 1}', '{"a": }'),
            ("m.toml", "a = 1", "a = = 1"),
            ("m.yaml", "a: 1\n", "a: [1, 2\n"),
        ],
    )
    async def test_each_supported_language_is_checked(
        self, patch, name: str, valid: str, broken: str  # noqa: ANN001
    ) -> None:
        tool, workspace, ctx = patch
        target = workspace / name
        target.write_text(valid, encoding="utf-8")

        # The valid form goes through, so the gate is not simply refusing
        # everything with a recognised extension.
        ok = await tool.run({"path": name, "edits": [{"old": valid.strip(), "new": valid.strip()}]}, ctx)
        assert ok.success, ok.error

        result = await tool.run({"path": name, "edits": [{"old": valid.strip(), "new": broken}]}, ctx)
        assert not result.success, f"{name} should have been refused"
        assert "does not parse" in result.error
        assert target.read_text() == valid, "nothing may be written"

    async def test_an_extension_with_no_checker_is_not_guessed_at(
        self, patch  # noqa: ANN001
    ) -> None:
        tool, workspace, ctx = patch
        target = workspace / "notes.txt"
        target.write_text("hello\n", encoding="utf-8")
        result = await tool.run(
            {"path": "notes.txt", "edits": [{"old": "hello", "new": "def f(:"}]}, ctx
        )
        assert result.success, "a .txt file has no parser and must not be judged"
        assert target.read_text() == "def f(:\n"

    async def test_write_file_is_gated_too(self, tmp_path: Path) -> None:
        """Writing a whole broken file is the same mistake as patching one."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        ctx = ctx_for(workspace, tmp_path / "home")
        tool = WriteFileTool()

        refused = await tool.run({"path": "bad.py", "content": "def f(:\n"}, ctx)
        assert not refused.success
        assert "does not parse" in refused.error
        assert not (workspace / "bad.py").exists()

        allowed = await tool.run(
            {"path": "bad.py", "content": "def f(:\n", "allow_syntax_errors": True}, ctx
        )
        assert allowed.success

    async def test_appending_is_not_gated(self, tmp_path: Path) -> None:
        """A partial append is not expected to parse on its own -- the file
        is mid-edit by construction, and judging it would refuse valid work."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        ctx = ctx_for(workspace, tmp_path / "home")
        tool = WriteFileTool()
        result = await tool.run(
            {"path": "m.py", "content": "def half(\n", "mode": "append"}, ctx
        )
        assert result.success, result.error
