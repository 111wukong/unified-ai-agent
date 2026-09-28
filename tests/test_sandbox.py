"""Sandbox backends.

What can and cannot be verified here, stated plainly: the real Seatbelt
profile cannot be applied inside this test environment, because macOS
refuses to install a narrowing sandbox from an already-sandboxed process.
So the tests cover the parts that *are* verifiable -- the probe's honesty,
the generated profile's contents, mode semantics, argv rewriting, and the
fallback reporting -- and `uaa sandbox` performs the live escape check for a
user running from a normal terminal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unified_agent.sandbox import (
    DockerSandbox,
    NoSandbox,
    ProbeResult,
    SandboxMode,
    SandboxSelection,
    SeatbeltSandbox,
    build_sandbox,
    seatbelt_probe,
)


class TestProbeHonesty:
    def test_probe_returns_a_reason_either_way(self) -> None:
        result = seatbelt_probe()
        assert isinstance(result.ok, bool)
        assert result.detail, "a probe must explain itself, not just fail"

    def test_probe_explains_the_nested_sandbox_case(self) -> None:
        """If this environment is itself sandboxed, the message must say so.

        Otherwise the user reads "unavailable" and concludes the backend is
        broken, when the real answer is "run this from your terminal".
        """
        result = seatbelt_probe()
        assert result.detail, "a probe must explain itself in every outcome"
        if result.ok:
            assert "applied" in result.detail
            return
        # The failure must be *described*, not just reported as a number.
        assert not result.detail.startswith("exit ")
        assert any(
            token in result.detail
            for token in ("Operation not permitted", "not found", "not macOS", "killed by")
        )
        if "Operation not permitted" in result.detail or "killed by" in result.detail:
            assert "normal terminal" in result.detail

    def test_a_no_op_profile_would_prove_nothing(self) -> None:
        """The probe profile must actually restrict something.

        `(allow default)` applies anywhere and would always pass, which is
        exactly the false confidence a probe exists to prevent.
        """
        from unified_agent.sandbox.base import _PROBE_PROFILE

        assert "deny default" in _PROBE_PROFILE
        assert "allow default" not in _PROBE_PROFILE


class TestSeatbeltProfile:
    @pytest.fixture
    def sandbox(self, tmp_path: Path) -> SeatbeltSandbox:
        return SeatbeltSandbox(home=tmp_path / "home")

    def test_profile_is_deny_by_default(self, sandbox, workspace) -> None:  # noqa: ANN001
        profile = sandbox.render_profile(SandboxMode.WORKSPACE_WRITE, workspace)
        assert "(deny default)" in profile
        assert "(allow file-read*)" in profile, "reads stay open by design"
        assert "(allow network*)" in profile

    def test_workspace_is_writable_in_workspace_write(self, sandbox, workspace) -> None:  # noqa: ANN001
        paths = sandbox.writable_paths(SandboxMode.WORKSPACE_WRITE, workspace)
        assert str(Path(workspace).resolve()) in paths

    def test_workspace_is_not_writable_in_read_only(self, sandbox, workspace) -> None:  # noqa: ANN001
        """Read-only mode is the point of the mode: `git commit` must fail."""
        paths = sandbox.writable_paths(SandboxMode.READ_ONLY, workspace)
        assert str(Path(workspace).resolve()) not in paths

    def test_caches_and_temp_are_always_writable(self, sandbox, workspace) -> None:  # noqa: ANN001
        for mode in (SandboxMode.READ_ONLY, SandboxMode.WORKSPACE_WRITE):
            paths = sandbox.writable_paths(mode, workspace)
            assert any("tmp" in p for p in paths)
            assert any("Cache" in p or ".cache" in p for p in paths)

    def test_credential_stores_are_never_writable(self, sandbox, workspace) -> None:  # noqa: ANN001
        """The actual security boundary: readable, never writable."""
        for mode in SandboxMode:
            if mode is SandboxMode.FULL:
                continue
            writable = " ".join(sandbox.writable_paths(mode, workspace))
            for secret in (".ssh", ".aws", ".gnupg", ".netrc", "Keychains"):
                assert secret not in writable, f"{secret} must not be writable in {mode}"

    def test_ssh_known_hosts_is_the_documented_exception(self, sandbox, workspace) -> None:  # noqa: ANN001
        profile = sandbox.render_profile(SandboxMode.WORKSPACE_WRITE, workspace)
        assert "known_hosts" in profile

    def test_extra_write_dirs_are_honoured(self, tmp_path, workspace) -> None:  # noqa: ANN001
        sandbox = SeatbeltSandbox(home=tmp_path / "home", extra_write_dirs=["~/extra-dir"])
        paths = sandbox.writable_paths(SandboxMode.WORKSPACE_WRITE, workspace)
        assert any(p.endswith("extra-dir") for p in paths)

    def test_profile_is_cached_per_mode_and_workspace(self, sandbox, workspace) -> None:  # noqa: ANN001
        first = sandbox.profile_path(SandboxMode.WORKSPACE_WRITE, workspace)
        second = sandbox.profile_path(SandboxMode.WORKSPACE_WRITE, workspace)
        third = sandbox.profile_path(SandboxMode.READ_ONLY, workspace)
        assert first == second
        assert first != third

    def test_quotes_in_paths_are_escaped(self, tmp_path) -> None:  # noqa: ANN001
        """A path with a quote must not break the profile's syntax."""
        sandbox = SeatbeltSandbox(home=tmp_path / "home")
        weird = tmp_path / 'a"b'
        weird.mkdir()
        profile = sandbox.render_profile(SandboxMode.WORKSPACE_WRITE, weird)
        assert '\\"' in profile

    def test_wrap_rewrites_argv(self, sandbox, workspace, monkeypatch) -> None:  # noqa: ANN001
        # `wrap` deliberately no-ops when the probe fails, so force the
        # available branch here; the fallback path is covered separately.
        monkeypatch.setattr(
            type(sandbox), "probe", lambda self: ProbeResult(True, "stubbed")
        )
        sandbox._probe = None
        argv = sandbox.wrap(
            ["echo", "hi"], workspace=workspace, mode=SandboxMode.WORKSPACE_WRITE, env={}
        )
        assert argv[0].endswith("sandbox-exec")
        assert argv[-2:] == ["echo", "hi"]

    def test_full_mode_does_not_wrap(self, sandbox, workspace, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setattr(
            type(sandbox), "probe", lambda self: ProbeResult(True, "stubbed")
        )
        sandbox._probe = None
        argv = sandbox.wrap(["echo"], workspace=workspace, mode=SandboxMode.FULL, env={})
        assert argv == ["echo"]

    def test_wrap_is_a_no_op_when_unavailable(self, sandbox, workspace) -> None:  # noqa: ANN001
        """A broken sandbox must not brick every command -- but the caller
        has to warn, which is what `SandboxSelection.warning()` is for."""
        sandbox._probe = ProbeResult(False, "stubbed failure")
        assert sandbox.wrap(["echo"], workspace=workspace, mode=SandboxMode.WORKSPACE_WRITE, env={}) == [
            "echo"
        ]

    def test_caveats_are_specific_and_non_empty(self, sandbox) -> None:
        caveats = sandbox.caveats()
        assert len(caveats) >= 3
        joined = " ".join(caveats).lower()
        assert "network" in joined, "the egress limitation must be stated"
        assert "login" in joined, "the credential-write limitation must be stated"


class TestNoSandbox:
    def test_passes_argv_through(self, workspace) -> None:  # noqa: ANN001
        sandbox = NoSandbox()
        assert sandbox.wrap(["ls"], workspace=workspace, mode=SandboxMode.READ_ONLY, env={}) == [
            "ls"
        ]

    def test_says_it_is_not_isolating(self) -> None:
        assert NoSandbox().isolation == "none"
        caveats = " ".join(NoSandbox().caveats()).lower()
        assert "no process isolation" in caveats


class TestDockerSandbox:
    def test_wrap_builds_a_mount_and_workdir(self, workspace, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setattr(DockerSandbox, "probe", lambda self: ProbeResult(True, "stubbed"))
        sandbox = DockerSandbox(image="python:3.12-slim", network="none")
        sandbox._probe = None
        argv = sandbox.wrap(
            ["pytest", "-q"],
            workspace=workspace,
            mode=SandboxMode.WORKSPACE_WRITE,
            env={"PATH": "/usr/bin"},
        )
        assert "run" in argv and "--rm" in argv
        assert f"{Path(workspace).resolve()}:/workspace" in argv
        assert argv[-2:] == ["pytest", "-q"]
        assert argv[-3] == "python:3.12-slim"

    def test_read_only_mode_mounts_read_only(self, workspace, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setattr(DockerSandbox, "probe", lambda self: ProbeResult(True, "stubbed"))
        sandbox = DockerSandbox()
        sandbox._probe = None
        argv = sandbox.wrap(["ls"], workspace=workspace, mode=SandboxMode.READ_ONLY, env={})
        assert "--read-only" in argv

    def test_network_defaults_to_none(self) -> None:
        """A sandbox that leaves the network open is not much of a sandbox."""
        assert DockerSandbox().network == "none"

    def test_caveats_mention_the_cost_and_the_kernel(self) -> None:
        caveats = " ".join(DockerSandbox().caveats()).lower()
        assert "round-trip" in caveats
        assert "kernel" in caveats, "shared-kernel risk must be stated"


class TestSelection:
    def test_explicit_none_returns_none(self, tmp_path) -> None:  # noqa: ANN001
        selection = build_sandbox("none", home=tmp_path)
        assert selection.sandbox.name == "none"
        assert not selection.fell_back

    def test_fallback_records_a_reason(self) -> None:
        """A silent downgrade to no isolation is the failure this prevents.

        Constructed rather than probed: the real answer depends on the host,
        and a test whose assertions change with the environment is not
        testing the contract.
        """
        selection = SandboxSelection(
            NoSandbox(), "seatbelt", notes=["seatbelt unavailable: stubbed failure"]
        )
        assert selection.fell_back
        assert selection.notes, "falling back must leave a trail"
        assert "fell back" in selection.summary()

    def test_real_selection_always_reports_something(self, tmp_path) -> None:  # noqa: ANN001
        selection = build_sandbox("seatbelt", home=tmp_path)
        assert selection.sandbox is not None
        assert selection.summary()
        assert selection.sandbox.probe_detail

    def test_auto_never_raises(self, tmp_path) -> None:  # noqa: ANN001
        selection = build_sandbox("auto", home=tmp_path)
        assert selection.sandbox is not None
        assert selection.sandbox.probe_detail

    def test_unknown_backend_is_rejected(self, tmp_path) -> None:  # noqa: ANN001
        with pytest.raises(ValueError, match="unknown sandbox backend"):
            build_sandbox("magic", home=tmp_path)

    def test_summary_explains_a_fallback(self, tmp_path) -> None:  # noqa: ANN001
        selection = build_sandbox("docker", home=tmp_path)
        text = selection.summary()
        assert "backend:" in text
        if selection.fell_back:
            assert "fell back" in text


class TestSandboxWiring:
    """The sandbox must be reachable from the tools, not just constructible."""

    def test_shell_tools_accept_a_sandbox(self, workspace) -> None:  # noqa: ANN001
        from unified_agent.tools.shell import RunCommandTool, build_shell_tools

        sandbox = NoSandbox()
        tool = RunCommandTool(sandbox=sandbox, sandbox_mode=SandboxMode.READ_ONLY)
        assert tool.sandbox is sandbox
        assert tool.sandbox_mode is SandboxMode.READ_ONLY
        assert len(build_shell_tools(sandbox=sandbox)) == 3

    def test_git_tools_accept_a_sandbox(self) -> None:  # noqa: ANN001
        from unified_agent.tools.git import build_git_tools

        sandbox = NoSandbox()
        tools = build_git_tools(sandbox=sandbox, sandbox_mode=SandboxMode.WORKSPACE_WRITE)
        assert tools and all(t.sandbox is sandbox for t in tools)

    async def test_sandbox_is_applied_to_the_actual_argv(self, workspace, tmp_path) -> None:  # noqa: ANN001
        """Assert on what actually got executed, not on the wrapper existing."""
        from unified_agent.tools.base import ToolContext
        from unified_agent.tools.shell import RunCommandTool

        recorded: list[list[str]] = []

        class RecordingSandbox(NoSandbox):
            name = "recording"

            def wrap(self, argv, *, workspace, mode, env):  # noqa: ANN001, ANN201
                recorded.append(list(argv))
                return argv

        ctx = ToolContext(
            task_id="t",
            session_id="s",
            step_id="step_1",
            workspace=workspace,
            home=tmp_path,
            artifact_dir=tmp_path,
        )
        tool = RunCommandTool(sandbox=RecordingSandbox(), sandbox_mode=SandboxMode.READ_ONLY)
        result = await tool.run({"command": "echo sandboxed"}, ctx)

        assert result.success
        assert recorded == [["echo", "sandboxed"]]


class TestFallbackWarning:
    """`wrap` no-ops when the backend is unavailable, so the caller must say so."""

    def test_warning_when_the_requested_backend_is_unavailable(self) -> None:
        selection = SandboxSelection(
            NoSandbox(), "seatbelt", notes=["seatbelt unavailable: stubbed failure"]
        )
        warning = selection.warning()
        assert warning, "a silent downgrade is the bug this prevents"
        assert "unsandboxed" in warning

    def test_warning_when_it_fell_back_to_a_weaker_backend(self) -> None:
        selection = SandboxSelection(DockerSandbox(), "seatbelt", notes=[])
        warning = selection.warning()
        assert warning and "fell back" in warning

    def test_no_warning_when_none_was_requested(self) -> None:
        assert SandboxSelection(NoSandbox(), "none", notes=[]).warning() is None

    def test_no_warning_when_the_request_is_satisfied(self) -> None:
        sandbox = SeatbeltSandbox.__new__(SeatbeltSandbox)
        sandbox.__dict__.update(NoSandbox().__dict__)
        selection = SandboxSelection(sandbox, "seatbelt", notes=[])
        selection.sandbox.name = "seatbelt"
        selection.sandbox._probe = ProbeResult(True, "stubbed")
        assert selection.warning() is None


class TestBackendsIdentifyThemselves:
    """Each backend must report its own name and isolation strength.

    The bug these guard against: `Sandbox` was a dataclass, so its field
    defaults for `name`/`isolation` were assigned by `__init__` and shadowed
    the subclass class attributes. Every backend reported `name="none"` even
    while it was actively isolating -- a sandbox that works but says it does
    not is as damaging as the reverse.
    """

    def test_each_backend_reports_its_own_name(self, tmp_path: Path) -> None:
        assert NoSandbox().name == "none"
        assert SeatbeltSandbox(home=tmp_path).name == "seatbelt"
        assert DockerSandbox().name == "docker"

    def test_each_backend_reports_its_own_isolation(self, tmp_path: Path) -> None:
        assert NoSandbox().isolation == "none"
        assert "process" in SeatbeltSandbox(home=tmp_path).isolation
        assert "container" in DockerSandbox().isolation

    def test_describe_mentions_the_backend(self, tmp_path: Path) -> None:
        assert "seatbelt" in SeatbeltSandbox(home=tmp_path).describe()
        assert "docker" in DockerSandbox().describe()

    def test_a_subclass_attribute_is_not_shadowed_by_construction(self, tmp_path: Path) -> None:
        """Constructing twice must not change what the class reports."""
        first = SeatbeltSandbox(home=tmp_path)
        second = SeatbeltSandbox(home=tmp_path)
        assert first.name == second.name == SeatbeltSandbox.name == "seatbelt"


