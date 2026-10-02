"""Sandbox backends.

What can and cannot be verified here, stated plainly: the real Seatbelt
profile cannot be applied inside this test environment, because macOS
refuses to install a narrowing sandbox from an already-sandboxed process.
So the tests cover the parts that *are* verifiable -- the probe's honesty,
the generated profile's contents, mode semantics, argv rewriting, and the
fallback reporting -- and `wukong sandbox` performs the live escape check for a
user running from a normal terminal.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wukong.sandbox import (
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
        from wukong.sandbox.base import _PROBE_PROFILE

        assert "deny default" in _PROBE_PROFILE
        assert "allow default" not in _PROBE_PROFILE


class TestProbeDoesNotOverclaim:
    """The probe must not assert a cause it has not established.

    A signal death with no output was previously reported as "you are inside a
    sandbox", which is a guess. The reader acts on the explanation, so a wrong
    one sends them looking in the wrong place -- which is exactly what
    happened when a report came back showing SIGABRT and the hint blamed a
    nested sandbox nobody had confirmed.

    These test `describe_probe_failure` directly rather than
    `seatbelt_probe`. The message construction is the part with the logic, and
    reaching it through the probe would mean the test only runs on macOS --
    the host-dependent-test mistake this project has already made twice.
    """

    def test_a_signal_death_does_not_claim_a_cause(self) -> None:
        from wukong.sandbox import describe_probe_failure

        detail = describe_probe_failure(
            returncode=-6, stdout=b"", stderr=b"", nested=False
        )
        assert "SIGABRT" in detail
        assert "already inside a sandbox" not in detail
        assert "refusal" in detail

    def test_a_signal_death_inside_a_known_sandbox_does_say_so(self) -> None:
        from wukong.sandbox import describe_probe_failure

        detail = describe_probe_failure(
            returncode=-6, stdout=b"", stderr=b"", nested=True
        )
        assert "already inside a sandbox" in detail

    def test_operation_not_permitted_is_enough_on_its_own(self) -> None:
        """The message is evidence even without the environment markers."""
        from wukong.sandbox import describe_probe_failure

        detail = describe_probe_failure(
            returncode=71,
            stdout=b"",
            stderr=b"sandbox-exec: sandbox_apply: Operation not permitted",
            nested=False,
        )
        assert "already inside a sandbox" in detail

    def test_stderr_is_kept_verbatim(self) -> None:
        from wukong.sandbox import describe_probe_failure

        detail = describe_probe_failure(
            returncode=65, stdout=b"", stderr=b"sandbox-exec: no version specified", nested=False
        )
        assert "no version specified" in detail

    def test_stdout_is_used_when_stderr_is_empty(self) -> None:
        """sandbox-exec does not reliably pick a stream."""
        from wukong.sandbox import describe_probe_failure

        detail = describe_probe_failure(
            returncode=1, stdout=b"Invalid Iconset", stderr=b"", nested=False
        )
        assert "Invalid Iconset" in detail

    def test_a_plain_nonzero_exit_is_reported_as_an_exit(self) -> None:
        from wukong.sandbox import describe_probe_failure

        detail = describe_probe_failure(returncode=3, stdout=b"", stderr=b"", nested=False)
        assert detail == "exit 3"


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
        from wukong.tools.shell import RunCommandTool, build_shell_tools

        sandbox = NoSandbox()
        tool = RunCommandTool(sandbox=sandbox, sandbox_mode=SandboxMode.READ_ONLY)
        assert tool.sandbox is sandbox
        assert tool.sandbox_mode is SandboxMode.READ_ONLY
        assert len(build_shell_tools(sandbox=sandbox)) == 3

    def test_git_tools_accept_a_sandbox(self) -> None:  # noqa: ANN001
        from wukong.tools.git import build_git_tools

        sandbox = NoSandbox()
        tools = build_git_tools(sandbox=sandbox, sandbox_mode=SandboxMode.WORKSPACE_WRITE)
        assert tools and all(t.sandbox is sandbox for t in tools)

    async def test_sandbox_is_applied_to_the_actual_argv(self, workspace, tmp_path) -> None:  # noqa: ANN001
        """Assert on what actually got executed, not on the wrapper existing."""
        from wukong.tools.base import ToolContext
        from wukong.tools.shell import RunCommandTool

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




class TestVerificationReport:
    """`wukong sandbox --report` exists because the check cannot always run here.

    macOS refuses to install a narrowing profile from an already-sandboxed
    process, so the verification has to happen in the user's own terminal. A
    command they have to assemble from a description is a command that does
    not get run, and a result they have to copy back is a result that gets
    mistyped -- so the output is a single pasteable line and the result lands
    on disk.
    """

    def test_verdict_says_no_isolation_and_how_to_check(self) -> None:
        from wukong.cli import _sandbox_verdict

        selection = SandboxSelection(NoSandbox(), "auto", notes=[])
        verdict = _sandbox_verdict(selection, ProbeResult(False, "stubbed"), [])
        assert "NO ISOLATION ACTIVE" in verdict
        assert "normal terminal" in verdict, "the verdict must say what to do"

    def test_verdict_confirms_a_working_sandbox(self) -> None:
        from wukong.cli import _sandbox_verdict

        sandbox = NoSandbox()
        sandbox.name = "seatbelt"
        selection = SandboxSelection(sandbox, "seatbelt", notes=[])
        live = [
            {"label": "write outside the workspace", "ok": True, "result": "blocked", "detail": ""},
            {"label": "read a system file", "ok": True, "result": "allowed", "detail": ""},
        ]
        verdict = _sandbox_verdict(selection, ProbeResult(True, "ok"), live)
        assert "blocked a write" in verdict
        assert "allowed reads" in verdict

    def test_verdict_flags_an_escape_as_not_working(self) -> None:
        """A sandbox that reports active but lets a write through is worse
        than one that reports inactive: the user stops watching."""
        from wukong.cli import _sandbox_verdict

        sandbox = NoSandbox()
        sandbox.name = "seatbelt"
        selection = SandboxSelection(sandbox, "seatbelt", notes=[])
        live = [
            {"label": "write outside the workspace", "ok": False, "result": "ESCAPED", "detail": ""},
            {"label": "read a system file", "ok": True, "result": "allowed", "detail": ""},
        ]
        verdict = _sandbox_verdict(selection, ProbeResult(True, "ok"), live)
        assert "ESCAPED" in verdict
        assert "not working" in verdict

    def test_paths_are_quoted_for_pasting(self) -> None:
        """Project paths routinely contain spaces; an unquoted path is a
        command that fails the moment it is pasted."""
        from wukong.cli import _sh

        assert _sh("/Users/me/WorkBuddy AI/project") == "'/Users/me/WorkBuddy AI/project'"
        assert _sh("/plain/path") == "/plain/path"

    def test_the_report_round_trips(self, tmp_path: Path) -> None:
        """The file and the printed view must be the same data."""
        import json

        from wukong.cli import _sandbox_verdict
        from wukong.sandbox import build_sandbox

        settings_home = tmp_path / "home"
        settings_home.mkdir()
        selection = build_sandbox("auto", home=settings_home)
        payload = {
            "requested": selection.requested,
            "selected": selection.sandbox.name,
            "isolation": selection.sandbox.isolation,
            "fell_back": selection.fell_back,
            "notes": selection.notes,
            "caveats": selection.sandbox.caveats(),
            "live_check": [],
            "verdict": _sandbox_verdict(selection, ProbeResult(False, "stub"), []),
        }
        target = tmp_path / "sandbox-verify.json"
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert set(loaded) >= {
            "requested",
            "selected",
            "isolation",
            "fell_back",
            "caveats",
            "verdict",
        }
        assert loaded["caveats"], "the limitations must travel with the result"


class TestEnvironmentFingerprint:
    """A probe result is unreadable without knowing where it ran.

    "Seatbelt does not work on this machine" and "Seatbelt cannot be tested
    from here" lead to opposite decisions, and the probe alone cannot tell
    them apart -- both produce a failure.
    """

    def test_reports_the_expected_shape(self) -> None:
        from wukong.sandbox import environment_fingerprint

        fingerprint = environment_fingerprint()
        assert set(fingerprint) >= {
            "inside_parent_sandbox",
            "sandbox_markers",
            "term_program",
            "parent_chain",
            "parent_chain_readable",
        }

    def test_detects_a_sandboxing_parent(self, monkeypatch) -> None:  # noqa: ANN001
        from wukong.sandbox import environment_fingerprint

        monkeypatch.setenv("CODEBUDDY_SANDBOX_BROKER_TRACE_ID", "x")
        fingerprint = environment_fingerprint()
        assert fingerprint["inside_parent_sandbox"] is True
        assert "CODEBUDDY_SANDBOX_BROKER_TRACE_ID" in fingerprint["sandbox_markers"]

    def test_a_clean_environment_reports_clean(self, monkeypatch) -> None:  # noqa: ANN001
        from wukong import sandbox as sandbox_pkg
        from wukong.sandbox import environment_fingerprint

        for key in list(os.environ):
            if key.startswith(sandbox_pkg.SANDBOX_MARKER_PREFIXES):
                monkeypatch.delenv(key, raising=False)
        fingerprint = environment_fingerprint()
        assert fingerprint["inside_parent_sandbox"] is False
        assert fingerprint["sandbox_markers"] == []

    def test_the_builtin_profile_probe_answers_without_raising(self) -> None:
        from wukong.sandbox import builtin_profile_probe

        result = builtin_profile_probe()
        assert isinstance(result.ok, bool)
        assert result.detail


class TestVerdictDistinguishesEnvironments:
    """The three cases must read differently, because they mean different
    things: broken here, untestable here, or working."""

    def _selection(self):  # noqa: ANN202
        sandbox = NoSandbox()
        sandbox.name = "none"
        return SandboxSelection(sandbox, "auto", notes=[])

    def test_nested_run_says_the_result_is_uninformative(self) -> None:
        from wukong.cli import _sandbox_verdict

        verdict = _sandbox_verdict(
            self._selection(),
            ProbeResult(False, "stubbed"),
            [],
            {"inside_parent_sandbox": True, "sandbox_markers": ["CODEBUDDY_X", "Y"]},
            ProbeResult(False, "refused"),
        )
        assert "says nothing about this machine" in verdict
        assert "Terminal.app" in verdict

    def test_clean_run_with_a_refused_builtin_profile_says_seatbelt_is_unusable(
        self,
    ) -> None:
        from wukong.cli import _sandbox_verdict

        verdict = _sandbox_verdict(
            self._selection(),
            ProbeResult(False, "refused"),
            [],
            {"inside_parent_sandbox": False, "sandbox_markers": []},
            ProbeResult(False, "sandbox-exec: sandbox_apply: Operation not permitted"),
        )
        assert "not a usable backend here" in verdict
        assert "path fence" in verdict, "say what still protects the user"

    def test_clean_run_without_a_builtin_answer_stays_neutral(self) -> None:
        from wukong.cli import _sandbox_verdict

        verdict = _sandbox_verdict(
            self._selection(),
            ProbeResult(False, "stubbed"),
            [],
            {"inside_parent_sandbox": False, "sandbox_markers": []},
            None,
        )
        assert "normal terminal" in verdict

    def test_a_working_probe_short_circuits(self) -> None:
        from wukong.cli import _sandbox_verdict

        verdict = _sandbox_verdict(
            self._selection(),
            ProbeResult(True, "ok"),
            [],
            {"inside_parent_sandbox": True, "sandbox_markers": ["X"]},
            ProbeResult(True, "ok"),
        )
        assert "works here" in verdict

    def test_the_verdict_still_defaults_environment_to_absent(self) -> None:
        """Callers that predate the environment argument must not crash."""
        from wukong.cli import _sandbox_verdict

        assert _sandbox_verdict(self._selection(), ProbeResult(False, "x"), [])


class TestProfileBisect:
    """The experiment is shipped, not run, because the failure is on the
    user's machine. That makes its *conclusion* logic the thing most worth
    testing -- a wrong conclusion sends them down the wrong path, and unlike
    the profiles it can be tested anywhere.
    """

    def _result(self, key: str, ok: bool, live_ok: bool | None = None):  # noqa: ANN202
        from wukong.sandbox import Result

        live = []
        if live_ok is not None:
            live = [{"label": "write outside the allowed subpath", "ok": live_ok,
                     "result": "blocked" if live_ok else "created", "expected": "blocked"}]
        return Result(key, "q", ok, "applied" if ok else "refused", live)

    def test_the_candidate_list_has_two_controls_and_a_live_check(self) -> None:
        from wukong.sandbox import candidates

        keys = [c.key for c in candidates(Path("/tmp/ws"), Path("/tmp/home"))]
        assert keys[0] == "control-allow-default", "a no-op control comes first"
        assert "allow-default-deny-network" in keys, "the second control"
        assert "allow-default-deny-write-escape" in keys, "the live escape check"
        assert keys[-1] == "generated-full-profile", "reproduce last"

    def test_the_candidate_fix_is_tested_live(self) -> None:
        from wukong.sandbox import candidates

        by_key = {c.key: c for c in candidates(Path("/tmp/ws"), Path("/tmp/home"))}
        assert by_key["allow-default-deny-write-reallow"].live is not None
        assert by_key["allow-default-deny-write-escape"].live is not None
        assert "(deny file-write*)" in by_key["allow-default-deny-write-reallow"].profile

    def test_no_op_control_failing_means_the_environment_cannot_diagnose(self) -> None:
        from wukong.sandbox import conclude

        text = conclude([self._result("control-allow-default", False)])
        assert "cannot apply any profile" in text
        assert "Terminal.app" in text

    def test_a_refused_narrowing_profile_is_not_read_as_deny_default(self) -> None:
        """The bug this guards: `(allow default)` passing proves only that a
        no-op is accepted. Reading that as the control misdiagnoses a nested
        environment as "deny-default is the trigger"."""
        from wukong.sandbox import conclude

        text = conclude(
            [
                self._result("control-allow-default", True),
                self._result("allow-default-deny-network", False),
                self._result("deny-default-alone", False),
            ]
        )
        assert "cannot install any restrictive profile" in text
        assert "deny default" not in text.lower().split("cannot")[0]

    def test_deny_default_alone_is_identified_as_the_trigger(self) -> None:
        from wukong.sandbox import conclude

        text = conclude(
            [
                self._result("control-allow-default", True),
                self._result("allow-default-deny-network", True),
                self._result("deny-default-alone", False),
                self._result("generated-full-profile", False),
            ]
        )
        assert "`(deny default)` is refused" in text
        assert "explicit denies" in text

    def test_a_working_deny_default_points_at_the_generated_profile(self) -> None:
        from wukong.sandbox import conclude

        text = conclude(
            [
                self._result("control-allow-default", True),
                self._result("allow-default-deny-network", True),
                self._result("deny-default-alone", True),
                self._result("generated-full-profile", False),
            ]
        )
        assert "specific problem rather than a structural one" in text

    def test_an_escape_failure_blocks_adopting_the_fallback(self) -> None:
        """Applying is not enough; it has to still block the write."""
        from wukong.sandbox import conclude

        text = conclude(
            [
                self._result("control-allow-default", True),
                self._result("allow-default-deny-network", True),
                self._result("deny-default-alone", True),
                self._result("generated-full-profile", True),
                self._result("allow-default-deny-write-escape", True, live_ok=False),
            ]
        )
        assert "must not be adopted" in text

    def test_everything_applying_is_reported_as_environment_specific(self) -> None:
        from wukong.sandbox import conclude

        text = conclude(
            [
                self._result("control-allow-default", True),
                self._result("allow-default-deny-network", True),
                self._result("deny-default-alone", True),
                self._result("generated-full-profile", True),
                self._result("allow-default-deny-write-escape", True, live_ok=True),
            ]
        )
        assert "environment-specific" in text


class TestBisectFailureModes:
    """A malformed profile and a refused profile must not read the same.

    Capability-based rather than platform-based: where `sandbox-exec` is
    absent the check is "the failure is still reported", and where it is
    present the check is the stronger one. A platform branch would make these
    pass vacuously on the runner.
    """

    def _seatbelt_available(self) -> bool:
        from wukong.sandbox.diagnose import SEATBELT_BIN

        return Path(SEATBELT_BIN).exists()

    def test_a_malformed_profile_is_labelled_as_such(self) -> None:
        from wukong.sandbox import Candidate, run_candidate

        result = run_candidate(
            Candidate(
                key="broken",
                question="?",
                profile="(version 1)\n(deny default)\n(allow nonsense-op)\n",
            )
        )
        assert not result.ok
        assert result.detail
        if self._seatbelt_available():
            # rc=65 with a parse message, never a bare refusal.
            assert "MALFORMED" in result.detail or "unbound variable" in result.detail

    def test_an_unclosed_paren_is_reported_as_syntax(self) -> None:
        from wukong.sandbox import Candidate, run_candidate

        result = run_candidate(
            Candidate(key="unclosed", question="?", profile="(version 1)\n(deny default\n")
        )
        assert not result.ok
        if self._seatbelt_available():
            assert "syntax" in result.detail.lower() or "MALFORMED" in result.detail

    def test_a_valid_profile_is_not_labelled_malformed(self) -> None:
        """The distinction has to hold in both directions: a profile refused
        for permission reasons must not be reported as malformed."""
        from wukong.sandbox import Candidate, run_candidate

        result = run_candidate(
            Candidate(key="valid", question="?", profile="(version 1)\n(deny default)\n")
        )
        assert "MALFORMED" not in result.detail

    def test_every_candidate_produces_a_result(self, tmp_path: Path) -> None:
        from wukong.sandbox import candidates, run_candidate

        for candidate in candidates(tmp_path / "ws", tmp_path / "home"):
            result = run_candidate(candidate)
            assert result.key == candidate.key
            assert result.detail, "a failure must always say something"
