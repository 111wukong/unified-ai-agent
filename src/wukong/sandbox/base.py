"""Sandbox backends.

The threat model is copied from the macOS Seatbelt profiles used by Codex
and Gemini CLI, because it is the right one for this product:

    **anti-tampering, not anti-exfiltration.**

Reads stay wide open -- an agent that cannot read the toolchain, the config
and the code is useless. What is locked down is *writes*. That is why the
Seatbelt profile denies by default but allows `file-read*` wholesale, and
why the write allowlist is the actual security boundary.

Modes, mirroring the two profiles those tools ship:

* `read-only`      -- the project tree (including `.git`) is not writable.
                      `git log/diff/show/blame` work; `commit/checkout/
                      fetch` and file edits fail. Use for "analyse this".
* `workspace-write` -- the project tree is writable, plus caches and temp.
                      The normal development mode.
* `full`           -- no sandbox. Path fence and command guard still apply.

## Availability is probed, never assumed

`available` runs a real probe, because the binary existing does not mean it
works. Two cases where it does not:

* a restrictive profile cannot be applied from inside an already-sandboxed
  process (`sandbox_apply: Operation not permitted`) -- so this backend is
  unavailable in any container or nested-sandbox environment;
* the Docker daemon may be installed but not running.

A backend that silently does nothing is worse than no backend, because the
user believes they are protected. So every fallback records a
`fallback_reason`, and `wukong sandbox` prints it.

Honest limitations, to be shown wherever this is offered:

* Network is open (the agent must reach its model API). This is **not** an
  egress firewall; a prompt-injected command can still phone home.
* Login flows fail inside the sandbox -- `npm login`, `gh auth login`,
  `aws configure` all *write* credential files. Run them outside.
* On macOS, keychain writes via Mach IPC are not blocked.
* `sandbox-exec` is deprecated by Apple. The Seatbelt subsystem is fully
  functional today, but if it is ever removed this backend stops working.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from pathlib import Path


class SandboxMode(str, Enum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    FULL = "full"


@dataclass
class ProbeResult:
    ok: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok


class Sandbox:
    """Base class. `probe()` must actually try the thing.

    Deliberately a plain class, not a dataclass. As a dataclass, the field
    defaults for `name` and `isolation` are assigned by `__init__` and
    *shadow* the subclass class attributes -- so every backend reported
    `name="none"` even while it was actively isolating. A sandbox that works
    but says it does not is as damaging as one that says it does and does
    not: it teaches the user to distrust the report.
    """

    name: str = "none"
    isolation: str = "none"

    def __init__(self) -> None:
        self.fallback_reason: str | None = None
        self._probe: ProbeResult | None = None

    def probe(self) -> ProbeResult:
        return ProbeResult(True, "no isolation to verify")

    @property
    def available(self) -> bool:
        if self._probe is None:
            self._probe = self.probe()
        return self._probe.ok

    @property
    def probe_detail(self) -> str:
        if self._probe is None:
            self._probe = self.probe()
        return self._probe.detail

    def wrap(
        self,
        argv: list[str],
        *,
        workspace: Path,
        mode: SandboxMode,
        env: dict[str, str],
    ) -> list[str]:
        return argv

    def describe(self) -> str:
        return f"{self.name} (isolation: {self.isolation})"

    def caveats(self) -> list[str]:
        return []


class NoSandbox(Sandbox):
    """No process isolation. The path fence and command guard are separate
    layers and are *not* disabled by this -- they still run."""

    name = "none"
    isolation = "none"

    def __init__(self) -> None:
        super().__init__()

    def caveats(self) -> list[str]:
        return [
            "no process isolation: a command can write anywhere your user can",
            "the path fence and command guard still apply, but they are "
            "string-level checks, not kernel enforcement",
            "on macOS enable Seatbelt with `wukong config set sandbox.backend seatbelt`",
        ]


# ---------------------------------------------------------------------------
# macOS Seatbelt
# ---------------------------------------------------------------------------

_SEATBELT_BIN = "/usr/bin/sandbox-exec"

# Always writable, regardless of mode: caches, temp and device nodes.
_ALWAYS_WRITABLE_DIRS = [
    "/private/tmp",
    "/private/var/tmp",
    "~/Library/Caches",
    "~/.cache",
    "~/.npm",
    "~/.cargo",
    "~/.gradle",
    "~/.m2",
    "~/.bun",
    "~/.deno",
    "~/.local/share/uv",
    "~/.local/state",
]

# Writable only in workspace-write and full: toolchains and agent state.
_WORKSPACE_MODE_EXTRA_DIRS = [
    "~/.nvm",
    "~/.pyenv",
    "~/.volta",
    "~/go",
    "~/.claude",
    "~/.codex",
    "~/.config/openai",
    "~/.config/anthropic",
    "~/.opencode",
    "~/.gemini",
    "~/.qwen",
]

# Readable but never writable: credential stores. Read access keeps
# git-over-ssh and the AWS SDK working; write access would let a
# prompt-injected command persist itself.
_READ_ONLY_SECRETS = [
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "~/.docker",
    "~/.kube",
    "~/.config/gcloud",
    "~/.azure",
    "~/.config/gh",
    "~/.netrc",
    "~/.npmrc",
    "~/.pypirc",
    "~/.gem/credentials",
    "~/Library/Keychains",
]

# ...with one documented exception: ssh must be able to record new host keys.
_SECRET_EXCEPTIONS = ["~/.ssh/known_hosts"]

_WRITABLE_LITERALS = [
    "/dev/null",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/dtracehelper",
    "/dev/tty",
]

# A minimal restrictive profile. If this cannot be applied, no useful
# profile can be.
_PROBE_PROFILE = "(version 1)\n(deny default)\n(allow process*)\n"


def _sb_string(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _expand(path: str) -> str:
    return str(Path(path).expanduser())


#: Environment variables that mean "a parent process is sandboxing us".
#: Their presence is the difference between "Seatbelt does not work on this
#: machine" and "Seatbelt cannot be tested from here", and those two
#: conclusions lead to opposite decisions.
SANDBOX_MARKER_PREFIXES = ("CODEBUDDY_", "SANDBOX_CENTER_", "WORKBUDDY_")

#: Markers that name the sandbox directly, rather than merely coming from the
#: app that applies it. Listed first: there are dozens of `CODEBUDDY_*`
#: variables, and an alphabetical cut would hide the two that actually matter.
SPECIFIC_MARKER_PREFIXES = ("CODEBUDDY_SANDBOX_", "SANDBOX_CENTER_", "WORKBUDDY_FS_")


def environment_fingerprint() -> dict[str, Any]:
    """Where was this run? The probe result is meaningless without it.

    A restrictive-profile probe fails for two very different reasons: the host
    refuses it, or the process is already sandboxed and cannot install a
    narrower one. Reporting "seatbelt: unavailable" without saying which is
    how a reader concludes the feature is broken when the answer is "run this
    somewhere else".
    """
    found = [k for k in os.environ if k.startswith(SANDBOX_MARKER_PREFIXES)]
    markers = sorted(found, key=lambda k: (not k.startswith(SPECIFIC_MARKER_PREFIXES), k))
    parents: list[str] = []
    try:
        pid = os.getpid()
        for _ in range(6):
            out = subprocess.run(
                ["ps", "-o", "comm=,ppid=", "-p", str(pid)],
                capture_output=True,
                timeout=5,
                check=False,
            )
            line = (out.stdout or b"").decode("utf-8", errors="replace").strip()
            if not line:
                break
            parts = line.rsplit(None, 1)
            parents.append(parts[0])
            if len(parts) < 2 or parts[1] == "1":
                break
            pid = int(parts[1])
    except (OSError, subprocess.SubprocessError, ValueError):
        parents = []

    return {
        "inside_parent_sandbox": bool(markers),
        "sandbox_marker_count": len(markers),
        "sandbox_markers": markers[:12],
        "term_program": os.environ.get("TERM_PROGRAM", ""),
        "shell": os.environ.get("SHELL", ""),
        "parent_chain": parents,
        "parent_chain_readable": bool(parents),
    }


def builtin_profile_probe() -> ProbeResult:
    """Can *any* restrictive profile be applied, including a system one?

    `sandbox-exec -n no-network` uses a profile Apple ships. If that fails the
    same way a generated one does, the problem is not the generated profile --
    which rules out the most likely false conclusion.
    """
    if sys.platform != "darwin" or not Path(_SEATBELT_BIN).exists():
        return ProbeResult(False, "not applicable")
    try:
        result = subprocess.run(
            [_SEATBELT_BIN, "-n", "no-network", "/usr/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ProbeResult(False, f"{type(exc).__name__}: {exc}")
    if result.returncode == 0:
        return ProbeResult(True, "the system's own no-network profile applied")
    stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    return ProbeResult(False, stderr or f"exit {result.returncode}")


def seatbelt_probe() -> ProbeResult:
    """Can a *restrictive* profile actually be applied here?

    `(allow default)` profiles apply anywhere, so probing with one would
    always succeed and prove nothing. The probe must be restrictive, because
    the failure mode is precisely "a narrowing profile is refused".
    """
    if sys.platform != "darwin":
        return ProbeResult(False, f"not macOS (platform={sys.platform})")
    if not Path(_SEATBELT_BIN).exists():
        return ProbeResult(False, f"{_SEATBELT_BIN} not found")
    with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as fh:
        fh.write(_PROBE_PROFILE)
        probe_path = fh.name
    try:
        result = subprocess.run(
            [_SEATBELT_BIN, "-f", probe_path, "/usr/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ProbeResult(False, f"{type(exc).__name__}: {exc}")
    finally:
        with contextlib.suppress(OSError):
            os.unlink(probe_path)

    if result.returncode == 0:
        return ProbeResult(True, "restrictive profile applied successfully")
    return ProbeResult(
        False,
        describe_probe_failure(
            returncode=result.returncode,
            stdout=result.stdout or b"",
            stderr=result.stderr or b"",
            nested=environment_fingerprint()["inside_parent_sandbox"],
        ),
    )


def describe_probe_failure(
    *, returncode: int, stdout: bytes, stderr: bytes, nested: bool
) -> str:
    """Explain a failed probe. Pure, so it is testable on any host.

    Extracted because the message *is* the logic here: a signal death was
    previously reported as "you are inside a sandbox", which asserts a cause
    the probe had not established. Keeping this inline meant the only way to
    test it was to run on macOS and actually be sandboxed.
    """
    out = (stdout or b"").decode("utf-8", errors="replace").strip()
    err = (stderr or b"").decode("utf-8", errors="replace").strip()
    message = " | ".join(part for part in (err, out) if part)

    if not message:
        if returncode < 0:
            import signal as _signal

            try:
                message = f"killed by {_signal.Signals(-returncode).name}"
            except ValueError:
                message = f"killed by signal {-returncode}"
        else:
            message = f"exit {returncode}"

    if nested or "Operation not permitted" in message:
        return message + (
            " -- this process is already inside a sandbox, and macOS will not "
            "let it install a narrower one. Run `wukong sandbox` from a normal "
            "terminal to verify."
        )
    if returncode < 0:
        # A signal death with no output is a refusal, not evidence of nesting.
        return message + (
            " -- sandbox-exec was refused without a message. That is a refusal "
            "rather than a crash: this build will not install a restrictive "
            "profile for this process. Run `wukong sandbox` from a normal terminal "
            "to compare."
        )
    return message


class SeatbeltSandbox(Sandbox):
    """macOS `sandbox-exec`. No per-command overhead once the profile exists."""

    name = "seatbelt"
    isolation = "process (write-allowlist)"

    def __init__(self, *, home: Path, extra_write_dirs: list[str] | None = None) -> None:
        super().__init__()
        self.home = Path(home)
        self.extra_write_dirs = extra_write_dirs or []
        self._profiles: dict[tuple[str, str], Path] = {}

    def probe(self) -> ProbeResult:
        return seatbelt_probe()

    # -- profile generation ----------------------------------------------
    def profile_path(self, mode: SandboxMode, workspace: Path) -> Path:
        key = (mode.value, str(workspace))
        cached = self._profiles.get(key)
        if cached and cached.exists():
            return cached
        target_dir = self.home / "sandbox"
        target_dir.mkdir(parents=True, exist_ok=True)
        digest = abs(hash(str(workspace))) % 10**8
        profile = target_dir / f"{mode.value}-{digest}.sb"
        profile.write_text(self.render_profile(mode, workspace), encoding="utf-8")
        self._profiles[key] = profile
        return profile

    def writable_paths(self, mode: SandboxMode, workspace: Path) -> list[str]:
        paths = [_expand(p) for p in _ALWAYS_WRITABLE_DIRS]
        if mode is not SandboxMode.READ_ONLY:
            paths.append(str(Path(os.path.realpath(workspace))))
            paths += [_expand(p) for p in _WORKSPACE_MODE_EXTRA_DIRS]
            paths += [_expand(p) for p in self.extra_write_dirs]
        return sorted(set(paths))

    def render_profile(self, mode: SandboxMode, workspace: Path) -> str:
        lines = [
            "(version 1)",
            ";; Generated by wukong.",
            ";; Threat model: anti-tampering, not anti-exfiltration --",
            ";; reads are open, writes are allowlisted.",
            "(deny default)",
            "",
            ";; process, ipc and read: permissive on purpose. A sandbox that",
            ";; breaks the toolchain gets turned off, and then it protects nothing.",
            "(allow process*)",
            "(allow signal (target self))",
            "(allow sysctl-read)",
            "(allow mach-lookup)",
            "(allow ipc-posix-shm)",
            "(allow system-socket)",
            "(allow network*)",
            "(allow file-read*)",
            "",
            f";; writes: allowlist only (mode: {mode.value})",
            "(allow file-write*",
        ]
        for path in self.writable_paths(mode, workspace):
            lines.append(f"  (subpath {_sb_string(path)})")
        for path in sorted(set(_expand(p) for p in _SECRET_EXCEPTIONS) | set(_WRITABLE_LITERALS)):
            lines.append(f"  (literal {_sb_string(path)})")
        lines.append(")")
        lines.append("")
        lines.append(";; credential stores are readable but not writable, via the")
        lines.append(";; default deny above. Listed for auditability:")
        for path in _READ_ONLY_SECRETS:
            lines.append(f";;   {_expand(path)}")
        lines.append("")
        return "\n".join(lines)

    def wrap(self, argv, *, workspace, mode, env):  # noqa: ANN001, ANN201
        if mode is SandboxMode.FULL or not self.available:
            return argv
        profile = self.profile_path(mode, workspace)
        return [_SEATBELT_BIN, "-f", str(profile), *argv]

    def caveats(self) -> list[str]:
        return [
            "network is open: this is not an egress firewall",
            "`npm login` / `gh auth login` / `aws configure` fail inside -- run them outside",
            "keychain writes via Mach IPC are not blocked",
            "`sandbox-exec` is deprecated by Apple (the Seatbelt subsystem still works)",
        ]


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


class DockerSandbox(Sandbox):
    """Container isolation. Honest about the cost: on macOS every command
    pays the VM round-trip, which is why this is not the default there."""

    name = "docker"
    isolation = "container (shared kernel)"

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        network: str = "none",
        extra_mounts: list[str] | None = None,
        docker_bin: str | None = None,
    ) -> None:
        super().__init__()
        self.image = image
        self.network = network
        self.extra_mounts = extra_mounts or []
        self.docker_bin = docker_bin or shutil.which("docker") or "docker"

    def probe(self) -> ProbeResult:
        if shutil.which(self.docker_bin) is None and not Path(self.docker_bin).exists():
            return ProbeResult(False, f"{self.docker_bin} not on PATH")
        try:
            result = subprocess.run(
                [self.docker_bin, "info"], capture_output=True, timeout=15, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ProbeResult(False, f"{type(exc).__name__}: {exc}")
        if result.returncode == 0:
            return ProbeResult(True, "daemon reachable")
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        return ProbeResult(False, f"daemon not reachable: {stderr[:200] or 'docker info failed'}")

    def wrap(self, argv, *, workspace, mode, env):  # noqa: ANN001, ANN201
        if mode is SandboxMode.FULL or not self.available:
            return argv
        workspace = Path(os.path.realpath(workspace))
        cmd = [
            self.docker_bin,
            "run",
            "--rm",
            "-i",
            "--network",
            self.network,
            "-v",
            f"{workspace}:/workspace",
            "-w",
            "/workspace",
        ]
        if mode is SandboxMode.READ_ONLY:
            cmd += ["--read-only", "--tmpfs", "/tmp"]
        for key in ("PATH", "LANG", "LC_ALL", "TERM", "PYTHONUNBUFFERED"):
            if value := env.get(key):
                cmd += ["-e", f"{key}={value}"]
        for mount in self.extra_mounts:
            cmd += ["-v", mount]
        return [*cmd, self.image, *argv]

    def caveats(self) -> list[str]:
        return [
            f"image `{self.image}` must contain the tools your commands need",
            f"network mode is `{self.network}`",
            "every command pays a container round-trip (seconds on macOS)",
            "paths outside the workspace are not visible",
            "pure containers share the host kernel -- use a microVM for truly "
            "untrusted code",
        ]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


@dataclass
class SandboxSelection:
    sandbox: Sandbox
    requested: str
    notes: list[str] = field(default_factory=list)

    @property
    def fell_back(self) -> bool:
        return self.requested not in ("auto", self.sandbox.name)

    def summary(self) -> str:
        lines = [f"backend: {self.sandbox.describe()}"]
        if self.fell_back:
            lines.append(
                f"requested `{self.requested}` but fell back to "
                f"`{self.sandbox.name}`: {self.sandbox.probe_detail}"
            )
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)

    def warning(self) -> str | None:
        """A one-line warning when isolation is weaker than requested.

        `wrap()` passes argv through unchanged when the backend is
        unavailable. That is the right runtime behaviour -- a broken sandbox
        must not brick every command -- but it means the user's configured
        isolation silently does not apply. So the caller has to say so.
        """
        if self.requested == "none":
            return None
        if self.sandbox.name == self.requested and self.sandbox.available:
            return None
        if self.sandbox.name == "none":
            return (
                f"sandbox requested ({self.requested}) is unavailable -- "
                "commands are running unsandboxed. Run `wukong sandbox` for details."
            )
        if self.fell_back:
            return (
                f"sandbox fell back from {self.requested} to {self.sandbox.name}: "
                f"{self.sandbox.probe_detail}"
            )
        return None


def build_sandbox(
    backend: str = "auto",
    *,
    home: Path,
    extra_write_dirs: list[str] | None = None,
    docker_image: str = "python:3.12-slim",
    docker_network: str = "none",
    docker_mounts: list[str] | None = None,
) -> SandboxSelection:
    """Pick a backend. `auto` prefers Seatbelt on macOS: it is free per
    command, and a sandbox you keep enabled protects more than a stronger
    one you turn off.

    The fallback is *recorded*, never silent.
    """
    if backend not in ("auto", "seatbelt", "docker", "none"):
        raise ValueError(f"unknown sandbox backend {backend!r}: auto|seatbelt|docker|none")

    notes: list[str] = []
    if backend in ("auto", "seatbelt"):
        seatbelt = SeatbeltSandbox(home=home, extra_write_dirs=extra_write_dirs)
        if seatbelt.available:
            if backend == "seatbelt":
                return SandboxSelection(seatbelt, backend, notes)
            return SandboxSelection(seatbelt, backend, notes)
        notes.append(f"seatbelt unavailable: {seatbelt.probe_detail}")

    if backend in ("auto", "docker"):
        docker = DockerSandbox(
            image=docker_image, network=docker_network, extra_mounts=docker_mounts
        )
        if docker.available:
            return SandboxSelection(docker, backend, notes)
        notes.append(f"docker unavailable: {docker.probe_detail}")

    if backend == "none":
        return SandboxSelection(NoSandbox(), backend, notes)

    notes.append(
        "no sandbox backend could be enabled -- commands run unsandboxed. "
        "The path fence and command guard still apply."
    )
    return SandboxSelection(NoSandbox(), backend, notes)


__all__ = [
    "Sandbox",
    "SandboxMode",
    "SandboxSelection",
    "ProbeResult",
    "NoSandbox",
    "SeatbeltSandbox",
    "DockerSandbox",
    "SANDBOX_MARKER_PREFIXES",
    "SPECIFIC_MARKER_PREFIXES",
    "build_sandbox",
    "builtin_profile_probe",
    "describe_probe_failure",
    "environment_fingerprint",
    "seatbelt_probe",
]
