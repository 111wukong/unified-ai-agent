"""Diagnose why a Seatbelt profile is refused, on a machine we cannot reach.

The situation this exists for: the profile is refused on the user's machine
and not on ours, so the experiment has to be shipped rather than run. A
bisect is the right experiment, but a bisect that only isolates the cause
still leaves the user waiting for a second round trip -- so this one also
tests the *candidate fix* end to end, live write included.

What the failure modes mean, established by experiment:

* ``rc=65`` plus a message -- the profile is malformed. ``sandbox-exec``
  reports ``unbound variable: X`` or ``syntax error: expecting ')'``. A
  parse problem is never a permission problem, so these two are
  distinguishable and the distinction is worth keeping.
* ``rc=71`` / ``SIGABRT`` with no usable message -- the profile parsed and was
  refused at apply time. This is the interesting case: the syntax is fine, so
  the question is which *rule* the system will not accept.
* ``rc=0`` -- applied.

The profiles below are ordered so that the first failure isolates the rule,
and the last one is the alternative construction under test.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEATBELT_BIN = "/usr/bin/sandbox-exec"

#: The probe target. `/usr/bin/true` proves the profile applies without
#: depending on anything the profile allows or denies.
PROBE_COMMAND = "/usr/bin/true"


@dataclass
class Candidate:
    key: str
    question: str
    profile: str
    #: Optional live check: (description, shell command, path that must exist)
    live: tuple[str, str, str] | None = None


@dataclass
class Result:
    key: str
    question: str
    ok: bool
    detail: str
    live: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "question": self.question,
            "ok": self.ok,
            "detail": self.detail,
            "live": self.live,
        }


def candidates(workspace: Path, home: Path) -> list[Candidate]:
    """The experiment, in the order that isolates the cause fastest."""
    scratch = Path(tempfile.gettempdir()) / "uaa-bisect-writable"
    forbidden = home / "uaa-bisect-should-not-exist"

    return [
        Candidate(
            key="control-allow-default",
            question="Can any profile be applied at all?",
            profile="(version 1)\n(allow default)\n",
        ),
        Candidate(
            key="deny-default-alone",
            question="Is `(deny default)` itself the trigger?",
            profile="(version 1)\n(deny default)\n",
        ),
        Candidate(
            key="deny-default-plus-read",
            question="Does adding one allow rule change the outcome?",
            profile="(version 1)\n(deny default)\n(allow file-read*)\n",
        ),
        Candidate(
            key="allow-default-deny-network",
            question="Does a narrowing rule on top of `allow default` apply?",
            profile="(version 1)\n(allow default)\n(deny network*)\n",
        ),
        Candidate(
            key="allow-default-deny-write",
            question="Can writes be denied globally on top of `allow default`?",
            profile="(version 1)\n(allow default)\n(deny file-write*)\n",
        ),
        Candidate(
            key="allow-default-deny-write-reallow",
            question="THE CANDIDATE FIX: deny writes, re-allow one subpath.",
            profile=(
                "(version 1)\n"
                "(allow default)\n"
                "(deny file-write*)\n"
                f'(allow file-write* (subpath "{scratch}"))\n'
            ),
            live=(
                "write inside the allowed subpath",
                f"mkdir -p {scratch} && touch {scratch}/allowed.txt",
                str(scratch / "allowed.txt"),
            ),
        ),
        Candidate(
            key="allow-default-deny-write-escape",
            question="Does the candidate fix actually block a write elsewhere?",
            profile=(
                "(version 1)\n"
                "(allow default)\n"
                "(deny file-write*)\n"
                f'(allow file-write* (subpath "{scratch}"))\n'
            ),
            live=(
                "write outside the allowed subpath",
                f"mkdir -p {forbidden.parent} && touch {forbidden}",
                str(forbidden),
            ),
        ),
        Candidate(
            key="generated-full-profile",
            question="Does the profile this project generates apply?",
            profile=_generated_profile(workspace, home),
        ),
    ]


def _generated_profile(workspace: Path, home: Path) -> str:
    """The real thing, so the bisect reproduces the reported failure."""
    from unified_agent.sandbox.base import SandboxMode, SeatbeltSandbox

    sandbox = SeatbeltSandbox(home=home)
    return sandbox.render_profile(SandboxMode.WORKSPACE_WRITE, workspace)


def run_candidate(candidate: Candidate, *, timeout_s: float = 15.0) -> Result:
    """Apply the profile and, where given, perform the live check."""
    path = Path(tempfile.mkdtemp(prefix="uaa-bisect-")) / "profile.sb"
    path.write_text(candidate.profile, encoding="utf-8")

    try:
        probe = subprocess.run(
            [SEATBELT_BIN, "-f", str(path), PROBE_COMMAND],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Result(candidate.key, candidate.question, False, f"{type(exc).__name__}: {exc}")

    ok = probe.returncode == 0
    detail = _explain(probe)
    live: list[dict[str, Any]] = []

    if ok and candidate.live is not None:
        label, command, target = candidate.live
        expected_exists = "allowed" in label
        try:
            subprocess.run(
                [SEATBELT_BIN, "-f", str(path), "/bin/sh", "-c", command],
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        exists = Path(target).exists()
        if exists and not expected_exists:
            Path(target).unlink(missing_ok=True)
        live.append(
            {
                "label": label,
                "ok": exists is expected_exists,
                "result": "created" if exists else "blocked",
                "expected": "created" if expected_exists else "blocked",
            }
        )

    return Result(candidate.key, candidate.question, ok, detail, live)


def _explain(probe: subprocess.CompletedProcess[bytes]) -> str:
    out = (probe.stdout or b"").decode("utf-8", errors="replace").strip()
    err = (probe.stderr or b"").decode("utf-8", errors="replace").strip()
    message = " | ".join(part for part in (err, out) if part)
    if probe.returncode == 0:
        return "applied"
    if probe.returncode == 65:
        return f"MALFORMED PROFILE (rc=65): {message or 'no message'}"
    if message:
        return message[:300]
    if probe.returncode < 0:
        return f"refused at apply time, killed by signal {-probe.returncode} (no message)"
    return f"refused at apply time (rc={probe.returncode}, no message)"


def diagnose(workspace: Path, home: Path) -> dict[str, Any]:
    results = [run_candidate(candidate) for candidate in candidates(workspace, home)]
    return {
        "results": [result.as_dict() for result in results],
        "conclusion": conclude(results),
    }


def conclude(results: list[Result]) -> str:
    """Turn the table into the sentence a reader needs.

    Two controls, not one. `(allow default)` passing only proves that a
    *no-op* profile is accepted; it says nothing about whether a profile that
    actually narrows something can be installed. Treating it as the control is
    how a nested environment gets misdiagnosed as "deny-default is the
    trigger" -- so `allow-default-deny-network` is the second control, and if
    that fails the answer is "this environment cannot tell us".
    """
    by_key = {result.key: result for result in results}
    control = by_key.get("control-allow-default")
    narrowing = by_key.get("allow-default-deny-network")
    deny_alone = by_key.get("deny-default-alone")
    generated = by_key.get("generated-full-profile")
    escape = by_key.get("allow-default-deny-write-escape")

    if control is None or not control.ok:
        return (
            "Even `(allow default)` was refused, so this environment cannot apply any "
            "profile at all -- it cannot diagnose the problem. Re-run from Terminal.app."
        )
    if narrowing is not None and not narrowing.ok:
        return (
            "`(allow default)` applies but a profile that actually narrows something is "
            "refused, so this environment cannot install any restrictive profile -- it "
            "cannot tell us why the generated one is refused. Re-run from Terminal.app."
        )
    if deny_alone is not None and not deny_alone.ok:
        return (
            "`(deny default)` is refused while narrowing profiles built on "
            "`(allow default)` apply, so an allowlist profile cannot be used on this "
            "machine. Build the profile as `(allow default)` plus explicit denies "
            "instead -- the two candidates above show whether that construction "
            "applies and whether it still blocks the write it should."
        )
    if generated is not None and not generated.ok:
        return (
            "`(deny default)` applies here, so the generated profile has a specific "
            "problem rather than a structural one. Compare its rules against the "
            "candidates above that did apply."
        )
    if escape is not None and escape.live and not escape.live[0]["ok"]:
        return (
            "The deny-then-re-allow construction does not block a write outside the "
            "allowlist, so it must not be adopted even though it applies."
        )
    if generated is not None and generated.ok:
        return "The generated profile applies here, so the refusal is environment-specific."
    return "Inconclusive; read the table."


__all__ = ["Candidate", "Result", "candidates", "conclude", "diagnose", "run_candidate"]
