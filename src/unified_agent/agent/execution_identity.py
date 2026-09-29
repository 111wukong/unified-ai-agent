"""What was approved is what runs.

Between the moment a human approves a call and the moment it executes, the
world can change. PATH can gain a directory, a `pytest` earlier on it can be
replaced, the file a patch targets can be edited by someone else. The approval
was for a *specific* thing, so the thing is fingerprinted when the request is
raised and checked again immediately before it runs. A mismatch is refused
rather than executed.

This is the problem OpenClaw solves by binding cwd, argv, env and the resolved
executable's real path -- plus a content hash for writable files -- to the
approval. The version here is deliberately narrower, covering the two places
where "what you approved" and "what actually happens" can differ in this
runtime:

* **a command**, whose `argv[0]` is resolved through PATH at execution time,
  so the binary that runs need not be the one that was on PATH when the request
  was shown; and
* **a file about to be overwritten**, which someone else may have changed while
  the request sat waiting for an answer.

It is not a general sandbox. It closes the gap between the approval and the
execution, which is the one gap an approval gate creates by existing.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

#: Tools whose arguments come from the model and name an executable to run.
COMMAND_TOOLS = frozenset({"run_command"})
#: Tools that replace a file's contents outright.
FILE_TOOLS = frozenset({"write_file", "apply_patch"})


@dataclass(frozen=True)
class ExecutionIdentity:
    """A fingerprint of the thing an approval is about."""

    kind: str  # "executable" | "file"
    target: str
    digest: str
    #: Shown in the approval prompt, so the human sees *which* binary or file
    #: they are approving rather than only the command text.
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "target": self.target,
            "digest": self.digest,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ExecutionIdentity | None:
        if not payload or not payload.get("target"):
            return None
        return cls(
            kind=str(payload.get("kind", "")),
            target=str(payload["target"]),
            digest=str(payload.get("digest", "")),
            detail=str(payload.get("detail", "")),
        )


def _digest_file(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _resolve(raw: str, workspace: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = workspace / path
    return Path(os.path.realpath(path))


def fingerprint(
    tool_name: str, arguments: dict, *, workspace: Path, env: dict[str, str]
) -> ExecutionIdentity | None:
    """Fingerprint what this call will act on, or None if there is nothing to bind.

    Returning None is not a failure: a read-only call has nothing to drift, and
    a command whose executable cannot be resolved *now* cannot be resolved
    later either, so there is nothing to compare.
    """
    if tool_name in COMMAND_TOOLS:
        command = arguments.get("command")
        if not isinstance(command, str):
            return None
        try:
            argv = shlex.split(command)
        except ValueError:
            return None
        if not argv:
            return None
        resolved = shutil.which(argv[0], path=env.get("PATH"))
        if not resolved:
            return None
        path = Path(os.path.realpath(resolved))
        try:
            digest = _digest_file(path)
        except OSError:
            return None
        return ExecutionIdentity(
            kind="executable",
            target=str(path),
            digest=digest,
            detail=f"runs {path}",
        )

    if tool_name in FILE_TOOLS:
        raw = arguments.get("path")
        if not isinstance(raw, str) or not raw:
            return None
        path = _resolve(raw, workspace)
        if not path.exists():
            # Creating a file has no pre-image to protect, and the approval is
            # for the content, which the model already fixed.
            return None
        try:
            content = path.read_bytes()
        except OSError:
            return None
        return ExecutionIdentity(
            kind="file",
            target=str(path),
            digest=hashlib.sha256(content).hexdigest()[:16],
            detail=f"replaces {path.name}",
        )

    return None


def verify(
    identity: ExecutionIdentity, *, workspace: Path, env: dict[str, str]
) -> str | None:
    """Why the approved thing is no longer what would run, or None if it is."""
    if identity.kind == "executable":
        path = Path(identity.target)
        if not path.exists():
            return (
                f"{path} is gone. It was on PATH when this was approved, so "
                "what would run now is something else."
            )
        try:
            current = _digest_file(path)
        except OSError as exc:
            return f"{path} could not be re-read: {exc}"
        if current != identity.digest:
            return (
                f"{path} changed since this was approved "
                f"(was {identity.digest}, now {current}). The binary that would "
                "run is not the one you looked at."
            )
        return None

    if identity.kind == "file":
        path = Path(identity.target)
        if not path.exists():
            return (
                f"{path} was deleted since this was approved. The change you "
                "approved would now create it instead of replacing it."
            )
        try:
            content = path.read_bytes()
        except OSError as exc:
            return f"{path} could not be re-read: {exc}"
        current = hashlib.sha256(content).hexdigest()[:16]
        if current != identity.digest:
            return (
                f"{path} was edited after this was approved "
                f"(was {identity.digest}, now {current}). Applying the change "
                "would overwrite an edit nobody reviewed."
            )
        return None

    return None


__all__ = [
    "COMMAND_TOOLS",
    "FILE_TOOLS",
    "ExecutionIdentity",
    "fingerprint",
    "verify",
]
