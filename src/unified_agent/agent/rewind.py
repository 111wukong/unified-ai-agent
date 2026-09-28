"""Rewinding a task's file changes.

The gap this closes: the runtime can replay its own *state* perfectly and
still leave a broken working tree behind. It knows exactly which step
corrupted a file and cannot put the file back. Events describe what the agent
did; only a checkpoint describes what was there before.

The mechanism follows from the project's existing write-ahead discipline
rather than adding a subsystem:

    ledger(begin) -> checkpoint(pre-image) -> execute -> ledger(end)

The pre-image is captured before the tool runs, so a crash mid-write still
leaves a copy. It is stored as a verbatim artifact (not redacted -- a
redacted copy restores a corrupted file) and referenced by a
`FILE_CHECKPOINT` event, so a rewind is computed from the event log like
everything else and is auditable afterwards.

Deliberately not a shadow git repository: this has to work in a directory
that is not a git repo (the demo project that found this gap was not one),
and it must not depend on git being installed. It also means a rewind shows
up in `uaa task events` next to the write it undoes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unified_agent.observability.events import EventType


@dataclass
class RewindStep:
    """One file, and what rewinding it means."""

    path: str
    action: str  # restore | delete | already | blocked
    artifact: str | None = None
    bytes: int = 0
    current_bytes: int | None = None
    reason: str = ""

    def describe(self) -> str:
        if self.action == "blocked":
            return f"blocked   {self.path}  -- {self.reason}"
        if self.action == "already":
            return f"already   {self.path}  (unchanged since the checkpoint)"
        if self.action == "delete":
            return f"delete    {self.path}  (did not exist before the task)"
        was = f"{self.current_bytes} -> {self.bytes} bytes" if self.current_bytes is not None else f"{self.bytes} bytes"
        return f"restore   {self.path}  ({was})"


@dataclass
class RewindPlan:
    task_id: str
    steps: list[RewindStep] = field(default_factory=list)

    @property
    def actionable(self) -> list[RewindStep]:
        return [s for s in self.steps if s.action in {"restore", "delete"}]

    @property
    def blocked(self) -> list[RewindStep]:
        return [s for s in self.steps if s.action == "blocked"]

    def render(self) -> str:
        if not self.steps:
            return "nothing to rewind: this task recorded no file checkpoints"
        lines = [step.describe() for step in self.steps]
        if self.blocked:
            lines.append("")
            lines.append(
                f"{len(self.blocked)} file(s) cannot be restored. Their changes "
                "cannot be undone from this task's record."
            )
        return "\n".join(lines)


def plan_rewind(store: Any, task_id: str, *, from_seq: int = 0) -> RewindPlan:
    """Work out what a rewind would do, without doing it.

    For each file, the checkpoint that matters is the **earliest one at or
    after `from_seq`** -- that is the state of the file immediately before
    the first write in the window. Later checkpoints for the same path are
    the intermediate states, and restoring one of those would rewind only
    part of the way.

    Planning is separated from applying so `--dry-run` is the same code path
    as the real thing. A preview computed by different logic is a preview
    that can disagree with what happens.
    """
    earliest: dict[str, Any] = {}
    for event in store.events(task_id):
        if event.type is not EventType.FILE_CHECKPOINT:
            continue
        if event.seq < from_seq:
            continue
        payload = event.payload
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            continue
        # Events are in ascending seq order, so the first one seen per path
        # is the earliest in the window.
        earliest.setdefault(path, payload)

    plan = RewindPlan(task_id=task_id)
    for path, payload in sorted(earliest.items()):
        target = Path(path)
        current = _read_bytes(target)

        if not payload.get("existed"):
            if current is None:
                plan.steps.append(
                    RewindStep(path=path, action="already", reason="never created")
                )
            else:
                plan.steps.append(
                    RewindStep(path=path, action="delete", current_bytes=len(current))
                )
            continue

        if not payload.get("restorable", True) or not payload.get("artifact"):
            plan.steps.append(
                RewindStep(
                    path=path,
                    action="blocked",
                    reason=str(payload.get("reason") or "no snapshot was stored"),
                )
            )
            continue

        original = Path(str(payload["artifact"])).read_text(
            encoding="utf-8", errors="replace"
        )
        if current is not None and current == original.encode("utf-8"):
            # Someone already put it back, or the write turned out to be a
            # no-op. Either way there is nothing to do, and saying so beats
            # rewriting the file with identical bytes.
            plan.steps.append(RewindStep(path=path, action="already", bytes=len(original)))
            continue
        plan.steps.append(
            RewindStep(
                path=path,
                action="restore",
                artifact=str(payload["artifact"]),
                bytes=len(original),
                current_bytes=len(current) if current is not None else None,
            )
        )
    return plan


def apply_rewind(plan: RewindPlan) -> list[str]:
    """Write the pre-images back. Returns one line per file actually changed.

    A blocked file is left exactly as it is: a partial restore that silently
    skips a file would be worse than a rewind that reports what it could not
    do.
    """
    changed: list[str] = []
    for step in plan.actionable:
        target = Path(step.path)
        if step.action == "delete":
            target.unlink(missing_ok=True)
            changed.append(f"deleted {step.path}")
            continue
        assert step.artifact is not None  # guaranteed by plan_rewind
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            Path(step.artifact).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )
        changed.append(f"restored {step.path}")
    return changed


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = ["RewindPlan", "RewindStep", "apply_rewind", "digest", "plan_rewind"]
