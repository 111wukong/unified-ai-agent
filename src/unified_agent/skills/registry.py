"""Skill discovery and lifecycle.

Status ladder (from the spec, kept because it is the right shape):

    candidate -> validated -> approved -> active -> deprecated

Only `active` skills appear in the prompt index and can be loaded. The
ladder exists so that a skill the agent wrote for itself cannot become
executable without a human moving it forward -- the "禁止自动升级为正式技能"
requirement, enforced by state rather than by convention.

**Where the status actually comes from** (this used to be the hole):

    DB row          >  frontmatter `metadata.status`  >  per-directory default

The DB row wins because `promote` is the only thing that writes one, and
`promote` is a human action. The per-directory default is `candidate` under
the agent's own candidate root and `active` everywhere else, and that
ordering matters: **the candidate root is the one directory an agent can
write to, so a file there must not be able to promote itself.** A skill
cannot become runnable by editing its own frontmatter.

Discovery covers the candidate root as well as the authored directories.
Leaving it out of the scan (the previous behaviour) did keep candidates
unrunnable, but it also made them invisible -- the human gate had nothing
to gate, because nobody could see the thing waiting for review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from unified_agent.skills.loader import (
    SecurityReport,
    Skill,
    SkillError,
    parse_skill_file,
    review_skill,
)

STATUS_ORDER = ["candidate", "validated", "approved", "active", "deprecated"]


@dataclass
class LoadResult:
    loaded: list[Skill]
    errors: list[tuple[Path, str]]
    reports: list[SecurityReport] = field(default_factory=list)


class SkillRegistry:
    def __init__(
        self,
        dirs: list[Path] | None = None,
        *,
        candidate_dirs: list[Path] | None = None,
        status_overrides: dict[str, str] | None = None,
        on_promote: Callable[[Skill], None] | None = None,
    ) -> None:
        self.dirs = [Path(d) for d in (dirs or [])]
        # The agent's own output. Scanned so it can be reviewed, but every
        # skill found here starts at `candidate` no matter what it claims.
        self.candidate_dirs = [Path(d) for d in (candidate_dirs or [])]
        self.status_overrides = dict(status_overrides or {})
        self.on_promote = on_promote
        self._skills: dict[str, Skill] = {}

    # -- discovery --------------------------------------------------------
    def discover(self, *, known_tools: set[str] | None = None) -> LoadResult:
        loaded: list[Skill] = []
        errors: list[tuple[Path, str]] = []
        reports: list[SecurityReport] = []

        for base, is_candidate in self._scan_roots():
            if not base.is_dir():
                continue
            source = "candidate" if is_candidate else ("local" if base.name == "skills" else "user")
            for skill_md in sorted(base.glob("*/SKILL.md")):
                try:
                    skill = parse_skill_file(skill_md, source=source)
                except SkillError as exc:
                    errors.append((skill_md, str(exc)))
                    continue
                skill.status = self._resolve_status(skill, is_candidate=is_candidate)
                report = review_skill(skill, known_tools=known_tools)
                reports.append(report)
                if report.blocked:
                    errors.append((skill_md, f"blocked by security review: {report.worst}"))
                    continue
                self._skills[skill.name] = skill
                loaded.append(skill)
        return LoadResult(loaded=loaded, errors=errors, reports=reports)

    def _scan_roots(self) -> list[tuple[Path, bool]]:
        roots = [(d, False) for d in self.dirs]
        roots += [(d, True) for d in self.candidate_dirs]
        return roots

    def _resolve_status(self, skill: Skill, *, is_candidate: bool) -> str:
        explicit = self.status_overrides.get(skill.name)
        if explicit in STATUS_ORDER:
            return explicit
        if is_candidate:
            # Deliberately ignores the file's own claim. See the module docstring.
            return "candidate"
        declared = skill.metadata.get("status")
        return declared if declared in STATUS_ORDER else "active"

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    # -- access -----------------------------------------------------------
    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list(self, *, status: str | None = None) -> list[Skill]:
        items = sorted(self._skills.values(), key=lambda s: s.name)
        if status:
            items = [s for s in items if s.status == status]
        return items

    def active(self) -> list[Skill]:
        return self.list(status="active")

    def names(self) -> list[str]:
        return [s.name for s in self.active()]

    def pending_review(self) -> list[Skill]:
        """Skills that need a human decision before they can run.

        `deprecated` is excluded: retiring something is a decision already
        taken, not work waiting to be done. Ordered by how close each one is
        to running, so a review that is nearly finished gets finished.
        """
        order = {status: i for i, status in enumerate(STATUS_ORDER)}
        return sorted(
            (s for s in self._skills.values() if s.status not in {"active", "deprecated"}),
            key=lambda s: (-order.get(s.status, 99), s.name),
        )

    # -- prompt surface ---------------------------------------------------
    def index_prompt(self) -> str:
        """Metadata-only index. Bodies are pulled via the load_skill tool."""
        skills = self.active()
        if not skills:
            return ""
        lines = [
            "# Available skills",
            "",
            "Each skill below is a vetted procedure. When a task matches a skill's",
            "description, call `load_skill` with its name *before* improvising.",
            "",
        ]
        lines += [s.index_line() for s in skills]
        return "\n".join(lines)

    # -- lifecycle --------------------------------------------------------
    def promote(self, name: str, to: str) -> Skill:
        """Move a skill along the ladder. One rung at a time.

        Skipping is refused, not just moving backwards. The whole point of
        `candidate -> validated -> approved -> active` is that each rung is a
        separate judgement (does it parse / is it correct / does a human
        accept it / may it run). Allowing `candidate -> active` would make
        the intermediate rungs decorative.

        `deprecated` is reachable from anywhere -- retiring something must
        never be blocked by process.

        The new status is handed to `on_promote` so it can outlive the
        process. Without that the ladder is a variable: it resets to
        whatever the directory implies on the next run, which is how this
        was broken before.
        """
        skill = self.get(name)
        if skill is None:
            raise SkillError(f"unknown skill {name!r}")
        if to not in STATUS_ORDER:
            raise SkillError(f"unknown status {to!r}; expected one of {STATUS_ORDER}")
        if to == "deprecated":
            skill.status = to
            self._persist(skill)
            return skill

        current, target = STATUS_ORDER.index(skill.status), STATUS_ORDER.index(to)
        if target < current:
            raise SkillError(
                f"cannot move {name!r} backwards from {skill.status!r} to {to!r}"
            )
        if target > current + 1:
            raise SkillError(
                f"cannot skip from {skill.status!r} to {to!r}; the ladder is "
                f"{' -> '.join(STATUS_ORDER[:4])} and each rung needs its own review"
            )
        skill.status = to
        self._persist(skill)
        return skill

    def _persist(self, skill: Skill) -> None:
        if self.on_promote is not None:
            self.on_promote(skill)


def status_table(skills: list[Any]) -> list[tuple[str, str, str]]:
    """(name, status, note) rows for a terminal listing."""
    rows: list[tuple[str, str, str]] = []
    for skill in skills:
        note = "agent-written" if skill.source == "candidate" else skill.source
        rows.append((skill.name, skill.status, note))
    return rows
