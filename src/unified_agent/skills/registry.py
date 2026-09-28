"""Skill discovery and lifecycle.

Status ladder (from the spec, kept because it is the right shape):

    candidate -> validated -> approved -> active -> deprecated

Only `active` skills appear in the prompt index and can be loaded. The
ladder exists so that a skill the agent wrote for itself cannot become
executable without a human moving it forward -- the "禁止自动升级为正式技能"
requirement, enforced by state rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    reports: list[SecurityReport]


class SkillRegistry:
    def __init__(self, dirs: list[Path] | None = None) -> None:
        self.dirs = [Path(d) for d in (dirs or [])]
        self._skills: dict[str, Skill] = {}

    # -- discovery --------------------------------------------------------
    def discover(self, *, known_tools: set[str] | None = None) -> LoadResult:
        loaded: list[Skill] = []
        errors: list[tuple[Path, str]] = []
        reports: list[SecurityReport] = []

        for base in self.dirs:
            if not base.is_dir():
                continue
            source = "local" if base.name == "skills" else "user"
            for skill_md in sorted(base.glob("*/SKILL.md")):
                try:
                    skill = parse_skill_file(skill_md, source=source, status="active")
                except SkillError as exc:
                    errors.append((skill_md, str(exc)))
                    continue
                report = review_skill(skill, known_tools=known_tools)
                reports.append(report)
                if report.blocked:
                    errors.append((skill_md, f"blocked by security review: {report.worst}"))
                    continue
                self._skills[skill.name] = skill
                loaded.append(skill)
        return LoadResult(loaded=loaded, errors=errors, reports=reports)

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

    def allowed_tools_for(self, names: list[str]) -> set[str]:
        out: set[str] = set()
        for name in names:
            skill = self.get(name)
            if skill and skill.status == "active":
                out |= skill.allowed_tools
        return out

    def tool_catalog_block(self, name: str) -> str:
        skill = self.get(name)
        return f"\n\n# Skill: {skill.name}\n{skill.render()}" if skill else ""

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
        """
        skill = self.get(name)
        if skill is None:
            raise SkillError(f"unknown skill {name!r}")
        if to not in STATUS_ORDER:
            raise SkillError(f"unknown status {to!r}; expected one of {STATUS_ORDER}")
        if to == "deprecated":
            skill.status = to
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
        return skill
