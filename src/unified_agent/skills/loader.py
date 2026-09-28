"""SKILL.md loader.

Format is the published Agent Skills spec (agentskills.io), not a bespoke
`skill.yaml` + `instructions.md` pair. Two concrete payoffs:

* skills written for Claude Code / other compatible agents load here
  unchanged, and vice versa;
* `allowed-tools` already exists in the spec and maps one-to-one onto this
  runtime's permission engine, so "skills cannot silently gain privileges"
  is a spec-level guarantee rather than a feature we had to invent.

Spec rules enforced: name (1-64, lowercase alnum + hyphen, no leading/
trailing/consecutive hyphens, must equal the directory name), description
(1-1024), compatibility (<=500), metadata (map of string -> string).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Prompt-injection and exfiltration shapes. Not exhaustive and not a
# sandbox -- it is a tripwire that forces an explicit human review.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "instruction override"),
    (r"(?i)disregard\s+(the\s+)?(system|previous)", "instruction override"),
    (r"忽略(之前|上面|以上|先前)的?(所有)?(指令|指示|规则)", "instruction override"),
    (r"(?i)you\s+are\s+now\s+(a|an|the)\b", "persona override"),
    (r"(?i)(exfiltrat|upload|send)\s+.*(\.env|credential|secret|api[_-]?key|token)", "credential exfiltration"),
    (r"(?i)\.ssh/id_|\.aws/credentials|\.netrc", "credential path access"),
    (r"(?i)(curl|wget|fetch)\s+https?://[^\s]+.*\|\s*(ba)?sh", "remote code execution"),
    (r"(?i)\brm\s+-rf\s+[/~]", "destructive command"),
    (r"(?i)base64\s+-d\s*\|", "obfuscated payload"),
    (r"(?i)do\s+not\s+(tell|inform|mention\s+to)\s+the\s+user", "concealment from user"),
    (r"不要(告诉|告知|通知)用户", "concealment from user"),
]

_DANGEROUS_SCRIPT_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-rf\b", "recursive delete"),
    (r"\bsudo\b", "privilege escalation"),
    (r"\bcurl\b[^|]*\|\s*(ba)?sh", "pipe-to-shell"),
    (r"\bwget\b[^|]*\|\s*(ba)?sh", "pipe-to-shell"),
    (r"(?i)os\.system\s*\(|subprocess\.(call|run|Popen)\s*\(", "subprocess execution"),
    (r"(?i)\beval\s*\(|\bexec\s*\(", "dynamic code execution"),
    (r"(?i)requests\.(post|put)\s*\(", "outbound network call"),
    (r"(?i)open\s*\(\s*['\"](~|/etc|/Users)", "absolute-path file access"),
]


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path
    source: str = "local"
    status: str = "active"
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: set[str] = field(default_factory=set)
    frontmatter: dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()[:16]

    def render(self, *, max_chars: int = 20_000) -> str:
        body = self.body
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n\n[... skill body truncated at {max_chars} chars]"
        header = f"# Skill: {self.name}\n\n{self.description}\n"
        if self.allowed_tools:
            header += f"\nPre-approved tools: {', '.join(sorted(self.allowed_tools))}\n"
        return f"{header}\n---\n\n{body}"

    def index_line(self) -> str:
        """Metadata-only listing (~100 tokens/skill), per progressive disclosure."""
        return f"- {self.name}: {self.description.strip()}"


class SkillError(Exception):
    pass


def parse_skill_file(path: Path, *, source: str = "local", status: str = "active") -> Skill:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text, path)
    problems = _validate_frontmatter(frontmatter, path)
    if problems:
        raise SkillError(f"{path}: " + "; ".join(problems))

    allowed_raw = frontmatter.get("allowed-tools") or ""
    allowed = set(allowed_raw.split()) if isinstance(allowed_raw, str) else set(allowed_raw or [])

    metadata = frontmatter.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise SkillError(f"{path}: metadata must be a mapping")
    metadata = {str(k): str(v) for k, v in metadata.items()}

    return Skill(
        name=str(frontmatter["name"]),
        description=str(frontmatter["description"]),
        body=body,
        path=path,
        source=source,
        status=status,
        license=frontmatter.get("license"),
        compatibility=frontmatter.get("compatibility"),
        metadata=metadata,
        allowed_tools=allowed,
        frontmatter=frontmatter,
    )


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        raise SkillError(f"{path}: SKILL.md must start with YAML frontmatter (---)")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillError(f"{path}: frontmatter is not closed with ---")
    try:
        loaded = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"{path}: invalid YAML frontmatter: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SkillError(f"{path}: frontmatter must be a YAML mapping")
    return loaded, parts[2].strip()


def _validate_frontmatter(fm: dict[str, Any], path: Path) -> list[str]:
    problems: list[str] = []
    name = fm.get("name")
    if not name:
        problems.append("missing required field 'name'")
    elif not isinstance(name, str):
        problems.append("'name' must be a string")
    else:
        if not (1 <= len(name) <= 64):
            problems.append(f"'name' must be 1-64 chars (got {len(name)})")
        if not _NAME_RE.match(name):
            problems.append(
                f"'name' {name!r} must be lowercase letters, digits and single hyphens"
            )
        if path.parent.name and name != path.parent.name:
            problems.append(
                f"'name' {name!r} must match its directory name {path.parent.name!r}"
            )

    description = fm.get("description")
    if not description:
        problems.append("missing required field 'description'")
    elif not isinstance(description, str):
        problems.append("'description' must be a string")
    elif len(description) > 1024:
        problems.append(f"'description' must be <=1024 chars (got {len(description)})")

    compatibility = fm.get("compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, str):
            problems.append("'compatibility' must be a string")
        elif not (1 <= len(compatibility) <= 500):
            problems.append("'compatibility' must be 1-500 chars")

    metadata = fm.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        problems.append("'metadata' must be a mapping")

    allowed = fm.get("allowed-tools")
    if allowed is not None and not isinstance(allowed, (str, list)):
        problems.append("'allowed-tools' must be a space-separated string or a list")
    return problems


# ---------------------------------------------------------------------------
# security review
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    severity: str  # P0 | P1 | P2
    kind: str
    detail: str
    where: str = ""

    def __str__(self) -> str:
        loc = f" [{self.where}]" if self.where else ""
        return f"{self.severity} {self.kind}{loc}: {self.detail}"


@dataclass
class SecurityReport:
    skill: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst(self) -> str:
        for level in ("P0", "P1", "P2"):
            if any(f.severity == level for f in self.findings):
                return level
        return "clean"

    @property
    def blocked(self) -> bool:
        return any(f.severity == "P0" for f in self.findings)

    def render(self) -> str:
        if not self.findings:
            return f"{self.skill}: clean (no findings)"
        lines = [f"{self.skill}: worst={self.worst}"]
        lines += [f"  {f}" for f in self.findings]
        return "\n".join(lines)


def review_skill(
    skill: Skill,
    *,
    known_tools: set[str] | None = None,
    skill_dirs: list[Path] | None = None,
) -> SecurityReport:
    """Static review performed before a skill may be activated."""
    report = SecurityReport(skill=skill.name)

    # 1. tool allowlist must reference tools that exist
    if known_tools is not None and skill.allowed_tools:
        unknown = sorted(skill.allowed_tools - known_tools)
        if unknown:
            report.findings.append(
                Finding("P1", "unknown-tools", f"allowed-tools names unknown tools: {unknown}")
            )

    # 2. instruction-level tripwires
    haystack = f"{skill.description}\n{skill.body}"
    for pattern, label in _INJECTION_PATTERNS:
        match = re.search(pattern, haystack)
        if match:
            severity = "P0" if label in {"instruction override", "credential exfiltration"} else "P1"
            report.findings.append(
                Finding(severity, label, f"matched {match.group(0)[:80]!r}")
            )

    # 3. bundled scripts
    scripts_dir = skill.path.parent / "scripts"
    if scripts_dir.is_dir():
        for script in sorted(scripts_dir.rglob("*")):
            if not script.is_file():
                continue
            try:
                content = script.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for pattern, label in _DANGEROUS_SCRIPT_PATTERNS:
                for match in re.finditer(pattern, content):
                    report.findings.append(
                        Finding(
                            "P1",
                            f"script:{label}",
                            f"matched {match.group(0)[:60]!r}",
                            where=str(script.relative_to(skill.path.parent)),
                        )
                    )

    # 4. path escape attempts in references
    for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", skill.body):
        if raw.startswith(("http://", "https://", "#", "mailto:")):
            continue
        candidate = (skill.path.parent / raw).resolve()
        base = skill.path.parent.resolve()
        if base not in candidate.parents and candidate != base:
            report.findings.append(
                Finding("P1", "path-escape", f"reference {raw!r} resolves outside the skill dir")
            )

    # 5. oversized body
    if len(skill.body) > 100_000:
        report.findings.append(
            Finding("P2", "oversized", f"body is {len(skill.body)} chars; spec recommends <500 lines")
        )
    return report
