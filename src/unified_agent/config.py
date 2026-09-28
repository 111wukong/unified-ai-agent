"""Configuration.

Deliberate deviation from the original spec: the spec asked for
`pydantic-settings` as the single config source. That works for scalars and
falls apart for *nested* structure (model aliases, per-effect permission
tables, MCP server lists) -- you end up encoding dicts in env vars.

So: structure lives in a TOML file, environment overrides a small set of
scalars, and secrets are *only* ever read from the environment (never
written to disk by this tool).
"""

from __future__ import annotations

import json
import os
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence, get_args, get_origin

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

from unified_agent.errors import ConfigError
from unified_agent.sandbox.base import SandboxMode
from unified_agent.types import Decision, EffectClass

# --------------------------------------------------------------------------
# model aliases
# --------------------------------------------------------------------------

ProviderKind = Literal["openai_compat", "anthropic", "mock"]

DEFAULT_KEY_ENV: dict[ProviderKind, str] = {
    "openai_compat": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mock": "",
}

# Rough list prices, USD per million tokens. Only used to *enforce* a cost
# budget; treat as a starting point and override in config.toml.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4o": (2.5, 10.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-1": (15.0, 75.0),
    "deepseek-chat": (0.27, 1.1),
    "deepseek-reasoner": (0.55, 2.19),
    "qwen-max": (1.6, 6.4),
    "qwen-plus": (0.4, 1.2),
}


class ModelSpec(BaseModel):
    """One named model alias the runtime can route to."""

    provider: ProviderKind = "openai_compat"
    model: str = "gpt-4.1"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 4096
    timeout_s: float = 180.0
    # USD per million tokens (input, output). 0 disables cost accounting.
    price_in: float | None = None
    price_out: float | None = None
    # Capability overrides; unset values fall back to the adapter's
    # declared defaults for the provider.
    capabilities: dict[str, Any] = Field(default_factory=dict)

    def key_env(self) -> str:
        if self.api_key_env:
            return self.api_key_env
        return DEFAULT_KEY_ENV.get(self.provider, "")

    def api_key(self) -> str | None:
        env = self.key_env()
        if not env:
            return None
        return os.environ.get(env) or None

    def resolved_prices(self) -> tuple[float, float]:
        if self.price_in is not None and self.price_out is not None:
            return self.price_in, self.price_out
        return DEFAULT_PRICES.get(self.model, (0.0, 0.0))


# --------------------------------------------------------------------------
# permissions
# --------------------------------------------------------------------------


class FsPolicy(BaseModel):
    """Filesystem fence.

    Roots are resolved against the workspace root and canonicalized
    (symlinks followed) before containment checks -- see tools/permissions.py.
    """

    read_roots: list[str] = Field(default_factory=lambda: ["."])
    write_roots: list[str] = Field(default_factory=lambda: ["."])
    # Extra glob patterns (relative to $HOME or absolute) that are always
    # refused, even for READ_ONLY tools.
    extra_deny: list[str] = Field(default_factory=list)


class ShellPolicy(BaseModel):
    """Command fence.

    `allow` is a list of argv *prefixes*. An empty `allow` means "any
    command, subject to the effect-class decision". `deny_patterns` are
    regexes matched against the raw command string and always win.
    """

    allow: list[str] = Field(
        default_factory=lambda: [
            "pytest",
            "ruff",
            "mypy",
            "python",
            "python3",
            "node",
            "npm",
            "pnpm",
            "git",
            "ls",
            "cat",
            "head",
            "tail",
            "wc",
            "find",
            "grep",
            "rg",
            "echo",
            "mkdir",
            "touch",
            "cp",
            "mv",
            "sort",
            "uniq",
            "diff",
            "tree",
        ]
    )
    deny_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]",
            r"\bsudo\b",
            r"\bchmod\s+777\b",
            r"\bchown\b",
            r"\bcurl\b[^|]*\|\s*(ba)?sh",
            r"\bwget\b[^|]*\|\s*(ba)?sh",
            r"\bmkfs\b",
            r"\bdd\s+if=",
            r">\s*/dev/[sh]d",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bkillall\b",
            r"\blaunchctl\b",
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]
    )
    # Shell metacharacters are refused unless this is on: they are the
    # standard way a "safe" allowlisted command smuggles a second one.
    allow_metacharacters: bool = False
    #: argv prefixes that are always denied, whatever the allowlist says.
    #:
    #: The allowlist matches on the command *name*, so "allow `python3`" is
    #: indistinguishable from "allow everything" -- and `git config
    #: core.hooksPath` writes the file that decides what runs on the next
    #: commit. The risk lives in the arguments, so the rules have to match the
    #: arguments. Prefixes are matched against the parsed argv, so
    #: `git   config` with extra spaces is caught too.
    #:
    #: Codex CLI's ExecPolicy calls this the forbidden tier; the shape is the
    #: same and so is the conclusion about interpreters.
    forbidden_prefixes: list[str] = Field(
        default_factory=lambda: [
            # git's own escape hatches. `config` writes `.git/config`
            # (hooksPath, pager, credential.helper, alias.*), `-c` sets one for
            # a single command, and the rest install or rewrite things that run
            # later. `git status` / `diff` / `log` are unaffected.
            "git config",
            "git -c",
            "git --config-env",
            "git hook",
            "git hooks",
            "git alias",
            "git filter-branch",
            "git filter-repo",
            "git submodule",
            # interpreters asked to run a *string* rather than a file
            "python -c",
            "python3 -c",
            "python -",
            "python3 -",
            "node -e",
            "node --eval",
            "node -p",
            "node --print",
            "perl -e",
            "ruby -e",
            "php -r",
            "lua -e",
            "bash -c",
            "sh -c",
            "zsh -c",
            "dash -c",
            "fish -c",
            "osascript -e",
            # commands whose entire purpose is to run another command
            "env",
            "xargs",
            "nohup",
            "watch",
            "sudo",
            "doas",
            "su",
        ]
    )
    #: argv prefixes that may run, but never without a per-call confirmation.
    #:
    #: The tier that makes pre-approval honest. `--approve execute_local` and
    #: `--yes` say "run commands without asking me each time", which is a
    #: reasonable thing to want -- but it is not a statement that the agent
    #: may run arbitrary code unattended, and an interpreter on the allowlist
    #: makes those two indistinguishable. Codex CLI's ExecPolicy has the same
    #: middle tier, for the same reason.
    #:
    #: These are denied by nothing and allowed by nothing: they always ask.
    confirm_prefixes: list[str] = Field(
        default_factory=lambda: [
            # interpreters: "python3 script.py" runs whatever was just written
            "python",
            "python3",
            "node",
            "deno",
            "bun",
            "perl",
            "ruby",
            "php",
            "lua",
            "Rscript",
            # package managers: `npm test` runs package.json scripts
            "npm",
            "pnpm",
            "yarn",
            "npx",
            "pip",
            "pip3",
            "uv",
            "poetry",
            # build tools: they run build scripts from the project
            "make",
            "cmake",
            "gradle",
            "mvn",
            "cargo",
            "go",
            # shells
            "bash",
            "sh",
            "zsh",
            "dash",
            "fish",
            # git operations that leave the machine or rewrite where it points
            "git push",
            "git remote",
            "git fetch",
            "git clone",
            "git clean",
            "git reset",
        ]
    )
    #: Arguments that make a confirm-prefix command inert.
    #:
    #: If *every* argument after the command is one of these, the invocation
    #: cannot execute anything -- `python3 --version` reports a version and
    #: stops. Without this the tier fires on it, and a guard that is annoying
    #: is a guard people switch off, which is a worse outcome than the one it
    #: was protecting against.
    inert_args: list[str] = Field(
        default_factory=lambda: ["--version", "-V", "-VV", "--help", "-h"]
    )
    #: Flags that turn an otherwise-inert command into a runner. Checked
    #: anywhere in the argv, because these are not prefixes: `find . -exec ...`
    #: puts the danger after the arguments.
    forbidden_flags: list[str] = Field(
        default_factory=lambda: [
            "-exec",
            "-execdir",
            "-ok",
            "-okdir",
            "-delete",  # `find / -delete` is not a file-listing command
            "-fprint",
            "-fprintf",
            "-fls",
            "--to-command",  # tar
            "--checkpoint-action",  # tar
        ]
    )
    default_timeout_s: float = 120.0
    # Ceiling on captured stdout+stderr, per command. Applied in
    # `tools/shell.py::_exec`; output past it is cut, not silently dropped
    # (the cut is announced in the text the model reads).
    max_output_bytes: int = 200_000


class NetworkPolicy(BaseModel):
    allow_domains: list[str] = Field(default_factory=list)
    allow_all: bool = False
    # Ceiling on a response body, in bytes. Applied in `tools/net.py`; the
    # result is flagged `truncated` so the model knows it did not see all of it.
    max_response_bytes: int = 2_000_000


class PermissionConfig(BaseModel):
    """Per-effect default decision. Matches the spec's table exactly."""

    defaults: dict[EffectClass, Decision] = Field(
        default_factory=lambda: {
            EffectClass.READ_ONLY: Decision.ALLOW,
            EffectClass.WRITE_LOCAL: Decision.ALLOW,
            EffectClass.EXECUTE_LOCAL: Decision.CONFIRM,
            EffectClass.NETWORK: Decision.CONFIRM,
            EffectClass.EXTERNAL_SIDE_EFFECT: Decision.CONFIRM,
            EffectClass.SYSTEM_ADMIN: Decision.DENY,
        }
    )
    fs: FsPolicy = Field(default_factory=FsPolicy)
    shell: ShellPolicy = Field(default_factory=ShellPolicy)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)

    @model_validator(mode="before")
    @classmethod
    def _hoist_effect_keys(cls, data: Any) -> Any:
        """Allow `read_only = "allow"` directly under [permissions].

        Writing the effect names as bare keys is far more readable than a
        nested [permissions.defaults] table, so accept both.

        Only a *string* is treated as a decision. `network` is both an effect
        class and the name of the `[permissions.network]` policy table, and
        popping it unconditionally moved a network allowlist into the
        decision map -- which then failed validation. That went unnoticed
        because the old TOML dumper never wrote `[permissions.network]`: the
        serializer's omission was hiding the validator's overreach.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        explicit = dict(data.get("defaults") or {})
        for effect in EffectClass:
            for key in (effect.value, effect.value.replace("_", "")):
                if key in data and isinstance(data[key], str):
                    explicit[effect] = data.pop(key)
        if explicit:
            data["defaults"] = explicit
        return data


# --------------------------------------------------------------------------
# agent budgets
# --------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Hard budgets. The runtime enforces these; observability merely reports.

    Every one of these is a *stop* condition, not a warning. That is the
    difference between a runaway loop costing $40 and one costing $0.40.
    """

    max_steps: int = 30
    max_plan_steps: int = 12
    max_tokens: int = 400_000
    max_cost_usd: float = 2.0
    wall_clock_s: float = 900.0
    # Model-call retries on retryable provider errors (not tool retries).
    model_retry_attempts: int = 3
    # Planner output repair attempts before giving up.
    planner_repair_attempts: int = 2
    # Tool output longer than this is truncated (head+tail) and offloaded.
    tool_output_chars: int = 8_000
    # Observations older than the newest N get compacted into a summary.
    keep_recent_observations: int = 6
    # Fraction of the context window reserved for the model's own output.
    output_reserve_ratio: float = 0.15
    # Post-task memory + skill-candidate extraction. Off by default because
    # it costs one extra model call per task, and that is a cost the user
    # should opt into. `uaa run --reflect` and the HTTP layer both set it.
    reflect: bool = False


class McpServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # Effect class assigned to every tool this server exposes. MCP servers
    # are third-party code: default to the cautious end.
    effect_class: EffectClass = EffectClass.EXECUTE_LOCAL
    requires_confirmation: bool = True
    enabled: bool = True
    startup_timeout_s: float = 30.0


class MultiAgentConfig(BaseModel):
    """Orchestrator-worker delegation.

    Off by default, and that is the point. The measured cost of multi-agent
    is roughly 15x the tokens of a plain chat (about 4x for a single agent),
    and the same research says explicitly that coding tasks are a poor fit --
    there are fewer genuinely parallel subtasks than in open-ended research.
    What it does win at is work that is highly parallel *and* wider than one
    context window. So this is an explicit choice behind a hard budget gate,
    not a default.
    """

    enabled: bool = False
    # Alias for sub-agents. Empty means "inherit the orchestrator's".
    # Splitting them is the standard cost shape: a strong model plans and
    # synthesises, cheap ones do the reading. It is also the only way to get
    # the fan-out under a budget you would accept.
    model: str = ""
    # Hard cap per delegation. The classic early failure is a model spawning
    # fifty sub-agents for a question that needed one.
    max_agents: int = 5
    # How many run at once. The 3-5 range is where the measured speedup is.
    parallelism: int = 3
    # Per sub-agent, not shared out of the parent's pool: one sub-agent that
    # will not stop searching must not be able to spend the whole run.
    max_steps_per_agent: int = 12
    # Effects a sub-agent may use without asking. Read-only by default: the
    # orchestrator is the only writer of shared state, so sub-agents cannot
    # trample each other's edits or the orchestrator's plan.
    allowed_effects: list[EffectClass] = Field(
        default_factory=lambda: [EffectClass.READ_ONLY]
    )
    # Characters of each sub-agent answer that come back into the
    # orchestrator's context. The full body goes to a file: pasting every
    # report into the conversation is the "telephone game" that makes the
    # orchestrator worse than a single agent.
    max_summary_chars: int = 1_200


class A2AConfig(BaseModel):
    """Agent-to-agent exposure.

    Off by default. Serving an endpoint that accepts work from other agents
    is a decision, and so is calling one -- both directions are opt-in.

    `allow_hosts` is empty by default, which means "call nobody". Fetching a
    remote Agent Card is an outbound request to a URL a peer supplies, so it
    is the same SSRF shape the A2A spec names for webhooks, and an allowlist
    is the only form of it worth enabling.
    """

    enabled: bool = False
    name: str = "unified-ai-agent"
    #: Where this agent says it lives, for the Agent Card's `url`. Empty
    #: means "derive from the request", which is wrong behind a proxy: the
    #: card would advertise an internal address a peer cannot reach.
    public_url: str = ""
    description: str = (
        "A local-first agent runtime: event-sourced task execution, enforced "
        "budgets, write-ahead tool ledger and a human approval gate."
    )
    # Hosts a remote agent may be reached at, e.g. ["partner.example.com"].
    allow_hosts: list[str] = Field(default_factory=list)
    # Escape hatch for a lab setup. Named so that turning it on is obviously
    # a decision: it disables the private-address block entirely.
    allow_private_networks: bool = False
    timeout_s: float = 30.0
    # How much of a remote agent's answer to accept, in bytes.
    max_response_bytes: int = 1_000_000


class MemoryConfig(BaseModel):
    """Long-term memory.

    The embedding model is optional on purpose. Without one the store still
    works: vectors degrade to a lexical hash (fine for dedupe, not for
    semantics) and the contradiction judge degrades to exact-duplicate
    detection. A memory feature that silently stops working when a second
    service is not configured is worse than one that degrades predictably.
    """

    embedding_model: str = ""
    embedding_base_url: str | None = None
    embedding_api_key_env: str = "OPENAI_API_KEY"
    hashing_dim: int = 512
    neighbour_limit: int = 5
    # How deep the vector branch reads before reciprocal rank fusion merges
    # it with FTS. Floored at the caller's `limit`: a ranked list shorter
    # than the result set gives RRF nothing to compare.
    vector_limit: int = 10
    # Drop vector hits at or below this cosine similarity. 0.0 only removes
    # "nothing in common"; it is not a relevance threshold.
    min_similarity: float = 0.0
    reconcile: bool = True
    #: Whether the curator may supersede an existing memory.
    #:
    #: Off, and that is the important part. Superseding hides a memory from
    #: search, and the judgement behind it -- "this new fact replaces that old
    #: one" -- has no reliable prior: two facts are often complementary rather
    #: than contradictory ("lives in New York" and "moved to San Francisco").
    #: Getting it wrong loses a fact silently and permanently, whereas a
    #: visible contradiction is merely inconvenient. Measured elsewhere: mem0
    #: removed write-time reconciliation entirely and gained 26 points on
    #: LongMemEval.
    #:
    #: Turning it on is a legitimate choice for a store that must stay small;
    #: it is simply not the default, and the default is the one that cannot
    #: lose data.
    allow_supersede: bool = False


class SandboxConfig(BaseModel):
    """Process isolation. `auto` prefers macOS Seatbelt: it is free per
    command, and a sandbox you leave on protects more than a stronger one
    you turn off because it costs three seconds a command."""

    backend: Literal["auto", "seatbelt", "docker", "none"] = "auto"
    mode: SandboxMode = SandboxMode.WORKSPACE_WRITE
    extra_write_dirs: list[str] = Field(default_factory=list)
    docker_image: str = "python:3.12-slim"
    docker_network: str = "none"
    docker_mounts: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    home: Path
    workspace: Path
    default_model: str = "default"
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    skill_dirs: list[str] = Field(default_factory=lambda: ["skills"])

    # -- derived paths ----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.home / "uaa.db"

    @property
    def log_dir(self) -> Path:
        return self.home / "logs"

    @property
    def artifact_dir(self) -> Path:
        return self.home / "artifacts"

    @property
    def state_dir(self) -> Path:
        return self.home / "state"

    @property
    def sandbox_dir(self) -> Path:
        return self.home / "sandbox"

    def model(self, alias: str | None = None) -> ModelSpec:
        name = alias or self.default_model
        if name not in self.models:
            raise ConfigError(
                f"model alias {name!r} not configured; known: {sorted(self.models)}"
            )
        return self.models[name]

    def ensure_dirs(self) -> None:
        for d in (
            self.home,
            self.log_dir,
            self.artifact_dir,
            self.state_dir,
            self.sandbox_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def secret_values(self) -> list[str]:
        """Every configured credential value, for log redaction."""
        found: list[str] = []
        for spec in self.models.values():
            key = spec.api_key()
            if key and len(key) >= 8:
                found.append(key)
        for server in self.mcp_servers:
            for value in server.env.values():
                if value and len(value) >= 8:
                    found.append(value)
        return found


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

DEFAULT_CONFIG_TOML = """\
# unified-ai-agent configuration
default_model = "default"

[models.default]
provider = "openai_compat"
model = "gpt-4.1"
api_key_env = "OPENAI_API_KEY"
temperature = 0.2

[models.fast]
provider = "openai_compat"
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"

# DeepSeek / 通义千问 / Ollama / LM Studio all speak the OpenAI wire format.
[models.deepseek]
provider = "openai_compat"
model = "deepseek-chat"
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"

[models.qwen]
provider = "openai_compat"
model = "qwen-plus"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"

[models.local]
provider = "openai_compat"
model = "qwen2.5"
base_url = "http://127.0.0.1:1234/v1"
api_key_env = ""

[models.claude]
provider = "anthropic"
model = "claude-sonnet-4-5"
api_key_env = "ANTHROPIC_API_KEY"

# Offline deterministic model. Used by the test suite and `uaa demo`.
[models.mock]
provider = "mock"
model = "mock-react"

[permissions]
# READ_ONLY / WRITE_LOCAL allow, EXECUTE_LOCAL / NETWORK / EXTERNAL_SIDE_EFFECT
# confirm, SYSTEM_ADMIN deny. Relax per-effect from the CLI with
# `uaa run --approve execute_local`.
[permissions.fs]
read_roots = ["."]
write_roots = ["."]

[sandbox]
# auto | seatbelt | docker | none
# `auto` prefers macOS Seatbelt (sandbox-exec): zero per-command cost.
backend = "auto"
# read-only | workspace-write | full
# `read-only` makes the project tree (including .git) unwritable -- use it
# for "analyse this repo" tasks.
mode = "workspace-write"

[memory]
# Leave empty to use the offline lexical fallback. Set to a real embedding
# model (e.g. "text-embedding-3-small") for semantic search.
embedding_model = ""
# embedding_base_url = "https://api.openai.com/v1"
embedding_api_key_env = "OPENAI_API_KEY"
# Run the contradiction judge when a new memory is written.
reconcile = true

[agent]
max_steps = 30
max_tokens = 400000
max_cost_usd = 2.0
wall_clock_s = 900
"""


def default_home() -> Path:
    override = os.environ.get("UAA_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".uaa"


def write_default_config(path: Path, *, force: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return path


def load_settings(
    *,
    home: Path | None = None,
    workspace: Path | None = None,
    config_file: Path | None = None,
    create_if_missing: bool = False,
) -> Settings:
    home = Path(home).expanduser() if home else default_home()
    workspace = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    cfg_path = Path(config_file) if config_file else home / "config.toml"

    raw: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:  # pragma: no cover - user error
            raise ConfigError(f"{cfg_path} is not valid TOML: {exc}") from exc
    elif create_if_missing:
        write_default_config(cfg_path)
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    models_raw = raw.pop("models", None)
    models: dict[str, ModelSpec] = {}
    if isinstance(models_raw, dict):
        for alias, spec in models_raw.items():
            if not isinstance(spec, dict):
                raise ConfigError(f"models.{alias} must be a table")
            models[alias] = ModelSpec(**spec)
    if not models:
        models = {"mock": ModelSpec(provider="mock", model="mock-react")}
        if "default_model" not in raw:
            raw["default_model"] = "mock"

    # `home` and `workspace` are decided by the caller, not by the file. They
    # are filtered out rather than passed through because passing them twice
    # is a `TypeError: got multiple values for keyword argument`, which is a
    # traceback instead of a sentence.
    file_settings = {
        k: v for k, v in raw.items() if k in Settings.model_fields and k not in _NOT_FILE_SETTINGS
    }
    settings = Settings(home=home, workspace=workspace, models=models, **file_settings)
    # env scalar overrides
    if env_model := os.environ.get("UAA_DEFAULT_MODEL"):
        settings.default_model = env_model
    if env_ws := os.environ.get("UAA_WORKSPACE"):
        settings.workspace = Path(env_ws).resolve()

    if settings.default_model not in settings.models:
        raise ConfigError(
            f"default_model {settings.default_model!r} is not in [models]; "
            f"known: {sorted(settings.models)}"
        )
    return settings


#: Supplied by the process, never by the file. They are excluded from the
#: dumper because `load_settings` passes them explicitly *before* `**raw`:
#: a config file containing them raises "got multiple values for keyword
#: argument" rather than doing anything useful.
_NOT_FILE_SETTINGS = frozenset({"home", "workspace"})


def config_to_toml(settings: Settings) -> str:
    """Serialize the whole editable configuration back to TOML.

    Generic over the model fields rather than a hand-listed set of sections.
    The hand-listed version was not merely incomplete, it **lost data**:
    `uaa config set` is a read-modify-write, so every section the dumper did
    not know about was deleted from the file the first time the user changed
    anything. A hand-written `[permissions.network] allow_domains` vanished
    the moment someone edited a step budget.

    Everything except `home`/`workspace` round-trips, and
    `tests/test_config_round_trip.py` asserts that over the whole surface --
    so a new field the dumper forgets fails a test instead of quietly
    deleting a user's setting.
    """
    lines: list[str] = []
    _emit_table(lines, "", settings)
    return "\n".join(lines).rstrip() + "\n"


def _emit_table(lines: list[str], prefix: str, model: BaseModel) -> None:
    scalars: list[tuple[str, Any]] = []
    tables: list[tuple[str, BaseModel]] = []
    arrays: list[tuple[str, list[Any]]] = []
    # `dict[str, ModelSpec]` is a table per key, not a scalar: rendering it
    # with `_dump_scalar` emits a pydantic repr that is not valid TOML, and
    # the file this tool writes then cannot be read back.
    keyed_tables: list[tuple[str, dict[Any, BaseModel]]] = []
    # A map of enums (`permissions.defaults`) gets its own table so the
    # security decisions are one per line. It cannot be hoisted to bare keys
    # under `[permissions]`, which is the documented shorthand: `network` is
    # both an effect class and the `[permissions.network]` table, and TOML
    # refuses to have the same key be both a value and a table.
    enum_maps: list[tuple[str, dict[Any, Any]]] = []
    for name in type(model).model_fields:
        if not prefix and name in _NOT_FILE_SETTINGS:
            continue
        value = getattr(model, name)
        # `None` means "unset" for every optional field here (a base_url, a
        # price, a key env var). Writing it would emit a bare `None`, which
        # is not TOML, and the loader would then reject the file this tool
        # just wrote.
        if value is None:
            continue
        if isinstance(value, BaseModel):
            tables.append((name, value))
        elif isinstance(value, dict) and any(
            isinstance(v, BaseModel) for v in value.values()
        ):
            keyed_tables.append((name, value))
        elif isinstance(value, dict) and value and all(
            isinstance(v, Enum) for v in value.values()
        ):
            enum_maps.append((name, value))
        elif isinstance(value, list) and any(isinstance(v, BaseModel) for v in value):
            arrays.append((name, value))
        else:
            scalars.append((name, value))

    if prefix:
        lines.append(f"[{prefix}]")
    for name, value in scalars:
        lines.append(f"{name} = {_dump_scalar(value)}")
    if prefix or scalars:
        lines.append("")
    # Sub-tables come after every scalar of this table: TOML has no way back
    # to the parent once a new `[header]` has been opened.
    for name, value in tables:
        _emit_table(lines, f"{prefix}.{name}" if prefix else name, value)
    for name, mapping in keyed_tables:
        for key, item in mapping.items():
            child = f"{prefix}.{name}.{key}" if prefix else f"{name}.{key}"
            _emit_table(lines, child, item)
    for name, mapping in enum_maps:
        lines.append(f"[{prefix}.{name}]" if prefix else f"[{name}]")
        for key, item in mapping.items():
            lines.append(f"{_bare_key(_key(key))} = {_dump_scalar(item)}")
        lines.append("")
    for name, items in arrays:
        header = f"{prefix}.{name}" if prefix else name
        for item in items:
            lines.append(f"[[{header}]]")
            for key in type(item).model_fields:
                lines.append(f"{key} = {_dump_scalar(getattr(item, key))}")
            lines.append("")


def _dump_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return json.dumps(value.value)
    if isinstance(value, str):
        # `json.dumps` is a correct TOML basic-string escaper: it handles
        # backslashes, quotes and control characters the same way.
        return json.dumps(value)
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, list):
        return "[" + ", ".join(_dump_scalar(v) for v in value) + "]"
    if isinstance(value, dict):
        pairs = ", ".join(
            f"{json.dumps(_key(k))} = {_dump_scalar(v)}" for k, v in value.items()
        )
        return "{" + pairs + "}"
    return str(value)


def _key(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def _bare_key(key: str) -> str:
    """Quote a TOML key only when it has to be quoted."""
    return key if key and all(c.isalnum() or c in "_-" for c in key) else json.dumps(key)


class ConfigEditor:
    """Read-modify-write for `uaa config set`."""

    def __init__(self, settings: Settings, config_file: Path | None = None) -> None:
        self.settings = settings
        self.config_file = config_file or (settings.home / "config.toml")

    def set(self, dotted_key: str, raw_value: str) -> Any:
        value = _coerce(raw_value)
        parts = dotted_key.split(".")
        if not parts or not all(parts):
            raise ConfigError(f"unsupported config path {dotted_key!r}")

        # `models` is a table per alias, which a dotted path cannot express
        # generically (the middle segment is a user-chosen key, not a field).
        if parts[0] == "models":
            if len(parts) == 3:
                return self._set_model_field(parts[1], parts[2], value)
            if len(parts) == 4 and parts[2] == "capabilities":
                return self._set_model_capability(parts[1], parts[3], value)
            raise ConfigError(
                f"{dotted_key!r}: expected models.<alias>.<field> or "
                "models.<alias>.capabilities.<key>"
            )

        container, field = self._walk(parts, dotted_key)
        return _assign(container, field, value, dotted_key, self.config_file)

    def _set_model_field(self, alias: str, field: str, value: Any) -> Any:
        if alias not in self.settings.models:
            self.settings.models[alias] = ModelSpec()
        if field not in ModelSpec.model_fields:
            raise ConfigError(f"unknown model field {field!r}")
        spec = self.settings.models[alias]
        return _assign(spec, field, value, f"models.{alias}.{field}", self.config_file)

    def _set_model_capability(self, alias: str, key: str, value: Any) -> Any:
        """`models.<alias>.capabilities.<key>`.

        A string-keyed map of scalars is expressible as a deeper path even
        though the map itself is not a table -- and this one matters: a model
        with a 1M context window whose `max_context_tokens` stays at the 128k
        default makes the context builder compact eight times too eagerly, and
        nothing in the output says why.
        """
        from unified_agent.models.base import ModelCapabilities

        if alias not in self.settings.models:
            self.settings.models[alias] = ModelSpec()
        spec = self.settings.models[alias]
        merged = dict(spec.capabilities) | {key: value}
        try:
            # Validate through the capability model so an unknown key or a bad
            # value is rejected here, not at the first request.
            validated = ModelCapabilities().merged(merged)
        except (ValueError, ValidationError) as exc:
            raise ConfigError(f"models.{alias}.capabilities.{key}: {exc}") from exc
        spec.capabilities = {
            name: getattr(validated, name)
            for name in merged
        }
        return value

    def _walk(self, parts: list[str], dotted_key: str) -> tuple[Any, str]:
        """Resolve a dotted path to (the object holding the field, field name).

        Walks any depth rather than a hand-listed set of two-part paths. The
        hand-listed version supported `agent.*` only, so `memory.vector_limit`
        and `multi_agent.enabled` were documented knobs the CLI refused --
        and a setting reachable only by hand-editing TOML is one most people
        never find. `permissions.network.*` is three deep.
        """
        if len(parts) == 1:
            if parts[0] not in Settings.model_fields:
                raise ConfigError(f"unknown setting {dotted_key!r}")
            return self.settings, parts[0]

        current: Any = self.settings
        for depth, part in enumerate(parts[:-1]):
            fields = getattr(type(current), "model_fields", None)
            if fields is None or part not in fields:
                raise ConfigError(f"unsupported config path {dotted_key!r}")
            current = getattr(current, part)
            if not isinstance(current, BaseModel):
                raise ConfigError(
                    f"{'.'.join(parts[: depth + 1])!r} is not a nested table"
                )

        final = parts[-1]
        fields = getattr(type(current), "model_fields", None) or {}
        if final not in fields:
            raise ConfigError(f"unknown {parts[-2]} field {final!r}")
        return current, final

    def save(self) -> Path:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(config_to_toml(self.settings), encoding="utf-8")
        return self.config_file


def _assign(
    container: BaseModel, field: str, value: Any, dotted_key: str, config_file: Path
) -> Any:
    """Validate one field and set it, refusing shapes a CLI cannot express.

    Two things this fixes over a bare `setattr`:

    * **`setattr` does not validate.** Pydantic only validates on construction
      unless `validate_assignment` is on, so `config set sandbox.mode
      read-only` used to store the raw string and the field's type silently
      stopped being `SandboxMode` -- equal today, broken the first time
      something uses `is` or `model_dump(mode="json")`.
    * **A list of scalars is expressible; a list of tables is not.** Several
      documented settings are lists (`a2a.allow_hosts`,
      `permissions.network.allow_domains`, `multi_agent.allowed_effects`), and
      refusing all lists made every one of them reachable only by editing
      TOML by hand. A comma-separated value covers them; `mcp_servers` stays
      hand-edited because each entry is a table.
    """
    annotation = type(container).model_fields[field].annotation
    current = getattr(container, field)

    if isinstance(current, BaseModel):
        raise ConfigError(
            f"{dotted_key!r} is a nested table; set one of its fields instead, "
            f"e.g. {dotted_key}.<field>"
        )
    if isinstance(current, dict):
        raise ConfigError(
            f"{dotted_key!r} is a mapping; edit {config_file} directly"
        )
    if _is_model_list(annotation):
        raise ConfigError(
            f"{dotted_key!r} is a list of tables; edit {config_file} directly"
        )
    if _is_scalar_list(annotation):
        # `_coerce` maps "" and "none" to None, which is how a scalar field is
        # unset. For a list, "unset" is the empty list -- otherwise clearing
        # an allowlist would be the one operation the CLI refuses.
        if value is None:
            value = []
        elif isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]

    adapter = TypeAdapter(annotation)
    try:
        coerced = adapter.validate_python(value)
    except ValidationError as exc:
        detail = exc.errors()[0]
        raise ConfigError(
            f"{dotted_key}: {detail.get('msg', 'invalid value')} (got {value!r})"
        ) from exc
    setattr(container, field, coerced)
    return coerced


def _is_scalar_list(annotation: Any) -> bool:
    """`list[str]`, `list[EffectClass]` and friends -- settable from a CSV."""
    origin = get_origin(annotation)
    if origin not in (list, Sequence):
        return False
    args = get_args(annotation)
    if not args:
        return False
    return all(
        isinstance(arg, type) and not issubclass(arg, BaseModel) for arg in args
    )


def _is_model_list(annotation: Any) -> bool:
    """`list[McpServerConfig]` -- each entry is a table, so hand-edit."""
    origin = get_origin(annotation)
    if origin not in (list, Sequence):
        return False
    args = get_args(annotation)
    return bool(args) and all(
        isinstance(arg, type) and issubclass(arg, BaseModel) for arg in args
    )


def _coerce(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw
