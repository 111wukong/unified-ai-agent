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

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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
    default_timeout_s: float = 120.0
    max_output_bytes: int = 200_000


class NetworkPolicy(BaseModel):
    allow_domains: list[str] = Field(default_factory=list)
    allow_all: bool = False
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
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        explicit = dict(data.get("defaults") or {})
        for effect in EffectClass:
            for key in (effect.value, effect.value.replace("_", "")):
                if key in data:
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

    settings = Settings(
        home=home,
        workspace=workspace,
        models=models,
        **{k: v for k, v in raw.items() if k in Settings.model_fields},
    )

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


def config_to_toml(settings: Settings) -> str:
    """Serialize the parts a human should edit back to TOML."""

    def dump(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, list):
            return "[" + ", ".join(dump(v) for v in value) + "]"
        return str(value)

    lines: list[str] = [f"default_model = {dump(settings.default_model)}", ""]
    for alias, spec in settings.models.items():
        lines.append(f"[models.{alias}]")
        data = spec.model_dump(exclude_none=True)
        data.pop("capabilities", None)
        for key, value in data.items():
            lines.append(f"{key} = {dump(value)}")
        lines.append("")

    perms = settings.permissions
    lines.append("[permissions]")
    for effect in EffectClass:
        lines.append(f'"{effect.value}" = {dump(perms.defaults[effect].value)}')
    lines.append("")
    lines.append("[permissions.fs]")
    lines.append(f"read_roots = {dump(perms.fs.read_roots)}")
    lines.append(f"write_roots = {dump(perms.fs.write_roots)}")
    lines.append("")
    lines.append("[permissions.shell]")
    lines.append(f"allow = {dump(perms.shell.allow)}")
    lines.append(f"allow_metacharacters = {dump(perms.shell.allow_metacharacters)}")
    lines.append("")
    lines.append("[agent]")
    for key, value in settings.agent.model_dump().items():
        lines.append(f"{key} = {dump(value)}")
    lines.append("")
    return "\n".join(lines)


class ConfigEditor:
    """Read-modify-write for `uaa config set`."""

    def __init__(self, settings: Settings, config_file: Path | None = None) -> None:
        self.settings = settings
        self.config_file = config_file or (settings.home / "config.toml")

    def set(self, dotted_key: str, raw_value: str) -> Any:
        value = _coerce(raw_value)
        parts = dotted_key.split(".")
        if len(parts) == 1:
            if parts[0] not in Settings.model_fields:
                raise ConfigError(f"unknown setting {dotted_key!r}")
            setattr(self.settings, parts[0], value)
        elif parts[0] == "models" and len(parts) == 3:
            alias, field = parts[1], parts[2]
            if alias not in self.settings.models:
                self.settings.models[alias] = ModelSpec()
            if field not in ModelSpec.model_fields:
                raise ConfigError(f"unknown model field {field!r}")
            setattr(self.settings.models[alias], field, value)
        elif parts[0] == "agent" and len(parts) == 2:
            if parts[1] not in AgentConfig.model_fields:
                raise ConfigError(f"unknown agent field {parts[1]!r}")
            setattr(self.settings.agent, parts[1], value)
        else:
            raise ConfigError(f"unsupported config path {dotted_key!r}")
        return value

    def save(self) -> Path:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(config_to_toml(self.settings), encoding="utf-8")
        return self.config_file


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
