"""Configuration round-trip.

`wukong config set` is a read-modify-write: it loads the file, changes one key,
and writes the whole thing back. So any field the serializer does not know
about is not merely missing from the output -- it is **deleted from the
file** the first time the user changes anything else.

That was real. The serializer hand-listed its sections (`models`,
`permissions`, `agent`), so `[permissions.network] allow_domains`,
`[sandbox]`, `[memory]` and `[multi_agent]` all disappeared on the next
`config set`. The sharpest edge was a hand-written network allowlist: it
vanished the moment someone edited a step budget, and nothing said so.

The first test here is deliberately whole-surface rather than a list of the
sections that exist today: a field added later and forgotten by the dumper
has to fail a test, not silently eat a user's setting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wukong.config import (
    ConfigEditor,
    ModelSpec,
    Settings,
    config_to_toml,
    load_settings,
)
from wukong.errors import ConfigError
from wukong.sandbox.base import SandboxMode
from wukong.types import Decision, EffectClass


def comprehensive() -> Settings:
    """A settings object with a non-default value in every section."""
    settings = Settings(
        home=Path("/tmp/wukong-round-trip-home"),
        workspace=Path("/tmp/wukong-round-trip-ws"),
    )
    settings.default_model = "deepseek"
    settings.skill_dirs = ["skills", ".agent/skills"]
    settings.models = {
        "deepseek": ModelSpec(
            provider="openai_compat",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
            temperature=0.35,
            price_in=0.27,
            price_out=1.1,
            capabilities={"json_schema": False},
        )
    }

    settings.permissions.network.allow_domains = ["example.com", "api.example.org"]
    settings.permissions.network.max_response_bytes = 123_456
    settings.permissions.shell.max_output_bytes = 65_432
    settings.permissions.shell.allow_metacharacters = True
    settings.permissions.defaults[EffectClass.NETWORK] = Decision.ALLOW

    settings.agent.max_steps = 17
    settings.agent.reflect = True

    settings.sandbox.mode = SandboxMode.READ_ONLY
    settings.sandbox.backend = "none"

    settings.memory.vector_limit = 21
    settings.memory.embedding_model = "text-embedding-3-small"

    settings.multi_agent.enabled = True
    settings.multi_agent.parallelism = 4
    settings.multi_agent.model = "deepseek"
    settings.multi_agent.max_agents = 7

    return settings


def test_every_section_survives_a_round_trip(tmp_path: Path) -> None:
    original = comprehensive()
    text = config_to_toml(original)

    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    reloaded = load_settings(home=tmp_path, workspace=tmp_path, config_file=path)

    before = original.model_dump(exclude={"home", "workspace"})
    after = reloaded.model_dump(exclude={"home", "workspace"})
    assert after == before


def test_the_sections_that_used_to_be_dropped_are_in_the_file(tmp_path: Path) -> None:
    text = config_to_toml(comprehensive())

    for section in (
        "[permissions.network]",
        "[sandbox]",
        "[memory]",
        "[multi_agent]",
        "[models.deepseek]",
        "[agent]",
    ):
        assert section in text, f"{section} is missing from the serialized config"


def test_a_hand_written_network_allowlist_survives_an_unrelated_set(tmp_path: Path) -> None:
    """The concrete data-loss bug, end to end.

    Someone hand-writes a network allowlist, then runs `wukong config set` for
    something unrelated. The allowlist must still be there.
    """
    path = tmp_path / "config.toml"
    path.write_text(
        'default_model = "mock"\n'
        "\n"
        "[models.mock]\n"
        'provider = "mock"\n'
        'model = "mock-react"\n'
        "\n"
        "[permissions.network]\n"
        'allow_domains = ["internal.example.com"]\n',
        encoding="utf-8",
    )

    settings = load_settings(home=tmp_path, workspace=tmp_path, config_file=path)
    editor = ConfigEditor(settings, config_file=path)
    editor.set("agent.max_steps", "11")
    editor.save()

    reloaded = load_settings(home=tmp_path, workspace=tmp_path, config_file=path)
    assert reloaded.permissions.network.allow_domains == ["internal.example.com"]
    assert reloaded.agent.max_steps == 11


def test_home_and_workspace_are_never_written(tmp_path: Path) -> None:
    """They come from the caller.

    `load_settings` passes them explicitly before `**raw`, so a file that
    contains them raises `TypeError: got multiple values for keyword
    argument` -- a traceback, from a file this tool wrote itself.
    """
    text = config_to_toml(comprehensive())

    assert "\nworkspace =" not in text
    assert not text.startswith("home =")
    # And a file that does contain them is still readable.
    path = tmp_path / "config.toml"
    path.write_text('home = "/nope"\nworkspace = "/nope"\n', encoding="utf-8")
    reloaded = load_settings(home=tmp_path, workspace=tmp_path, config_file=path)
    assert reloaded.home == tmp_path


class TestConfigSet:
    def test_any_nested_section_is_settable(self, tmp_path: Path) -> None:
        """It used to be `agent.*` and `models.*` only.

        A knob you can only reach by editing TOML by hand is a knob most
        people never find -- `memory.vector_limit` and `multi_agent.enabled`
        were both documented and both unreachable from the CLI.
        """
        settings = Settings(home=tmp_path, workspace=tmp_path)
        editor = ConfigEditor(settings, config_file=tmp_path / "config.toml")

        assert editor.set("multi_agent.enabled", "true") is True
        assert editor.set("memory.vector_limit", "33") == 33
        assert editor.set("sandbox.mode", "read-only") == "read-only"
        assert editor.set("permissions.network.max_response_bytes", "1000") == 1000

        assert settings.multi_agent.enabled is True
        assert settings.memory.vector_limit == 33
        assert settings.sandbox.mode is SandboxMode.READ_ONLY

    def test_a_scalar_cannot_be_assigned_to_a_table(self, tmp_path: Path) -> None:
        """Otherwise pydantic raises and the user sees a traceback."""
        settings = Settings(home=tmp_path, workspace=tmp_path)
        editor = ConfigEditor(settings, config_file=tmp_path / "config.toml")

        with pytest.raises(ConfigError, match="is a nested table"):
            editor.set("permissions.shell", "3")
        with pytest.raises(ConfigError, match="list of tables"):
            editor.set("mcp_servers", "nope")
        with pytest.raises(ConfigError, match="is a mapping"):
            editor.set("permissions.defaults", "allow")
        # `models` is a table per alias, so the path needs the alias too.
        with pytest.raises(ConfigError, match=r"models\.<alias>\.<field>"):
            editor.set("models", "nope")

    def test_a_list_of_scalars_is_settable_from_a_comma_separated_value(
        self, tmp_path: Path
    ) -> None:
        """Several documented settings are lists.

        Refusing every list made `a2a.allow_hosts`,
        `permissions.network.allow_domains` and `multi_agent.allowed_effects`
        reachable only by hand-editing TOML -- and the first two are security
        settings, which is the worst place to have a setting people do not
        find.
        """
        settings = Settings(home=tmp_path, workspace=tmp_path)
        editor = ConfigEditor(settings, config_file=tmp_path / "config.toml")

        assert editor.set("a2a.allow_hosts", "a.example.com, b.example.com") == [
            "a.example.com",
            "b.example.com",
        ]
        assert settings.a2a.allow_hosts == ["a.example.com", "b.example.com"]

        assert editor.set("permissions.network.allow_domains", "x.test") == ["x.test"]
        assert editor.set("skill_dirs", "skills, .agent/skills") == ["skills", ".agent/skills"]

        # Enums coerce, and a bad value names the field.
        assert editor.set("multi_agent.allowed_effects", "read_only,network") == [
            EffectClass.READ_ONLY,
            EffectClass.NETWORK,
        ]
        with pytest.raises(ConfigError, match="multi_agent.allowed_effects"):
            editor.set("multi_agent.allowed_effects", "read_only,nonsense")

        # An empty value is an empty list, not an empty string.
        assert editor.set("a2a.allow_hosts", "") == []

    def test_unknown_paths_still_fail_loudly(self, tmp_path: Path) -> None:
        settings = Settings(home=tmp_path, workspace=tmp_path)
        editor = ConfigEditor(settings, config_file=tmp_path / "config.toml")

        with pytest.raises(ConfigError, match="unsupported config path"):
            editor.set("nonsense.field", "1")
        with pytest.raises(ConfigError, match="unknown agent field"):
            editor.set("agent.nonsense", "1")
        with pytest.raises(ConfigError, match="unknown setting"):
            editor.set("nonsense", "1")
