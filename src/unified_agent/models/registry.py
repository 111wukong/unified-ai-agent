"""Model registry + factory."""

from __future__ import annotations

from typing import Iterable

from unified_agent.config import ModelSpec, Settings
from unified_agent.errors import ConfigError, ModelError
from unified_agent.models.base import ChatModel, TextProtocolModel

_BUILDERS: dict[str, str] = {
    "openai_compat": "unified_agent.models.openai_compat:OpenAICompatModel",
    "anthropic": "unified_agent.models.anthropic:AnthropicModel",
    "mock": "unified_agent.models.mock:MockModel",
}


def build_model(alias: str, spec: ModelSpec) -> ChatModel:
    target = _BUILDERS.get(spec.provider)
    if target is None:
        raise ConfigError(
            f"models.{alias}: unknown provider {spec.provider!r}; "
            f"known: {sorted(_BUILDERS)}"
        )
    module_path, cls_name = target.split(":")
    module = __import__(module_path, fromlist=[cls_name])
    model: ChatModel = getattr(module, cls_name)(spec)

    if spec.provider != "mock" and not model.capabilities.native_tool_calling:
        model = TextProtocolModel(model)
    return model


class ModelRegistry:
    """Lazily-built, cached model instances keyed by alias."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[str, ChatModel] = {}

    def aliases(self) -> list[str]:
        return sorted(self.settings.models)

    def get(self, alias: str | None = None) -> ChatModel:
        name = alias or self.settings.default_model
        if name in self._cache:
            return self._cache[name]
        spec = self.settings.model(name)
        if spec.provider != "mock" and not spec.api_key():
            raise ModelError(
                f"models.{name}: provider {spec.provider!r} needs an API key. "
                f"Set {spec.key_env() or 'the configured key env var'} in the environment, "
                f"or use --model mock to run offline.",
                retryable=False,
            )
        model = build_model(name, spec)
        self._cache[name] = model
        return model

    def try_get(self, alias: str) -> ChatModel | None:
        try:
            return self.get(alias)
        except (ConfigError, ModelError):
            return None

    def summarizer(self) -> ChatModel | None:
        """Prefer a cheap alias for compaction; fall back to the default."""
        for candidate in ("fast", "mini", "small", "haiku"):
            if candidate in self.settings.models:
                model = self.try_get(candidate)
                if model is not None:
                    return model
        try:
            return self.get()
        except ModelError:
            return None

    async def aclose(self) -> None:
        for model in self._cache.values():
            await model.aclose()
        self._cache.clear()


def available_providers() -> Iterable[str]:
    return _BUILDERS.keys()
