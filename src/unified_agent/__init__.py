"""unified-ai-agent — local-first AI agent runtime.

Design thesis: do not merge four frameworks. Pick one durable core
(event-sourced state machine) and express the other paradigms as thin,
optional layers on top of it. The moat is what none of them do well:
resume semantics, context budgeting, and skill promotion.
"""

__version__ = "0.1.0"

from unified_agent.types import (  # noqa: F401
    EffectClass,
    Message,
    ModelResponse,
    ToolCall,
    TokenUsage,
)

__all__ = [
    "__version__",
    "EffectClass",
    "Message",
    "ModelResponse",
    "ToolCall",
    "TokenUsage",
]
