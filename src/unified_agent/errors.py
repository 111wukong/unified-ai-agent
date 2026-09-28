"""Exception hierarchy.

Rule: anything that can be shown to the model is an exception the runtime
catches and converts into an observation. Anything the model must never
see (permission denials that reveal the policy, config errors) propagates
with a redacted message.
"""

from __future__ import annotations


class UAAError(Exception):
    """Base class for all runtime errors."""


class ConfigError(UAAError):
    """Bad or missing configuration."""


class ModelError(UAAError):
    """Provider call failed (network, auth, rate limit, malformed response)."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolError(UAAError):
    """Tool raised while executing. Converted to a failed observation."""


class PermissionDenied(UAAError):
    """Policy refused the call. Never retried, never escalated."""

    def __init__(self, message: str, *, tool: str = "", effect: str = "") -> None:
        super().__init__(message)
        self.tool = tool
        self.effect = effect


class ConfirmationRequired(UAAError):
    """Policy demands human approval before the call may run."""

    def __init__(self, message: str, *, tool: str, arguments: dict, effect: str) -> None:
        super().__init__(message)
        self.tool = tool
        self.arguments = arguments
        self.effect = effect


class BudgetExceeded(UAAError):
    """Hard budget (steps / tokens / cost / wall clock) exhausted."""


class SchemaValidationError(UAAError):
    """Model output did not match the required schema after repair attempts."""
