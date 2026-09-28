"""Secret redaction.

Two rules that the original spec only half-stated:

1. Redaction applies to *artifacts and event payloads*, not just log lines.
   Offloading a 200 KB tool output to disk is useless if that output
   contains the API key you just `cat`-ed.
2. Redaction happens on the way *in* to the store, not on the way out to
   the terminal. A secret that reached the database is already leaked.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

MASK = "***REDACTED***"

# High-confidence credential shapes. Deliberately conservative: false
# positives here destroy useful debug output.
_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_\-]{30,}\b"),  # telegram bot token
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),  # JWT
    re.compile(r"(?i)\b(?:authorization|api[-_]?key|token|secret|password)\b\s*[:=]\s*[\"']?([^\s\"',}]{8,})"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.=]{12,}"),
    # key=value style in .env content
    re.compile(r"(?m)^([A-Z0-9_]{3,}(?:KEY|TOKEN|SECRET|PASSWORD))\s*=\s*\S+"),
]


class Redactor:
    """Value-aware + pattern-aware redaction.

    Value-aware matters most: the runtime *knows* the API keys it is
    configured with, so it can mask them even when they appear in an
    unexpected shape (e.g. base64 in a URL query).
    """

    def __init__(self, secrets: Iterable[str] = (), *, extra_patterns: Iterable[str] = ()) -> None:
        self._secrets = sorted({s for s in secrets if s and len(s) >= 8}, key=len, reverse=True)
        self._patterns = list(_PATTERNS) + [re.compile(p) for p in extra_patterns]

    def __call__(self, text: str) -> str:
        if not text:
            return text
        out = text
        for secret in self._secrets:
            if secret in out:
                out = out.replace(secret, MASK)
        for pattern in self._patterns:
            out = pattern.sub(_replace, out)
        return out

    def deep(self, value: Any) -> Any:
        """Recursively redact strings inside dicts/lists."""
        if isinstance(value, str):
            return self(value)
        if isinstance(value, dict):
            return {k: self.deep(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.deep(v) for v in value]
        return value


def _replace(match: re.Match[str]) -> str:
    text = match.group(0)
    # Preserve the leading key name when the pattern captured it.
    if match.re.groups and match.group(1):
        prefix = text[: match.start(1) - match.start()]
        return f"{prefix}{MASK}"
    return MASK


DEFAULT_REDACTOR = Redactor()
