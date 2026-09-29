"""Which key to use, and why a provider said no.

Two things a single-key adapter cannot express, both of which cost real time to
diagnose:

* **A provider can refuse for reasons that retrying will never fix.** An
  exhausted balance answers 402 today and will answer 402 in an hour. Treating
  that like a flaky network means the retry loop burns the task's budget and
  then reports a generic failure -- and the user, seeing "provider error",
  goes looking for a bug in their code. Hermes keeps a per-credential
  `failure_reason` for exactly this; the classification here is the same idea.
* **One key is a single point of failure.** A pool is several keys for the same
  alias, tried in order, with the ones that have failed permanently set aside
  rather than retried on every call.

Nothing here is hidden state: the pool is inspectable, and `uaa doctor` reports
what each credential's last failure was.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

#: Failures that retrying cannot fix. A key that hits one of these is set aside
#: for the life of the process rather than retried on every call.
PERMANENT = frozenset({"billing", "auth", "forbidden", "model_not_found"})

#: Failures worth trying again, possibly on another key.
TRANSIENT = frozenset({"rate_limit", "quota", "provider_error", "timeout", "unknown"})


def classify_failure(status: int | None, body: str = "") -> str:
    """Why the provider refused, in terms someone can act on.

    Deliberately conservative about calling something a quota problem: a 429
    with the word "quota" in it is ambiguous -- it can mean "this key is out of
    credit" or "you are going too fast" -- so the wording decides. Getting that
    wrong in the other direction is worse: setting a working key aside because
    of one burst of traffic.
    """
    text = (body or "").lower()
    if status == 402:
        return "billing"
    if status == 401:
        return "auth"
    if status == 403:
        # A 403 is often "this key may not use this model", which is not
        # something a retry or another key fixes for the same account.
        return "forbidden"
    if status == 404:
        return "model_not_found"
    if status == 429:
        for hint in ("insufficient", "exceeded your current quota", "balance", "arrears"):
            if hint in text:
                return "quota"
        return "rate_limit"
    if status is not None and status >= 500:
        return "provider_error"
    if status in (408, 409, 425):
        return "timeout"
    if status is not None and status >= 400:
        return "bad_request"
    return "unknown"


def explain(reason: str, *, provider: str, model: str) -> str:
    """A sentence a user can act on, for the reasons that need one."""
    if reason == "billing":
        return (
            f"{provider} says the account has no credit (HTTP 402). This is not a "
            f"bug and retrying will not help -- top up the account, or point "
            f"{model!r} at a key that has credit."
        )
    if reason == "quota":
        return (
            f"{provider} says the quota is exhausted. Retrying will not help "
            "until it resets or the plan is raised."
        )
    if reason == "auth":
        return (
            f"{provider} rejected the key (HTTP 401). Check the environment "
            "variable is set in the process that runs the agent, and that the key "
            "belongs to this endpoint."
        )
    if reason == "forbidden":
        return f"{provider} refused this key for {model!r} (HTTP 403)."
    if reason == "model_not_found":
        return (
            f"{provider} does not know a model called {model!r}. The display name "
            "and the API id are often different -- check the provider's /models."
        )
    return ""


@dataclass
class Credential:
    env: str
    value: str
    failure_reason: str = ""
    failures: int = 0

    @property
    def usable(self) -> bool:
        return self.failure_reason not in PERMANENT

    @property
    def hint(self) -> str:
        """Masked, so a status report never prints a key."""
        if len(self.value) <= 8:
            return "…"
        return f"{self.value[:4]}…{self.value[-4:]}"


@dataclass
class CredentialPool:
    """Ordered credentials for one alias.

    Resolution happens at construction so a missing environment variable is
    visible immediately rather than at the first call, and the *names* that were
    set are kept even for the ones that failed to resolve -- "you did not set
    it" and "it is set and rejected" are different problems.
    """

    envs: list[str] = field(default_factory=list)
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    credentials: list[Credential] = field(default_factory=list)

    def __post_init__(self) -> None:
        for env in self.envs:
            if not env:
                continue
            value = (self.environ.get(env) or "").strip()
            if value:
                self.credentials.append(Credential(env=env, value=value))

    @property
    def missing(self) -> list[str]:
        """Env names that are configured but empty. Reported, not hidden."""
        present = {cred.env for cred in self.credentials}
        return [env for env in self.envs if env and env not in present]

    def usable(self) -> list[Credential]:
        return [cred for cred in self.credentials if cred.usable]

    def mark_failed(self, credential: Credential, reason: str) -> None:
        credential.failures += 1
        if reason in PERMANENT:
            credential.failure_reason = reason

    def status(self) -> list[dict[str, object]]:
        """For `uaa doctor`. Never includes a key."""
        return [
            {
                "env": cred.env,
                "key": cred.hint,
                "usable": cred.usable,
                "failures": cred.failures,
                "reason": cred.failure_reason,
            }
            for cred in self.credentials
        ]


__all__ = [
    "PERMANENT",
    "TRANSIENT",
    "Credential",
    "CredentialPool",
    "classify_failure",
    "explain",
]
