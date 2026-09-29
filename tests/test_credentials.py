"""Credential selection, and saying why a provider refused.

Two things a single-key adapter cannot express, both of which cost real time to
diagnose: a refusal that retrying will never fix (an empty balance), and one key
being a single point of failure.

Hermes keeps a per-credential `failure_reason` for the same reason; the point of
the classification is that "provider error" sends a user looking for a bug in
their code when the account is simply out of credit.
"""

from __future__ import annotations

import pytest

from unified_agent.config import ModelSpec
from unified_agent.models.credentials import (
    PERMANENT,
    CredentialPool,
    classify_failure,
    explain,
)
from unified_agent.models.openai_compat import OpenAICompatModel


class TestClassification:
    @pytest.mark.parametrize(
        ("status", "body", "expected"),
        [
            (402, "", "billing"),
            (401, "", "auth"),
            (403, "", "forbidden"),
            (404, "", "model_not_found"),
            (429, "slow down", "rate_limit"),
            (429, "You exceeded your current quota", "quota"),
            (429, "Insufficient balance", "quota"),
            (500, "", "provider_error"),
            (503, "", "provider_error"),
            (408, "", "timeout"),
            (422, "", "bad_request"),
            (None, "", "unknown"),
        ],
    )
    def test_the_reason_is_named(self, status, body, expected) -> None:  # noqa: ANN001
        assert classify_failure(status, body) == expected

    def test_a_plain_429_is_not_called_a_quota_problem(self) -> None:
        """Getting this wrong in the other direction is worse: setting a
        working key aside because of one burst of traffic."""
        assert classify_failure(429, "Too many requests") == "rate_limit"
        assert "rate_limit" not in PERMANENT

    def test_the_permanent_set_is_the_one_no_retry_fixes(self) -> None:
        assert PERMANENT == {"billing", "auth", "forbidden", "model_not_found"}
        assert "rate_limit" not in PERMANENT
        assert "provider_error" not in PERMANENT

    def test_billing_says_what_to_do(self) -> None:
        text = explain("billing", provider="deepseek", model="deepseek-flash")
        assert "no credit" in text
        assert "not a bug" in text, "the whole point is to stop someone hunting for one"
        assert "retrying will not help" in text

    def test_an_unknown_model_name_points_at_the_id(self) -> None:
        """The display name and the API id are routinely different, and this is
        the error that results."""
        text = explain("model_not_found", provider="deepseek", model="DeepSeek-V4.1-Flash")
        assert "/models" in text

    def test_reasons_without_a_useful_sentence_explain_nothing(self) -> None:
        assert explain("rate_limit", provider="x", model="y") == ""


class TestCredentialPool:
    def test_keys_resolve_in_the_order_given(self) -> None:
        pool = CredentialPool(envs=["A", "B"], environ={"A": "first", "B": "second"})
        assert [c.value for c in pool.credentials] == ["first", "second"]

    def test_a_missing_variable_is_reported_not_ignored(self) -> None:
        """'You did not set it' and 'it is set and rejected' are different
        problems, and only one of them is worth retrying."""
        pool = CredentialPool(envs=["A", "B"], environ={"A": "x"})
        assert pool.missing == ["B"]
        assert len(pool.credentials) == 1

    def test_a_blank_variable_counts_as_missing(self) -> None:
        pool = CredentialPool(envs=["A"], environ={"A": "   "})
        assert pool.missing == ["A"]

    def test_a_permanently_failed_key_is_set_aside(self) -> None:
        pool = CredentialPool(envs=["A", "B"], environ={"A": "dead", "B": "alive"})
        pool.mark_failed(pool.credentials[0], "billing")
        assert [c.value for c in pool.usable()] == ["alive"]

    def test_a_transient_failure_leaves_the_key_in_play(self) -> None:
        pool = CredentialPool(envs=["A"], environ={"A": "key"})
        pool.mark_failed(pool.credentials[0], "rate_limit")
        assert [c.value for c in pool.usable()] == ["key"]
        assert pool.credentials[0].failures == 1

    def test_the_status_report_never_contains_a_key(self) -> None:
        secret = "sk-abcdefghijklmnop"
        pool = CredentialPool(envs=["A"], environ={"A": secret})
        pool.mark_failed(pool.credentials[0], "billing")
        rendered = repr(pool.status())
        assert secret not in rendered
        assert "sk-a" in rendered and "mnop" in rendered, "a hint, so you can tell keys apart"

    def test_a_short_value_does_not_leak_through_the_hint(self) -> None:
        pool = CredentialPool(envs=["A"], environ={"A": "abc"})
        assert pool.status()[0]["key"] == "…"


class TestSpecKeyEnvs:
    def test_the_single_key_form_still_works(self) -> None:
        spec = ModelSpec(provider="openai_compat", api_key_env="ONE")
        assert spec.key_envs() == ["ONE"]

    def test_the_pool_form_is_appended_to_the_single_one(self) -> None:
        spec = ModelSpec(provider="openai_compat", api_key_env="ONE", api_key_envs=["TWO", "THREE"])
        assert spec.key_envs() == ["ONE", "TWO", "THREE"]

    def test_duplicates_are_dropped(self) -> None:
        """A name listed twice would make the pool report a phantom second
        credential, and 'two keys, both dead' reads differently from 'one key'."""
        spec = ModelSpec(provider="openai_compat", api_key_env="ONE", api_key_envs=["ONE", "TWO"])
        assert spec.key_envs() == ["ONE", "TWO"]

    def test_a_provider_default_is_used_when_nothing_is_configured(self) -> None:
        spec = ModelSpec(provider="anthropic")
        assert spec.key_envs() == ["ANTHROPIC_API_KEY"]


class TestAdapterRefusal:
    def model(self, spec: ModelSpec) -> OpenAICompatModel:  # noqa: ANN001
        return OpenAICompatModel(spec)

    def response(self, status: int, body: str = "") -> object:
        """Just enough of an httpx.Response for `_refusal` and `_error_text`."""

        class Fake:
            status_code = status
            text = body

            @staticmethod
            def json() -> dict:
                import json

                try:
                    return json.loads(body)
                except ValueError:
                    raise

        return Fake()

    def test_a_402_is_not_retryable_and_is_named(self) -> None:
        model = self.model(ModelSpec(provider="openai_compat", api_key_env="K"))
        message, retryable = model._refusal(self.response(402, '{"error":"no balance"}'))
        assert retryable is False
        assert "no credit" in message

    def test_a_429_is_retryable(self) -> None:
        model = self.model(ModelSpec(provider="openai_compat", api_key_env="K"))
        _, retryable = model._refusal(self.response(429, "slow down"))
        assert retryable is True

    def test_a_refusal_is_attributed_to_the_key_in_use(self, monkeypatch) -> None:  # noqa: ANN001
        """Otherwise the pool learns nothing and retries the dead key."""
        monkeypatch.setenv("K", "a-key")
        model = self.model(ModelSpec(provider="openai_compat", api_key_env="K"))
        model._in_use = model.pool.credentials[0]

        model._refusal(self.response(402))

        assert model.pool.credentials[0].failure_reason == "billing"
        assert model.pool.usable() == []

    def test_the_pool_reports_what_uaa_doctor_would_show(self, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setenv("PRIMARY", "sk-primary-key-0001")
        monkeypatch.setenv("BACKUP", "sk-backup-key-0002")
        spec = ModelSpec(provider="openai_compat", api_key_env="PRIMARY", api_key_envs=["BACKUP"])
        model = self.model(spec)
        model._in_use = model.pool.credentials[0]
        model._refusal(self.response(402))

        rows = model.pool.status()
        assert [row["env"] for row in rows] == ["PRIMARY", "BACKUP"]
        assert rows[0]["usable"] is False and rows[0]["reason"] == "billing"
        assert rows[1]["usable"] is True
        assert "sk-primary-key-0001" not in repr(rows)
