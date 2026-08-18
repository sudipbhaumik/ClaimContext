"""Tests for claimcontext.config — validates Settings loads correctly."""

import os
from unittest.mock import patch

import pytest

from claimcontext.config import Settings, get_settings


def test_settings_loads_with_defaults() -> None:
    s = Settings()
    assert s.app_name == "claimcontext"
    assert s.environment == "development"
    assert s.log_level == "INFO"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.qdrant_collection == "claimcontext"


def test_missing_optional_secrets_do_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    # delenv, not just patch.dict(clear=False) — a real key set in the dev
    # shell (e.g. OPENAI_API_KEY for other work) must not leak into this
    # test's assertion, or the test's pass/fail depends on whoever's running
    # it, not on the code under test.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.anthropic_api_key is None
    assert s.openai_api_key is None


def test_env_var_overrides_default() -> None:
    with patch.dict(os.environ, {"APP_NAME": "test-app", "ENVIRONMENT": "staging"}):
        s = Settings()
    assert s.app_name == "test-app"
    assert s.environment == "staging"


def test_get_settings_returns_cached_instance() -> None:
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_retrieval_defaults_are_sane() -> None:
    s = Settings()
    assert s.top_k > 0
    assert s.rerank_top_n <= s.top_k
    assert 0.0 < s.refuse_threshold < 1.0
    assert s.chunk_overlap < s.chunk_size
