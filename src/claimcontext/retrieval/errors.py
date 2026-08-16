from __future__ import annotations

from typing import Literal

StalenessReason = Literal["empty", "model_mismatch"]


class IndexStalenessError(Exception):
    """Raised by Retriever.check_index_staleness() in two distinct cases:
    - Empty collection: index has no points; ingestion has not run.
    - Model mismatch: stored embedding_model differs from config.embedding_model;
      vectors are semantically incompatible — reindex required.
    Both refuse serving; the error message directs the operator to the correct fix.
    `reason` lets callers (e.g. /ready) report which case fired without parsing
    the message text.
    """

    def __init__(self, message: str, reason: StalenessReason) -> None:
        super().__init__(message)
        self.reason = reason


class LLMError(Exception):
    """Raised by LLMClient.complete() on any provider-level failure
    (timeout, auth error, network error, API error). Wraps the original."""


class ConfigurationError(Exception):
    """Raised at LLMClient construction time when a required config value
    (API key, base URL) is absent. Fails fast — not at call time."""
