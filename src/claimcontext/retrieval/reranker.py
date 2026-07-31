"""Cross-encoder reranker for spec-2c.

bge-reranker-base applies sigmoid internally — its predict() output is already in [0, 1].
A score of ~0.97+ means the passage is clearly relevant to the query.
A score of ~0.001 means clearly not relevant (wrong passage for an in-domain query).
A score of ~0.5 is the model's uncertainty floor (off-corpus queries land here).

This is different from the raw logit range (-11 to +11) of a plain CrossEncoder trained
for other tasks. Do not apply sigmoid to the output — it is already applied.

The refuse_threshold must exceed 0.5 to reject off-corpus queries; spec-4 calibrates
the exact value against the golden set.
"""

from __future__ import annotations

import logging

from claimcontext.config import Settings
from claimcontext.retrieval.models import RetrievalResult

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_name = settings.reranker_model
        self._model = None  # lazy-loaded on first rerank() call

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading reranker model %s", self._model_name)
            # Force CPU: MPS (Apple Silicon) produces non-deterministic float results
            # for cross-encoder predict(), causing the same query to score differently
            # across process invocations. A score of 0.487 in one run and 0.563 in the
            # next would flip the refuse gate — unacceptable in a regulated domain.
            # CPU is slower but deterministic; the reranker is not on the hot path.
            self._model = CrossEncoder(self._model_name, device="cpu")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Re-score candidates with cross-encoder. Returns same list, re-ordered,
        .score replaced with cross-encoder score (already in [0,1] for bge-reranker).

        Uses section + text as the passage so the reranker sees the same content as
        the BM25 index (section heading is part of chunk identity for identifier queries).
        """
        if not candidates:
            return []

        self._load()

        pairs = [
            (query, f"{r.section} {r.text}".strip() if r.section else r.text) for r in candidates
        ]
        scores = self._model.predict(pairs)  # type: ignore[attr-defined]

        log.debug(
            "rerank query=%r candidates=%d top_score=%.4f",
            query[:60],
            len(candidates),
            float(max(scores)),
        )

        reranked = [
            r.model_copy(update={"score": float(s)})
            for r, s in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
