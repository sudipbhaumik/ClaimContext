"""BM25 sparse retrieval for spec-2b.

Tokenisation decision (directly determines whether Proof 1 passes):
    Identifier-preserving: split on whitespace, then for each word check whether
    it looks like a hyphenated identifier ([a-z0-9]+(-[a-z0-9]+)+). If yes, keep
    it as a single token; otherwise split further on any non-alphanumeric character.

    "WR-001"  -> ["wr-001"]   (one token -- only the endorsement chunk matches)
    "POL-3301" -> ["pol-3301"] (one token -- only the policy doc matches)

    The original split-on-non-alphanumeric failed Proof 1: "001" appeared as a
    fragment across claim IDs, dates, and amounts, so CLM-1001-notes outranked
    the endorsement. Applied identically at index time (build) and query time.
    Do not change the tokeniser without re-running Proof 1.

The full Qdrant payload (all RetrievalResult fields including embedding_model and
chunker_version) is cached during build(). Sparse-only RRF hits carry identical
metadata completeness to dense results — no partial defaults.

allowed_ids is the entitlement hook for spec-3. Always None in spec-2b; spec-3
will populate it with the set of chunk_ids the caller is entitled to see.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel

from claimcontext.config import Settings

log = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
_HYPHENATED_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")
_NONALNUM = re.compile(r"\W+")


def _tokenise(text: str) -> list[str]:
    tokens: list[str] = []
    for word in _WHITESPACE.split(text.lower()):
        if not word:
            continue
        if _HYPHENATED_ID.match(word):
            tokens.append(word)
        else:
            tokens.extend(t for t in _NONALNUM.split(word) if t)
    return tokens


class SparseResult(BaseModel):
    chunk_id: str
    doc_id: str
    rank: int  # 1-indexed position in BM25 result list
    bm25_score: float


class BM25Index:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bm25: Any | None = None  # BM25Okapi instance after build()
        self._chunk_ids: list[str] = []  # corpus order (index = BM25 doc index)
        self._payload_cache: dict[str, dict] = {}  # chunk_id → full Qdrant payload

    # ── Plumbing: Claude writes ───────────────────────────────────────────────

    def build(self) -> None:
        """Scroll all chunks from Qdrant, tokenise text, fit BM25Okapi in memory.

        Idempotent — replaces previous index state. Called once at startup after
        check_index_staleness() passes; call again after an ingest run to refresh.

        Pagination: QdrantClient.scroll() returns (points, next_page_offset).
        Repeat until next_page_offset is None to exhaust the collection.
        """
        from qdrant_client import QdrantClient
        from rank_bm25 import BM25Okapi

        client = QdrantClient(
            url=self._settings.qdrant_url,
            timeout=self._settings.qdrant_timeout_seconds,
        )

        chunk_ids: list[str] = []
        tokenised_corpus: list[list[str]] = []
        payload_cache: dict[str, dict] = {}

        offset = None
        total = 0
        while True:
            points, next_offset = client.scroll(
                collection_name=self._settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                p = pt.payload or {}
                chunk_id = str(pt.id)
                # Include section heading: structure-aware chunker stores the
                # heading in "section" not "text" (e.g. "ENDORSEMENT WR-001 —
                # WIND-DRIVEN RAIN COVERAGE EXTENSION" for the endorsement chunk).
                # BM25 must index both fields so exact-identifier queries hit the
                # primary document, not only policy cross-references.
                text = p.get("text", "")
                section = p.get("section", "")
                searchable = f"{section} {text}" if section else text
                chunk_ids.append(chunk_id)
                tokenised_corpus.append(_tokenise(searchable))
                payload_cache[chunk_id] = p
                total += 1

            if next_offset is None:
                break
            offset = next_offset

        self._chunk_ids = chunk_ids
        self._payload_cache = payload_cache
        self._bm25 = BM25Okapi(tokenised_corpus)
        log.info("BM25 index built: %d chunks", total)

    @property
    def payload_cache(self) -> dict[str, dict]:
        return self._payload_cache

    @property
    def corpus_size(self) -> int:
        return len(self._chunk_ids)

    # ── Core query path: I author ─────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_n: int,
        allowed_ids: frozenset[str] | None = None,
    ) -> list[SparseResult]:
        """Return top_n BM25 results for query.

        BM25Okapi.get_scores() returns a score array in corpus order (not ranked).
        We rank by score descending, optionally filter to allowed_ids, and return
        the top_n as SparseResult with 1-indexed rank positions.

        allowed_ids is the spec-3 entitlement hook — always None in spec-2b.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before search()")

        tokens = _tokenise(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Only keep chunks with a positive BM25 score. Zero-score chunks did not
        # match the query at all; including them in RRF would assign false sparse
        # ranks and allow non-matching documents to benefit from the sparse channel.
        indexed: list[tuple[int, float]] = [
            (i, float(scores[i]))
            for i in range(len(self._chunk_ids))
            if scores[i] > 0 and (allowed_ids is None or self._chunk_ids[i] in allowed_ids)
        ]
        indexed.sort(key=lambda x: x[1], reverse=True)

        results: list[SparseResult] = []
        for rank_0, (corpus_idx, score) in enumerate(indexed[:top_n]):
            chunk_id = self._chunk_ids[corpus_idx]
            results.append(
                SparseResult(
                    chunk_id=chunk_id,
                    doc_id=self._payload_cache.get(chunk_id, {}).get("doc_id", ""),
                    rank=rank_0 + 1,
                    bm25_score=score,
                )
            )

        log.debug(
            "BM25 search query=%r → %d results (top score=%.4f)",
            query[:60],
            len(results),
            results[0].bm25_score if results else 0.0,
        )
        return results
