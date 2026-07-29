"""Reciprocal Rank Fusion — hand-implemented per CLAUDE.md §4.

Why rank-based, not score-based: cosine (dense) and BM25 (sparse) scores live on
incompatible scales. A cosine of 0.70 and a BM25 of 12.3 cannot be added or averaged
meaningfully. Rank position is scale-invariant — 1/(k+rank) is a smooth bounded reward
that rewards top-ranked documents from either list without caring about their raw score.

Reference: Cormack, Clarke, Buettcher (2009). k=60 is the standard paper default.
"""

from __future__ import annotations

import logging

from claimcontext.retrieval.models import RetrievalResult

log = logging.getLogger(__name__)


def rrf(
    dense: list[RetrievalResult],
    sparse_chunk_ids: list[str],  # ordered by BM25 rank (index 0 = rank 1)
    k: int = 60,
    payload_cache: dict[str, dict] | None = None,
) -> list[RetrievalResult]:
    """Fuse dense and sparse ranked lists by Reciprocal Rank Fusion.

    For each unique chunk_id appearing in either list:
        rrf_score = sum(1 / (k + rank_i))   for each list i containing the chunk

    Ranks are 1-indexed. A chunk absent from one list contributes 0 from that list.

    The output .score field carries the RRF score (not cosine, not BM25) so
    spec-2c reranker sees a single uniform signal.

    Chunks present only in the sparse list are reconstructed from payload_cache.
    payload_cache must map chunk_id -> dict of RetrievalResult fields; it holds
    every field scrolled from Qdrant at build time (full metadata, no defaults).

    Args:
        dense: ranked results from dense vector search (index 0 = rank 1)
        sparse_chunk_ids: chunk_ids from BM25 search in rank order (index 0 = rank 1)
        k: RRF constant (default 60 per paper)
        payload_cache: chunk_id -> full payload dict; required if any chunk_id in
            sparse_chunk_ids is not present in dense

    Returns:
        list[RetrievalResult] sorted descending by rrf_score, .score = rrf_score
    """
    # Step 1: accumulate RRF scores keyed by chunk_id
    scores: dict[str, float] = {}

    for rank_0, result in enumerate(dense):
        rank = rank_0 + 1  # 1-indexed
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)

    for rank_0, chunk_id in enumerate(sparse_chunk_ids):
        rank = rank_0 + 1
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    # Step 2: build a lookup of RetrievalResult by chunk_id from the dense list
    dense_by_id: dict[str, RetrievalResult] = {r.chunk_id: r for r in dense}

    # Step 3: assemble output — for sparse-only chunks, reconstruct from payload cache
    results: list[RetrievalResult] = []
    for chunk_id, rrf_score in scores.items():
        if chunk_id in dense_by_id:
            r = dense_by_id[chunk_id]
            results.append(r.model_copy(update={"score": rrf_score}))
        else:
            # sparse-only: full payload was scrolled at build time
            if payload_cache is None or chunk_id not in payload_cache:
                log.warning(
                    "payload_cache missing chunk_id=%s — skipping sparse-only hit", chunk_id
                )
                continue
            p = payload_cache[chunk_id]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=p.get("doc_id", ""),
                    doc_type=p.get("doc_type", ""),
                    policy_number=p.get("policy_number"),
                    claim_number=p.get("claim_number"),
                    page=int(p.get("page", 1)),
                    section=p.get("section", ""),
                    score=rrf_score,
                    text=p.get("text", ""),
                    embedding_model=p.get("embedding_model", ""),
                    chunker_version=p.get("chunker_version", ""),
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)

    log.debug(
        "rrf: %d dense + %d sparse → %d fused (top score=%.5f)",
        len(dense),
        len(sparse_chunk_ids),
        len(results),
        results[0].score if results else 0.0,
    )
    return results
