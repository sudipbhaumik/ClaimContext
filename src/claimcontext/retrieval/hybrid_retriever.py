"""HybridRetriever: dense + sparse (BM25) fused by RRF.

Exposes the identical search() signature as spec-2a's Retriever so ask(),
the CLI, and spec-2c reranker need no changes when switching retrieval modes.

query_filter  → passed through to dense Qdrant search (spec-3 entitlement)
allowed_ids   → passed through to BM25Index.search() (spec-3 entitlement)
Both are None in spec-2b; spec-3 populates them without touching this interface.
"""

from __future__ import annotations

import logging

from claimcontext.config import Settings
from claimcontext.observability.tracing import get_tracer
from claimcontext.retrieval.models import RetrievalResult
from claimcontext.retrieval.retriever import Retriever
from claimcontext.retrieval.rrf import rrf
from claimcontext.retrieval.sparse import BM25Index

log = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._dense = Retriever(settings)
        self._sparse = BM25Index(settings)

    def check_index_staleness(self) -> None:
        """Validate index currency, then build the in-memory BM25 index."""
        self._dense.check_index_staleness()
        self._sparse.build()

    def warm_up(self) -> None:
        """Force the embedding model to load now, not on first real query."""
        self._dense.warm_up()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        query_filter: object = None,  # dense-side Qdrant filter (spec-3)
        allowed_ids: frozenset[str] | None = None,  # sparse-side entitlement (spec-3)
    ) -> list[RetrievalResult]:
        """Retrieve via dense + sparse, fuse with RRF, return top_k results.

        rrf_fetch_k: fetch more candidates than top_k from each side so RRF has
        headroom to promote sparse-only results (e.g. WR-001 which dense misses).
        A BM25 rank-1 chunk absent from dense scores 1/(60+1) ≈ 0.0164, beating
        any dense-only chunk below roughly rank 14. The WR-001 endorsement wins
        regardless of how far dense buried it.
        """
        k = top_k if top_k is not None else self._settings.top_k
        fetch_k = min(k * 3, max(self._sparse.corpus_size, 1))
        tracer = get_tracer()

        with tracer.span(
            "retrieval.hybrid_search",
            as_type="retriever",
            input={"query": query, "top_k": k},
        ) as obs:
            dense_results = self._dense.search(query, top_k=fetch_k, query_filter=query_filter)
            sparse_results = self._sparse.search(query, top_n=fetch_k, allowed_ids=allowed_ids)

            sparse_ids = [r.chunk_id for r in sparse_results]

            fused = rrf(
                dense=dense_results,
                sparse_chunk_ids=sparse_ids,
                k=self._settings.rrf_k,
                payload_cache=self._sparse.payload_cache,
            )

            log.info(
                "hybrid search query=%r → fused %d → returning top %d",
                query[:60],
                len(fused),
                k,
            )
            result = fused[:k]
            if obs is not None:
                obs.update(output={"fused_count": len(fused), "returned_count": len(result)})
            return result
