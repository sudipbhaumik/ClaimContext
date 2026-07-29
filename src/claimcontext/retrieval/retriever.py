"""Dense retriever for spec-2a.

Wraps Qdrant vector search. The Embedder instance is module-level (lazy-loaded on
first search call) so tests can mock it cheaply.

query_filter is threaded through to Qdrant unchanged — spec-3 will inject an
entitlement filter without modifying this file.
"""

from __future__ import annotations

import logging

from claimcontext.config import Settings
from claimcontext.ingestion.embedder import Embedder
from claimcontext.retrieval.errors import IndexStalenessError
from claimcontext.retrieval.models import RetrievalResult

log = logging.getLogger(__name__)


class Retriever:
    def __init__(self, settings: Settings) -> None:
        from qdrant_client import QdrantClient

        self._settings = settings
        self._client = QdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout_seconds,
        )
        self._embedder = Embedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )

    def check_index_staleness(self) -> None:
        """Sample one point from the collection and validate embedding_model.

        Raises IndexStalenessError in two distinct cases:
        - Empty collection: "index is empty — run ingestion first"
        - Model mismatch: "index built with <stored>, config says <config> — reindex required"
        Same exception type, different messages, different operator remediations.
        """
        results, _ = self._client.scroll(
            collection_name=self._settings.qdrant_collection,
            limit=1,
            with_payload=True,
        )

        if not results:
            raise IndexStalenessError(
                f"index is empty (collection={self._settings.qdrant_collection!r}) — "
                "run ingestion first: python -m claimcontext"
            )

        payload = results[0].payload or {}
        stored_model = payload.get("embedding_model", "<unknown>")
        config_model = self._settings.embedding_model

        if stored_model != config_model:
            raise IndexStalenessError(
                f"index built with embedding_model={stored_model!r}, "
                f"config says {config_model!r} — reindex required: "
                "delete data/ingest/.hash_store.json and re-run python -m claimcontext"
            )

        log.debug("index staleness check passed (model=%s)", stored_model)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        query_filter: object = None,  # qdrant_client.models.Filter | None
    ) -> list[RetrievalResult]:
        """Embed query and run dense vector search over Qdrant.

        query_filter is passed through to Qdrant unchanged (None = no filter).
        spec-3 will inject an entitlement filter here without touching this method.
        """
        k = top_k if top_k is not None else self._settings.top_k

        query_vector = self._embedder.embed([query])[0]

        # qdrant-client 1.8+ replaced .search() with .query_points()
        response = self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=query_vector,
            limit=k,
            with_payload=True,
            query_filter=query_filter,  # type: ignore[arg-type]
        )

        results: list[RetrievalResult] = []
        for sp in response.points:
            p = sp.payload or {}
            results.append(
                RetrievalResult(
                    chunk_id=str(sp.id),
                    doc_id=p.get("doc_id", ""),
                    doc_type=p.get("doc_type", ""),
                    policy_number=p.get("policy_number"),
                    claim_number=p.get("claim_number"),
                    page=int(p.get("page", 1)),
                    section=p.get("section", ""),
                    score=float(sp.score),
                    text=p.get("text", ""),
                    embedding_model=p.get("embedding_model", ""),
                    chunker_version=p.get("chunker_version", ""),
                )
            )

        log.debug(
            "search query=%r top_k=%d → %d results (top score=%.4f)",
            query[:60],
            k,
            len(results),
            results[0].score if results else 0.0,
        )
        return results
