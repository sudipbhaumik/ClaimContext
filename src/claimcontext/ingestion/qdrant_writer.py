"""Qdrant upsert/delete wiring for spec-1b.

Creates the collection on first use. Upserts are batched and retried via
tenacity. Deletion uses a payload filter on doc_id so all chunks for a document
are removed atomically before the new chunks are inserted.

Payload index creation is idempotent — safe to call on every startup.
"""

from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from claimcontext.ingestion.models import Chunk

log = logging.getLogger(__name__)

_INDEXED_FIELDS = [
    "doc_id",
    "doc_type",
    "policy_number",
    "claim_number",
    "region",
    "assigned_adjuster",
    "chunker_version",
    "embedding_model",
]


class QdrantWriter:
    def __init__(self, url: str, collection: str, vector_dim: int, timeout: int) -> None:
        from qdrant_client import QdrantClient

        self._collection = collection
        self._dim = vector_dim
        self._client = QdrantClient(url=url, timeout=timeout)

    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            log.info("created Qdrant collection %r (dim=%d)", self._collection, self._dim)
        else:
            log.debug("Qdrant collection %r already exists", self._collection)

        for field in _INDEXED_FIELDS:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        if not chunks:
            return
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk.model_dump(exclude={"chunk_index"}),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        log.debug("upserted %d chunks for doc_id=%r", len(points), chunks[0].doc_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def delete_chunks_for_doc(self, doc_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        log.info("deleted old chunks for doc_id=%r", doc_id)
