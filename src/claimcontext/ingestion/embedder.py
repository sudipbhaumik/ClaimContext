"""Embedding wrapper for spec-1b.

Wraps sentence-transformers SentenceTransformer. Model is loaded once on first
call to embed() (lazy load) so --dry-run never instantiates the model.

normalize_embeddings=True is correct for cosine-distance Qdrant collections:
bge vectors must be L2-normalised so that cosine similarity equals dot product,
which is what Qdrant's cosine metric actually computes.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: _ST | None = None

    def _load(self) -> _ST:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        d = self._load().get_sentence_embedding_dimension()
        if d is None:
            raise RuntimeError(f"model {self._model_name!r} returned None for embedding dim")
        return int(d)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches. Returns one float vector per text."""
        if not texts:
            return []
        model = self._load()
        all_vectors: list[list[float]] = []
        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            t0 = time.monotonic()
            vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            elapsed = time.monotonic() - t0
            log.debug(
                "embedded batch %d-%d (%d texts) in %.2fs",
                batch_start,
                batch_start + len(batch),
                len(batch),
                elapsed,
            )
            all_vectors.extend(v.tolist() for v in vecs)
        return all_vectors
