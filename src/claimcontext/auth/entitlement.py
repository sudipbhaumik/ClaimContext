"""Entitlement resolution for spec-3 access control.

Single resolution path: Principal → EntitlementScope → (qdrant_filter, allowed_ids).

Both the dense-side Qdrant Filter and the sparse-side allowed_ids set are derived from
the same EntitlementScope, not computed independently. This prevents the two retrieval
paths from diverging on what "entitled" means — if they were computed separately, a bug
in either could create a subset mismatch and let non-entitled chunks through on one path.

Dense filtering is a TRUE pre-filter: Qdrant excludes non-entitled chunks from the ANN
search before any scoring occurs. Non-entitled vectors are never touched.

BM25 filtering is post-score: BM25Index scores the full in-memory corpus, then the
allowed_ids hook drops non-entitled chunks before they enter RRF fusion. Both paths
converge before fusion, so no unentitled chunk reaches reranking or the LLM.

Production path (not built): replace in-memory BM25 with Qdrant native sparse vectors
(SPLADE / BM42) or entitlement-partitioned BM25 indices to achieve filter-before-score
on both paths.

Authoring note (§3 Python-mastery split): build_qdrant_filter() and build_allowed_ids()
are authored by the human; this module's structure was reviewed and accepted as-is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from claimcontext.auth import models as _models

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntitlementScope:
    """Derived from a Principal; drives both retrieval filter paths."""

    adjuster_id: str
    region: str

    def as_filter(self) -> Filter:
        """Build the Qdrant Filter for dense pre-filtering.

        Authored by human (§3 Python-mastery split).
        """
        return Filter(
            must=[
                FieldCondition(key="region", match=MatchValue(value=self.region)),
                FieldCondition(key="assigned_adjuster", match=MatchValue(value=self.adjuster_id)),
            ]
        )

    def collect_allowed_ids(
        self, qdrant_url: str, collection: str, timeout: int | None = None
    ) -> frozenset[str]:
        """Scroll Qdrant with the entitlement filter; return entitled chunk_ids.

        Used as the allowed_ids argument to BM25Index.search() (sparse-side filter).
        Authored by human (§3 Python-mastery split) — the entitlement logic below
        is unchanged; `timeout` is a hardening addition (spec-7b finding): every
        other QdrantClient in this codebase sets an explicit timeout, this one
        didn't, and a live test hang (an unreachable Qdrant blocking a connection
        for 9.5 hours instead of failing fast) showed that omission is not
        theoretical — the OS cannot be trusted to refuse a dead connection quickly.
        """
        client = QdrantClient(url=qdrant_url, timeout=timeout)
        chunk_ids: list[str] = []
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=collection,
                scroll_filter=self.as_filter(),  # type: ignore[arg-type]
                with_payload=["chunk_id"],
                limit=256,
                offset=offset,
            )
            for point in results:
                if point.payload and "chunk_id" in point.payload:
                    chunk_ids.append(point.payload["chunk_id"])
            if offset is None:
                break

        log.debug(
            "entitlement: adjuster=%r region=%r entitled_chunks=%d",
            self.adjuster_id,
            self.region,
            len(chunk_ids),
        )
        return frozenset(chunk_ids)


def build_entitlement_scope(principal: _models.Principal) -> EntitlementScope:
    """Resolve a Principal to an EntitlementScope.

    This is the single entry point for entitlement resolution. Both as_filter()
    (dense path) and collect_allowed_ids() (sparse path) are methods on the returned
    scope object — they cannot diverge because they share the same adjuster_id + region.
    """
    return EntitlementScope(adjuster_id=principal.adjuster_id, region=principal.region)
