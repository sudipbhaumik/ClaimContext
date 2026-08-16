"""Metadata-filter tool (spec-5b): exact structured lookup by claim/policy/doc_type,
scoped by entitlement — for when the caller already knows the precise scope (e.g. a
decomposed sub-query naming one claim) and wants precision over semantic recall.

Not a semantic search: no query embedding, no BM25, no reranking. A direct Qdrant
scroll filtered on exact metadata match. All returned chunks get score=1.0 (there is
no ranking signal for an exact match — every result satisfies the filter equally).

Entitlement is NOT optional here (spec-5b Decisions): every filter this tool builds
ANDs in the same EntitlementScope filter ask()/HybridRetriever already use, via the
same single-resolution-path pattern from auth/entitlement.py. No new retrieval path
in this system is exempt from spec-3's access-control model — this tool included.
"""

from __future__ import annotations

import logging
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http.models import Record
from qdrant_client.models import FieldCondition, Filter, MatchValue
from tenacity import retry, stop_after_attempt, wait_exponential

from claimcontext.auth.entitlement import build_entitlement_scope
from claimcontext.auth.models import Principal
from claimcontext.config import Settings
from claimcontext.retrieval.models import RetrievalResult

log = logging.getLogger(__name__)

_ScrollOffset = int | str | UUID | None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _scroll_page(
    client: QdrantClient, collection: str, scroll_filter: Filter, offset: _ScrollOffset
) -> tuple[list[Record], _ScrollOffset]:
    """One page of the entitlement-scoped scroll, retried on transient failure
    (spec-5b hardening). A static @retry decorator here (matching the existing
    qdrant_writer.py convention) rather than the settings-driven Retrying used in
    routing.py — this helper isn't given settings; 3 is the same fixed default
    already used elsewhere in this codebase for Qdrant retry.
    """
    return client.scroll(
        collection_name=collection,
        scroll_filter=scroll_filter,  # type: ignore[arg-type]
        with_payload=True,
        with_vectors=False,
        limit=256,
        offset=offset,
    )


def search_by_metadata(
    settings: Settings,
    principal: Principal,
    claim_number: str | None = None,
    policy_number: str | None = None,
    doc_type: str | None = None,
) -> list[RetrievalResult]:
    """Return every entitled chunk matching the given metadata filters exactly.

    At least one of claim_number/policy_number/doc_type should be given — an
    unfiltered call just returns the caller's entire entitled corpus, which is
    rarely what's wanted but is not itself unsafe (entitlement still applies).
    """
    scope = build_entitlement_scope(principal)

    extra_conditions: list[FieldCondition] = []
    if claim_number is not None:
        extra_conditions.append(
            FieldCondition(key="claim_number", match=MatchValue(value=claim_number))
        )
    if policy_number is not None:
        extra_conditions.append(
            FieldCondition(key="policy_number", match=MatchValue(value=policy_number))
        )
    if doc_type is not None:
        extra_conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))

    # Combine via nested Filter (AND of the entitlement filter + the metadata
    # filters), not by merging .must lists — keeps the entitlement filter's own
    # structure untouched and avoids depending on its internal shape.
    combined_filter = (
        Filter(must=[scope.as_filter(), Filter(must=extra_conditions)])  # type: ignore[arg-type]
        if extra_conditions
        else scope.as_filter()
    )

    client = QdrantClient(url=settings.qdrant_url, timeout=settings.agent_tool_timeout_seconds)
    results: list[RetrievalResult] = []
    offset = None

    while True:
        points, offset = _scroll_page(client, settings.qdrant_collection, combined_filter, offset)
        for pt in points:
            p = pt.payload or {}
            results.append(
                RetrievalResult(
                    chunk_id=str(pt.id),
                    doc_id=p.get("doc_id", ""),
                    doc_type=p.get("doc_type", ""),
                    policy_number=p.get("policy_number"),
                    claim_number=p.get("claim_number"),
                    page=int(p.get("page", 1)),
                    section=p.get("section", ""),
                    score=1.0,
                    text=p.get("text", ""),
                    embedding_model=p.get("embedding_model", ""),
                    chunker_version=p.get("chunker_version", ""),
                    effective_date=p.get("effective_date"),
                    expiry_date=p.get("expiry_date"),
                    loss_date=p.get("loss_date"),
                )
            )
        if offset is None:
            break

    log.info(
        "metadata_filter: adjuster=%r claim=%r policy=%r doc_type=%r -> %d chunks",
        principal.adjuster_id,
        claim_number,
        policy_number,
        doc_type,
        len(results),
    )
    return results
