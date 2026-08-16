"""Routing and decomposition logic for the spec-5a agent orchestrator.

Core decision logic (§3 Python-mastery split: I author, Claude wrote this at the
user's explicit direction for spec-5a — see spec-5a build order step 3).

Decided design (specs/spec-5a-agent-orchestrator.md, "Build order" step 3):
  - The router does NOT re-implement Tier-3 detection. `ask()`'s existing
    `_is_tier3_query()` guard stays the single source of truth; Tier-3 queries
    route to "single" and get refused inside the `ask()` call the graph makes.
  - The router DOES pre-filter two things `ask()` has no equivalent guard for:
      1. Topic/scope — is this a claims question at all? (CLAUDE.md §6B input
         guardrail; `ask()` has no topic check today, so this is new coverage,
         not a duplicate of anything.)
      2. Cross-entitlement, when a claim number is named explicitly in the query
         text. This is a genuine shortcut (avoids a wasted retrieval round trip
         when the answer is already cheaply knowable), not the authoritative
         enforcement point — `ask()`'s EntitlementScope pre-filter remains that,
         unconditionally, for every query this router routes to "single"/"multi".
         A miss here (query doesn't name a claim explicitly) is not a leak: it
         just means `ask()` catches it downstream instead, with the identical
         response shape either way.

Any refusal manufactured directly by this module (without calling ask()) reuses
ask.py's exact `_REFUSE_MESSAGE`, `_query_hash()`, and `_audit()` — imported, not
duplicated — so the response and audit-log shape are byte-identical to what
`ask()` would have produced for the same case (§6B: no new distinguishable
refusal path introduced at this layer).
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from claimcontext.auth.models import Principal
from claimcontext.config import Settings
from claimcontext.retrieval.ask import (  # noqa: PLC2701 — intentional reuse, see module docstring
    _audit,
    _query_hash,
)
from claimcontext.retrieval.errors import LLMError
from claimcontext.retrieval.llm_client import LLMClient

log = logging.getLogger(__name__)

_CLAIM_ID_PATTERN = re.compile(r"\bCLM-\d{4}\b", re.IGNORECASE)

_SCOPE_CHECK_SYSTEM = (
    "You classify whether a question is about insurance claims, policies, "
    "coverage, or claim documents. Answer with exactly one word: YES if the "
    "question is a claims/policy/coverage question, NO if it is about anything "
    "else (general knowledge, other topics, small talk)."
)

_MULTI_PART_SYSTEM = (
    "You classify whether a claims question asks ONE thing or asks TWO OR "
    "MORE genuinely separate things (e.g. two different claim numbers, or two "
    'unrelated coverage questions joined by "and"). A single question about '
    "a single claim is SINGLE even if it is phrased in detail or mentions "
    "multiple details about that one thing. Answer with exactly one word: "
    "SINGLE or MULTI.\n\n"
    "Example (SINGLE): What is the current reserve amount set on claim "
    "CLM-1001?\n"
    "Example (SINGLE): What did the adjuster conclude about coverage for "
    "CLM-1004, and what was the basis for that conclusion?\n"
    "Example (MULTI): What did the adjuster conclude about coverage for "
    "CLM-1004, and why was CLM-1003 denied?"
)

_DECOMPOSE_SYSTEM = (
    "Split a compound insurance-claims question into its separate questions, "
    "one full question per line, no numbering, no extra text. Every claim "
    "number in the original must appear in exactly one output line — never "
    "drop one. Do not invent facts or ask about anything not in the original "
    "question.\n\n"
    "Example input:\n"
    "What did the adjuster conclude about coverage for CLM-1004, and why was "
    "CLM-1003 denied?\n\n"
    "Example output:\n"
    "What did the adjuster conclude about coverage for CLM-1004?\n"
    "Why was CLM-1003 denied?"
)


def extract_claim_ids(query: str) -> list[str]:
    """Regex-extract claim numbers (e.g. "CLM-1004") literally named in the query.

    Advisory only — a miss (no claim number found) is not a security gap, it just
    means the cross-entitlement pre-filter has nothing to check and the query
    routes through normally, where ask()'s EntitlementScope is the real gate.
    """
    return sorted({m.group(0).upper() for m in _CLAIM_ID_PATTERN.finditer(query)})


def _claim_owner(claim_id: str, settings: Settings) -> tuple[str, str] | None:
    """Look up (region, assigned_adjuster) for a claim number via a single cheap
    Qdrant scroll (limit=1, no vector search, no reranking) — not a full retrieval.

    Returns None if the claim number doesn't exist in the corpus at all (not a
    security-relevant case; just means the pre-filter can't say anything, and the
    query proceeds to ask() as normal).
    """
    client = QdrantClient(url=settings.qdrant_url)
    results, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="claim_number", match=MatchValue(value=claim_id))]
        ),
        with_payload=["region", "assigned_adjuster"],
        limit=1,
    )
    if not results or not results[0].payload:
        return None
    payload = results[0].payload
    region = payload.get("region")
    adjuster = payload.get("assigned_adjuster")
    if region is None or adjuster is None:
        return None
    return (region, adjuster)


def _is_cross_entitlement(query: str, principal: Principal, settings: Settings) -> bool:
    """True only when EVERY named, resolvable claim is confirmed cross-entitlement.

    Deliberately not "any" — a query naming one entitled claim and one
    non-entitled claim (e.g. "what about CLM-1001, and why was CLM-1003
    denied?") must NOT be refused outright here. That's exactly the mixed-
    entitlement multi-part case: it should route to "multi", decompose into
    per-claim sub-queries, and let each sub-query's own ask() call apply the
    real EntitlementScope gate — which correctly answers the entitled claim's
    part and refuses the other's. Refusing the whole query here for a mix
    would incorrectly deny an adjuster their own entitled data just because it
    was asked about alongside a claim they don't own.
    """
    claim_ids = extract_claim_ids(query)
    if not claim_ids:
        return False

    owners = [(cid, _claim_owner(cid, settings)) for cid in claim_ids]
    known_owners = [(cid, owner) for cid, owner in owners if owner is not None]
    if not known_owners:
        return False

    all_cross = all(
        region != principal.region or adjuster != principal.adjuster_id
        for _, (region, adjuster) in known_owners
    )
    if all_cross:
        # Hash claim IDs before logging — same discipline as ask.py's audit log
        # (_query_hash): a plaintext claim ID in the general log is exactly the
        # kind of disclosure surface §6B is designed to avoid. Region is the
        # PRINCIPAL's own region (not the target claim's), so it carries no
        # claim-existence information and is safe to log as-is.
        hashed_ids = [_query_hash(cid) for cid, _ in known_owners]
        log.info(
            "router: cross-entitlement pre-filter fired — all named claims "
            "(hashed: %s) are outside principal_region=%s",
            hashed_ids,
            principal.region,
        )
    return all_cross


def _call_llm_with_retry(system: str, user: str, llm: LLMClient, settings: Settings) -> str:
    """Retry an agent-internal LLM call (scope check, multi-part check, decompose)
    on LLMError, bounded by settings.agent_retry_attempts (spec-5b hardening).

    Uses a runtime Retrying object, not a static @retry decorator (the codebase's
    existing pattern in qdrant_writer.py), because the retry count must come from
    settings — config, not a hardcoded magic number (CLAUDE.md §2A.2).
    """
    last_error: LLMError | None = None
    for attempt in Retrying(
        stop=stop_after_attempt(settings.agent_retry_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(LLMError),
        reraise=True,
    ):
        with attempt:
            try:
                return llm.complete(system, user)
            except LLMError as exc:
                last_error = exc
                raise
    # Unreachable — Retrying either returns above or reraises — but keeps mypy
    # happy about the function always returning str or raising.
    assert last_error is not None
    raise last_error


def _is_in_scope(query: str, llm: LLMClient, settings: Settings) -> bool:
    """Cheap topic/scope check (CLAUDE.md §6B input guardrail). Not perfectly
    reliable (see ask.py's own documented Tier-3-pattern caveats for the same
    class of heuristic) — false negatives here are caught later by the refuse
    gate inside ask() anyway (weak/no retrieval context → refuse), so this is a
    cost-saving early exit, not the only line of defense against off-topic
    queries reaching generation.
    """
    response = _call_llm_with_retry(_SCOPE_CHECK_SYSTEM, query, llm, settings).strip().upper()
    return response.startswith("YES")


def _is_multi_part(query: str, llm: LLMClient, settings: Settings) -> bool:
    """Cheap single-vs-multi classification. Two or more distinct claim numbers
    named explicitly is a fast, free signal independent of the LLM call; only
    fall back to asking the model when that signal is absent or ambiguous.
    """
    if len(extract_claim_ids(query)) >= 2:
        return True
    response = _call_llm_with_retry(_MULTI_PART_SYSTEM, query, llm, settings).strip().upper()
    return response.startswith("MULTI")


def classify_route(
    query: str,
    principal: Principal,
    llm: LLMClient,
    settings: Settings,
) -> Literal["single", "multi", "refuse"]:
    """The routing decision. See module docstring for the design this implements."""
    if _is_cross_entitlement(query, principal, settings):
        return "refuse"
    if not _is_in_scope(query, llm, settings):
        return "refuse"
    if _is_multi_part(query, llm, settings):
        return "multi"
    return "single"


def manufacture_refusal_audit(query: str, principal: Principal) -> None:
    """Log the audit entry for a router-manufactured refusal, in the same shape
    ask() itself would log for an entitlement-denied query — see module docstring.
    """
    _audit(
        adjuster_id=principal.adjuster_id,
        region=principal.region,
        query_hash=_query_hash(query),
        chunks_retrieved=0,
        decision="denied",
    )


def decompose_query(query: str, settings: Settings, llm: LLMClient) -> list[str]:
    """Split a multi-part query into claim-scoped sub-queries.

    Falls back to [query] unchanged if the LLM returns nothing usable — a
    decompose failure must not lose the query entirely.

    Deliberately does NOT cap the result at settings.agent_max_sub_queries here
    (spec-5b — a silent cap would truncate a legitimate answer's coverage without
    telling anyone). The caller (graph.py's multi_node) checks the length against
    the budget and escalates if it's exceeded — a visible, distinct outcome
    instead of a silent, undersized answer.
    """
    raw = _call_llm_with_retry(_DECOMPOSE_SYSTEM, query, llm, settings)
    sub_queries = [line.strip() for line in raw.splitlines() if line.strip()]
    if not sub_queries:
        log.warning("decompose_query: LLM returned no usable sub-queries, falling back to original")
        return [query]
    return sub_queries
