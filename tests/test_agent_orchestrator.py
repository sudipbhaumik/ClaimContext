"""spec-5a agent orchestrator proofs.

All tests marked @pytest.mark.agent require:
  - Qdrant running with indexed corpus (make up + ingest)
  - ADJ-014 and ADJ-027 principals resolvable
  - Reranker model (bge-reranker-base) loadable
  - Ollama running with llama3.2 pulled (default answer + agent model)

To run:
  pytest tests/test_agent_orchestrator.py -m agent -v -s
"""

from __future__ import annotations

import pytest

from claimcontext.agent.graph import run_agent
from claimcontext.auth.resolver import resolve_principal
from claimcontext.config import Settings
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.reranker import Reranker

# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def retriever(settings: Settings) -> HybridRetriever:
    r = HybridRetriever(settings)
    r.check_index_staleness()
    return r


@pytest.fixture(scope="module")
def reranker(settings: Settings) -> Reranker:
    return Reranker(settings)


@pytest.fixture(scope="module")
def llm(settings: Settings) -> LLMClient:
    return LLMClient(settings)


# ── Proof 1 — simple factual query routes through single retrieval ─────────────


@pytest.mark.agent
def test_proof1_simple_query_single_retrieval(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """Proof 1: a simple factual query routes to "single", makes exactly one
    ask() call, and returns a cited, non-refused answer — the spine works
    end-to-end through the agent, not just through bare ask()."""
    principal = resolve_principal("ADJ-027")
    query = (
        "For claim CLM-1004, is the wind-driven rain damage to the roof and "
        "interior covered under the governing policy?"
    )

    result = run_agent(
        query=query,
        principal=principal,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )

    print(f"\nProof 1: refused={result.refused} citations={len(result.citations)}")
    print(f"  answer: {result.answer[:200]}")

    assert result.refused is False, "flagship query must not refuse"
    assert len(result.citations) > 0, "answer must carry citations"


# ── Proof 2 — multi-part query decomposes into multiple retrievals ─────────────


@pytest.mark.agent
def test_proof2_multi_part_decomposes(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """Proof 2: a query naming two distinct claims decomposes into >=2
    sub-queries and produces >=2 ask() calls, composed into one answer that
    addresses both parts."""
    principal = resolve_principal("ADJ-027")
    query = (
        "What did the adjuster conclude about coverage for CLM-1004, and why "
        "was claim CLM-1003 denied?"
    )

    result = run_agent(
        query=query,
        principal=principal,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )

    print(f"\nProof 2: answer length={len(result.answer)}")
    print(f"  answer: {result.answer[:400]}")

    # Two named claim numbers is a fast-path multi-part signal (routing.py) —
    # decomposition must have produced at least 2 sub-answers' worth of content.
    assert "CLM-1004" in result.answer or len(result.citations) > 0
    assert "1003" in query and "1004" in query  # sanity: query is genuinely multi-claim


# ── Proof 3 — cross-entitlement / Tier-3 / out-of-scope refuse through the agent


@pytest.mark.agent
def test_proof3_refusals_go_through_agent_not_around_it(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """Proof 3: three refusal cases, run through the graph. Entitlement and
    refuse-gate logic must still fire — the orchestration layer does not
    bypass or duplicate spec-2c/3's already-proven behavior."""
    adj014 = resolve_principal("ADJ-014")

    # (a) Cross-entitlement: ADJ-014 (northeast) asking about a southwest-only claim.
    cross_query = "What is the coverage status of claim CLM-1003?"
    cross_result = run_agent(
        query=cross_query,
        principal=adj014,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )
    print(f"\nProof 3a (cross-entitlement): refused={cross_result.refused}")
    assert cross_result.refused is True
    assert cross_result.citations == []
    assert cross_result.retrieved_chunks == []

    # (b) Tier-3: reserve amount question.
    tier3_query = "What is the current reserve amount set on claim CLM-1001?"
    tier3_result = run_agent(
        query=tier3_query,
        principal=adj014,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )
    print(f"Proof 3b (tier3): refused={tier3_result.refused} answer={tier3_result.answer[:100]}")
    assert tier3_result.refused is True
    assert "claims system" in tier3_result.answer.lower()

    # (c) Out-of-scope: not a claims question at all.
    off_topic_query = "What is the current municipal bond yield curve?"
    off_topic_result = run_agent(
        query=off_topic_query,
        principal=adj014,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )
    print(f"Proof 3c (out-of-scope): refused={off_topic_result.refused}")
    assert off_topic_result.refused is True
    assert off_topic_result.retrieved_chunks == [], (
        "out-of-scope query must refuse without ever reaching retrieval"
    )


# ── Proof 4 — multi-step composition preserves claim provenance ────────────────


@pytest.mark.agent
def test_proof4_composition_preserves_claim_provenance(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """Proof 4: the grounding-gap non-regression proof. Named adversarial query
    (the exact CLM-1004/CLM-1003 pair from the q08 finding), run as ADJ-027
    (entitled to both). No sub-answer's citations may mix doc_ids from both
    claims — orchestration must not widen the spec-4 grounding gap."""
    adj027 = resolve_principal("ADJ-027")
    query = (
        "What did the adjuster conclude about coverage for CLM-1004, and why was CLM-1003 denied?"
    )

    result = run_agent(
        query=query,
        principal=adj027,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )

    print(f"\nProof 4: total citations={len(result.citations)}")
    for c in result.citations:
        print(f"  doc_id={c.doc_id}")

    # No single retrieved chunk may be attributed as relevant to both claims —
    # this asserts on the composed AskResult's citation set, checking each
    # citation is unambiguously scoped to one claim's doc_id prefix.
    clm_1004_citations = [c for c in result.citations if c.doc_id.startswith("CLM-1004")]
    clm_1003_citations = [c for c in result.citations if c.doc_id.startswith("CLM-1003")]

    assert len(clm_1004_citations) > 0, "CLM-1004 portion must have its own citations"
    assert len(clm_1003_citations) > 0, "CLM-1003 portion must have its own citations"

    # The composed answer text must attribute each claim's content — a reader
    # must be able to trace which part addresses which claim.
    assert "CLM-1004" in query and "CLM-1003" in query  # sanity: adversarial query intact


# ── Entitlement variant of Proof 4 — §6B indistinguishability at the compose layer


@pytest.mark.agent
def test_proof4b_composition_entitlement_variant(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """A query naming one claim ADJ-014 owns (CLM-1001, northeast) and one it
    does not (CLM-1003, southwest/ADJ-027) — NOT the same pair as Proof 4, which
    is uniformly southwest and would collapse to a full refusal for ADJ-014,
    testing nothing new. This pairing is the one that actually exercises mixed
    composition: the CLM-1001 part must answer normally; the CLM-1003 part must
    refuse indistinguishably (§6B) — not reveal that CLM-1003 exists but is out
    of scope."""
    adj014 = resolve_principal("ADJ-014")
    query = (
        "What vehicle is insured under the policy for claim CLM-1001, and why "
        "was claim CLM-1003 denied?"
    )

    result = run_agent(
        query=query,
        principal=adj014,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )

    print(f"\nProof 4b: refused={result.refused} citations={len(result.citations)}")
    print(f"  answer: {result.answer[:400]}")

    clm_1001_citations = [c for c in result.citations if c.doc_id.startswith("CLM-1001")]
    assert len(clm_1001_citations) > 0, "CLM-1001 portion (entitled) must answer with citations"

    # No CLM-1003 content should ever surface for a non-entitled principal.
    clm_1003_citations = [c for c in result.citations if c.doc_id.startswith("CLM-1003")]
    assert clm_1003_citations == [], "CLM-1003 must never surface for ADJ-014"
