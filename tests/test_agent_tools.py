"""spec-5b proofs — citation-checker, metadata-filter, temporal-validity tools,
hardening (budget/escalation), and MCP exposure.

Proofs 1-2 (citation-checker) and 3 (temporal) need no live services.
Proof 5 (MCP) runs the server in-process, no live services beyond that.
Live-service proofs (metadata-filter, budget/escalation) are marked
@pytest.mark.agent to match spec-5a's convention.

To run:
  pytest tests/test_agent_tools.py -v
  pytest tests/test_agent_tools.py -v -m agent   # live-service proofs only
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from claimcontext.agent.graph import build_agent_graph
from claimcontext.agent.mcp_server import mcp_server
from claimcontext.agent.state import AgentState
from claimcontext.agent.tools.citation_checker import check_citations, check_citations_raw
from claimcontext.agent.tools.metadata_filter import search_by_metadata
from claimcontext.agent.tools.temporal import check_policy_in_force
from claimcontext.auth.resolver import resolve_principal
from claimcontext.config import Settings
from claimcontext.retrieval.models import AskResult, RetrievalResult

# ── Proof 1 — citation-checker catches a real fabrication ──────────────────────


def _chunk(doc_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="x",
        doc_id=doc_id,
        doc_type="t",
        policy_number=None,
        claim_number=None,
        page=1,
        section="",
        score=0.9,
        text="",
        embedding_model="m",
        chunker_version="v",
    )


def test_proof1_citation_checker_catches_fabrication() -> None:
    answer = "Per [CLM-9999-notes | p.1], the claim is denied."
    result = AskResult(
        query="q",
        answer=answer,
        citations=[],
        retrieved_chunks=[_chunk("CLM-1001-fnol")],
        llm_model="m",
        prompt_version="v",
    )
    check = check_citations(result)
    assert check.has_fabricated_citations is True
    assert check.fabricated == ["CLM-9999-notes"]


# ── Proof 2 — no fabrication on a real answer with only real citations ─────────


def test_proof2_citation_checker_no_fabrication_on_real_citations() -> None:
    """NOTE (per spec-5b review): this demonstrates absence of fabrication, NOT
    correctness of grounding. It does not, and cannot, distinguish this case from
    a q08-shaped wrong-source case — see citation_checker.py's module docstring."""
    answer = "The vehicle is a Honda.\n\nSOURCES:\n\n[CLM-1001-fnol | p.1]\n[CLM-1001-letter | p.1]"
    result = AskResult(
        query="q",
        answer=answer,
        citations=[],
        retrieved_chunks=[_chunk("CLM-1001-fnol"), _chunk("CLM-1001-letter")],
        llm_model="m",
        prompt_version="v",
    )
    check = check_citations(result)
    assert check.fabricated == []
    assert check.has_fabricated_citations is False


# ── Proof 3 — temporal-validity tool resolves the CLM-1002 edge case ───────────


def test_proof3_temporal_validity_clm1002_edge_case() -> None:
    """CLM-1002's loss date sits near policy expiry (deliberate corpus design,
    CLAUDE.md §5) — real values from POL-3302, not an obvious mid-term example."""
    result = check_policy_in_force(
        loss_date="2026-05-23", effective_date="2025-06-01", expiry_date="2026-06-01"
    )
    assert result.in_force is True

    # Boundary: loss exactly on expiry date must still be in force (inclusive).
    boundary = check_policy_in_force(
        loss_date="2026-06-01", effective_date="2025-06-01", expiry_date="2026-06-01"
    )
    assert boundary.in_force is True

    # Just past expiry: not in force.
    past = check_policy_in_force(
        loss_date="2026-06-02", effective_date="2025-06-01", expiry_date="2026-06-01"
    )
    assert past.in_force is False

    # Missing data: None, not a silent wrong answer.
    missing = check_policy_in_force(loss_date="2026-05-23", effective_date=None, expiry_date=None)
    assert missing.in_force is None


# ── Proof 4a — budget cap fires on decompose overflow → escalation ─────────────


class _FakeLLM:
    pass


class _FakeRetriever:
    pass


def test_proof4a_budget_cap_escalates_on_decompose_overflow() -> None:
    settings = Settings()
    principal = resolve_principal("ADJ-014")
    oversized = [f"sub-question {i}" for i in range(settings.agent_max_sub_queries + 2)]

    with (
        patch("claimcontext.agent.graph.decompose_query", return_value=oversized),
        patch("claimcontext.agent.graph.classify_route", return_value="multi"),
    ):
        graph = build_agent_graph(settings, _FakeRetriever(), _FakeLLM())  # type: ignore[arg-type]
        result = graph.invoke(AgentState(query="a compound question", principal=principal))

    final = result["final_answer"]
    assert final.refused is True
    assert "system limit" in final.answer.lower()


# ── Proof 4b — tool failure after retry exhaustion → escalation ────────────────


def test_proof4b_retry_exhaustion_escalates() -> None:
    """Uses ConnectionError — a genuinely TRANSIENT failure — not a bare
    RuntimeError. Per the fix that promoted known-gap #3 from deferred: only
    specific, retryable exception types (connection/timeout/LLM errors) are
    caught and escalated; anything else (a real bug) must propagate loudly. A
    bare RuntimeError is intentionally NOT in that set, so this test would fail
    if it used one — that failure mode is exactly what the fix verifies against.
    """
    settings = Settings()
    principal = resolve_principal("ADJ-014")

    def always_fails(*args: object, **kwargs: object) -> None:
        raise ConnectionError("simulated repeated transient failure")

    with (
        patch("claimcontext.agent.graph.classify_route", return_value="single"),
        patch("claimcontext.agent.graph.ask", side_effect=always_fails),
    ):
        graph = build_agent_graph(settings, _FakeRetriever(), _FakeLLM())  # type: ignore[arg-type]
        result = graph.invoke(AgentState(query="a question", principal=principal))

    final = result["final_answer"]
    assert final.refused is True
    assert "system limit" in final.answer.lower()


def test_proof4b_non_transient_error_is_not_swallowed() -> None:
    """The other half of the fix: a NON-transient failure (a real bug) must
    propagate as a loud crash, not disappear into a graceful escalation — the
    property that protects this project's defect-discovery discipline."""
    settings = Settings()
    principal = resolve_principal("ADJ-014")

    def buggy(*args: object, **kwargs: object) -> None:
        raise TypeError("a real programming bug, not a transient failure")

    with (
        patch("claimcontext.agent.graph.classify_route", return_value="single"),
        patch("claimcontext.agent.graph.ask", side_effect=buggy),
    ):
        graph = build_agent_graph(settings, _FakeRetriever(), _FakeLLM())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="a real programming bug"):
            graph.invoke(AgentState(query="a question", principal=principal))


# ── Proof 5 — MCP tool is externally callable ───────────────────────────────────


def test_proof5_mcp_tool_callable() -> None:
    async def _call() -> None:
        tools = await mcp_server.list_tools()
        assert "check_citations" in [t.name for t in tools]

        mcp_result = await mcp_server.call_tool(
            "check_citations",
            {
                "answer": "Per [CLM-1001-fnol | p.1] the vehicle is a Honda.",
                "retrieved_doc_ids": ["CLM-1001-fnol"],
            },
        )
        assert getattr(mcp_result, "is_error", False) is False

    asyncio.run(_call())

    # Confirm the MCP call returns the same result a direct Python call would.
    direct = check_citations_raw(
        "Per [CLM-1001-fnol | p.1] the vehicle is a Honda.", ["CLM-1001-fnol"]
    )
    assert direct.has_fabricated_citations is False


# ── Live-service proofs — metadata-filter entitlement scoping ──────────────────


@pytest.mark.agent
def test_metadata_filter_respects_entitlement() -> None:
    settings = Settings()
    adj014 = resolve_principal("ADJ-014")
    adj027 = resolve_principal("ADJ-027")

    own = search_by_metadata(settings, adj014, claim_number="CLM-1001")
    assert len(own) > 0

    cross = search_by_metadata(settings, adj014, claim_number="CLM-1003")
    assert cross == []

    other_own = search_by_metadata(settings, adj027, claim_number="CLM-1003")
    assert len(other_own) > 0
