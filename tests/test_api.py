"""spec-7a FastAPI serving proofs.

All tests marked @pytest.mark.api require:
  - Qdrant running with indexed corpus (make up + ingest)
  - ADJ-014 and ADJ-027 principals resolvable
  - Reranker model (bge-reranker-base) loadable
  - Ollama running with llama3.2 pulled (default answer + agent model)

To run:
  pytest tests/test_api.py -m api -v -s
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from claimcontext.api.app import AppState, app
from claimcontext.api.schemas import PUBLIC_REFUSAL_MESSAGE
from claimcontext.retrieval.errors import IndexStalenessError

# ── Shared fixture ───────────────────────────────────────────────────────────
# module-scoped: pays the retriever/embedder/reranker startup cost once for
# the whole file, same pattern as test_agent_orchestrator.py's fixtures.


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# ── Proof 1 — real end-to-end HTTP request, live services ──────────────────


@pytest.mark.api
def test_proof1_real_request_returns_cited_answer(client: TestClient) -> None:
    resp = client.post(
        "/ask",
        json={
            "query": (
                "For claim CLM-1004, is the wind-driven rain damage to the roof "
                "and interior covered under the governing policy?"
            ),
            "adjuster_id": "ADJ-027",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["citations"], "expected at least one citation for a grounded answer"
    assert body["answer"]


# ── Proof 2 — all refusal types indistinguishable at the HTTP boundary ─────


@pytest.mark.api
def test_proof2_all_refusal_types_indistinguishable(client: TestClient) -> None:
    cases = {
        "unknown_adjuster": {"query": "irrelevant", "adjuster_id": "ADJ-999"},
        "cross_entitlement": {
            "query": "For claim CLM-1004, is the damage covered?",
            "adjuster_id": "ADJ-014",  # CLM-1004 belongs to ADJ-027's region
        },
        "weak_context": {
            "query": "What is the airspeed velocity of an unladen swallow?",
            "adjuster_id": "ADJ-014",
        },
        "tier3": {
            "query": "What is the current reserve amount on claim CLM-1002?",
            "adjuster_id": "ADJ-014",
        },
    }
    responses = {name: client.post("/ask", json=payload) for name, payload in cases.items()}

    statuses = {name: r.status_code for name, r in responses.items()}
    bodies = {name: r.json() for name, r in responses.items()}

    assert len(set(statuses.values())) == 1, f"status codes differ: {statuses}"
    first = next(iter(bodies.values()))
    for name, body in bodies.items():
        assert body == first, f"{name} body differs from the others: {body} vs {first}"
    assert first["refused"] is True
    assert first["citations"] == []


@pytest.mark.api
def test_proof2_escalation_matches_other_refusal_types(client: TestClient) -> None:
    """Force the decompose-cap escalation path (agent_max_sub_queries=0) and
    confirm its response is byte-identical to the other four refusal types."""
    state: AppState = app.state.claimcontext
    original_cap = state.settings.agent_max_sub_queries
    state.settings.agent_max_sub_queries = 0
    try:
        resp = client.post(
            "/ask",
            json={
                "query": (
                    "For claim CLM-1004, is the damage covered, and separately "
                    "what does the FNOL say about the reported cause?"
                ),
                "adjuster_id": "ADJ-027",
            },
        )
    finally:
        state.settings.agent_max_sub_queries = original_cap

    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert body["answer"] == PUBLIC_REFUSAL_MESSAGE


# ── Proof 3 — /ready reports staleness/dependency failure; /health stays alive ──


@pytest.mark.api
def test_proof3_ready_reports_staleness_health_stays_alive(client: TestClient) -> None:
    state: AppState = app.state.claimcontext

    class _FailingRetriever:
        def check_index_staleness(self) -> None:
            raise IndexStalenessError("simulated staleness", reason="model_mismatch")

    original_retriever = state.retriever
    state.retriever = _FailingRetriever()  # type: ignore[assignment]
    try:
        ready_resp = client.get("/ready")
        health_resp = client.get("/health")
    finally:
        state.retriever = original_retriever

    assert ready_resp.status_code == 503
    ready_body = ready_resp.json()
    assert ready_body["status"] == "not_ready"
    assert ready_body["checks"]["index"] == "model_mismatch"

    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}


@pytest.mark.api
def test_proof3_ready_ok_when_index_healthy(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "checks": {"index": "ok"}}


@pytest.mark.api
def test_proof3_mid_flight_staleness_maps_to_503(client: TestClient) -> None:
    """A staleness error surfacing during /ask (not just /ready) must map to a
    clean 503, never an uncaught 500 — spec-7a's mid-flight decision."""
    state: AppState = app.state.claimcontext

    class _FailingRetriever:
        def check_index_staleness(self) -> None:
            raise IndexStalenessError("simulated mid-flight staleness", reason="empty")

    original_retriever = state.retriever
    state.retriever = _FailingRetriever()  # type: ignore[assignment]
    try:
        resp = client.post("/ask", json={"query": "anything", "adjuster_id": "ADJ-014"})
    finally:
        state.retriever = original_retriever

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Service temporarily unavailable. Please try again shortly."}


# ── Proof 4 — resources constructed once, not per request ──────────────────


@pytest.mark.api
def test_proof4_resources_constructed_once(client: TestClient) -> None:
    state: AppState = app.state.claimcontext
    retriever_id, llm_id, reranker_id = id(state.retriever), id(state.llm), id(state.reranker)

    client.post(
        "/ask",
        json={
            "query": "For claim CLM-1004, is the damage covered under the policy?",
            "adjuster_id": "ADJ-027",
        },
    )

    state = app.state.claimcontext
    assert id(state.retriever) == retriever_id
    assert id(state.llm) == llm_id
    assert id(state.reranker) == reranker_id
