"""spec-6 trajectory eval proofs.

All tests marked @pytest.mark.agent_eval require:
  - Qdrant running with indexed corpus (make up + ingest)
  - ADJ-014 and ADJ-027 principals resolvable
  - Reranker model (bge-reranker-base) loadable
  - Ollama running with llama3.2 pulled

To run:
  pytest tests/test_agent_eval.py -v -m agent_eval
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from claimcontext.agent.trajectory import run_agent_with_trajectory
from claimcontext.agent_eval.runner import load_trajectory_golden_set, run_trajectory_eval
from claimcontext.agent_eval.schema import TrajectoryGoldenEntry, is_correct_trajectory
from claimcontext.auth.resolver import resolve_principal
from claimcontext.config import Settings
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.reranker import Reranker
from claimcontext.retrieval.retriever import Retriever

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


@pytest.fixture(scope="module")
def golden_entries(settings: Settings) -> dict[str, TrajectoryGoldenEntry]:
    return {e.entry_id: e for e in load_trajectory_golden_set(settings.agent_eval_golden_set_path)}


# ── Proof 1 — simple query trajectory captured correctly ───────────────────────


@pytest.mark.agent_eval
def test_proof1_simple_trajectory(
    settings: Settings, retriever: HybridRetriever, llm: LLMClient, reranker: Reranker
) -> None:
    principal = resolve_principal("ADJ-014")
    traj = run_agent_with_trajectory(
        query="What vehicle is insured under the policy for claim CLM-1001?",
        principal=principal,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )
    print(f"\nProof 1: route={traj.route} steps={len(traj.steps)} escalated={traj.escalated}")
    assert traj.route == "single"
    assert len(traj.steps) == 1
    assert traj.escalated is False


# ── Proof 2 — multi-part trajectory: cross-step provenance holds ───────────────


@pytest.mark.agent_eval
def test_proof2_multi_part_cross_step_provenance(
    settings: Settings, retriever: HybridRetriever, llm: LLMClient, reranker: Reranker
) -> None:
    """Trajectory-level view of spec-5a Proof 4's non-regression property:
    checked here at each STEP, not just the final composed answer. NOTE: this
    checks CROSS-step mixing only (claim A's step doesn't cite claim B) — NOT
    within-step purity (KI-1/q08, deferred, out of scope — see schema.py)."""
    principal = resolve_principal("ADJ-027")
    traj = run_agent_with_trajectory(
        query=(
            "What did the adjuster conclude about coverage for CLM-1004, "
            "and why was CLM-1003 denied?"
        ),
        principal=principal,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )
    print(f"\nProof 2: route={traj.route} steps={len(traj.steps)}")
    for step in traj.steps:
        print(f"  {step.query[:50]!r} citations={step.citations_doc_ids}")

    assert traj.route == "multi"
    assert len(traj.steps) == 2

    # Cross-step check directly: no claim's doc_ids should appear in more than
    # one step's citation list.
    seen: dict[str, int] = {}
    for i, step in enumerate(traj.steps):
        for doc_id in step.citations_doc_ids:
            claim = "-".join(doc_id.split("-")[:2]) if doc_id.startswith("CLM-") else None
            if claim is None:
                continue
            if claim in seen and seen[claim] != i:
                pytest.fail(f"{claim} cited in both step {seen[claim]} and step {i}")
            seen[claim] = i


# ── Proof 3 — one deliberately-broken case per failure mode ────────────────────


@pytest.mark.agent_eval
def test_proof3_spec_failure_caught(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
    golden_entries: dict[str, TrajectoryGoldenEntry],
) -> None:
    """Monkeypatch classify_route to force the wrong route for sf01's query —
    confirm the eval reports it failed (route mismatch), not silently green."""
    entry = golden_entries["sf01"]
    principal = resolve_principal(entry.adjuster_id)

    # Baseline: unbroken, must pass.
    baseline = run_agent_with_trajectory(
        query=entry.query,
        principal=principal,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )
    assert is_correct_trajectory(entry, baseline) is True

    # Broken: force route="multi" for a query the entry expects "single".
    with patch("claimcontext.agent.graph.classify_route", return_value="multi"):
        broken = run_agent_with_trajectory(
            query=entry.query,
            principal=principal,
            settings=settings,
            retriever=retriever,
            llm=llm,
            reranker=reranker,
        )
    print(f"\nProof 3 (spec): broken route={broken.route}")
    assert broken.route == "multi"
    assert is_correct_trajectory(entry, broken) is False, (
        "harness must catch the wrong-route regression, not pass it silently"
    )


@pytest.mark.agent_eval
def test_proof3_execution_failure_caught(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
    golden_entries: dict[str, TrajectoryGoldenEntry],
) -> None:
    """Force ask() to raise a transient exception past retry exhaustion —
    confirm escalation fires and the eval reports the entry failed, since it
    expects no escalation. Clean discrimination: expects False, gets True,
    correctly fails — not the trivially-passable 'expects True, gets True'."""
    entry = golden_entries["ex01"]
    principal = resolve_principal(entry.adjuster_id)

    def always_fails(*args: object, **kwargs: object) -> None:
        raise ConnectionError("simulated repeated transient failure")

    with patch("claimcontext.agent.graph.ask", side_effect=always_fails):
        broken = run_agent_with_trajectory(
            query=entry.query,
            principal=principal,
            settings=settings,
            retriever=retriever,
            llm=llm,
            reranker=reranker,
        )
    print(f"\nProof 3 (execution): escalated={broken.escalated} reason={broken.escalation_reason}")
    assert broken.escalated is True
    assert broken.escalation_reason == "tool_failure"
    assert is_correct_trajectory(entry, broken) is False


@pytest.mark.agent_eval
def test_proof3_environmental_failure_caught(
    settings: Settings,
    llm: LLMClient,
    reranker: Reranker,
    golden_entries: dict[str, TrajectoryGoldenEntry],
) -> None:
    """A GENUINE environmental failure, not a mocked exception: point at a
    real, unreachable Qdrant address (connection refused) — distinct mechanism
    from the execution proof's code-level exception injection, verified to
    raise qdrant_client.http.exceptions.ResponseHandlingException (already in
    graph.py's _RETRYABLE_EXCEPTIONS)."""
    entry = golden_entries["env01"]
    principal = resolve_principal(entry.adjuster_id)

    broken_settings = settings.model_copy(
        update={"qdrant_url": "http://localhost:19999", "qdrant_timeout_seconds": 2}
    )
    broken_retriever = Retriever(broken_settings)  # never connects until .search() is called

    broken = run_agent_with_trajectory(
        query=entry.query,
        principal=principal,
        settings=broken_settings,
        retriever=broken_retriever,
        llm=llm,
        reranker=reranker,
    )
    print(
        f"\nProof 3 (environmental): escalated={broken.escalated} reason={broken.escalation_reason}"
    )
    assert broken.escalated is True
    assert broken.escalation_reason == "tool_failure"
    assert is_correct_trajectory(entry, broken) is False


@pytest.mark.agent_eval
def test_proof3_alignment_failure_caught(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
    golden_entries: dict[str, TrajectoryGoldenEntry],
) -> None:
    """Defense-in-depth / layering proof — NOT a leak-detection proof. See
    specs/spec-6-agent-eval.md for why disabling only the router's
    _is_cross_entitlement pre-filter cannot itself produce a leak (ask()'s
    real EntitlementScope gate is unconditional and independent of the
    router's decision). What this proves: even when the router's shortcut
    regresses, (a) the eval catches the route mismatch, AND (b) the real gate
    downstream still held — no leak occurred despite the regression."""
    entry = golden_entries["al01"]
    principal = resolve_principal(entry.adjuster_id)

    # Baseline: unbroken, router's pre-filter catches it, route="refuse".
    baseline = run_agent_with_trajectory(
        query=entry.query,
        principal=principal,
        settings=settings,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )
    assert baseline.route == "refuse"
    assert is_correct_trajectory(entry, baseline) is True

    # Broken: router's shortcut disabled — must NOT be read as "the leak
    # detector fired"; it's the ROUTE MISMATCH the harness catches.
    with patch("claimcontext.agent.routing._is_cross_entitlement", return_value=False):
        broken = run_agent_with_trajectory(
            query=entry.query,
            principal=principal,
            settings=settings,
            retriever=retriever,
            llm=llm,
            reranker=reranker,
        )
    print(f"\nProof 3 (alignment): route={broken.route} refused={broken.final_answer.refused}")

    # (a) The harness catches the regression via route mismatch.
    assert broken.route != "refuse", "router's shortcut should have been bypassed"
    assert is_correct_trajectory(entry, broken) is False

    # (b) Layering held: ask()'s own EntitlementScope gate still refused,
    # independent of the router's regression — the actual security property.
    assert broken.final_answer.refused is True
    assert broken.final_answer.citations == []


# ── Proof 4 — CI gate: full golden set run, unbroken, must be green ────────────


@pytest.mark.agent_eval
def test_proof4_ci_gate_passes_on_unbroken_code(
    settings: Settings, retriever: HybridRetriever, llm: LLMClient, reranker: Reranker
) -> None:
    report = run_trajectory_eval(settings, retriever, llm, reranker)
    print(f"\nProof 4: gate_passed={report.gate_passed} by_category={report.pass_rate_by_category}")
    assert report.gate_passed is True
    assert all(rate == 1.0 for rate in report.pass_rate_by_category.values())
