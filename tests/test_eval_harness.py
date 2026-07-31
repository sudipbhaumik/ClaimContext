"""spec-4 eval harness proofs.

All tests marked @pytest.mark.eval require:
  - Qdrant running with indexed corpus (make up + ingest)
  - ADJ-014 and ADJ-027 principals resolvable
  - Reranker model (bge-reranker-base) loadable
  - Ollama running with mistral pulled (default judge), OR OPENAI_API_KEY / ANTHROPIC_API_KEY

To run:
  pytest tests/test_eval_harness.py -m eval -v -s
  make eval
"""

from __future__ import annotations

import pytest

from claimcontext.config import Settings
from claimcontext.eval.calibration import (
    run_calibration,
)
from claimcontext.eval.ragas_adapter import ragas_judge_available, score_with_ragas
from claimcontext.eval.runner import _build_report, load_golden_set, run_eval
from claimcontext.eval.schema import (
    ExpectedBehavior,
    GoldenEntry,
    is_correct_outcome,
)
from claimcontext.eval.scorecard import format_comparison, print_scorecard
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


@pytest.fixture(scope="module")
def golden_entries(settings: Settings) -> list[GoldenEntry]:
    return load_golden_set(settings.eval_golden_set_path)


# ── Proof 1 — harness runs end-to-end and prints a scorecard ──────────────────


@pytest.mark.eval
def test_proof1_harness_end_to_end(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """Proof 1: harness runs against all golden entries, returns a valid EvalReport,
    scorecard prints, prompt_version is recorded."""
    report = run_eval(settings=settings, retriever=retriever, llm=llm, reranker=reranker)

    # RAGAS scoring requires API key — skip if unavailable
    if ragas_judge_available(settings):
        report = score_with_ragas(report, settings)
        # Rebuild aggregates after RAGAS populates scores
        report = _build_report(settings, report.entries)

    print_scorecard(report)

    assert len(report.entries) == 11, f"expected 11 entries, got {len(report.entries)}"
    assert report.golden_set_version == settings.eval_golden_set_version
    assert report.prompt_version not in ("", "error", "unknown"), (
        f"prompt_version not recorded: {report.prompt_version!r}"
    )

    # All entries must complete without None ask_result
    for entry in report.entries:
        assert entry.ask_result is not None, f"{entry.entry_id} has null ask_result"

    gate = "PASS" if report.gate_passed else "FAIL"
    print(f"\nProof 1 complete: {len(report.entries)} entries, gate={gate}")
    print(f"  prompt_version: {report.prompt_version}")
    print(f"  refusal_accuracy: {report.refusal_accuracy:.3f}")


# ── Proof 2 — correct refusals score as PASS; §6B response indistinguishability ──


@pytest.mark.eval
def test_proof2_refusal_scoring_and_indistinguishability(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """Proof 2: correct refusals are PASS; wrong answers on refusal cases are FAIL;
    §6B: q04 (entitlement-denial path) produces same response shape as weak-context refusal."""
    report = run_eval(settings=settings, retriever=retriever, llm=llm, reranker=reranker)

    refusal_entries = [r for r in report.entries if r.expected_behavior != ExpectedBehavior.ANSWER]
    assert len(refusal_entries) == 3, f"expected 3 refusal entries, got {len(refusal_entries)}"

    for r in refusal_entries:
        print(
            f"  {r.entry_id} [{r.expected_behavior.value}]: "
            f"refused={r.ask_result.refused} correct={r.refusal_correct}"
        )

    # All refusal entries must be scored (refusal_correct is not None)
    for r in refusal_entries:
        assert r.refusal_correct is not None, f"{r.entry_id} has null refusal_correct"

    # Refusal entries must NOT appear in RAGAS — they have no RAGAS scores
    for r in refusal_entries:
        assert r.context_precision is None, f"{r.entry_id} should have null context_precision"
        assert r.context_recall is None, f"{r.entry_id} should have null context_recall"
        assert r.faithfulness is None, f"{r.entry_id} should have null faithfulness"
        assert r.answer_relevance is None, f"{r.entry_id} should have null answer_relevance"

    # §6B check: q04 (entitlement-denial path) must produce same response shape
    # as a weak-context refusal. The predicate checks outcome only; the shape check
    # closes the §6B property on real golden-set data (spec-3 proved it on probes).
    q04 = next((r for r in report.entries if r.entry_id == "q04"), None)
    assert q04 is not None
    assert q04.ask_result.refused is True, "q04 must be refused"
    assert q04.ask_result.citations == [], "q04 must have empty citations (§6B)"
    assert q04.ask_result.retrieved_chunks == [], "q04 must have empty retrieved_chunks (§6B)"

    # §6B structural shape: ALL refused entries must have empty citations and chunks.
    # This holds for every refusal path — entitlement-denial, weak-context, and tier3.
    for r in refusal_entries:
        if r.ask_result.refused:
            assert r.ask_result.citations == [], f"{r.entry_id} must have empty citations (§6B)"
            assert r.ask_result.retrieved_chunks == [], (
                f"{r.entry_id} must have empty retrieved_chunks (§6B)"
            )

    # §6B byte-identical assertion: non-tier3 refusals (entitlement-denial and weak-context)
    # must carry the same _REFUSE_MESSAGE — the user cannot distinguish why they were refused.
    # Tier-3 refusals carry a different routing message (to the claims system of record);
    # this is intentional — the routing affordance is not a disclosure.
    non_tier3_refusals = [
        r for r in refusal_entries
        if r.expected_behavior != ExpectedBehavior.TIER3 and r.ask_result.refused
    ]
    if len(non_tier3_refusals) >= 2:
        ref_answer = non_tier3_refusals[0].ask_result.answer
        for r in non_tier3_refusals[1:]:
            assert r.ask_result.answer == ref_answer, (
                f"§6B violation: {r.entry_id} refusal message differs from "
                f"{non_tier3_refusals[0].entry_id}.\n"
                f"  {non_tier3_refusals[0].entry_id}: {ref_answer[:80]!r}\n"
                f"  {r.entry_id}: {r.ask_result.answer[:80]!r}"
            )
        print(
            f"  §6B confirmed: {len(non_tier3_refusals)} non-tier3 refusals "
            f"produce identical response"
        )

    # Penalty test: a wrong answer on a refusal case must mark passed=False
    # Synthesise a fake result where q04 returned an answer instead of refusing
    fake_ask = q04.ask_result.model_copy(update={"refused": False})
    q04_entry_golden = GoldenEntry(
        entry_id="q04",
        question=q04.question,
        ground_truth_answer=None,
        ground_truth_context=[],
        expected_behavior=ExpectedBehavior.REFUSE,
        question_type="entitlement_refusal",
        adjuster_id="ADJ-014",
    )
    assert not is_correct_outcome(q04_entry_golden, fake_ask, settings.tier3_refusal_marker), (
        "wrong answer on a REFUSE entry must fail the predicate"
    )
    print("  Penalty check: wrong answer on REFUSE entry correctly fails predicate")

    print(f"\nProof 2 complete: refusal_accuracy={report.refusal_accuracy:.3f}")


# ── Proof 3 — degraded config produces measurably worse scores ────────────────


@pytest.mark.eval
def test_proof3_degradation_detects_regression(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """Proof 3: dropping the reranker produces measurably worse scores.
    If this proof fails, the harness cannot detect regressions — every green run
    afterward is meaningless."""
    # Baseline: full pipeline with reranker
    baseline_report = run_eval(settings=settings, retriever=retriever, llm=llm, reranker=reranker)

    # Degraded: no reranker (raw RRF scores drive context selection)
    degraded_report = run_eval(settings=settings, retriever=retriever, llm=llm, reranker=None)

    # Run RAGAS on both if API key available
    if ragas_judge_available(settings):
        baseline_report = score_with_ragas(baseline_report, settings)
        baseline_report = _build_report(settings, baseline_report.entries)
        degraded_report = score_with_ragas(degraded_report, settings)
        degraded_report = _build_report(settings, degraded_report.entries)

    comparison = format_comparison(baseline_report, degraded_report)
    print(comparison)

    # At least one RAGAS metric must be lower in degraded run
    # (or refusal accuracy if a refusal case regresses)
    if ragas_judge_available(settings):
        metrics = [
            (
                "context_precision",
                baseline_report.mean_context_precision,
                degraded_report.mean_context_precision,
            ),
            (
                "context_recall",
                baseline_report.mean_context_recall,
                degraded_report.mean_context_recall,
            ),
            ("faithfulness", baseline_report.mean_faithfulness, degraded_report.mean_faithfulness),
            (
                "answer_relevance",
                baseline_report.mean_answer_relevance,
                degraded_report.mean_answer_relevance,
            ),
        ]
        regressions = [(name, b, d) for name, b, d in metrics if d < b]
        assert len(regressions) >= 1, (
            "No metric degraded when reranker was dropped. The harness cannot detect "
            "regressions — check retrieval quality and RAGAS judge availability.\n"
            f"Baseline: cp={baseline_report.mean_context_precision:.3f} "
            f"cr={baseline_report.mean_context_recall:.3f} "
            f"f={baseline_report.mean_faithfulness:.3f} "
            f"ar={baseline_report.mean_answer_relevance:.3f}\n"
            f"Degraded: cp={degraded_report.mean_context_precision:.3f} "
            f"cr={degraded_report.mean_context_recall:.3f} "
            f"f={degraded_report.mean_faithfulness:.3f} "
            f"ar={degraded_report.mean_answer_relevance:.3f}"
        )
        print(f"\nProof 3: regression detected in {[n for n, _, _ in regressions]}")
    else:
        pytest.skip("RAGAS judge not available — Proof 3 requires RAGAS scores")


# ── Proof 4 — threshold calibration: three-band score distribution ────────────


@pytest.mark.eval
def test_proof4_threshold_calibration(
    settings: Settings,
    retriever: HybridRetriever,
    reranker: Reranker,
    golden_entries: list[GoldenEntry],
) -> None:
    """Proof 4: score distribution across answerable / clause-reference / off-corpus bands.
    Reports where q06's FNOL score lands — which determines whether q06 should be REFUSE or ANSWER.
    """
    cal_report = run_calibration(settings=settings, retriever=retriever, reranker=reranker)

    # Hard constraints
    assert cal_report.off_corpus_max < 0.10, (
        f"off-corpus max {cal_report.off_corpus_max:.4f} too high — "
        f"reranker is not distinguishing off-corpus queries"
    )
    assert cal_report.answerable_min > 0.50, (
        f"answerable min {cal_report.answerable_min:.4f} too low — "
        f"answerable queries are not scoring above the uncertainty band"
    )
    assert 0.10 < cal_report.recommended_threshold < 0.90, (
        f"recommended threshold {cal_report.recommended_threshold} out of reasonable range"
    )
    assert len(cal_report.threshold_justification) > 50, "justification too short"

    # q06 FNOL score — the key calibration discovery for q06's label
    q06_entry = next((e for e in golden_entries if e.entry_id == "q06"), None)
    assert q06_entry is not None
    q06_score = None
    if q06_entry:
        fetch_k = min(settings.top_k * 3, 30)
        candidates = retriever.search(q06_entry.question, top_k=fetch_k)
        if candidates:
            reranked = reranker.rerank(q06_entry.question, candidates)
            q06_score = reranked[0].score if reranked else 0.0

    print("\nProof 4: Calibration complete")
    print(f"  answerable min:       {cal_report.answerable_min:.4f}")
    print(f"  clause-reference max: {cal_report.clause_reference_max:.4f}")
    print(f"  off-corpus max:       {cal_report.off_corpus_max:.4f}")
    print(f"  recommended threshold:{cal_report.recommended_threshold}")

    if q06_score is not None:
        print(f"\n  q06 FNOL score: {q06_score:.4f}")
        current_threshold = settings.refuse_threshold
        would_refuse = q06_score < current_threshold
        print(f"  current threshold:  {current_threshold}")
        print(f"  q06 gate outcome:   {'REFUSE' if would_refuse else 'PASS (answer from FNOL)'}")
        if not would_refuse:
            print(
                f"  ⚠  q06 scores {q06_score:.4f} > threshold {current_threshold} — "
                f"gate will NOT fire. Evaluate whether 'a collision was reported; "
                f"no investigation conducted yet' is a correct partial answer. "
                f"If yes, relabel q06 as ANSWER and document the reasoning."
            )


# ── Proof 5 — CI gate fails on impossibly high threshold ──────────────────────


@pytest.mark.eval
def test_proof5_ci_gate_fails_and_passes(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """Proof 5: gate fails when threshold is impossibly high; restores to PASS at real thresholds.
    Gate failure propagates to pytest exit code via assertion."""
    # Run eval once (without RAGAS to keep test fast)
    report = run_eval(settings=settings, retriever=retriever, llm=llm, reranker=reranker)

    # Inject impossibly high RAGAS scores as float so _build_report gates can fire
    # Force context_precision to 0.0 on all ANSWER entries to trigger the gate
    patched_entries = []
    for r in report.entries:
        if r.expected_behavior == ExpectedBehavior.ANSWER and r.passed:
            patched_entries.append(
                r.model_copy(
                    update={
                        "context_precision": 0.0,
                        "context_recall": 0.0,
                        "faithfulness": 0.0,
                        "answer_relevance": 0.0,
                    }
                )
            )
        else:
            patched_entries.append(r)

    # Override threshold to 0.99 (impossibly high)
    high_threshold_settings = settings.model_copy(update={"eval_context_precision_threshold": 0.99})
    degraded_report = _build_report(high_threshold_settings, patched_entries)

    assert not degraded_report.gate_passed, "gate must fail with impossibly high threshold"
    assert "context_precision" in degraded_report.gate_failures, (
        f"context_precision must be in gate_failures: {degraded_report.gate_failures}"
    )
    print(f"\nProof 5a: gate correctly failed — failures: {degraded_report.gate_failures}")

    # Restore real thresholds → gate must pass (or at least not fail on context_precision)
    # If no RAGAS scores are available, aggregate means are 0.0 and real thresholds (0.60)
    # will also fail — skip the pass-assertion if API key is absent
    if ragas_judge_available(settings):
        ragas_report = score_with_ragas(report, settings)
        final_report = _build_report(settings, ragas_report.entries)
        assert final_report.gate_passed, (
            f"gate must pass with real thresholds after RAGAS scoring. "
            f"Failures: {final_report.gate_failures}. "
            f"Scores: cp={final_report.mean_context_precision:.3f} "
            f"cr={final_report.mean_context_recall:.3f} "
            f"f={final_report.mean_faithfulness:.3f} "
            f"ar={final_report.mean_answer_relevance:.3f}"
        )
        print_scorecard(final_report)
        print("Proof 5b: gate passes at real thresholds ✓")
    else:
        print("Proof 5b: skipped (no RAGAS API key) — gate-pass assertion requires RAGAS scores")
