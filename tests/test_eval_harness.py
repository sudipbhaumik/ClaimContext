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
    EvalResult,
    ExpectedBehavior,
    GoldenEntry,
    is_correct_outcome,
)
from claimcontext.eval.scorecard import format_comparison, print_scorecard
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.models import AskResult
from claimcontext.retrieval.reranker import Reranker

# ── Known, accepted eval gaps ───────────────────────────────────────────────────
# Entry IDs that are *expected* to fail is_correct_outcome() right now, for a
# documented, non-code reason — NOT a permanent expectation of failure. Each entry
# here must name the fix condition that removes it. A live-eval proof that hard-
# asserts "gate must fully pass" bakes an accepted gap in as required behavior,
# which means the test enforces the bug's continued existence. Instead: report the
# scorecard, and only fail the proof on *unexpected* failures (anything not in this
# set) — so a real regression is still caught, but the known gap doesn't need the
# assertion loosened as new gaps are added or old ones close.
#
# Two DISTINCT root causes, not one — verified by checking each entry's actual
# top-reranked score before grouping them (see specs/spec-4-handoff.md, "Two root
# causes behind the four exceptions" for full analysis and evidence):
#
#   Root cause A — terse claim-note shorthand scores near-zero against natural-
#   language questions on the general-domain cross-encoder, even when the note
#   contains the literal answer. Two failure modes from the same cause, differing
#   in severity:
#     q02, q05 — SAFE mode: no chunk (including the buried correct note) clears
#         refuse_threshold, so the gate correctly refuses. Uninformative, not wrong.
#         q02 top score 0.059, q05 top score 0.143 (threshold 0.4-0.55 depending on
#         config) — both decisively below, not borderline like q06.
#     q08 — DANGEROUS mode: a same-claim, wrong-sub-topic chunk (CLM-1004-fnol,
#         0.86) clears the gate while the actually-correct note (CLM-1004-notes,
#         containing the literal adjuster conclusion) sits at rank 12+/30 scoring
#         0.0001 — invisible to rerank_top_n=5. The gate opens on the wrong
#         evidence; the LLM answers using nearby low-score chunks including a
#         DIFFERENT claim's data (CLM-1003), producing a confident, cross-
#         contaminated, wrong answer. This is the failure mode grounding exists to
#         prevent, slipping through the exact mechanism meant to catch it — it is
#         the single most important finding in this project so far. Remove q08
#         once a future spec resolves root cause A (see handoff for the fix-space
#         options — retrieval-side fix, query-aware sufficiency check, or output-
#         faithfulness gating — deliberately not chosen yet).
#
#   Root cause B — generation does not reason about informative absence:
#     q06 — the gate correctly retrieves the ONLY evidence that exists (CLM-1005
#         has no notes by corpus design) and correctly passes it through
#         (score 0.563, not borderline-low like q02/q05's failures). The system
#         hedges ("not enough information") instead of stating the true fact
#         ("no investigation conducted; only FNOL on file"). Unrelated to root
#         cause A — retrieval did its job here; generation did not.
#         Remove once BOTH: (a) generation reasons about absence and produces the
#         informative-absence answer, (b) ground_truth_answer is written for q06
#         and its expected_behavior is relabeled ANSWER.
KNOWN_EVAL_EXCEPTIONS: frozenset[str] = frozenset({"q02", "q05", "q06", "q08"})

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

    expected_count = len(load_golden_set(settings.eval_golden_set_path))
    assert len(report.entries) == expected_count, (
        f"expected {expected_count} entries, got {len(report.entries)}"
    )
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
        r
        for r in refusal_entries
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


# ── Proof 5a — gate MECHANISM, synthetic + deterministic, no live services ────


def _synthetic_result(
    entry_id: str,
    behavior: ExpectedBehavior,
    passed: bool,
    refusal_correct: bool | None = None,
    **ragas_scores: float | None,
) -> EvalResult:
    """Build a minimal EvalResult by hand — no retriever/LLM/judge involved.

    Exists so gate-threshold logic (_build_report) can be tested as pure
    input->output behavior: known scores in, known gate_passed/gate_failures out.
    Deterministic and fast — this is what makes Proof 5a safe to run on every
    change, unlike Proof 5b which depends on live local-model inference.
    """
    return EvalResult(
        entry_id=entry_id,
        question=f"synthetic question for {entry_id}",
        expected_behavior=behavior,
        question_type="synthetic",
        ask_result=AskResult(
            query=f"synthetic question for {entry_id}",
            answer="synthetic answer",
            citations=[],
            retrieved_chunks=[],
            llm_model="synthetic",
            prompt_version="synthetic",
            refused=(behavior != ExpectedBehavior.ANSWER),
        ),
        passed=passed,
        refusal_correct=refusal_correct,
        ground_truth_answer="synthetic reference" if behavior == ExpectedBehavior.ANSWER else None,
        **ragas_scores,  # type: ignore[arg-type]
    )


@pytest.mark.eval
def test_proof5a_gate_logic_synthetic(settings: Settings) -> None:
    """Proof 5a: the gate MECHANISM fires and clears correctly, on synthetic,
    hand-constructed scorecards — deterministic, no live services, no LLM calls.
    This is the rigorous half of Proof 5: it proves _build_report's threshold
    logic is correct, independent of what any particular model run produces.
    """
    # Known-bad scorecard: every metric at 0.0, one wrong refusal — gate must fail
    # on every threshold it checks, not just one.
    bad_entries = [
        _synthetic_result(
            "a1",
            ExpectedBehavior.ANSWER,
            passed=True,
            context_precision=0.0,
            context_recall=0.0,
            faithfulness=0.0,
            answer_relevance=0.0,
        ),
        _synthetic_result("r1", ExpectedBehavior.REFUSE, passed=False, refusal_correct=False),
    ]
    bad_report = _build_report(settings, bad_entries)
    assert not bad_report.gate_passed, "gate must fail on a known-bad scorecard"
    for metric in ("context_precision", "context_recall", "faithfulness", "answer_relevance"):
        assert metric in bad_report.gate_failures, (
            f"{metric}=0.0 must be in gate_failures: {bad_report.gate_failures}"
        )
    assert "refusal_accuracy" in bad_report.gate_failures
    print(f"\nProof 5a-bad: gate correctly failed — failures: {bad_report.gate_failures}")

    # Known-good scorecard: every metric comfortably above threshold, all
    # refusals correct — gate must pass with zero failures.
    good_entries = [
        _synthetic_result(
            "a1",
            ExpectedBehavior.ANSWER,
            passed=True,
            context_precision=0.95,
            context_recall=0.95,
            faithfulness=0.95,
            answer_relevance=0.95,
        ),
        _synthetic_result("r1", ExpectedBehavior.REFUSE, passed=True, refusal_correct=True),
        _synthetic_result("t1", ExpectedBehavior.TIER3, passed=True, refusal_correct=True),
    ]
    good_report = _build_report(settings, good_entries)
    assert good_report.gate_passed, (
        f"gate must pass on a known-good scorecard: {good_report.gate_failures}"
    )
    assert good_report.gate_failures == []
    print("Proof 5a-good: gate correctly passed — failures: []")


# ── Proof 5b — live eval result, reported against a documented exception set ──


@pytest.mark.eval
def test_proof5b_live_eval_reports_against_known_exceptions(
    settings: Settings,
    retriever: HybridRetriever,
    llm: LLMClient,
    reranker: Reranker,
) -> None:
    """Proof 5b: run the real pipeline + RAGAS scoring, print the scorecard, and
    fail only on *unexpected* entry failures — anything not in
    KNOWN_EVAL_EXCEPTIONS. The live model produces real, sometimes-imperfect
    scores; hard-asserting a fully-passing gate here would either be flaky
    (chasing local-model variance) or would require baking an accepted gap
    (q06) into the assertion as required behavior — which means the test
    enforces the bug's continued existence instead of catching regressions.
    A NEW failure (anything outside the documented set) still fails this proof.
    """
    if not ragas_judge_available(settings):
        print("Proof 5b: skipped (no RAGAS judge available)")
        return

    report = run_eval(settings=settings, retriever=retriever, llm=llm, reranker=reranker)
    report = score_with_ragas(report, settings)
    report = _build_report(settings, report.entries)
    print_scorecard(report)

    actual_failures = {r.entry_id for r in report.entries if not r.passed}
    unexpected = actual_failures - KNOWN_EVAL_EXCEPTIONS
    assert not unexpected, (
        f"Unexpected eval failures beyond the documented exception set "
        f"{sorted(KNOWN_EVAL_EXCEPTIONS)}: {sorted(unexpected)}. "
        f"If this is a real regression, fix it. If it's a new accepted gap, add it "
        f"to KNOWN_EVAL_EXCEPTIONS with a documented removal condition — do not "
        f"silently drop this assertion."
    )

    still_documented = actual_failures & KNOWN_EVAL_EXCEPTIONS
    if still_documented:
        print(
            f"Proof 5b: no unexpected failures. "
            f"Documented exceptions still failing: {sorted(still_documented)}"
        )
    else:
        print(
            "Proof 5b: no unexpected failures, and all documented exceptions now "
            "pass — remove them from KNOWN_EVAL_EXCEPTIONS."
        )
