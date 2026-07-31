"""Threshold calibration — measures reranker score distribution across query bands.

Runs three bands of queries through retrieval + reranker (no LLM call).
Collects the top reranked score per query. Reports the distribution and
calls the recommend_threshold() function from schema.py.

Called via: uv run python -m claimcontext.eval.calibration
or:          make eval-calibrate
"""

from __future__ import annotations

import logging

from claimcontext.config import Settings, get_settings
from claimcontext.eval.schema import CalibrationReport, recommend_threshold
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.reranker import Reranker

log = logging.getLogger(__name__)

# ── Calibration query sets (defined by spec-4 author) ─────────────────────────
# These query bands characterise three distinct score regions for bge-reranker-base.
#
# Design constraints:
#   ANSWERABLE: natural-language adjuster questions with clear answers in the corpus.
#     Must be phrased as questions, not keyword strings (cross-encoder is trained on
#     question-passage pairs; keyword strings score artificially low).
#   CLAUSE_REFERENCE: queries that reference clause numbers/section IDs where the
#     policy text exists BUT the clause identifier alone doesn't match the natural
#     content well. The corpus has the document; the query is structurally mismatched.
#     Must NOT contain text that verbatim matches document content (that would make
#     it answerable). "What does Exclusion 2.1 say?" is not clause-reference — it's
#     answerable. True clause-reference: "does Section 4.2 apply here?" where 4.2
#     doesn't exist in the corpus or the question framing is abstract.
#   OFF_CORPUS: completely outside the insurance/claims domain.
#
# All queries run WITHOUT entitlement filter — calibration measures raw reranker
# score distribution, not per-adjuster scores. Entitlement filtering happens in
# ask(), not here.

ANSWERABLE_QUERIES = [
    # Natural-language questions with grounded answers in ADJ-014 northeast corpus
    "Is the wind-driven rain damage covered under the policy for claim CLM-1004?",
    "Why was claim CLM-1003 denied?",
    "Was the policy still active when CLM-1002 loss occurred?",
    "What vehicle is insured under the policy for claim CLM-1001?",
    "What did the adjuster conclude about CLM-1004 coverage?",
    "What repair items were authorized for claim CLM-1001?",
    "What does the flood exclusion in POL-4403 say?",
    "When did Endorsement WR-001 take effect on POL-5504?",
    "What damage was reported in the CLM-1001 FNOL?",
    "What are the coverage limits under policy POL-3301?",
]

CLAUSE_REFERENCE_QUERIES = [
    # Queries that cite clause numbers for content that doesn't exist in the corpus,
    # or use abstract framing that doesn't match the document's natural language.
    # Expected: low scores because the specific clause/section identifier has no
    # matching content (these section numbers don't appear in the corpus).
    "What does Section 4.2 of the deductible schedule say?",
    "What are the limits under Schedule III of the policy?",
    "What does paragraph 1(b) of the endorsement provide?",
    "What does Section V definitions clause say about 'occurrence'?",
    "What are the reporting requirements under Section 8.1?",
    "What does the arbitration clause in Section 12 require?",
]

OFF_CORPUS_QUERIES = [
    "What is the CEO's annual compensation package?",
    "What is the quarterly earnings per share guidance?",
    "What is the current municipal bond yield curve?",
    "What were the sports scores from last night?",
    "What is on the restaurant menu and what are the prices?",
]


def _top_reranker_score(
    query: str,
    retriever: HybridRetriever,
    reranker: Reranker,
    settings: Settings,
) -> float:
    """Retrieve candidates and return the top cross-encoder score. No LLM call."""
    fetch_k = min(settings.top_k * 3, 30)
    candidates = retriever.search(query, top_k=fetch_k)
    if not candidates:
        return 0.0
    reranked = reranker.rerank(query, candidates)
    return reranked[0].score if reranked else 0.0


def run_calibration(
    settings: Settings | None = None,
    retriever: HybridRetriever | None = None,
    reranker: Reranker | None = None,
) -> CalibrationReport:
    """Measure top-reranked score distribution across three query bands.

    All three band results plus the recommended threshold are returned in
    CalibrationReport. Prints a human-readable summary to the logger.
    """
    if settings is None:
        settings = get_settings()

    if retriever is None:
        retriever = HybridRetriever(settings)
        retriever.check_index_staleness()

    if reranker is None:
        reranker = Reranker(settings)

    log.info(
        "calibration: running %d queries across 3 bands",
        len(ANSWERABLE_QUERIES) + len(CLAUSE_REFERENCE_QUERIES) + len(OFF_CORPUS_QUERIES),
    )

    answerable_scores: list[float] = []
    for q in ANSWERABLE_QUERIES:
        score = _top_reranker_score(q, retriever, reranker, settings)
        log.info("answerable  score=%.4f  query=%r", score, q[:60])
        answerable_scores.append(score)

    clause_scores: list[float] = []
    for q in CLAUSE_REFERENCE_QUERIES:
        score = _top_reranker_score(q, retriever, reranker, settings)
        log.info("clause-ref  score=%.4f  query=%r", score, q[:60])
        clause_scores.append(score)

    off_corpus_scores: list[float] = []
    for q in OFF_CORPUS_QUERIES:
        score = _top_reranker_score(q, retriever, reranker, settings)
        log.info("off-corpus  score=%.4f  query=%r", score, q[:60])
        off_corpus_scores.append(score)

    report = CalibrationReport(
        answerable_scores=answerable_scores,
        clause_reference_scores=clause_scores,
        off_corpus_scores=off_corpus_scores,
        answerable_min=min(answerable_scores) if answerable_scores else 0.0,
        clause_reference_max=max(clause_scores) if clause_scores else 0.0,
        off_corpus_max=max(off_corpus_scores) if off_corpus_scores else 0.0,
    )

    # Populate recommendation fields
    rec_threshold, rec_justification = recommend_threshold(report)
    report = report.model_copy(
        update={
            "recommended_threshold": rec_threshold,
            "threshold_justification": rec_justification,
        }
    )

    _print_calibration(report)
    return report


def _print_calibration(report: CalibrationReport) -> None:
    w = 70
    print("\n" + "━" * w)
    print("THRESHOLD CALIBRATION — bge-reranker-base score distribution")
    print("━" * w)
    print(f"\nAnswerable band ({len(report.answerable_scores)} queries):")
    for s in report.answerable_scores:
        print(f"  {s:.4f}")
    print(f"  min={report.answerable_min:.4f}")

    print(f"\nClause-reference band ({len(report.clause_reference_scores)} queries):")
    for s in report.clause_reference_scores:
        print(f"  {s:.4f}")
    print(f"  max={report.clause_reference_max:.4f}")

    print(f"\nOff-corpus band ({len(report.off_corpus_scores)} queries):")
    for s in report.off_corpus_scores:
        print(f"  {s:.4f}")
    print(f"  max={report.off_corpus_max:.4f}")

    print(f"\nRecommended refuse_threshold: {report.recommended_threshold}")
    print(f"\nJustification:\n  {report.threshold_justification}")
    print("\n⚠  Scores are directional. LLM-as-judge biases (verbosity, position,")
    print("   self-preference) apply to RAGAS metrics, not to these reranker scores.")
    print("   Reranker calibration is model-intrinsic — no judge involved here.")
    print("━" * w + "\n")


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_calibration()
