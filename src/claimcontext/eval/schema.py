"""Pydantic models and core logic for the spec-4 eval harness.

Authored by: human (scoring contract, refusal predicate, threshold recommendation).
The scoring predicate and threshold recommendation function are the correctness-
critical pieces — they determine what counts as pass/fail and what threshold to use.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from claimcontext.retrieval.models import AskResult


class ExpectedBehavior(str, Enum):  # noqa: UP042
    ANSWER = "answer"  # system must return a grounded answer
    REFUSE = "refuse"  # system must refuse (any reason — entitlement or weak-context)
    TIER3 = "tier3"  # system must refuse with system-of-record language, not corpus-absence


class GoldenEntry(BaseModel):
    entry_id: str
    question: str
    ground_truth_answer: str | None  # None for REFUSE / TIER3 entries
    ground_truth_context: list[str]  # doc_ids that must appear in retrieved context
    expected_behavior: ExpectedBehavior
    question_type: str  # "flagship" | "denial" | "temporal" | ...
    adjuster_id: str  # principal to run this entry as
    notes: str | None = None  # human-authored notes; not used in scoring


class EvalResult(BaseModel):
    entry_id: str
    question: str
    expected_behavior: ExpectedBehavior
    question_type: str
    ask_result: AskResult
    passed: bool
    refusal_correct: bool | None = None  # None for ANSWER entries
    ground_truth_answer: str | None = None  # populated by runner; used by RAGAS adapter
    # RAGAS metrics — None for refused/failed entries:
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None


class EvalReport(BaseModel):
    golden_set_version: str
    prompt_version: str
    timestamp: datetime
    entries: list[EvalResult]
    # Aggregate scores (ANSWER entries that passed only):
    mean_context_precision: float
    mean_context_recall: float
    mean_faithfulness: float
    mean_answer_relevance: float
    # CI gate:
    gate_passed: bool
    gate_failures: list[str]
    refusal_accuracy: float  # correct_refusals / total_refusal_entries


class CalibrationReport(BaseModel):
    answerable_scores: list[float]
    clause_reference_scores: list[float]
    off_corpus_scores: list[float]
    answerable_min: float
    clause_reference_max: float
    off_corpus_max: float
    recommended_threshold: float = 0.0
    threshold_justification: str = ""


# ── Refusal predicate (I author this) ─────────────────────────────────────────


def _contains_tier3_marker(answer: str, marker: str) -> bool:
    return marker.lower() in answer.lower()


def is_correct_outcome(
    entry: GoldenEntry,
    result: AskResult,
    tier3_marker: str = "claims system",
) -> bool:
    """Binary predicate: did the system do the right thing for this entry?

    Checks the user-facing outcome only — not which internal path fired.
    Two internal paths produce refused=True:
      - Entitlement-denial (access-control layer, before retrieval)
      - Reranker-gate (weak-context, after retrieval, low score)
    Both are correct refusals for a REFUSE entry. The predicate does not
    distinguish them because the user-facing response must be indistinguishable
    between the two (§6B property confirmed in spec-3 Proof 5, re-asserted
    in Proof 2 of this harness on the actual q04 golden-set entry).

    For TIER3: refused=True is necessary but not sufficient — the answer must
    also carry the system-of-record reason, not the generic weak-context message.
    This is the only place we inspect answer content for classification, and
    only to distinguish two correct refusal types.
    """
    if entry.expected_behavior == ExpectedBehavior.ANSWER:
        return not result.refused

    if entry.expected_behavior == ExpectedBehavior.REFUSE:
        return result.refused

    if entry.expected_behavior == ExpectedBehavior.TIER3:
        return result.refused and _contains_tier3_marker(result.answer, tier3_marker)

    raise ValueError(f"Unknown expected_behavior: {entry.expected_behavior!r}")


# ── Threshold recommendation (I author this) ──────────────────────────────────


def recommend_threshold(report: CalibrationReport) -> tuple[float, str]:
    """Recommend a refuse_threshold from the measured score distribution.

    Score bands observed from the golden set (all scores on CPU — see reranker.py):
      - off-corpus:          ~0.001       (no relevant documents exist)
      - clause-reference:    ~0.02–0.05   (document exists, query cites clause numbers)
      - weak-partial (q06):  ~0.563       (FNOL exists, no investigation notes)
      - answerable:          ~0.85+       (sufficient grounded evidence)

    The recommended threshold is the midpoint of the safe band [clause_max, ans_min].
    The robustness of the band is the finding, not the point value.

    THRESHOLD AND LABEL ARE THE SAME DECISION.
    q06 sits at 0.563, just above the current 0.55 threshold. Setting threshold >0.563
    would make the gate fire on q06 — which is silently deciding q06 should refuse,
    without naming that as a labeling decision. Do not raise the threshold "for margin"
    without first deciding what q06 should do:
      - If q06 should answer (informative-absence response once generation is fixed):
        keep threshold at 0.55. Remediation is generation, not gate.
      - If q06 should refuse: raise threshold above 0.563 explicitly, update q06
        expected_behavior, and write the justification for that decision.
    Current recommendation: keep 0.55. q06 should answer; generation is what's weak.

    Hard constraints:
      1. threshold > off_corpus_max  → all off-corpus queries refused
      2. threshold < answerable_min  → all answerable queries pass
    """
    off_max = report.off_corpus_max
    clause_max = report.clause_reference_max
    ans_min = report.answerable_min

    # Midpoint of the safe band [clause_max, ans_min].
    # NOTE: if q06 should answer (expected), this value must stay below 0.563 so
    # the gate passes q06 to the LLM. The midpoint of [clause_max, ans_min] satisfies
    # this as long as ans_min < 0.85 and clause_max is well below 0.55.
    recommended = round((clause_max + ans_min) / 2, 2)

    safe_band_width = round(ans_min - clause_max, 3)

    justification = (
        f"Score bands — off-corpus max: {off_max:.4f}, "
        f"clause-reference max: {clause_max:.4f}, "
        f"weak-partial (q06) ~0.563, "
        f"answerable min: {ans_min:.4f}. "
        f"Safe band width: {safe_band_width:.3f} "
        f"(threshold robust anywhere in [{clause_max:.2f}, {ans_min:.2f}]). "
        f"Recommended {recommended} (midpoint). "
        f"WARNING: raising threshold above 0.563 would flip q06 from answer-attempt "
        f"to refuse — that is a label decision, not a margin tweak. "
        f"Current policy: keep threshold at 0.55; q06 remediation is generation "
        f"(informative-absence reasoning), not the refuse gate. "
        f"Clause-reference queries ({clause_max:.2f}) are refused by this threshold — "
        f"product decision, not calibration. Do not change to make a test green."
    )
    return recommended, justification
