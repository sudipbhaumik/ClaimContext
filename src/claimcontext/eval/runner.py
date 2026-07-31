"""Eval runner — calls ask() for every GoldenEntry and returns EvalResult list."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from claimcontext.auth.models import Principal
from claimcontext.auth.resolver import resolve_principal
from claimcontext.config import Settings
from claimcontext.eval.schema import (
    EvalReport,
    EvalResult,
    ExpectedBehavior,
    GoldenEntry,
    is_correct_outcome,
)
from claimcontext.retrieval.ask import ask
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.models import AskResult
from claimcontext.retrieval.reranker import Reranker
from claimcontext.retrieval.retriever import Retriever

log = logging.getLogger(__name__)


def load_golden_set(path: str) -> list[GoldenEntry]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(GoldenEntry.model_validate_json(line))
    log.info("loaded %d golden entries from %s", len(entries), path)
    return entries


def run_eval(
    settings: Settings,
    retriever: Retriever | HybridRetriever,
    llm: LLMClient,
    reranker: Reranker | None = None,
) -> EvalReport:
    """Run the full harness: load golden set → ask() per entry → score → report."""
    entries = load_golden_set(settings.eval_golden_set_path)
    results: list[EvalResult] = []

    for entry in entries:
        log.info("eval entry=%s behavior=%s", entry.entry_id, entry.expected_behavior)
        principal = _resolve(entry.adjuster_id)

        try:
            result = ask(
                query=entry.question,
                retriever=retriever,
                llm=llm,
                settings=settings,
                reranker=reranker,
                principal=principal,
            )
        except Exception as exc:
            log.error("ask() failed for entry %s: %s", entry.entry_id, exc)
            # Build a synthetic refused result so the run completes
            result = AskResult(
                query=entry.question,
                answer=f"[ERROR: {exc}]",
                citations=[],
                retrieved_chunks=[],
                llm_model=settings.llm_model,
                prompt_version="error",
                refused=True,
                adjuster_id=entry.adjuster_id,
            )

        correct = is_correct_outcome(entry, result, settings.tier3_refusal_marker)

        is_refusal_entry = entry.expected_behavior in (
            ExpectedBehavior.REFUSE,
            ExpectedBehavior.TIER3,
        )

        eval_result = EvalResult(
            entry_id=entry.entry_id,
            question=entry.question,
            expected_behavior=entry.expected_behavior,
            question_type=entry.question_type,
            ask_result=result,
            passed=correct,
            refusal_correct=correct if is_refusal_entry else None,
            ground_truth_answer=entry.ground_truth_answer,  # for RAGAS adapter
            # RAGAS scores populated later by ragas_adapter
        )
        results.append(eval_result)

    return _build_report(settings, results)


def _resolve(adjuster_id: str) -> Principal:
    return resolve_principal(adjuster_id)


def _build_report(settings: Settings, results: list[EvalResult]) -> EvalReport:
    """Aggregate results into EvalReport. RAGAS metrics must be populated first."""
    # Refusal accuracy
    refusal_entries = [r for r in results if r.expected_behavior != ExpectedBehavior.ANSWER]
    correct_refusals = sum(1 for r in refusal_entries if r.refusal_correct)
    refusal_accuracy = correct_refusals / len(refusal_entries) if refusal_entries else 1.0

    # Aggregate RAGAS metrics from ANSWER entries that passed
    scored = [
        r
        for r in results
        if r.expected_behavior == ExpectedBehavior.ANSWER
        and r.passed
        and r.context_precision is not None
    ]

    def _mean(vals: list[float | None]) -> float:
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else 0.0

    mean_cp = _mean([r.context_precision for r in scored])
    mean_cr = _mean([r.context_recall for r in scored])
    mean_f = _mean([r.faithfulness for r in scored])
    mean_ar = _mean([r.answer_relevance for r in scored])

    # Gate evaluation
    gate_failures: list[str] = []
    thresholds = {
        "context_precision": (mean_cp, settings.eval_context_precision_threshold),
        "context_recall": (mean_cr, settings.eval_context_recall_threshold),
        "faithfulness": (mean_f, settings.eval_faithfulness_threshold),
        "answer_relevance": (mean_ar, settings.eval_answer_relevance_threshold),
    }
    for name, (score, threshold) in thresholds.items():
        if score < threshold:
            gate_failures.append(name)

    if refusal_accuracy < settings.eval_refusal_accuracy_threshold:
        gate_failures.append("refusal_accuracy")

    prompt_versions = {
        r.ask_result.prompt_version for r in results if r.ask_result.prompt_version != "error"
    }
    prompt_version = next(iter(prompt_versions), "unknown")

    return EvalReport(
        golden_set_version=settings.eval_golden_set_version,
        prompt_version=prompt_version,
        timestamp=datetime.now(tz=UTC),
        entries=results,
        mean_context_precision=mean_cp,
        mean_context_recall=mean_cr,
        mean_faithfulness=mean_f,
        mean_answer_relevance=mean_ar,
        gate_passed=len(gate_failures) == 0,
        gate_failures=gate_failures,
        refusal_accuracy=refusal_accuracy,
    )


def save_report(report: EvalReport, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    log.info("eval report saved to %s", path)


def load_report(path: str) -> EvalReport:
    with open(path, encoding="utf-8") as f:
        return EvalReport.model_validate(json.load(f))
