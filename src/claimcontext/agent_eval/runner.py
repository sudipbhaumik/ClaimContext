"""Trajectory eval runner — calls run_agent_with_trajectory() for every
TrajectoryGoldenEntry and returns a TrajectoryEvalReport. Structurally mirrors
spec-4's eval/runner.py (same project convention)."""

from __future__ import annotations

import logging

from claimcontext.agent.trajectory import AgentTrajectory, run_agent_with_trajectory
from claimcontext.agent_eval.schema import (
    FailureMode,
    TrajectoryEvalReport,
    TrajectoryEvalResult,
    TrajectoryGoldenEntry,
    is_correct_trajectory,
)
from claimcontext.auth.resolver import resolve_principal
from claimcontext.config import Settings
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.reranker import Reranker
from claimcontext.retrieval.retriever import Retriever

log = logging.getLogger(__name__)


def load_trajectory_golden_set(path: str) -> list[TrajectoryGoldenEntry]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(TrajectoryGoldenEntry.model_validate_json(line))
    log.info("loaded %d trajectory golden entries from %s", len(entries), path)
    return entries


def _failure_reason(entry: TrajectoryGoldenEntry, trajectory: AgentTrajectory) -> str:
    if trajectory.route != entry.expected_route:
        return f"route: expected {entry.expected_route!r}, got {trajectory.route!r}"
    if trajectory.escalated != entry.expected_escalation:
        return (
            f"escalation: expected {entry.expected_escalation}, "
            f"got {trajectory.escalated} (reason={trajectory.escalation_reason!r})"
        )
    return "cross-step provenance violated: a claim's citations span multiple steps"


def run_trajectory_eval(
    settings: Settings,
    retriever: Retriever | HybridRetriever,
    llm: LLMClient,
    reranker: Reranker | None = None,
) -> TrajectoryEvalReport:
    """Run the full harness: load golden set → run_agent_with_trajectory() per
    entry → score → report."""
    entries = load_trajectory_golden_set(settings.agent_eval_golden_set_path)
    results: list[TrajectoryEvalResult] = []

    for entry in entries:
        log.info("trajectory eval entry=%s failure_mode=%s", entry.entry_id, entry.failure_mode)
        principal = resolve_principal(entry.adjuster_id)

        trajectory = run_agent_with_trajectory(
            query=entry.query,
            principal=principal,
            settings=settings,
            retriever=retriever,
            llm=llm,
            reranker=reranker,
        )

        passed = is_correct_trajectory(entry, trajectory)
        result = TrajectoryEvalResult(
            entry_id=entry.entry_id,
            query=entry.query,
            failure_mode=entry.failure_mode,
            trajectory=trajectory,
            passed=passed,
            failure_reason=None if passed else _failure_reason(entry, trajectory),
        )
        results.append(result)

    return _build_report(results)


def _build_report(results: list[TrajectoryEvalResult]) -> TrajectoryEvalReport:
    pass_rate_by_category: dict[str, float] = {}
    gate_failures: list[str] = []

    for mode in FailureMode:
        category_results = [r for r in results if r.failure_mode == mode]
        if not category_results:
            continue
        passed_count = sum(1 for r in category_results if r.passed)
        rate = passed_count / len(category_results)
        pass_rate_by_category[mode.value] = rate
        if rate < 1.0:
            gate_failures.append(mode.value)

    return TrajectoryEvalReport(
        entries=results,
        pass_rate_by_category=pass_rate_by_category,
        gate_passed=len(gate_failures) == 0,
        gate_failures=gate_failures,
    )
