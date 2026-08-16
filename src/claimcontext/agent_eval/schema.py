"""Pydantic models and core logic for the spec-6 trajectory eval harness.

Authored by: human (trajectory-correctness predicate, failure-mode tagging) —
the scoring predicate is the correctness-critical piece, same authoring-split
rule spec-4's schema.py was built under (§3 Python-mastery split).

Distinct from spec-4's eval/schema.py: that harness scores AskResult content
(did it answer correctly, was the answer grounded). This harness scores the
AGENT'S PATH — which route it took, how many steps, whether it escalated, and
whether orchestration kept claims separated across steps. Two different units
of correctness; this one does not re-litigate spec-4's job.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from claimcontext.agent.trajectory import AgentTrajectory


class FailureMode(str, Enum):  # noqa: UP042 — matches spec-4's ExpectedBehavior convention
    """CLAUDE.md §7, Phase 5's required taxonomy. Every golden entry is tagged
    with which of these it's designed to catch — the scorecard reports
    pass/fail per category, not just an aggregate number."""

    SPEC = "spec"  # wrong decision given correct inputs (e.g. bad routing)
    EXECUTION = "execution"  # a step failed to execute (e.g. tool call failure)
    ENVIRONMENTAL = "environmental"  # the environment failed (e.g. Qdrant unreachable)
    ALIGNMENT = "alignment"  # the agent did something it should structurally never do


class TrajectoryGoldenEntry(BaseModel):
    entry_id: str
    query: str
    adjuster_id: str
    expected_route: str  # "single" | "multi" | "refuse"
    expected_escalation: bool
    failure_mode: FailureMode
    notes: str | None = None  # human-authored notes; not used in scoring


class TrajectoryEvalResult(BaseModel):
    entry_id: str
    query: str
    failure_mode: FailureMode
    trajectory: AgentTrajectory
    passed: bool
    failure_reason: str | None = None  # populated when passed=False


class TrajectoryEvalReport(BaseModel):
    entries: list[TrajectoryEvalResult]
    # Per-category pass rate — the scorecard's headline, not an aggregate number
    # (CLAUDE.md §7: report is supposed to let you NAME which failure mode was
    # caught, not just say "X% passed").
    pass_rate_by_category: dict[str, float]
    gate_passed: bool
    gate_failures: list[str]  # category names that regressed


# ── Trajectory-correctness predicate (I author this) ───────────────────────────


def is_correct_trajectory(entry: TrajectoryGoldenEntry, trajectory: AgentTrajectory) -> bool:
    """Binary predicate: did the agent take the right PATH for this entry?

    Checks exactly three things:
      1. Route taken matches expected_route.
      2. Escalation state matches expected_escalation.
      3. Cross-step provenance: no step's citations include a doc_id that
         belongs to a DIFFERENT step's claim than the one it answered. This is
         an orchestration property (did compose/decompose keep claims
         separated) — NOT a check of whether a single step's own citations are
         the correct evidence for its own question. That's within-step
         purity, which is KNOWN_ISSUES.md KI-1 (the spec-4 q08 finding),
         already known, already deferred, and explicitly out of scope here.
         Conflating the two would make this predicate trip on the accepted
         q08 defect and report it as a spec-6 regression — it is neither new
         nor this spec's to fix.
    """
    if trajectory.route != entry.expected_route:
        return False

    if trajectory.escalated != entry.expected_escalation:
        return False

    return _cross_step_provenance_holds(trajectory)


def _claim_prefix(doc_id: str) -> str | None:
    """ "CLM-1004-fnol" -> "CLM-1004". Returns None for non-claim doc_ids
    (e.g. "POL-5504-policy") — policy docs aren't claim-scoped, so they can't
    violate cross-step claim separation."""
    parts = doc_id.split("-")
    if len(parts) >= 2 and parts[0] == "CLM":
        return f"{parts[0]}-{parts[1]}"
    return None


def _cross_step_provenance_holds(trajectory: AgentTrajectory) -> bool:
    """True iff no claim's doc_ids appear in more than one step's citations.

    Single-step trajectories ("single"/"refuse" routes, or a "multi" route
    that decomposed to one step) trivially hold — there's only one step, so
    there's no OTHER step to cross-contaminate with.
    """
    if len(trajectory.steps) <= 1:
        return True

    claim_to_steps: dict[str, set[int]] = {}
    for i, step in enumerate(trajectory.steps):
        for doc_id in step.citations_doc_ids:
            claim = _claim_prefix(doc_id)
            if claim is None:
                continue
            claim_to_steps.setdefault(claim, set()).add(i)

    return all(len(step_indices) <= 1 for step_indices in claim_to_steps.values())
