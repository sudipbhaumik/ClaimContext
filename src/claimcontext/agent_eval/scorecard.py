"""Human-readable trajectory eval report printing — structurally mirrors
spec-4's eval/scorecard.py."""

from __future__ import annotations

from claimcontext.agent_eval.schema import TrajectoryEvalReport

_W = 74


def print_trajectory_scorecard(report: TrajectoryEvalReport) -> None:
    print("\n" + "━" * _W)
    print("ClaimContext Agent Trajectory Evaluation")
    print("━" * _W)

    print(f"\n{'Failure mode':<16} {'Pass rate':>10}  {'Gate':>6}")
    print("-" * 40)
    for category, rate in report.pass_rate_by_category.items():
        gate = "PASS" if category not in report.gate_failures else "FAIL ✗"
        print(f"{category:<16} {rate:>10.3f}  {gate:>6}")

    hdr = f"{'ID':<6} {'Failure mode':<14} {'Route':<8} {'Result':<6}"
    print(f"\n{hdr}")
    print("-" * _W)
    for r in report.entries:
        result_str = "PASS" if r.passed else "FAIL"
        print(f"{r.entry_id:<6} {r.failure_mode.value:<14} {r.trajectory.route:<8} {result_str:<6}")
        if not r.passed:
            print(f"       -> {r.failure_reason}")

    print("\n" + "━" * _W)
    gate_str = "PASS" if report.gate_passed else "FAIL ✗"
    print(f"CI Gate: {gate_str}", end="")
    if report.gate_failures:
        print(f"  — failed categories: {', '.join(report.gate_failures)}")
    else:
        print()
    print("━" * _W + "\n")
