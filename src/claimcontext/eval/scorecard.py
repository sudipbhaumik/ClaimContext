"""Terminal scorecard formatter for EvalReport."""

from __future__ import annotations

from claimcontext.eval.schema import EvalReport, ExpectedBehavior

_W = 72
_BIAS_WARNING = (
    "⚠  Scores are directional. LLM-as-judge introduces verbosity, position,\n"
    "   and self-preference biases. Treat movements <0.05 as noise.\n"
    "   Judge: different-family model (not the answer LLM) — self-preference mitigated."
)


def print_scorecard(report: EvalReport) -> None:
    """Print a human-readable eval report to stdout."""
    print("\n" + "━" * _W)
    print("ClaimContext RAG Evaluation")
    print(f"  Golden set: {report.golden_set_version}  |  Prompt: {report.prompt_version}")
    print(f"  Timestamp:  {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
    print("━" * _W)

    # Aggregate metric table
    thresholds = {
        "context_precision": ("Context Precision", report.mean_context_precision),
        "context_recall": ("Context Recall", report.mean_context_recall),
        "faithfulness": ("Faithfulness", report.mean_faithfulness),
        "answer_relevance": ("Answer Relevance", report.mean_answer_relevance),
    }
    print(f"\n{'Metric':<24} {'Score':>6}  {'Gate':>6}")
    print("-" * 40)
    for key, (label, score) in thresholds.items():
        gate = "PASS" if key not in report.gate_failures else "FAIL ✗"
        print(f"{label:<24} {score:>6.3f}  {gate:>6}")

    # Refusal accuracy
    ra_gate = "PASS" if "refusal_accuracy" not in report.gate_failures else "FAIL ✗"
    print(f"{'Refusal Accuracy':<24} {report.refusal_accuracy:>6.3f}  {ra_gate:>6}")

    # Per-entry table
    hdr = f"{'ID':<5} {'Type':<24} {'Behavior':<8} {'Result':<8} {'CP':>5} {'CR':>5} {'F':>5} {'AR':>5}"  # noqa: E501
    print(f"\n{hdr}")
    print("-" * _W)
    for r in report.entries:
        result_str = "PASS" if r.passed else "FAIL"
        cp = f"{r.context_precision:.2f}" if r.context_precision is not None else "  —  "
        cr = f"{r.context_recall:.2f}" if r.context_recall is not None else "  —  "
        f_ = f"{r.faithfulness:.2f}" if r.faithfulness is not None else "  —  "
        ar = f"{r.answer_relevance:.2f}" if r.answer_relevance is not None else "  —  "
        beh = r.expected_behavior.value.upper()[:6]
        row = f"{r.entry_id:<5} {r.question_type:<24} {beh:<8} {result_str:<8} {cp:>5} {cr:>5} {f_:>5} {ar:>5}"  # noqa: E501
        print(row)

    # Refusal detail
    refusal_entries = [r for r in report.entries if r.expected_behavior != ExpectedBehavior.ANSWER]
    if refusal_entries:
        print(f"\nRefusal cases ({len(refusal_entries)} total):")
        for r in refusal_entries:
            status = "✓" if r.refusal_correct else "✗"
            refused_flag = "refused" if r.ask_result.refused else "ANSWERED (wrong)"
            print(f"  {status} {r.entry_id} [{r.expected_behavior.value}]: {refused_flag}")

    # Gate summary
    print("\n" + "━" * _W)
    if report.gate_passed:
        print("CI Gate: PASS ✓")
    else:
        print(f"CI Gate: FAIL ✗  — failed: {', '.join(report.gate_failures)}")

    print(f"\n{_BIAS_WARNING}")
    print("━" * _W + "\n")


def format_comparison(baseline: EvalReport, degraded: EvalReport) -> str:
    """Format a side-by-side comparison for Proof 3 (regression detection)."""
    lines = [
        "",
        "━" * _W,
        "REGRESSION DETECTION — baseline vs degraded",
        f"  Baseline:  {baseline.golden_set_version} / {baseline.prompt_version}",
        f"  Degraded:  {degraded.golden_set_version} / {degraded.prompt_version}",
        "━" * _W,
        f"{'Metric':<24} {'Baseline':>10}  {'Degraded':>10}  {'Delta':>8}",
        "-" * 56,
    ]
    pairs = [
        ("Context Precision", baseline.mean_context_precision, degraded.mean_context_precision),
        ("Context Recall", baseline.mean_context_recall, degraded.mean_context_recall),
        ("Faithfulness", baseline.mean_faithfulness, degraded.mean_faithfulness),
        ("Answer Relevance", baseline.mean_answer_relevance, degraded.mean_answer_relevance),
        ("Refusal Accuracy", baseline.refusal_accuracy, degraded.refusal_accuracy),
    ]
    for label, b, d in pairs:
        delta = d - b
        sign = "+" if delta >= 0 else ""
        lines.append(f"{label:<24} {b:>10.3f}  {d:>10.3f}  {sign}{delta:>7.3f}")
    lines.append("━" * _W + "\n")
    return "\n".join(lines)
