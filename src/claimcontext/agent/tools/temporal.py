"""Temporal-validity tool (spec-5b): was the policy in force on the date of loss?

CLAUDE.md §5 requires this be *derived* at query time, not precomputed into the
corpus (no policy_in_effect_on_loss boolean baked into ingestion) — this module is
that derivation. Dates are stored ISO-formatted (YYYY-MM-DD) at ingestion; verified
directly against the live Qdrant payload before writing this (spec-5b build order
step 1) rather than assumed.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class TemporalValidityResult(BaseModel):
    in_force: bool | None  # None when a required date is missing/unparseable
    explanation: str


def _parse(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def check_policy_in_force(
    loss_date: str | None,
    effective_date: str | None,
    expiry_date: str | None,
) -> TemporalValidityResult:
    """True iff effective_date <= loss_date <= expiry_date, all inclusive.

    Inclusive on both ends deliberately: a policy that expires on the exact date
    of loss, or takes effect on the exact date of loss, is still in force that day
    — there is no reason to treat either boundary as exclusive, and an off-by-one
    here would silently misjudge a claim.
    """
    loss = _parse(loss_date)
    effective = _parse(effective_date)
    expiry = _parse(expiry_date)

    missing = [
        name
        for name, value in (
            ("loss_date", loss),
            ("effective_date", effective),
            ("expiry_date", expiry),
        )
        if value is None
    ]
    if missing:
        return TemporalValidityResult(
            in_force=None,
            explanation=(
                f"Cannot determine temporal validity — missing or unparseable: "
                f"{', '.join(missing)}."
            ),
        )

    assert loss is not None and effective is not None and expiry is not None  # narrowed above
    in_force = effective <= loss <= expiry

    if in_force:
        explanation = (
            f"Policy was in force on the date of loss: effective {effective.isoformat()} "
            f"<= loss {loss.isoformat()} <= expiry {expiry.isoformat()}."
        )
    elif loss < effective:
        explanation = (
            f"Loss ({loss.isoformat()}) occurred before the policy took effect "
            f"({effective.isoformat()}) — not in force."
        )
    else:
        explanation = (
            f"Loss ({loss.isoformat()}) occurred after the policy expired "
            f"({expiry.isoformat()}) — not in force."
        )

    return TemporalValidityResult(in_force=in_force, explanation=explanation)
