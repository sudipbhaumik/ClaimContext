"""
Synthetic corpus generator for ClaimContext.

Usage:
    python scripts/generate_corpus.py [--seed 42] [--out data/documents] [--force]

Produces deterministic claim-world artifacts into data/documents/ and writes
data/documents/manifest.json. Zero LLM calls — all variety comes from the
committed clause library.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

# ── Path setup ───────────────────────────────────────────────────────────────
# scripts/ → clause_library; src/ → claimcontext.config
# Both needed when running as `python scripts/generate_corpus.py` outside uv.
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_repo_root / "src"))

from clause_library import (  # noqa: E402
    ADJUSTER_NAMES,
    AUTO_COVERAGE_CLAUSES,
    AUTO_EXCLUSION_CLAUSES,
    CONDITION_CLAUSES,
    ENDORSEMENT_WR001_BODY,
    FNOL_LOSS_DESCRIPTIONS_AUTO,
    FNOL_LOSS_DESCRIPTIONS_PROPERTY,
    GENERAL_CONDITIONS_TEXT,
    INSPECTOR_NAMES,
    INSURER_ADDRESS,
    INSURER_NAME,
    LETTER_ACKNOWLEDGEMENT_BODIES,
    LETTER_COVERAGE_APPROVAL_BODY,
    LETTER_DENIAL_BODY_FLOOD,
    LINE_ITEMS_AUTO,
    LINE_ITEMS_PROPERTY,
    LINE_ITEMS_PROPERTY_FLOOD,
    NOTE_PHRASINGS_COVERAGE_REVIEW,
    NOTE_PHRASINGS_DECISION,
    NOTE_PHRASINGS_ESTIMATE,
    NOTE_PHRASINGS_FNOL,
    NOTE_PHRASINGS_INSPECTION,
    POLICY_FOOTER_BOILERPLATE,
    POLICY_HEADER_BOILERPLATE,
    PROPERTY_COVERAGE_CLAUSES,
    PROPERTY_EXCLUSION_CLAUSES,
)

# ── Models ───────────────────────────────────────────────────────────────────


class ClaimWorld(BaseModel):
    claim_number: str
    policy_number: str
    lob: Literal["auto", "property"]
    region: Literal["northeast", "southwest"]
    assigned_adjuster: str
    lifecycle: Literal["mature", "new"]
    policyholder_name: str
    policyholder_address: str
    loss_date: date
    effective_date: date
    expiry_date: date
    loss_description: str
    vehicle: str | None = None
    coverage_outcome: str | None = None
    special: str


class ManifestEntry(BaseModel):
    file: str
    doc_id: str
    doc_type: Literal["policy", "endorsement", "claim_note", "claim_document"]
    policy_number: str | None = None
    claim_number: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    region: str
    assigned_adjuster: str
    lob: str
    version: str = "v1"


# ── Claim world definitions ───────────────────────────────────────────────────

CLAIM_WORLDS: list[ClaimWorld] = [
    ClaimWorld(
        claim_number="CLM-1001",
        policy_number="POL-3301",
        lob="auto",
        region="northeast",
        assigned_adjuster="ADJ-014",
        lifecycle="mature",
        policyholder_name="Margaret Chen",
        policyholder_address="47 Birchwood Lane, Hartford, CT 06101",
        loss_date=date(2026, 2, 10),
        effective_date=date(2025, 6, 1),
        expiry_date=date(2026, 6, 1),
        loss_description="Rear-end collision at intersection of Main St and Elm Ave",
        vehicle="2021 Honda Accord LX, VIN 1HGCV1F18MA012345",
        coverage_outcome="covered",
        special="happy_path",
    ),
    ClaimWorld(
        claim_number="CLM-1002",
        policy_number="POL-3302",
        lob="auto",
        region="northeast",
        assigned_adjuster="ADJ-014",
        lifecycle="mature",
        policyholder_name="Margaret Chen",
        policyholder_address="47 Birchwood Lane, Hartford, CT 06101",
        loss_date=date(2026, 5, 23),  # 9 days before expiry
        effective_date=date(2025, 6, 1),
        expiry_date=date(2026, 6, 1),
        loss_description="Sideswipe collision on I-95 northbound, other party fled scene",
        vehicle="2019 Toyota Camry SE, VIN 4T1B11HK3KU234567",
        coverage_outcome="covered",
        special="near_expiry",
    ),
    ClaimWorld(
        claim_number="CLM-1003",
        policy_number="POL-4403",
        lob="property",
        region="southwest",
        assigned_adjuster="ADJ-027",
        lifecycle="mature",
        policyholder_name="Robert Navarro",
        policyholder_address="2281 Desert View Drive, Phoenix, AZ 85001",
        loss_date=date(2026, 3, 5),
        effective_date=date(2025, 8, 1),
        expiry_date=date(2026, 8, 1),
        loss_description="Water damage from flooding in basement following heavy rainfall",
        vehicle=None,
        coverage_outcome="denied",
        special="denied_flood_exclusion",
    ),
    ClaimWorld(
        claim_number="CLM-1004",
        policy_number="POL-5504",
        lob="property",
        region="southwest",
        assigned_adjuster="ADJ-027",
        lifecycle="mature",
        policyholder_name="Sandra Okafor",
        policyholder_address="814 Cactus Bloom Road, Tucson, AZ 85701",
        loss_date=date(2026, 4, 12),
        effective_date=date(2025, 6, 1),
        expiry_date=date(2026, 6, 1),
        loss_description="Wind-driven rain damage to roof and interior following severe storm",
        vehicle=None,
        coverage_outcome="covered_via_endorsement",
        special="endorsement_overrides_exclusion",
    ),
    ClaimWorld(
        claim_number="CLM-1005",
        policy_number="POL-3305",
        lob="auto",
        region="northeast",
        assigned_adjuster="ADJ-014",
        lifecycle="new",
        policyholder_name="Daniel Park",
        policyholder_address="193 Maple Street, Boston, MA 02101",
        loss_date=date(2026, 6, 14),
        effective_date=date(2025, 12, 1),
        expiry_date=date(2026, 12, 1),
        loss_description="Single-vehicle collision with highway guardrail during adverse weather",
        vehicle="2022 Ford F-150 XLT, VIN 1FTFW1E51NFA56789",
        coverage_outcome=None,
        special="fnol_only",
    ),
]

# ── PDF builder (minimal valid PDF, no library dependency) ────────────────────


def _build_pdf(text: str) -> bytes:
    """Build a minimal valid single-page PDF from plain text."""
    all_lines: list[str] = []
    for raw in text.split("\n"):
        wrapped = textwrap.wrap(raw, 90) if raw.strip() else [""]
        all_lines.extend(wrapped)

    cmds: list[str] = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
    for line in all_lines[:52]:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("\r", "")
        cmds.append(f"({safe}) Tj T*")
    cmds.append("ET")
    stream = "\n".join(cmds).encode("latin-1", errors="replace")

    o1 = b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
    o2 = b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
    o3 = (
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>>\nendobj\n"
    )
    o4 = (
        f"4 0 obj\n<</Length {len(stream)}>>\nstream\n".encode() + stream + b"\nendstream\nendobj\n"
    )
    o5 = b"5 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n"

    header = b"%PDF-1.4\n"
    objs = [o1, o2, o3, o4, o5]

    pos = len(header)
    offsets: list[int] = []
    for o in objs:
        offsets.append(pos)
        pos += len(o)

    xref_pos = pos
    xref = b"xref\n0 6\n0000000000 65535 f\r\n"
    for off in offsets:
        xref += f"{off:010d} 00000 n\r\n".encode()

    trailer = f"trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    return header + b"".join(objs) + xref + trailer


# ── OCR noise ─────────────────────────────────────────────────────────────────


def _apply_ocr_noise(text: str, rng: random.Random) -> str:
    """Apply mild OCR-style noise: ~5% of words affected."""
    words = text.split(" ")
    result: list[str] = []
    substitution_chars = ["0", "l", "I", "|", "1"]
    for word in words:
        r = rng.random()
        if r < 0.02 and len(word) > 5:
            mid = rng.randint(2, len(word) - 2)
            result.append(word[:mid] + " " + word[mid:])
        elif r < 0.035 and len(word) > 4:
            mid = rng.randint(2, len(word) - 1)
            result.append(word[:mid] + "-\n" + word[mid:])
        elif r < 0.05 and word.isalpha() and len(word) > 2:
            chars = list(word)
            idx = rng.randint(0, len(chars) - 1)
            chars[idx] = rng.choice(substitution_chars)
            result.append("".join(chars))
        else:
            result.append(word)
    return " ".join(result)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _fmt_date(d: date) -> str:
    return d.strftime("%B %d, %Y")


def _iso(d: date) -> str:
    return d.isoformat()


def _adjuster_name(adjuster_id: str) -> str:
    return ADJUSTER_NAMES.get(adjuster_id, adjuster_id)


# ── Policy writer ─────────────────────────────────────────────────────────────


def _write_policy(cw: ClaimWorld, out_dir: Path, rng: random.Random, as_pdf: bool) -> ManifestEntry:
    if cw.lob == "auto":
        coverage_sections = "\n\n".join(rng.sample(AUTO_COVERAGE_CLAUSES, k=3))
        exclusion_pairs = AUTO_EXCLUSION_CLAUSES
    else:
        coverage_sections = "\n\n".join(rng.sample(PROPERTY_COVERAGE_CLAUSES, k=3))
        exclusion_pairs = PROPERTY_EXCLUSION_CLAUSES

    exclusions_text = "\n\n".join(f"{title}\n{body}" for title, body in exclusion_pairs)
    conditions_text = "\n\n".join(rng.sample(CONDITION_CLAUSES, k=3))
    conditions_section = GENERAL_CONDITIONS_TEXT.format(conditions=conditions_text)

    header = POLICY_HEADER_BOILERPLATE.format(
        insurer_name=INSURER_NAME,
        insurer_address=INSURER_ADDRESS,
        insurer_name_upper=INSURER_NAME.upper(),
    )
    footer = POLICY_FOOTER_BOILERPLATE.format(
        insurer_name=INSURER_NAME,
        policy_number=cw.policy_number,
        page=1,
        total_pages=2,
    )

    lob_label = "Automobile" if cw.lob == "auto" else "Homeowners Property"

    body = f"""{header}
INSURANCE POLICY DECLARATIONS

Policy Number:    {cw.policy_number}
Line of Business: {lob_label}
Policyholder:     {cw.policyholder_name}
Address:          {cw.policyholder_address}
Effective Date:   {_fmt_date(cw.effective_date)}
Expiry Date:      {_fmt_date(cw.expiry_date)}
Insurer:          {INSURER_NAME}

{"Vehicle:          " + cw.vehicle if cw.vehicle else ""}

{"=" * 80}
SECTION I — INSURING AGREEMENT AND COVERAGES
{"=" * 80}

{coverage_sections}

{"=" * 80}
SECTION II — EXCLUSIONS
{"=" * 80}

The following perils and circumstances are EXCLUDED from coverage under this policy.
Where an exclusion conflicts with an attached endorsement, the endorsement controls.

{exclusions_text}

{"=" * 80}
SECTION III — LIMITS AND DEDUCTIBLES
{"=" * 80}

Coverage A — Bodily Injury Liability:    $100,000 per person / $300,000 per occurrence
Coverage B — Property Damage Liability:  $100,000 per occurrence
Coverage C — Collision:                  Actual Cash Value, $500 deductible
Coverage D — Comprehensive:              Actual Cash Value, $250 deductible
Coverage E — Uninsured Motorists:        $100,000 per person / $300,000 per occurrence
Coverage F — Medical Payments:           $5,000 per person

{conditions_section}

{footer}"""

    ext = "pdf" if as_pdf else "txt"
    fname = f"CLM-{cw.claim_number[-4:]}-policy.{ext}"
    path = out_dir / "policies" / fname

    if as_pdf:
        path.write_bytes(_build_pdf(body))
    else:
        path.write_text(body, encoding="utf-8")

    return ManifestEntry(
        file=f"policies/{fname}",
        doc_id=f"{cw.policy_number}-policy",
        doc_type="policy",
        policy_number=cw.policy_number,
        claim_number=cw.claim_number,
        effective_date=_iso(cw.effective_date),
        expiry_date=_iso(cw.expiry_date),
        region=cw.region,
        assigned_adjuster=cw.assigned_adjuster,
        lob=cw.lob,
    )


# ── Endorsement writer ────────────────────────────────────────────────────────


def _write_endorsement(cw: ClaimWorld, out_dir: Path) -> ManifestEntry:
    endorsement_date = date(2025, 9, 1)
    body = ENDORSEMENT_WR001_BODY.format(
        endorsement_date=_fmt_date(endorsement_date),
        policy_number=cw.policy_number,
        policyholder_name=cw.policyholder_name,
    )
    fname = f"CLM-{cw.claim_number[-4:]}-endorsement-WR001.txt"
    path = out_dir / "policies" / fname
    path.write_text(body, encoding="utf-8")
    return ManifestEntry(
        file=f"policies/{fname}",
        doc_id=f"{cw.policy_number}-endorsement-WR001",
        doc_type="endorsement",
        policy_number=cw.policy_number,
        claim_number=cw.claim_number,
        effective_date=_iso(endorsement_date),
        expiry_date=_iso(cw.expiry_date),
        region=cw.region,
        assigned_adjuster=cw.assigned_adjuster,
        lob=cw.lob,
    )


# ── FNOL writer ───────────────────────────────────────────────────────────────


def _write_fnol(cw: ClaimWorld, out_dir: Path, rng: random.Random) -> ManifestEntry:
    if cw.lob == "auto":
        loss_desc = rng.choice(FNOL_LOSS_DESCRIPTIONS_AUTO)
        vehicle_line = f"Vehicle:             {cw.vehicle or 'N/A'}"
    else:
        loss_desc = rng.choice(FNOL_LOSS_DESCRIPTIONS_PROPERTY)
        vehicle_line = ""

    report_date = date(
        cw.loss_date.year,
        cw.loss_date.month,
        cw.loss_date.day + 1 if cw.loss_date.day < 28 else cw.loss_date.day,
    )

    body = f"""FIRST NOTICE OF LOSS

{"=" * 80}
Insurer:             {INSURER_NAME}
{"=" * 80}

Claim Number:        {cw.claim_number}
Policy Number:       {cw.policy_number}
Date of Loss:        {_fmt_date(cw.loss_date)}
Date Reported:       {_fmt_date(report_date)}

Policyholder:        {cw.policyholder_name}
Address:             {cw.policyholder_address}
{vehicle_line}
Assigned Adjuster:   {_adjuster_name(cw.assigned_adjuster)} ({cw.assigned_adjuster})
Region:              {cw.region.title()}

{"=" * 80}
DESCRIPTION OF LOSS
{"=" * 80}

{loss_desc}

{"=" * 80}
INITIAL LOSS SUMMARY
{"=" * 80}

Reported Cause:      {cw.loss_description}
Policy Status:       Active at time of loss
Coverage Review:     Pending adjuster assignment and inspection

{"=" * 80}
NOTICE

This document constitutes the initial notice of loss only. Coverage determination
is subject to investigation and review of the applicable policy terms. The filing
of this notice does not constitute an admission of coverage or liability.

{INSURER_NAME}
{INSURER_ADDRESS}
{"=" * 80}
"""

    fname = f"CLM-{cw.claim_number[-4:]}-fnol.txt"
    path = out_dir / "claim-docs" / fname
    path.write_text(body, encoding="utf-8")
    return ManifestEntry(
        file=f"claim-docs/{fname}",
        doc_id=f"{cw.claim_number}-fnol",
        doc_type="claim_document",
        policy_number=cw.policy_number,
        claim_number=cw.claim_number,
        region=cw.region,
        assigned_adjuster=cw.assigned_adjuster,
        lob=cw.lob,
    )


# ── Estimate writers ──────────────────────────────────────────────────────────


def _estimate_line_items(cw: ClaimWorld, rng: random.Random) -> list[tuple[str, int, float]]:
    """Return (description, qty, unit_cost) tuples for this claim."""
    if cw.lob == "auto":
        pool = LINE_ITEMS_AUTO
    elif cw.special == "denied_flood_exclusion":
        pool = LINE_ITEMS_PROPERTY_FLOOD
    else:
        pool = LINE_ITEMS_PROPERTY
    count = rng.randint(5, 8)
    selected = rng.sample(pool, k=min(count, len(pool)))
    return [(desc, 1, cost) for desc, cost in selected]


def _write_estimate_html(cw: ClaimWorld, out_dir: Path, rng: random.Random) -> ManifestEntry:
    items = _estimate_line_items(cw, rng)
    total = sum(qty * unit for _, qty, unit in items)
    inspector = rng.choice(INSPECTOR_NAMES)
    insp_date = date(cw.loss_date.year, cw.loss_date.month, min(cw.loss_date.day + 5, 28))

    rows = "\n".join(
        f"    <tr><td>{i + 1}</td><td>{desc}</td><td>{qty}</td>"
        f"<td>${unit:,.2f}</td><td>${qty * unit:,.2f}</td></tr>"
        for i, (desc, qty, unit) in enumerate(items)
    )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Damage Assessment — {cw.claim_number}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1 {{ font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #555; padding: 6px 10px; text-align: left; }}
    th {{ background: #ddd; }}
    .total {{ font-weight: bold; }}
  </style>
</head>
<body>
<h1>DAMAGE ASSESSMENT / REPAIR ESTIMATE</h1>
<p>
  <strong>Claim Number:</strong> {cw.claim_number}<br>
  <strong>Policy Number:</strong> {cw.policy_number}<br>
  <strong>Policyholder:</strong> {cw.policyholder_name}<br>
  <strong>Inspection Date:</strong> {_fmt_date(insp_date)}<br>
  {"<strong>Vehicle:</strong> " + (cw.vehicle or "") + "<br>" if cw.vehicle else ""}
  <strong>Inspector:</strong> {inspector}<br>
  <strong>Adjuster:</strong> {_adjuster_name(cw.assigned_adjuster)} ({cw.assigned_adjuster})
</p>

<table>
  <thead>
    <tr>
      <th>Item</th><th>Description</th><th>Qty</th><th>Unit Cost</th><th>Total</th>
    </tr>
  </thead>
  <tbody>
{rows}
    <tr class="total">
      <td colspan="4"><strong>TOTAL (before deductible)</strong></td>
      <td><strong>${total:,.2f}</strong></td>
    </tr>
  </tbody>
</table>

<p><em>This estimate is subject to final review and approval by the assigned adjuster.
Authorized payment will reflect applicable deductibles and policy limits.</em></p>
</body>
</html>"""

    fname = f"CLM-{cw.claim_number[-4:]}-estimate.html"
    path = out_dir / "claim-docs" / fname
    path.write_text(body, encoding="utf-8")
    return ManifestEntry(
        file=f"claim-docs/{fname}",
        doc_id=f"{cw.claim_number}-estimate",
        doc_type="claim_document",
        policy_number=cw.policy_number,
        claim_number=cw.claim_number,
        region=cw.region,
        assigned_adjuster=cw.assigned_adjuster,
        lob=cw.lob,
    )


def _write_estimate_txt(
    cw: ClaimWorld, out_dir: Path, rng: random.Random, apply_noise: bool = False
) -> ManifestEntry:
    items = _estimate_line_items(cw, rng)
    total = sum(qty * unit for _, qty, unit in items)
    inspector = rng.choice(INSPECTOR_NAMES)
    insp_date = date(cw.loss_date.year, cw.loss_date.month, min(cw.loss_date.day + 5, 28))

    rows = "\n".join(
        f"  {i + 1:<4} {desc:<50} {qty:<5} ${unit:>10,.2f}   ${qty * unit:>10,.2f}"
        for i, (desc, qty, unit) in enumerate(items)
    )

    body = f"""DAMAGE ASSESSMENT / REPAIR ESTIMATE

{"=" * 80}
Claim Number:        {cw.claim_number}
Policy Number:       {cw.policy_number}
Policyholder:        {cw.policyholder_name}
Inspection Date:     {_fmt_date(insp_date)}
{"Vehicle:             " + (cw.vehicle or "") if cw.vehicle else ""}
Inspector:           {inspector}
Adjuster:            {_adjuster_name(cw.assigned_adjuster)} ({cw.assigned_adjuster})
{"=" * 80}

LINE ITEMS
{"─" * 80}
  No.  Description                                        Qty    Unit Cost        Total
{"─" * 80}
{rows}
{"─" * 80}
       {"TOTAL (before deductible)":<52}          ${total:>10,.2f}
{"=" * 80}

NOTE: This estimate is subject to final review and approval. Authorized payment
will reflect applicable deductibles and policy limits per {cw.policy_number}.
"""

    if apply_noise:
        body = _apply_ocr_noise(body, rng)

    fname = f"CLM-{cw.claim_number[-4:]}-estimate.txt"
    path = out_dir / "claim-docs" / fname
    path.write_text(body, encoding="utf-8")
    return ManifestEntry(
        file=f"claim-docs/{fname}",
        doc_id=f"{cw.claim_number}-estimate",
        doc_type="claim_document",
        policy_number=cw.policy_number,
        claim_number=cw.claim_number,
        region=cw.region,
        assigned_adjuster=cw.assigned_adjuster,
        lob=cw.lob,
    )


# ── Correspondence letter writers ─────────────────────────────────────────────


def _write_letter(
    cw: ClaimWorld,
    out_dir: Path,
    rng: random.Random,
    letter_date: date,
    body_text: str,
    suffix: str = "letter",
) -> ManifestEntry:
    adjuster_name = _adjuster_name(cw.assigned_adjuster)
    salutation = "Mr." if cw.policyholder_name.split()[0] in ("Robert", "Daniel") else "Ms."
    last_name = cw.policyholder_name.split()[-1]
    full_body = f"""{_fmt_date(letter_date)}

{cw.policyholder_name}
{cw.policyholder_address}

Re: Claim Number {cw.claim_number}
    Policy Number {cw.policy_number}

Dear {salutation} {last_name},

{body_text}

Sincerely,

{adjuster_name}
Claims Adjuster — {cw.region.title()} Region
{INSURER_NAME}
{INSURER_ADDRESS}

─────────────────────────────────────────────────────────────────────────────
This letter is for informational purposes only. Policy terms govern in all
cases. To dispute this determination, contact our Claims Department in writing
within 30 days of this letter's date.
─────────────────────────────────────────────────────────────────────────────
"""
    fname = f"CLM-{cw.claim_number[-4:]}-{suffix}.txt"
    path = out_dir / "claim-docs" / fname
    path.write_text(full_body, encoding="utf-8")
    return ManifestEntry(
        file=f"claim-docs/{fname}",
        doc_id=f"{cw.claim_number}-{suffix}",
        doc_type="claim_document",
        policy_number=cw.policy_number,
        claim_number=cw.claim_number,
        region=cw.region,
        assigned_adjuster=cw.assigned_adjuster,
        lob=cw.lob,
    )


# ── Claim notes JSONL writer ──────────────────────────────────────────────────


def _write_notes_jsonl(cw: ClaimWorld, out_dir: Path, rng: random.Random) -> ManifestEntry:
    adjuster = cw.assigned_adjuster
    loss_dt = cw.loss_date
    inspector = rng.choice(INSPECTOR_NAMES)

    insp_date = date(loss_dt.year, loss_dt.month, min(loss_dt.day + 5, 28))
    est_date = date(loss_dt.year, loss_dt.month, min(loss_dt.day + 10, 28))
    dec_date = date(loss_dt.year, loss_dt.month, min(loss_dt.day + 18, 28))

    fnol_text = rng.choice(NOTE_PHRASINGS_FNOL).format(
        date=_fmt_date(loss_dt),
        loss_desc=cw.loss_description + ".",
        loss_date=_fmt_date(loss_dt),
        report_date=_fmt_date(date(loss_dt.year, loss_dt.month, min(loss_dt.day + 1, 28))),
    )
    insp_text = rng.choice(NOTE_PHRASINGS_INSPECTION).format(
        date=_fmt_date(insp_date),
        area="affected area",
        eta=_fmt_date(est_date),
    )

    if cw.special == "denied_flood_exclusion":
        cov_text = rng.choice(NOTE_PHRASINGS_COVERAGE_REVIEW).format(
            policy_number=cw.policy_number,
            loss_date=_fmt_date(loss_dt),
            exclusion_finding=(
                "Exclusion 2.1 (Flood) applies. "
                "Surface water intrusion confirmed by inspection. Coverage not triggered."
            ),
            outcome="DENIED",
            basis="Flood exclusion — Exclusion 2.1",
        )
        est_text = rng.choice(NOTE_PHRASINGS_ESTIMATE).format(estimator=inspector, amount=17_850.00)
        dec_text = rng.choice(NOTE_PHRASINGS_DECISION).format(
            outcome="DENIED — flood exclusion applies",
            date=_fmt_date(dec_date),
            reason="Exclusion 2.1 — Flood. Surface water entry confirmed.",
            amount=0.0,
        )
    elif cw.special == "endorsement_overrides_exclusion":
        cov_text = rng.choice(NOTE_PHRASINGS_COVERAGE_REVIEW).format(
            policy_number=cw.policy_number,
            loss_date=_fmt_date(loss_dt),
            exclusion_finding=(
                "Exclusion 2.3 (Wind-Driven Rain) reviewed. Endorsement WR-001 (eff. 2025-09-01) "
                "overrides Exclusion 2.3 where windstorm creates opening. Conditions met: "
                "storm caused roof breach; rain entered through that opening. Coverage applies."
            ),
            outcome="COVERED — endorsement WR-001 applies",
            basis="Endorsement WR-001 modifying Exclusion 2.3",
        )
        est_text = rng.choice(NOTE_PHRASINGS_ESTIMATE).format(estimator=inspector, amount=12_340.00)
        dec_text = rng.choice(NOTE_PHRASINGS_DECISION).format(
            outcome="COVERED under Endorsement WR-001",
            date=_fmt_date(dec_date),
            reason="N/A",
            amount=12_340.00,
        )
    elif cw.special == "near_expiry":
        cov_text = (
            f"Coverage review completed. Policy {cw.policy_number} "
            f"active at DOL {_fmt_date(loss_dt)}. "
            f"Note: loss occurred 9 days before policy expiry ({_fmt_date(cw.expiry_date)}). "
            "DOL falls within policy period. Coverage confirmed."
        )
        est_text = rng.choice(NOTE_PHRASINGS_ESTIMATE).format(estimator=inspector, amount=4_820.00)
        dec_text = rng.choice(NOTE_PHRASINGS_DECISION).format(
            outcome="COVERED",
            date=_fmt_date(dec_date),
            reason="N/A",
            amount=4_820.00,
        )
    else:
        cov_text = rng.choice(NOTE_PHRASINGS_COVERAGE_REVIEW).format(
            policy_number=cw.policy_number,
            loss_date=_fmt_date(loss_dt),
            exclusion_finding="No applicable exclusions identified. Coverage confirmed.",
            outcome="COVERED",
            basis="Section I — Collision coverage",
        )
        est_text = rng.choice(NOTE_PHRASINGS_ESTIMATE).format(estimator=inspector, amount=3_247.50)
        dec_text = rng.choice(NOTE_PHRASINGS_DECISION).format(
            outcome="COVERED",
            date=_fmt_date(dec_date),
            reason="N/A",
            amount=3_247.50,
        )

    clm = cw.claim_number[-4:]
    base_notes = [
        {"note_id": f"NOTE-{clm}-01", "note_date": _iso(loss_dt), "text": fnol_text},
        {"note_id": f"NOTE-{clm}-02", "note_date": _iso(insp_date), "text": insp_text},
        {"note_id": f"NOTE-{clm}-03", "note_date": _iso(insp_date), "text": cov_text},
        {"note_id": f"NOTE-{clm}-04", "note_date": _iso(est_date), "text": est_text},
        {"note_id": f"NOTE-{clm}-05", "note_date": _iso(dec_date), "text": dec_text},
    ]

    if rng.random() > 0.4:
        extra_text = (
            f"Follow-up contact with insured {_fmt_date(est_date)}. "
            "Insured confirmed they have not commenced permanent repairs "
            "pending our authorization. Advised to retain all contractor receipts."
        )
        base_notes.insert(
            3,
            {
                "note_id": f"NOTE-{clm}-03B",
                "note_date": _iso(est_date),
                "text": extra_text,
            },
        )

    notes = [
        {
            "claim_number": cw.claim_number,
            "author": adjuster,
            **note,
        }
        for note in base_notes
    ]

    fname = f"{cw.claim_number}.jsonl"
    path = out_dir / "claim-notes" / fname
    path.write_text(
        "\n".join(json.dumps(n) for n in notes) + "\n",
        encoding="utf-8",
    )
    return ManifestEntry(
        file=f"claim-notes/{fname}",
        doc_id=f"{cw.claim_number}-notes",
        doc_type="claim_note",
        policy_number=cw.policy_number,
        claim_number=cw.claim_number,
        region=cw.region,
        assigned_adjuster=cw.assigned_adjuster,
        lob=cw.lob,
    )


# ── Generator ─────────────────────────────────────────────────────────────────


def generate(out_dir: Path, seed: int = 42) -> list[ManifestEntry]:
    """
    Generate the full synthetic corpus into out_dir.
    Returns the list of ManifestEntry objects written.
    Deterministic: same seed → byte-identical output.
    """
    rng = random.Random(seed)
    entries: list[ManifestEntry] = []

    for subdir in ("policies", "claim-notes", "claim-docs"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    for cw in CLAIM_WORLDS:
        # Policy document (CLM-1001 and CLM-1004 as PDF, others as TXT)
        as_pdf = cw.claim_number in ("CLM-1001", "CLM-1004")
        entries.append(_write_policy(cw, out_dir, rng, as_pdf=as_pdf))

        # Endorsement (CLM-1004 only)
        if cw.special == "endorsement_overrides_exclusion":
            entries.append(_write_endorsement(cw, out_dir))

        # FNOL (all claims)
        entries.append(_write_fnol(cw, out_dir, rng))

        # Estimate and letter (mature claims only)
        if cw.lifecycle == "mature":
            if cw.claim_number == "CLM-1001":
                entries.append(_write_estimate_html(cw, out_dir, rng))
            elif cw.claim_number == "CLM-1004":
                # OCR-noised estimate for CLM-1004
                entries.append(_write_estimate_txt(cw, out_dir, rng, apply_noise=True))
            else:
                entries.append(_write_estimate_txt(cw, out_dir, rng, apply_noise=False))

            # Correspondence letter
            letter_date = date(
                cw.loss_date.year,
                cw.loss_date.month,
                min(cw.loss_date.day + 20, 28),
            )

            if cw.special == "denied_flood_exclusion":
                body_text = LETTER_DENIAL_BODY_FLOOD.format(
                    policy_number=cw.policy_number,
                    loss_date=_fmt_date(cw.loss_date),
                )
                entries.append(
                    _write_letter(cw, out_dir, rng, letter_date, body_text, "letter-final")
                )
                # Near-duplicate draft (minor wording variation)
                draft_body = body_text.replace(
                    "After careful review of your claim and the applicable policy provisions,",
                    "Following our review of your claim and the policy provisions,",
                ).replace(
                    "We recognize this outcome may be disappointing.",
                    "We understand this may be a difficult outcome.",
                )
                entries.append(
                    _write_letter(cw, out_dir, rng, letter_date, draft_body, "letter-draft")
                )
            elif cw.special == "endorsement_overrides_exclusion":
                body_text = LETTER_COVERAGE_APPROVAL_BODY.format(
                    policy_number=cw.policy_number,
                    authorized_amount=12_340.00,
                )
                entries.append(_write_letter(cw, out_dir, rng, letter_date, body_text, "letter"))
            else:
                body_text = rng.choice(LETTER_ACKNOWLEDGEMENT_BODIES)
                entries.append(_write_letter(cw, out_dir, rng, letter_date, body_text, "letter"))

        # Claim notes JSONL (mature claims only — CLM-1005 gets none)
        if cw.lifecycle == "mature":
            entries.append(_write_notes_jsonl(cw, out_dir, rng))

    # Corrupt PDF fixture (spec-1a error-isolation proof)
    corrupt_path = out_dir / "_corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4\nThis is not a valid PDF.\n%%EOF\n")

    # Manifest — written last, after all artifacts exist
    manifest_path = out_dir / "manifest.json"
    manifest_data = [e.model_dump() for e in entries]
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return entries


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    from claimcontext.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate ClaimContext synthetic corpus.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--out",
        default=settings.documents_dir,
        help=f"Output directory (default: {settings.documents_dir})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing corpus without prompting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists() and not args.force:
        print(
            f"Corpus already exists at {out_dir} (manifest.json found).\n"
            "Use --force to overwrite. "
            "Aborting to protect a corpus the golden eval set may depend on."
        )
        raise SystemExit(1)

    print(f"Generating corpus → {out_dir}  [seed={args.seed}]")
    entries = generate(out_dir, seed=args.seed)

    by_type: dict[str, int] = {}
    for e in entries:
        by_type[e.doc_type] = by_type.get(e.doc_type, 0) + 1

    print(f"\nCorpus written: {len(entries)} artifacts")
    for doc_type, count in sorted(by_type.items()):
        print(f"  {doc_type:<20} {count}")
    print(f"\nManifest: {manifest_path}")
    print(f"Seed used: {args.seed}")
    print("\nDone.")


if __name__ == "__main__":
    main()
