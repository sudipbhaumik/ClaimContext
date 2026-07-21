from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class SourceDocument(BaseModel):
    """Raw manifest entry + file bytes. Output of SourceReader.discover()."""

    # Tier 1 — identity & entitlement
    doc_id: str
    doc_type: Literal["policy", "endorsement", "claim_note", "claim_document"]
    policy_number: str | None
    claim_number: str | None
    region: str
    assigned_adjuster: str

    # Tier 2 — stable document facts
    effective_date: str | None
    expiry_date: str | None
    loss_date: str | None
    lob: str
    version: str

    # Source location
    file_path: Path
    raw_bytes: bytes


class ExtractedDocument(BaseModel):
    """Normalised text + full Tier-1/Tier-2 metadata. Input to spec-1b chunker.

    The `section` Tier-2 field is intentionally absent here — it requires the
    full document text context that only the chunker has. Spec-1b populates it.
    """

    # Tier 1
    doc_id: str
    doc_type: Literal["policy", "endorsement", "claim_note", "claim_document"]
    policy_number: str | None
    claim_number: str | None
    region: str
    assigned_adjuster: str

    # Tier 2 (section populated by spec-1b chunker)
    effective_date: str | None
    expiry_date: str | None
    loss_date: str | None
    lob: str
    version: str

    # Extraction output
    text: str  # full normalised text; PDFs contain <!-- PAGE n --> markers
    page_count: int  # 1 for non-PDF formats
    content_hash: str  # SHA-256 hex of raw_bytes


class ExtractResult(BaseModel):
    """Wrapper returned for every document — never raises. status drives spec-1b."""

    doc_id: str
    status: Literal["new", "updated", "skipped", "error"]
    document: ExtractedDocument | None = None
    error: str | None = None
