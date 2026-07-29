from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


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


class Chunk(BaseModel):
    """A single indexable unit produced by the chunker. Carries full Tier-1 + Tier-2 metadata."""

    # Tier 1 — identity & entitlement (filterable in Qdrant payload)
    chunk_id: str  # SHA-256(doc_id + "|" + str(chunk_index))[:16]
    doc_id: str
    doc_type: Literal["policy", "endorsement", "claim_note", "claim_document"]
    policy_number: str | None
    claim_number: str | None
    region: str
    assigned_adjuster: str

    # Tier 2 — stable document facts (filterable in Qdrant payload)
    page: int  # 1-indexed; 1 if no <!-- PAGE n --> markers present
    section: str  # heading path above this chunk; "" for unstructured docs
    effective_date: str | None
    expiry_date: str | None
    loss_date: str | None
    lob: str
    version: str
    embedding_model: str  # index versioning — §2A.4
    chunker_version: str  # index versioning — §2A.4

    # Content
    chunk_index: int  # 0-based position within doc (used to derive chunk_id)
    text: str  # chunk text; page markers stripped


class IngestReport(BaseModel):
    """Structured summary of one ingest run. Printed as JSON to stdout."""

    run_id: str  # UUID4
    started_at: str  # ISO-8601
    elapsed_seconds: float
    discovered: int  # total docs seen by spec-1a
    ingested_docs: int  # new + updated docs that produced chunks
    ingested_chunks: int  # total chunks upserted this run
    skipped: int  # unchanged docs (status="skipped")
    failed: int  # extraction failures from spec-1a (status="error")
    failures: list[dict[str, str]]  # [{"doc_id": ..., "error": ...}]
    embed_errors: int = Field(default=0)  # chunk/embed failures (separate from spec-1a errors)
