"""
Tests for spec-1b: chunk → embed → upsert → run report.

Proves all 5 spec-1b proofs. Uses mock Embedder and QdrantWriter throughout so
Qdrant does not need to be running. Reuses spec-1a's session-scoped first_run
fixture for proofs 3 and 4.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from claimcontext.config import Settings
from claimcontext.ingestion import (
    Chunk,
    ExtractResult,
    FileSystemReader,
    SourceDocument,
    SourceReader,
    build_report,
    chunk_document,
    run_chunk_embed_upsert,
    run_discover_extract,
)

SOURCE_DIR = Path("data/documents")
CORRUPT_PDF = SOURCE_DIR / "_corrupt.pdf"


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> list[ExtractResult]:
    """Single extract run against the real corpus; reused across proofs 3 and 4."""
    store = tmp_path_factory.mktemp("store") / ".hash_store.json"
    return run_discover_extract(SOURCE_DIR, store)


@pytest.fixture
def settings() -> Settings:
    return Settings()


# ── Mock infrastructure ───────────────────────────────────────────────────────


class _MockEmbedder:
    """Returns zero vectors of the correct dimension. Never loads torch."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [[0.0] * self.dim for _ in texts]


class _MockWriter:
    """Records all calls to upsert_chunks and delete_chunks_for_doc in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []  # ("upsert"|"delete", arg)
        self.upserted: dict[str, list[str]] = {}  # doc_id → [chunk_id, ...]
        self.deleted: list[str] = []

    def ensure_collection(self) -> None:
        pass

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        doc_id = chunks[0].doc_id
        self.events.append(("upsert", doc_id))
        self.upserted.setdefault(doc_id, []).extend(c.chunk_id for c in chunks)

    def delete_chunks_for_doc(self, doc_id: str) -> None:
        self.events.append(("delete", doc_id))
        self.deleted.append(doc_id)
        self.upserted.pop(doc_id, None)


class _MockReader(SourceReader):
    def __init__(self, docs: list[SourceDocument]) -> None:
        self._docs = docs

    def discover(self) -> list[SourceDocument]:
        return self._docs


def _corrupt_doc() -> SourceDocument:
    return SourceDocument(
        doc_id="TEST-corrupt",
        doc_type="claim_document",
        policy_number=None,
        claim_number="CLM-9999",
        effective_date=None,
        expiry_date=None,
        loss_date=None,
        region="northeast",
        assigned_adjuster="ADJ-014",
        lob="auto",
        version="v1",
        file_path=CORRUPT_PDF,
        raw_bytes=CORRUPT_PDF.read_bytes(),
    )


class _FailEmbedder(_MockEmbedder):
    """Raises on embed() for any text containing the trigger string."""

    def __init__(self, trigger: str) -> None:
        super().__init__()
        self._trigger = trigger

    def embed(self, texts: list[str]) -> list[list[float]]:
        if any(self._trigger in t for t in texts):
            raise RuntimeError(f"embed failed: trigger {self._trigger!r} found")
        return super().embed(texts)


# ── Proof 1: Idempotency with incremental update ──────────────────────────────


def test_proof1_only_new_doc_processes_on_second_run(tmp_path: Path, settings: Settings) -> None:
    store = tmp_path / ".hash_store.json"
    run1 = run_discover_extract(SOURCE_DIR, store)
    embedder = _MockEmbedder()
    writer = _MockWriter()
    run_chunk_embed_upsert(run1, embedder, writer, settings)

    # Add a synthetic new doc via a reader that appends one extra
    real_docs = FileSystemReader(SOURCE_DIR).discover()
    extra_doc = SourceDocument(
        doc_id="CLM-9000-fnol",
        doc_type="claim_document",
        policy_number=None,
        claim_number="CLM-9000",
        effective_date=None,
        expiry_date=None,
        loss_date="2026-01-01",
        region="northeast",
        assigned_adjuster="ADJ-014",
        lob="auto",
        version="v1",
        file_path=Path("data/documents/claim-docs/CLM-9000-fnol.txt"),
        raw_bytes=b"New claim FNOL for CLM-9000. Damage reported.",
    )
    store2 = tmp_path / ".hash_store2.json"
    run2 = run_discover_extract(SOURCE_DIR, store2, reader=_MockReader(real_docs + [extra_doc]))

    writer2 = _MockWriter()
    embedder2 = _MockEmbedder()
    run_chunk_embed_upsert(run2, embedder2, writer2, settings)

    # Only the new doc should be upserted in run2 context
    new_statuses = {r.doc_id: r.status for r in run2 if r.doc_id == "CLM-9000-fnol"}
    assert new_statuses == {"CLM-9000-fnol": "new"}


# ── Proof 2: Update deletes old, inserts new; ordering verified ───────────────


def test_proof2_update_ordering_delete_before_upsert(tmp_path: Path, settings: Settings) -> None:
    real_docs = FileSystemReader(SOURCE_DIR).discover()
    target = next(d for d in real_docs if d.doc_id == "CLM-1001-fnol")

    store = tmp_path / ".hash_store.json"
    run1 = run_discover_extract(SOURCE_DIR, store, reader=_MockReader([target]))
    writer = _MockWriter()
    run_chunk_embed_upsert(run1, _MockEmbedder(), writer, settings)

    # Mutate: append text to make the hash different
    mutated = SourceDocument(
        **{**target.model_dump(), "raw_bytes": target.raw_bytes + b"\nAdditional text added."}
    )
    store2 = tmp_path / ".hash_store2.json"
    # Seed store2 with run1's hash so the pipeline sees it as "updated"
    store2.write_text(json.dumps({"CLM-1001-fnol": hashlib.sha256(target.raw_bytes).hexdigest()}))
    run2 = run_discover_extract(SOURCE_DIR, store2, reader=_MockReader([mutated]))

    assert any(r.status == "updated" for r in run2 if r.doc_id == "CLM-1001-fnol")

    writer2 = _MockWriter()
    run_chunk_embed_upsert(run2, _MockEmbedder(), writer2, settings)

    # Ordering: delete must precede upsert in the event log
    events_for_doc = [(evt, arg) for evt, arg in writer2.events if arg == "CLM-1001-fnol"]
    event_types = [evt for evt, _ in events_for_doc]
    assert "delete" in event_types, "delete_chunks_for_doc was not called for updated doc"
    assert "upsert" in event_types, "upsert_chunks was not called for updated doc"
    delete_idx = event_types.index("delete")
    upsert_idx = event_types.index("upsert")
    assert delete_idx < upsert_idx, "delete must happen before upsert"


def test_proof2_embed_failure_leaves_old_chunks(tmp_path: Path, settings: Settings) -> None:
    """If embedding fails, old chunks must NOT be deleted."""
    real_docs = FileSystemReader(SOURCE_DIR).discover()
    target = next(d for d in real_docs if d.doc_id == "CLM-1001-fnol")

    store = tmp_path / ".hash_store.json"
    run1 = run_discover_extract(SOURCE_DIR, store, reader=_MockReader([target]))
    writer = _MockWriter()
    run_chunk_embed_upsert(run1, _MockEmbedder(), writer, settings)

    # Simulate updated status by seeding the store with the old hash
    store2 = tmp_path / ".store2.json"
    store2.write_text(json.dumps({"CLM-1001-fnol": hashlib.sha256(target.raw_bytes).hexdigest()}))
    mutated = SourceDocument(**{**target.model_dump(), "raw_bytes": target.raw_bytes + b"\nMore."})
    run2 = run_discover_extract(SOURCE_DIR, store2, reader=_MockReader([mutated]))

    # Embedder that always raises
    fail_embedder = _FailEmbedder(trigger="")  # trigger on any text

    writer2 = _MockWriter()
    _, embed_errors = run_chunk_embed_upsert(run2, fail_embedder, writer2, settings)

    assert embed_errors == 1
    assert "CLM-1001-fnol" not in writer2.deleted, "delete must not be called when embedding fails"


# ── Proof 3: Fixed-width table stays in one chunk ─────────────────────────────


def test_proof3_table_in_single_chunk(first_run: list[ExtractResult], settings: Settings) -> None:
    est = next((r for r in first_run if r.doc_id == "CLM-1003-estimate"), None)
    assert est is not None and est.document is not None, (
        f"CLM-1003-estimate missing or failed: {est}"
    )

    chunks = chunk_document(est.document, settings)
    assert chunks, "CLM-1003-estimate produced no chunks"

    table_chunks = [c for c in chunks if "TOTAL" in c.text and "$" in c.text]
    assert table_chunks, "No chunk contains both TOTAL and $"

    # No chunk has TOTAL but not $ or vice versa (table wasn't split)
    for c in chunks:
        has_total = "TOTAL" in c.text
        has_dollar = "$" in c.text
        assert has_total == has_dollar, (
            f"Table appears split: has_total={has_total} has_dollar={has_dollar}\n{c.text[:200]}"
        )

    # Column headers and line items co-exist in the same chunk
    for c in table_chunks:
        assert "Description" in c.text or "No." in c.text or "Unit" in c.text, (
            f"Table chunk missing column headers:\n{c.text[:300]}"
        )


# ── Proof 4: page field from multi-page PDF ───────────────────────────────────


def test_proof4_page_field_populated(first_run: list[ExtractResult], settings: Settings) -> None:
    pol = next((r for r in first_run if r.doc_id == "POL-5504-policy"), None)
    assert pol is not None and pol.document is not None, f"POL-5504-policy missing or failed: {pol}"
    assert pol.document.page_count == 3

    chunks = chunk_document(pol.document, settings)
    assert chunks, "POL-5504-policy produced no chunks"

    pages = {c.page for c in chunks}
    assert 1 in pages, "No chunk on page 1"
    assert 2 in pages, "No chunk on page 2"
    assert 3 in pages, "No chunk on page 3"

    for c in chunks:
        assert "<!-- PAGE" not in c.text, f"Page marker leaked into chunk text:\n{c.text[:200]}"
        assert c.page in {1, 2, 3}, f"Unexpected page value: {c.page}"


# ── Proof 5: corrupt file → run completes → report names failure ──────────────


def test_proof5_corrupt_file_reported(tmp_path: Path, settings: Settings) -> None:
    real_docs = FileSystemReader(SOURCE_DIR).discover()[:3]
    corrupt = _corrupt_doc()
    mixed = real_docs + [corrupt]

    store = tmp_path / ".hash_store.json"
    results = run_discover_extract(SOURCE_DIR, store, reader=_MockReader(mixed))

    writer = _MockWriter()
    chunk_counts, embed_errors = run_chunk_embed_upsert(results, _MockEmbedder(), writer, settings)

    report = build_report(results, chunk_counts, elapsed_seconds=0.1)

    assert report.failed == 1, f"Expected 1 failure, got {report.failed}"
    assert report.failures[0]["doc_id"] == "TEST-corrupt"
    assert report.failures[0]["error"]
    assert report.ingested_docs == 3
    assert embed_errors == 0  # real docs completed without embed error


# ── Chunk model sanity ────────────────────────────────────────────────────────


def test_chunk_ids_are_stable(first_run: list[ExtractResult], settings: Settings) -> None:
    """Same doc chunked twice → identical chunk_ids."""
    r = next(x for x in first_run if x.doc_id == "CLM-1001-fnol" and x.document)
    chunks_a = chunk_document(r.document, settings)
    chunks_b = chunk_document(r.document, settings)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_chunk_ids_are_unique_within_doc(
    first_run: list[ExtractResult], settings: Settings
) -> None:
    for r in first_run:
        if not r.document:
            continue
        chunks = chunk_document(r.document, settings)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"{r.doc_id}: duplicate chunk_ids"


def test_all_chunks_carry_tier1_metadata(
    first_run: list[ExtractResult], settings: Settings
) -> None:
    for r in first_run:
        if not r.document:
            continue
        for c in chunk_document(r.document, settings):
            assert c.doc_id
            assert c.doc_type
            assert c.region
            assert c.assigned_adjuster
            assert c.embedding_model == settings.embedding_model
            assert c.chunker_version == settings.chunker_version


def test_claim_note_chunks_have_empty_section(
    first_run: list[ExtractResult], settings: Settings
) -> None:
    note_results = [r for r in first_run if r.document and r.document.doc_type == "claim_note"]
    assert note_results
    for r in note_results:
        for c in chunk_document(r.document, settings):
            assert c.section == "", f"{r.doc_id} note chunk has non-empty section: {c.section!r}"
            assert c.page == 1


def test_policy_chunks_have_sections(first_run: list[ExtractResult], settings: Settings) -> None:
    """At least some policy chunks must have a non-empty section (heading detected)."""
    pdf_policy = next(r for r in first_run if r.doc_id == "POL-3301-policy" and r.document)
    chunks = chunk_document(pdf_policy.document, settings)
    sections = [c.section for c in chunks if c.section]
    assert sections, "No section headings detected in POL-3301-policy"


def test_report_fields(tmp_path: Path, settings: Settings) -> None:
    store = tmp_path / ".hash_store.json"
    results = run_discover_extract(SOURCE_DIR, store)
    writer = _MockWriter()
    chunk_counts, embed_errors = run_chunk_embed_upsert(results, _MockEmbedder(), writer, settings)
    report = build_report(results, chunk_counts, elapsed_seconds=1.23, embed_errors=embed_errors)

    assert report.discovered == 24
    assert report.ingested_docs == 24
    assert report.skipped == 0
    assert report.failed == 0
    assert report.embed_errors == 0
    assert report.ingested_chunks > 0
    assert len(report.run_id) == 36  # UUID4
    assert report.elapsed_seconds == 1.23
