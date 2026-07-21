"""
Tests for spec-1a: discover → extract → normalise → hash.

Proves all 4 spec-1a proofs:
1. Idempotency    — second run skips all documents unchanged since run 1.
2. Fault isolation — a corrupt document does not abort the run.
3. Multi-page PDF — page markers present; CLM-1004 policy Exclusion 2.3 present.
4. Table survival — fixed-width table content survives normalisation intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimcontext.ingestion import (
    ExtractResult,
    FileSystemReader,
    SourceDocument,
    SourceReader,
    run_discover_extract,
)

SOURCE_DIR = Path("data/documents")
CORRUPT_PDF = SOURCE_DIR / "_corrupt.pdf"


# ── Shared fixture ────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> list[ExtractResult]:
    """Single run against the real corpus; reused across proofs 3 and 4."""
    store = tmp_path_factory.mktemp("store") / ".hash_store.json"
    return run_discover_extract(SOURCE_DIR, store)


# ── Proof 1: Idempotency ──────────────────────────────────────────────────────


def test_first_run_produces_new_status(tmp_path: Path) -> None:
    store = tmp_path / ".hash_store.json"
    results = run_discover_extract(SOURCE_DIR, store)
    assert all(r.status in ("new", "error") for r in results), (
        "First run must produce only 'new' (or 'error') statuses"
    )


def test_second_run_skips_all_succeeded(tmp_path: Path) -> None:
    store = tmp_path / ".hash_store.json"
    run1 = run_discover_extract(SOURCE_DIR, store)
    run2 = run_discover_extract(SOURCE_DIR, store)

    succeeded = {r.doc_id for r in run1 if r.status == "new"}
    skipped = {r.doc_id for r in run2 if r.status == "skipped"}
    assert succeeded == skipped, (
        f"Documents new in run1 must be skipped in run2.\n  Not skipped: {succeeded - skipped}"
    )


def test_second_run_has_no_new_or_updated(tmp_path: Path) -> None:
    store = tmp_path / ".hash_store.json"
    run_discover_extract(SOURCE_DIR, store)
    run2 = run_discover_extract(SOURCE_DIR, store)
    unexpected = [r for r in run2 if r.status in ("new", "updated")]
    assert not unexpected, f"Run 2 must not re-process unchanged docs: {unexpected}"


# ── Proof 2: Fault isolation ──────────────────────────────────────────────────


class _MockReader(SourceReader):
    """Injects an arbitrary list of SourceDocuments, bypassing the manifest."""

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


def test_corrupt_pdf_yields_error_status(tmp_path: Path) -> None:
    store = tmp_path / ".hash_store.json"
    reader = _MockReader([_corrupt_doc()])
    results = run_discover_extract(SOURCE_DIR, store, reader=reader)

    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].error is not None
    assert results[0].document is None


def test_corrupt_pdf_does_not_abort_run(tmp_path: Path) -> None:
    """Corrupt doc injected alongside real docs — all real docs still process."""
    real_docs = FileSystemReader(SOURCE_DIR).discover()
    mixed = [real_docs[0], _corrupt_doc()] + real_docs[1:]

    store = tmp_path / ".hash_store.json"
    results = run_discover_extract(SOURCE_DIR, store, reader=_MockReader(mixed))

    error_results = [r for r in results if r.status == "error"]
    success_results = [r for r in results if r.status in ("new", "updated")]

    assert len(error_results) == 1, "Exactly one error expected (the corrupt doc)"
    assert error_results[0].doc_id == "TEST-corrupt"
    assert len(success_results) == len(real_docs), (
        "All real documents must complete despite the corrupt doc"
    )


def test_corrupt_doc_not_added_to_hash_store(tmp_path: Path) -> None:
    """Failed documents must NOT be persisted — they are retried next run."""
    store = tmp_path / ".hash_store.json"
    reader = _MockReader([_corrupt_doc()])
    run_discover_extract(SOURCE_DIR, store, reader=reader)

    data = json.loads(store.read_text())
    assert "TEST-corrupt" not in data, "Corrupt doc must not be stored in hash store"


# ── Proof 3: Multi-page PDF with page markers ─────────────────────────────────


def test_clm1004_policy_three_pages(first_run: list[ExtractResult]) -> None:
    pol = next((r for r in first_run if r.doc_id == "POL-5504-policy"), None)
    assert pol is not None, "POL-5504-policy missing from results"
    assert pol.document is not None, f"POL-5504-policy has status={pol.status}, error={pol.error}"
    assert pol.document.page_count == 3


def test_clm1004_policy_page_markers_present(first_run: list[ExtractResult]) -> None:
    pol = next(r for r in first_run if r.doc_id == "POL-5504-policy")
    assert pol.document is not None
    text = pol.document.text
    assert "<!-- PAGE 1 -->" in text
    assert "<!-- PAGE 2 -->" in text
    assert "<!-- PAGE 3 -->" in text


def test_clm1004_policy_exclusion_2_3_present(first_run: list[ExtractResult]) -> None:
    """Exclusion 2.3 was silently dropped by the old 52-line truncation; must be present now."""
    pol = next(r for r in first_run if r.doc_id == "POL-5504-policy")
    assert pol.document is not None
    text = pol.document.text
    assert "2.3" in text, "Exclusion 2.3 reference missing from CLM-1004 policy"
    assert "wind-driven rain" in text.lower(), "'wind-driven rain' missing from CLM-1004 policy"


def test_all_policy_pdfs_have_page_markers(first_run: list[ExtractResult]) -> None:
    # Only CLM-1001 and CLM-1004 policies are PDF; CLM-1002/1003/1005 are TXT
    pdf_policy_ids = {"POL-3301-policy", "POL-5504-policy"}
    pdf_results = [r for r in first_run if r.doc_id in pdf_policy_ids]
    assert len(pdf_results) == 2, f"Expected 2 PDF policies, got {[r.doc_id for r in pdf_results]}"
    for r in pdf_results:
        assert r.document is not None
        assert "<!-- PAGE 1 -->" in r.document.text, f"{r.doc_id} missing <!-- PAGE 1 --> marker"
        assert r.document.page_count >= 1


# ── Proof 4: Fixed-width table survives normalisation ────────────────────────


def test_clm1003_pdf_estimate_table_preserved(first_run: list[ExtractResult]) -> None:
    """CLM-1003 estimate is a PDF with a fixed-width line-item table."""
    est = next((r for r in first_run if r.doc_id == "CLM-1003-estimate"), None)
    assert est is not None, "CLM-1003-estimate missing from results"
    assert est.document is not None, f"CLM-1003-estimate error: {est.error}"
    text = est.document.text
    assert "TOTAL" in text
    assert "$" in text
    assert "Unit Cost" in text or "Unit" in text
    # Separator lines must not be stripped
    assert any(c in text for c in ("─", "═", "=")), "Table separator lines were stripped"


def test_clm1001_html_estimate_table_preserved(first_run: list[ExtractResult]) -> None:
    """CLM-1001 estimate is HTML with a <table> of line items."""
    est = next((r for r in first_run if r.doc_id == "CLM-1001-estimate"), None)
    assert est is not None, "CLM-1001-estimate missing from results"
    assert est.document is not None
    text = est.document.text
    assert "TOTAL" in text
    assert "$" in text
    assert "CLM-1001" in text or "Claim Number" in text
    # Table cell content (columns: Item, Description, Qty, Unit Cost, Total)
    assert "Unit Cost" in text or "Description" in text


# ── General correctness ───────────────────────────────────────────────────────


def test_all_24_documents_processed(first_run: list[ExtractResult]) -> None:
    assert len(first_run) == 24, f"Expected 24, got {len(first_run)}"


def test_no_errors_on_valid_corpus(first_run: list[ExtractResult]) -> None:
    errors = [r for r in first_run if r.status == "error"]
    assert not errors, f"Unexpected extraction errors: {[(r.doc_id, r.error) for r in errors]}"


def test_tier1_metadata_on_all_documents(first_run: list[ExtractResult]) -> None:
    for r in first_run:
        if r.document is None:
            continue
        assert r.document.doc_id
        assert r.document.doc_type in ("policy", "endorsement", "claim_note", "claim_document")
        assert r.document.region in ("northeast", "southwest")
        assert r.document.assigned_adjuster


def test_loss_date_on_claim_scoped_artifacts(first_run: list[ExtractResult]) -> None:
    for r in first_run:
        if r.document is None or not r.document.claim_number:
            continue
        assert r.document.loss_date is not None, (
            f"{r.doc_id} has claim_number but missing loss_date"
        )


def test_content_hash_is_sha256_hex(first_run: list[ExtractResult]) -> None:
    for r in first_run:
        if r.document is None:
            continue
        assert len(r.document.content_hash) == 64, (
            f"{r.doc_id} has malformed content_hash: {r.document.content_hash!r}"
        )
        assert all(c in "0123456789abcdef" for c in r.document.content_hash)


def test_hash_store_written_with_all_doc_ids(tmp_path: Path) -> None:
    store = tmp_path / ".hash_store.json"
    run_discover_extract(SOURCE_DIR, store)
    data = json.loads(store.read_text())
    assert len(data) == 24
    for doc_id, h in data.items():
        assert len(h) == 64, f"{doc_id}: malformed hash"


def test_jsonl_notes_render_note_ids(first_run: list[ExtractResult]) -> None:
    note_results = [r for r in first_run if r.document and r.document.doc_type == "claim_note"]
    assert note_results, "No claim_note results found"
    for r in note_results:
        assert r.document is not None
        assert "NOTE-" in r.document.text, f"{r.doc_id} missing NOTE- label in extracted text"


def test_clm1005_fnol_only_no_notes(first_run: list[ExtractResult]) -> None:
    clm1005 = [r for r in first_run if r.document and r.document.claim_number == "CLM-1005"]
    doc_types = {r.document.doc_type for r in clm1005 if r.document}
    assert "claim_note" not in doc_types, "CLM-1005 must not have claim_note artifacts"
    assert "claim_document" in doc_types, "CLM-1005 must have its FNOL"
