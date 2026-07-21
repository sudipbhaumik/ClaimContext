"""Ingestion pipeline: discover → extract → normalise → hash (spec-1a).

Public entry point: ``run_discover_extract``.
Spec-1b calls this and adds: chunk → embed → upsert → run report.

Change detection uses SHA-256 of raw bytes, not file mtime.

Why raw bytes, not mtime:
    File modification times are not portable — copying files, deploying to a
    new host, or mounting a network share can reset mtime without changing
    content. A database row has no mtime at all. SHA-256(raw_bytes) is
    content-addressed: the same bytes always produce the same hash regardless
    of how they arrived, so FileSystemReader and DatabaseReader can share the
    same change-detection logic unchanged.

Why raw bytes, not SHA-256(normalised_text):
    Hashing raw bytes means a bug in the normaliser cannot silently suppress a
    re-ingest. If the normaliser changes (even to fix a bug), the raw hash is
    unchanged and the document appears as "skipped" — which is wrong. Hashing
    raw bytes catches all source changes and forces a re-extract on any
    normaliser update (by incrementing chunker_version in config, spec-1b).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

from .extractor import extract_text
from .models import ExtractedDocument, ExtractResult, SourceDocument
from .normaliser import normalise
from .reader import FileSystemReader, SourceReader

log = logging.getLogger(__name__)


# ── Hash store ────────────────────────────────────────────────────────────────


def load_hash_store(path: Path) -> dict[str, str]:
    """Load ``{doc_id: sha256_hex}`` from disk. Returns ``{}`` if absent."""
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def save_hash_store(store: dict[str, str], path: Path) -> None:
    """Persist ``{doc_id: sha256_hex}`` to disk, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")


def _content_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


# ── Per-document processing ───────────────────────────────────────────────────


def _process(src: SourceDocument) -> ExtractedDocument:
    raw_text, page_count = extract_text(src.raw_bytes, src.file_path)
    text = normalise(raw_text)
    return ExtractedDocument(
        doc_id=src.doc_id,
        doc_type=src.doc_type,
        policy_number=src.policy_number,
        claim_number=src.claim_number,
        region=src.region,
        assigned_adjuster=src.assigned_adjuster,
        effective_date=src.effective_date,
        expiry_date=src.expiry_date,
        loss_date=src.loss_date,
        lob=src.lob,
        version=src.version,
        text=text,
        page_count=page_count,
        content_hash=_content_hash(src.raw_bytes),
    )


# ── Public entry point ────────────────────────────────────────────────────────


def run_discover_extract(
    source_dir: Path,
    hash_store_path: Path,
    reader: SourceReader | None = None,
) -> list[ExtractResult]:
    """Discover all documents and extract text for new/changed ones.

    Returns one ``ExtractResult`` per document. Never raises per-document
    errors — failures are captured as ``ExtractResult(status='error')``.

    Args:
        source_dir:      Root of the corpus (contains ``manifest.json``).
        hash_store_path: Path to the persistent ``{doc_id: hash}`` store.
        reader:          Optional ``SourceReader`` override; defaults to
                         ``FileSystemReader(source_dir)``. Pass a mock reader
                         in tests to inject synthetic or corrupt documents.
    """
    if reader is None:
        reader = FileSystemReader(source_dir)

    hash_store = load_hash_store(hash_store_path)
    source_docs = reader.discover()

    results: list[ExtractResult] = []
    updated_store = dict(hash_store)

    for src in source_docs:
        current_hash = _content_hash(src.raw_bytes)
        stored_hash = hash_store.get(src.doc_id)

        if stored_hash == current_hash:
            log.debug("skipping unchanged %s", src.doc_id)
            results.append(ExtractResult(doc_id=src.doc_id, status="skipped"))
            continue

        status: Literal["new", "updated"] = "updated" if stored_hash is not None else "new"
        try:
            doc = _process(src)
        except Exception as exc:
            log.error("extraction failed for %s: %s", src.doc_id, exc, exc_info=True)
            results.append(ExtractResult(doc_id=src.doc_id, status="error", error=str(exc)))
            # Do NOT update the hash store — failed docs are retried next run
            continue

        updated_store[src.doc_id] = current_hash
        results.append(ExtractResult(doc_id=src.doc_id, status=status, document=doc))
        log.info("%s %s (%d pages)", status, src.doc_id, doc.page_count)

    save_hash_store(updated_store, hash_store_path)

    n_new = sum(1 for r in results if r.status == "new")
    n_updated = sum(1 for r in results if r.status == "updated")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_errors = sum(1 for r in results if r.status == "error")
    log.info(
        "discover+extract: %d new, %d updated, %d skipped, %d errors",
        n_new,
        n_updated,
        n_skipped,
        n_errors,
    )
    return results
