"""CLI entry point: python -m claimcontext.ingest

Usage:
    python -m claimcontext.ingest [--source PATH] [--dry-run]

--dry-run discovers, extracts, and chunks but does NOT load the embedding model
or contact Qdrant. Safe to run without the torch stack installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("claimcontext.ingest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ClaimContext ingestion pipeline")
    parser.add_argument("--source", type=Path, help="Override ingest_source_dir from config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk only — do not embed or upsert; safe without torch/Qdrant",
    )
    args = parser.parse_args(argv)

    from claimcontext.config import get_settings
    from claimcontext.ingestion.pipeline import run_chunk_embed_upsert, run_discover_extract
    from claimcontext.ingestion.report import build_report

    settings = get_settings()
    source_dir = args.source or settings.ingest_source_dir
    hash_store = settings.ingest_hash_store_path

    t0 = time.monotonic()

    # spec-1a: discover + extract
    log.info("discover+extract from %s", source_dir)
    results = run_discover_extract(source_dir, hash_store)

    if args.dry_run:
        # Chunk only — Embedder and QdrantWriter are never instantiated.
        from claimcontext.ingestion.chunker import chunk_document

        total_chunks = 0
        for r in results:
            if r.status in ("new", "updated") and r.document is not None:
                chunks = chunk_document(r.document, settings)
                total_chunks += len(chunks)
                log.info(
                    "[dry-run] %s %s → %d chunks",
                    r.status,
                    r.doc_id,
                    len(chunks),
                )
                for c in chunks:
                    log.debug(
                        "  chunk %d  page=%d  section=%r  %r...",
                        c.chunk_index,
                        c.page,
                        c.section,
                        c.text[:60],
                    )
        elapsed = time.monotonic() - t0
        log.info("[dry-run] would upsert %d chunks in %.2fs", total_chunks, elapsed)
        return 0

    # spec-1b: chunk → embed → delete → upsert
    # Embedder and QdrantWriter instantiated HERE, after dry-run check.
    from claimcontext.ingestion.embedder import Embedder
    from claimcontext.ingestion.qdrant_writer import QdrantWriter

    embedder = Embedder(settings.embedding_model, settings.embedding_batch_size)
    writer = QdrantWriter(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_dim=embedder.dim,
        timeout=settings.qdrant_timeout_seconds,
    )
    writer.ensure_collection()

    chunk_counts, embed_errors = run_chunk_embed_upsert(results, embedder, writer, settings)

    elapsed = time.monotonic() - t0
    report = build_report(results, chunk_counts, elapsed, embed_errors=embed_errors)

    print(json.dumps(report.model_dump(), indent=2))
    return 0 if report.failed == 0 and report.embed_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
