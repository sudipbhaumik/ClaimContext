"""CLI entry point: python -m claimcontext

Subcommands:
  (no subcommand)   — run the ingestion pipeline (spec-1a + spec-1b)
  ask "<query>"     — retrieve + LLM answer with citations (spec-2a)

Usage:
    python -m claimcontext [--source PATH] [--dry-run]
    python -m claimcontext ask "<query>" [--top-k N] [--dry-run]
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
log = logging.getLogger("claimcontext")


def _run_ingest(args: argparse.Namespace) -> int:
    from claimcontext.config import get_settings
    from claimcontext.ingestion.pipeline import run_chunk_embed_upsert, run_discover_extract
    from claimcontext.ingestion.report import build_report

    settings = get_settings()
    source_dir = args.source or settings.ingest_source_dir
    hash_store = settings.ingest_hash_store_path

    t0 = time.monotonic()

    log.info("discover+extract from %s", source_dir)
    results = run_discover_extract(source_dir, hash_store)

    if args.dry_run:
        from claimcontext.ingestion.chunker import chunk_document

        total_chunks = 0
        for r in results:
            if r.status in ("new", "updated") and r.document is not None:
                chunks = chunk_document(r.document, settings)
                total_chunks += len(chunks)
                log.info("[dry-run] %s %s → %d chunks", r.status, r.doc_id, len(chunks))
        elapsed = time.monotonic() - t0
        log.info("[dry-run] would upsert %d chunks in %.2fs", total_chunks, elapsed)
        return 0

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


def _run_ask(args: argparse.Namespace) -> int:
    from claimcontext.config import get_settings
    from claimcontext.retrieval.errors import IndexStalenessError, LLMError

    settings = get_settings()

    # ── Entitlement (spec-3) ──────────────────────────────────────────────────
    # Identity comes from --adjuster-id flag, not from the query text.
    # resolve_principal() raises AuthorizationError on unknown IDs — fail fast.
    principal = None
    if args.adjuster_id is not None:
        from claimcontext.auth.errors import AuthorizationError
        from claimcontext.auth.resolver import resolve_principal

        try:
            principal = resolve_principal(args.adjuster_id)
        except AuthorizationError as exc:
            log.error("auth failed: %s", exc)
            return 1
        log.info(
            "principal resolved: adjuster=%r region=%r", principal.adjuster_id, principal.region
        )

    from claimcontext.retrieval.hybrid_retriever import HybridRetriever
    from claimcontext.retrieval.retriever import Retriever

    retriever: HybridRetriever | Retriever
    if settings.retrieval_mode == "hybrid":
        retriever = HybridRetriever(settings)
    else:
        retriever = Retriever(settings)

    try:
        retriever.check_index_staleness()
    except IndexStalenessError as exc:
        log.error("index check failed: %s", exc)
        return 1

    if args.dry_run:
        # Retrieve only — skip reranker and LLM call
        top_k = args.top_k if args.top_k is not None else settings.top_k
        results = retriever.search(args.query, top_k=top_k)
        output = [r.model_dump() for r in results]
        print(json.dumps(output, indent=2))
        return 0

    from claimcontext.retrieval.ask import ask
    from claimcontext.retrieval.llm_client import LLMClient
    from claimcontext.retrieval.reranker import Reranker

    llm = LLMClient(settings)
    reranker = Reranker(settings)
    try:
        result = ask(
            query=args.query,
            retriever=retriever,
            llm=llm,
            settings=settings,
            reranker=reranker,
            principal=principal,
        )
    except LLMError as exc:
        log.error("LLM call failed: %s", exc)
        return 1

    print(json.dumps(result.model_dump(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claimcontext",
        description="ClaimContext — grounded, cited Q&A for insurance claim adjusters",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # ── ingest (default — no subcommand) ─────────────────────────────────────
    ingest_parser = subparsers.add_parser("ingest", help="Run the ingestion pipeline")
    ingest_parser.add_argument("--source", type=Path, help="Override ingest_source_dir")
    ingest_parser.add_argument("--dry-run", action="store_true")

    # ── ask ───────────────────────────────────────────────────────────────────
    ask_parser = subparsers.add_parser("ask", help="Ask a question (retrieve + LLM answer)")
    ask_parser.add_argument("query", help="The question to answer")
    ask_parser.add_argument("--top-k", type=int, default=None, help="Override top_k from config")
    ask_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Retrieve only — print chunks, skip LLM call",
    )
    ask_parser.add_argument(
        "--adjuster-id",
        default=None,
        help="Apply entitlement filter for this adjuster (e.g. ADJ-014). "
        "Identity comes from the mock auth resolver, not the query text.",
    )

    args = parser.parse_args(argv)

    # Backwards-compat: bare `python -m claimcontext` (no subcommand) runs ingest
    if args.subcommand is None or args.subcommand == "ingest":
        if args.subcommand is None:
            # Inject defaults for ingest when called without subcommand
            args.source = None
            args.dry_run = False
        return _run_ingest(args)
    elif args.subcommand == "ask":
        return _run_ask(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
