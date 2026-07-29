"""Run-report builder for spec-1b."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from claimcontext.ingestion.models import ExtractResult, IngestReport


def build_report(
    results: list[ExtractResult],
    chunk_counts: dict[str, int],
    elapsed_seconds: float,
    embed_errors: int = 0,
) -> IngestReport:
    """Build an IngestReport from spec-1a results and spec-1b chunk counts.

    failures is sourced exclusively from spec-1a ExtractResult errors.
    embed_errors is a separate counter for chunk/embed failures in spec-1b.
    """
    failures = [
        {"doc_id": r.doc_id, "error": r.error or "unknown"} for r in results if r.status == "error"
    ]
    ingested_docs = sum(1 for r in results if r.status in ("new", "updated"))
    ingested_chunks = sum(chunk_counts.get(r.doc_id, 0) for r in results)

    return IngestReport(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=round(elapsed_seconds, 3),
        discovered=len(results),
        ingested_docs=ingested_docs,
        ingested_chunks=ingested_chunks,
        skipped=sum(1 for r in results if r.status == "skipped"),
        failed=len(failures),
        failures=failures,
        embed_errors=embed_errors,
    )
