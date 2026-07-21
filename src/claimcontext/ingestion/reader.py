from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from .models import SourceDocument

log = logging.getLogger(__name__)


class SourceReader(ABC):
    """Port: abstract source of SourceDocuments.

    All discover/extract work sits behind this interface so FileSystemReader,
    DatabaseReader, and EventAdapter are interchangeable without touching the
    pipeline. Change detection is content-hash based (not file-mtime), so the
    same logic works identically for a file, a DB row, or an event payload.
    """

    @abstractmethod
    def discover(self) -> list[SourceDocument]:
        """Return one SourceDocument per artifact to be ingested."""
        ...


class FileSystemReader(SourceReader):
    """Adapter: reads manifest.json and opens files from source_dir.

    Production adapters (NOT built in spec-1a):

    DatabaseReader — claim notes live in the claims management DB. The JSONL
        corpus files are DB-row-shaped so this adapter is near-trivial:
        ``SELECT * FROM claim_notes WHERE updated_at > last_run`` yields one
        SourceDocument per row. Content-hash change detection works identically
        (no mtime needed).

    EventAdapter — CDC / event-triggered path. Receives new-claim events,
        fetches bytes from the source system, yields SourceDocument on demand.
        Backpressure and DLQ shape documented in design doc §9.9.
    """

    def __init__(self, source_dir: Path) -> None:
        self._source_dir = source_dir

    def discover(self) -> list[SourceDocument]:
        manifest_path = self._source_dir / "manifest.json"
        entries: list[dict] = json.loads(manifest_path.read_text(encoding="utf-8"))
        docs: list[SourceDocument] = []
        for entry in entries:
            file_path = self._source_dir / entry["file"]
            raw_bytes = file_path.read_bytes()
            docs.append(
                SourceDocument(
                    doc_id=entry["doc_id"],
                    doc_type=entry["doc_type"],
                    policy_number=entry.get("policy_number"),
                    claim_number=entry.get("claim_number"),
                    effective_date=entry.get("effective_date"),
                    expiry_date=entry.get("expiry_date"),
                    loss_date=entry.get("loss_date"),
                    region=entry["region"],
                    assigned_adjuster=entry["assigned_adjuster"],
                    lob=entry["lob"],
                    version=entry.get("version", "v1"),
                    file_path=file_path,
                    raw_bytes=raw_bytes,
                )
            )
        log.info("discovered %d documents from %s", len(docs), self._source_dir)
        return docs
