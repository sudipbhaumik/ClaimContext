"""Format-specific text extractors.

PDF   — PyMuPDF (fitz). Primary choice over pypdf because it is layout-aware:
        it respects the reading order defined by the PDF content stream rather
        than naively concatenating character positions. <!-- PAGE n --> markers
        are inserted between pages so the chunker can populate the `page`
        Tier-2 field per chunk.

HTML  — stdlib html.parser. No external dependency; tables are converted to
        tab-separated rows so column content (item descriptions, dollar amounts)
        is preserved for retrieval.

TXT   — Plain UTF-8 read. No structural parsing; the normaliser handles cleanup.

JSONL — Structured path, distinct from the document path. Each JSON line is a
        claim-note DB row ({note_id, note_date, author, text}); it is rendered
        as a labelled text block rather than treated as a flat document. This
        mirrors how the DatabaseReader adapter would surface the same data.
"""

from __future__ import annotations

import json
import logging
from html.parser import HTMLParser
from pathlib import Path

log = logging.getLogger(__name__)


def extract_text(raw_bytes: bytes, file_path: Path) -> tuple[str, int]:
    """Dispatch extraction by file extension.

    Returns ``(text, page_count)``. page_count is 1 for non-PDF formats.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(raw_bytes)
    if suffix == ".html":
        return _extract_html(raw_bytes), 1
    if suffix == ".jsonl":
        return _extract_jsonl(raw_bytes), 1
    return _extract_txt(raw_bytes), 1


def _extract_pdf(raw_bytes: bytes) -> tuple[str, int]:
    import fitz  # type: ignore[import]

    doc = fitz.Document(stream=raw_bytes, filetype="pdf")
    n_pages: int = doc.page_count
    if n_pages == 0:
        doc.close()
        raise ValueError("PDF has no extractable pages (possibly corrupt or truncated)")

    parts: list[str] = []
    for i, page in enumerate(doc, start=1):
        page_text: str = page.get_text("text")
        parts.append(f"<!-- PAGE {i} -->\n{page_text}")
    doc.close()
    return "\n".join(parts), n_pages


def _extract_html(raw_bytes: bytes) -> str:
    text = raw_bytes.decode("utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(text)
    return parser.get_text()


def _extract_txt(raw_bytes: bytes) -> str:
    return raw_bytes.decode("utf-8", errors="replace")


def _extract_jsonl(raw_bytes: bytes) -> str:
    """Render each claim-note JSON line as a labelled text block.

    Output format::

        [NOTE-1001-01] 2026-02-10 — ADJ-014
        <note text>

        [NOTE-1001-02] 2026-02-10 — ADJ-014
        <note text>
    """
    raw = raw_bytes.decode("utf-8", errors="replace")
    blocks: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj: dict = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("skipping malformed JSONL line: %s", exc)
            continue
        note_id = obj.get("note_id", "")
        note_date = obj.get("note_date", "")
        author = obj.get("author", "")
        body = obj.get("text", "")
        header = f"[{note_id}] {note_date} — {author}" if note_id else f"{note_date} — {author}"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


class _HTMLTextExtractor(HTMLParser):
    """html.parser subclass: HTML → plain text with table rows as TSV lines.

    Tables are rendered as tab-separated rows so column structure (item #,
    description, qty, unit cost, total) survives into plain text. This matters
    for the chunker and BM25 retriever: dollar amounts and item names must stay
    on the same line as their row context.
    """

    _BLOCK_TAGS = frozenset(
        {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}
    )
    _SKIP_TAGS = frozenset({"style", "script", "head"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._row_cells: list[str] = []
        self._table_rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip = True
        elif tag == "table":
            self._table_rows = []
        elif tag == "tr":
            self._row_cells = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_parts = []
        elif tag == "br":
            (self._cell_parts if self._in_cell else self._parts).append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip = False
        elif tag == "table":
            for row in self._table_rows:
                self._parts.append("\t".join(cell.strip() for cell in row))
                self._parts.append("\n")
            self._table_rows = []
        elif tag == "tr":
            if self._row_cells:
                self._table_rows.append(self._row_cells[:])
        elif tag in ("td", "th"):
            self._in_cell = False
            self._row_cells.append("".join(self._cell_parts))
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        (self._cell_parts if self._in_cell else self._parts).append(data)

    def get_text(self) -> str:
        return "".join(self._parts)
