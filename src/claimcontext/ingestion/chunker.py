"""Structure-aware chunker for spec-1b.

Strategy (document types):
  Pass 1 — page cursor: strip <!-- PAGE n --> markers, track current page.
  Pass 2 — heading split: separator-aware heading detection divides text into
            candidate sections (label + lines).
  Pass 3 — table detection: annotate lines inside fixed-width table blocks so
            Pass 4 never splits mid-table.
  Pass 4 — token-bounded split with overlap: emit chunks; defer emit past table
            block boundaries; back-cursor chunk_overlap tokens for next chunk.

JSONL claim_note documents bypass all four passes: each [NOTE-xxx] block is
already atomic and becomes exactly one chunk regardless of token count.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import NamedTuple

import tiktoken

from claimcontext.config import Settings
from claimcontext.ingestion.models import Chunk, ExtractedDocument

log = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")

# ── Regex constants ────────────────────────────────────────────────────────────

# Separator line: all one repeated char (=, ─, ?, ─) with length >= 8
_RE_SEPARATOR = re.compile(r"^[\=\─\?\─]{8,}\s*$")

# Table line: contains money pattern OR is a separator OR has 3+ column gaps
_RE_MONEY = re.compile(r"\$\s*[\d,]+")
_RE_COL_GAPS = re.compile(r"  {3,}")  # 3+ spaces = column gap

# JSONL note block opener
_RE_NOTE_OPENER = re.compile(r"^\[NOTE-")

# Page marker
_RE_PAGE_MARKER = re.compile(r"^<!--\s*PAGE\s+(\d+)\s*-->")

# Long digit run (5+) — helps exclude policy numbers / dates from headings
_RE_LONG_DIGITS = re.compile(r"\d{5,}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _tok(text: str) -> int:
    return len(_ENCODING.encode(text))


def _is_separator(line: str) -> bool:
    return bool(_RE_SEPARATOR.match(line.strip()))


def _is_table_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _is_separator(line):
        return True
    if _RE_MONEY.search(s):
        return True
    return bool(_RE_COL_GAPS.search(s))


def _is_heading(line: str, prev_is_sep_or_blank: bool, next_is_sep_or_blank: bool) -> bool:
    s = line.strip()
    if not s or len(s) < 3:
        return False
    # 60-char ceiling: longest real heading in corpus is 56c; boilerplate prose runs longer.
    if len(s) > 60:
        return False
    if "$" in s or "@" in s:
        return False
    if _is_separator(line):
        return False
    if _RE_LONG_DIGITS.search(s):
        return False
    # Page footer lines — normaliser regex misses trailing text after "Page N of M".
    if s.startswith("Page ") and " of " in s:
        return False
    return prev_is_sep_or_blank and next_is_sep_or_blank


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    # Qdrant requires point IDs to be unsigned integers or UUIDs.
    # Take the first 32 hex chars of SHA-256 and format as UUID (8-4-4-4-12).
    h = hashlib.sha256(f"{doc_id}|{chunk_index}".encode()).hexdigest()[:32]
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ── Pass 1+2: page-cursor scan + heading split ────────────────────────────────


class _Section(NamedTuple):
    label: str  # heading text; "" for pre-heading content
    page: int  # page number at start of section
    lines: list[str]


def _split_into_sections(text: str) -> list[_Section]:
    """Scan lines, track page cursor, split at heading boundaries."""
    raw_lines = text.splitlines()
    n = len(raw_lines)

    current_page = 1
    current_label = ""
    buffer: list[str] = []
    sections: list[_Section] = []

    def _flush() -> None:
        nonlocal buffer
        # Strip leading/trailing blank lines from the buffer
        trimmed = buffer[:]
        while trimmed and not trimmed[0].strip():
            trimmed.pop(0)
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
        if trimmed:
            sections.append(_Section(label=current_label, page=current_page, lines=trimmed))
        buffer = []

    for i, raw in enumerate(raw_lines):
        # Pass 1: page marker
        m = _RE_PAGE_MARKER.match(raw.strip())
        if m:
            current_page = int(m.group(1))
            continue  # drop marker from text

        # Separator lines: structural punctuation, not content — but we need them
        # for heading context, so peek at neighbours without emitting them.
        if _is_separator(raw):
            continue  # drop separator; heading detection uses prev/next context

        # Heading detection: look at what came before and after in raw_lines
        # (using raw_lines index, not buffer) so separator context is visible.
        prev_raw = raw_lines[i - 1] if i > 0 else ""
        next_raw = raw_lines[i + 1] if i < n - 1 else ""
        prev_is_sep_or_blank = not prev_raw.strip() or _is_separator(prev_raw)
        next_is_sep_or_blank = not next_raw.strip() or _is_separator(next_raw)

        if _is_heading(raw, prev_is_sep_or_blank, next_is_sep_or_blank):
            _flush()
            current_label = raw.strip()
            continue

        buffer.append(raw)

    _flush()
    return sections


# ── Pass 3: table-line annotation ─────────────────────────────────────────────


def _annotate_tables(lines: list[str]) -> list[tuple[str, bool]]:
    """Return (line, in_table) pairs. A table block starts at the first table
    line and ends at the first blank line after >=2 consecutive table lines."""
    annotated: list[tuple[str, bool]] = []
    in_table = False
    consecutive_table = 0

    for line in lines:
        if _is_table_line(line):
            consecutive_table += 1
            in_table = True
            annotated.append((line, True))
        elif not line.strip():
            if in_table and consecutive_table >= 2:
                # End of table block on blank line
                annotated.append((line, True))  # include the closing blank in the unit
                in_table = False
                consecutive_table = 0
            else:
                in_table = False
                consecutive_table = 0
                annotated.append((line, False))
        else:
            consecutive_table = 0
            annotated.append((line, in_table))

    return annotated


# ── Pass 4: token-bounded split with overlap ──────────────────────────────────


def _split_section(
    annotated: list[tuple[str, bool]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split annotated lines into token-bounded chunks, respecting table blocks."""
    if not annotated:
        return []

    # Pre-compute per-line token counts so we can back-cursor cheaply.
    line_tokens = [_tok(line) for line, _ in annotated]

    chunks: list[str] = []
    buf_start = 0  # index into annotated

    def _emit(end: int) -> None:
        """Emit annotated[buf_start:end] as a chunk."""
        nonlocal buf_start
        text = "\n".join(line for line, _ in annotated[buf_start:end]).strip()
        if text:
            chunks.append(text)
        # Back-cursor: find a new buf_start that carries overlap tokens forward.
        overlap_tokens = 0
        new_start = end
        for j in range(end - 1, buf_start - 1, -1):
            overlap_tokens += line_tokens[j]
            if overlap_tokens >= chunk_overlap:
                new_start = j
                break
        else:
            new_start = buf_start  # overlap >= whole chunk: keep nothing new
        buf_start = new_start

    running = 0
    i = buf_start
    while i < len(annotated):
        line, in_table = annotated[i]
        running += line_tokens[i]

        if running > chunk_size:
            if in_table:
                # Inside a table: continue until table mode ends
                while i < len(annotated) and annotated[i][1]:
                    i += 1
                tok_count = sum(line_tokens[buf_start:i])
                if tok_count > chunk_size:
                    log.warning(
                        "oversized table chunk: %d tokens (chunk_size=%d)",
                        tok_count,
                        chunk_size,
                    )
                _emit(i)
                running = sum(line_tokens[buf_start:i])
            else:
                _emit(i)
                running = sum(line_tokens[buf_start:i])
        else:
            i += 1

    # Emit any remaining content
    if buf_start < len(annotated):
        _emit(len(annotated))

    return chunks


# ── JSONL note chunking ───────────────────────────────────────────────────────


def _chunk_notes(text: str) -> list[str]:
    """Each [NOTE-xxx] block → one chunk. No token splitting."""
    blocks: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        if _RE_NOTE_OPENER.match(line) and current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current = [line]
        else:
            current.append(line)

    if current:
        block = "\n".join(current).strip()
        if block:
            blocks.append(block)

    return blocks


# ── Public entry point ────────────────────────────────────────────────────────


def chunk_document(doc: ExtractedDocument, settings: Settings) -> list[Chunk]:
    """Chunk a normalised document into indexable units.

    Returns an empty list for documents with no extractable content.
    Each Chunk carries full Tier-1 + Tier-2 metadata; section is "" for
    unstructured doc types (claim_note, unheaded correspondence).
    """
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    raw_texts: list[tuple[str, int, str]] = []  # (text, page, section)

    if doc.doc_type == "claim_note":
        for block in _chunk_notes(doc.text):
            raw_texts.append((block, 1, ""))
    else:
        sections = _split_into_sections(doc.text)
        for sec in sections:
            annotated = _annotate_tables(sec.lines)
            for chunk_text in _split_section(annotated, chunk_size, chunk_overlap):
                raw_texts.append((chunk_text, sec.page, sec.label))

    chunks: list[Chunk] = []
    for idx, (text, page, section) in enumerate(raw_texts):
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc.doc_id, idx),
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                policy_number=doc.policy_number,
                claim_number=doc.claim_number,
                region=doc.region,
                assigned_adjuster=doc.assigned_adjuster,
                page=page,
                section=section,
                effective_date=doc.effective_date,
                expiry_date=doc.expiry_date,
                loss_date=doc.loss_date,
                lob=doc.lob,
                version=doc.version,
                embedding_model=settings.embedding_model,
                chunker_version=settings.chunker_version,
                chunk_index=idx,
                text=text,
            )
        )

    if not chunks:
        log.warning("chunk_document produced 0 chunks for %s", doc.doc_id)

    return chunks
