"""Citation-checker tool (spec-5b): detects FABRICATED citations only.

ask.py's prompt (prompts/rag_v1.txt, rule 5) instructs the LLM to end every answer
with a SOURCES section in the format "[doc_id | p.<page> | <section>]" — but models
don't follow this exactly in practice (observed variants: "[SOURCE: doc_id | p.1
SECTION]", "[doc_id | p.1 §Section]"). The parser below is therefore tolerant: it
scans the whole answer text for bracketed segments and extracts anything shaped like
a corpus doc_id, rather than requiring an exact "SOURCES:" section or exact
separator characters.

SCOPE — read this before using the result:
This tool detects INVENTED citations: a doc_id the LLM named that was never in
retrieved_chunks at all. It does NOT detect grounded-in-the-wrong-source citations
— a doc_id that genuinely was retrieved but is the wrong claim's evidence (e.g. the
spec-4 q08 finding: CLM-1003 content, real and retrieved, cited in a CLM-1004
answer). has_fabricated_citations=False on q08-shaped input, because every citation
named there really was retrieved. KI-1 (KNOWN_ISSUES.md) is NOT covered by this
tool and stays deferred. Do not read a False result as "this answer is grounded."
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from claimcontext.retrieval.models import AskResult

_DOC_ID_PATTERN = re.compile(r"\b([A-Z]{2,6}-\d{3,5}(?:-[A-Za-z0-9]+)*)\b")
_BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")


class CitationCheckResult(BaseModel):
    claimed_doc_ids: list[str]  # parsed from the LLM's answer text
    retrieved_doc_ids: list[str]  # from AskResult.retrieved_chunks
    fabricated: list[str]  # claimed but never retrieved
    has_fabricated_citations: bool  # fabricated != [] — NOT a grounding guarantee


def _extract_claimed_doc_ids(answer: str) -> list[str]:
    claimed: set[str] = set()
    for bracketed in _BRACKET_PATTERN.findall(answer):
        for match in _DOC_ID_PATTERN.finditer(bracketed):
            claimed.add(match.group(1))
    return sorted(claimed)


def check_citations_raw(answer: str, retrieved_doc_ids: list[str]) -> CitationCheckResult:
    """Primitive-typed core: compare claimed vs. retrieved doc_ids directly.

    Split out from check_citations() so the MCP tool (spec-5b — the one tool
    exposed via MCP) can wrap this directly with simple, JSON-serializable
    arguments instead of requiring a full nested AskResult over the wire.
    """
    claimed = _extract_claimed_doc_ids(answer)
    retrieved = sorted(set(retrieved_doc_ids))
    fabricated = sorted(set(claimed) - set(retrieved))

    return CitationCheckResult(
        claimed_doc_ids=claimed,
        retrieved_doc_ids=retrieved,
        fabricated=fabricated,
        has_fabricated_citations=bool(fabricated),
    )


def check_citations(result: AskResult) -> CitationCheckResult:
    """Compare the LLM's self-reported sources against what was actually retrieved.

    Refused entries (empty retrieved_chunks, no real answer) trivially have no
    fabrication — nothing was claimed and nothing could have been fabricated.
    """
    retrieved_doc_ids = [chunk.doc_id for chunk in result.retrieved_chunks]
    return check_citations_raw(result.answer, retrieved_doc_ids)
