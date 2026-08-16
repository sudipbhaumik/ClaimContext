"""HTTP request/response schemas for the FastAPI serving layer (spec-7a).

These are a deliberately narrower shape than the internal AskResult they're
built from — see AskResponse for what's withheld and why.
"""

from __future__ import annotations

from pydantic import BaseModel

from claimcontext.retrieval.models import AskResult

# Single, shared, non-disclosing answer text for every refusal-shaped outcome
# (weak-context, cross-entitlement, Tier-3, escalation, unknown adjuster_id).
# AskResult.answer already carries THREE distinct internal message strings
# (_REFUSE_MESSAGE, _TIER3_REFUSE_MESSAGE, _ESCALATE_MESSAGE — see ask.py/
# graph.py) that are useful internally but would leak refusal-type information
# verbatim if mirrored into the HTTP response. See spec-7a Proof 2.
PUBLIC_REFUSAL_MESSAGE = (
    "This request could not be answered. Please consult the source documents "
    "directly or escalate to a supervisor."
)


class AskRequest(BaseModel):
    query: str
    adjuster_id: str


class CitationOut(BaseModel):
    doc_id: str
    page: int
    section: str
    score: float
    text_excerpt: str


class AskResponse(BaseModel):
    """Public shape of an /ask response.

    Withheld relative to AskResult: retrieved_chunks (internal retrieval
    detail, not for a client), llm_model/prompt_version (internal
    operational detail), adjuster_id (client already knows who they are).
    `answer` and `citations` are normalized to a single shared value across
    all refusal reasons — see to_ask_response().
    """

    answer: str
    citations: list[CitationOut]
    refused: bool


def to_ask_response(result: AskResult) -> AskResponse:
    """Build the public HTTP response from the internal AskResult.

    When refused, replaces the internal (refusal-type-distinguishing) answer
    text with one shared, non-disclosing message and drops citations — the
    §6B indistinguishability property extended to the HTTP boundary (spec-7a
    Proof 2). citations are otherwise passed through unchanged.
    """
    if result.refused:
        return AskResponse(answer=PUBLIC_REFUSAL_MESSAGE, citations=[], refused=True)
    return AskResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                doc_id=c.doc_id,
                page=c.page,
                section=c.section,
                score=c.score,
                text_excerpt=c.text_excerpt,
            )
            for c in result.citations
        ],
        refused=False,
    )
