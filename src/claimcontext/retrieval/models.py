from __future__ import annotations

from pydantic import BaseModel


class Citation(BaseModel):
    """Source attribution for one retrieved chunk. Built from chunk payload metadata —
    never parsed from LLM output (LLM is not trusted to name its own sources)."""

    doc_id: str
    page: int
    section: str  # "" for unheaded chunks
    score: float  # cosine similarity from Qdrant
    text_excerpt: str  # chunk text[:200]


class RetrievalResult(BaseModel):
    """One scored chunk returned by Retriever.search(). Carries enough for
    spec-2b (RRF) and spec-2c (rerank) to re-score without re-fetching."""

    chunk_id: str
    doc_id: str
    doc_type: str
    policy_number: str | None
    claim_number: str | None
    page: int
    section: str
    score: float
    text: str
    embedding_model: str  # from payload — used by staleness guard
    chunker_version: str  # from payload


class AskResult(BaseModel):
    """Complete output of one ask() call."""

    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult]  # full top_k for 2b/2c to re-score
    llm_model: str
    prompt_version: str  # filename of prompt template used (§2A.3)
    refused: bool = False  # True when top rerank score < refuse_threshold (spec-2c)
    adjuster_id: str | None = None  # Principal.adjuster_id when entitlement was applied (spec-3)
