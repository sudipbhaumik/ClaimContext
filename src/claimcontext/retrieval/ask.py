"""ask() — the end-to-end RAG path: retrieve → rerank → refuse gate → LLM.

Build order (authoring discipline):
  1. Citation construction — pure data-shaping from chunk payload metadata.
     No LLM involved; crisp and fully testable from retrieval results alone.
  2. Context assembly — format retrieved chunks into source blocks that the LLM
     receives as reference text. Delimited and labeled so the model treats them
     as data, not instructions (§6B injection-awareness).
  3. Rerank — cross-encoder re-scores the fused candidates, replacing RRF scores
     with a joint query-passage relevance signal.
  4. Refuse gate — if the top reranked score is below refuse_threshold, refuse.
     The refusal message must not disclose whether records exist (§6B).
  5. LLM call — delegated to LLMClient.complete(); ask() assembles around it.

Citations are sourced from retrieved chunk metadata, NOT parsed from LLM output.
The LLM is instructed to produce a SOURCES list as a grounding signal for the
human reader, but we do not trust it for citation accuracy — the retrieval layer
already knows exactly which chunks were surfaced.
"""

from __future__ import annotations

import logging
from pathlib import Path

from claimcontext.config import Settings
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.models import AskResult, Citation, RetrievalResult
from claimcontext.retrieval.reranker import Reranker
from claimcontext.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

_EXCERPT_LEN = 200
_PROMPT_VERSION = "rag_v1.txt"

# §6B: the refusal message must not disclose whether records exist.
# "I don't have enough relevant information" is true and non-disclosing — it does not
# say "nothing was found" (which reveals corpus contents to probing queries).
_REFUSE_MESSAGE = (
    "I don't have enough relevant information in the available documents to answer "
    "this question reliably. Please consult the source documents directly or escalate "
    "to a supervisor."
)


# ── Step 1: Citation construction ─────────────────────────────────────────────
# Pure data-shaping from chunk payload metadata. No LLM, no I/O — just picking
# fields off RetrievalResult and truncating text. Testable in complete isolation.


def _build_citations(results: list[RetrievalResult]) -> list[Citation]:
    return [
        Citation(
            doc_id=r.doc_id,
            page=r.page,
            section=r.section,
            score=r.score,
            text_excerpt=r.text[:_EXCERPT_LEN],
        )
        for r in results
    ]


# ── Step 2: Context assembly ───────────────────────────────────────────────────
# Format retrieved chunks into labeled source blocks. Each block is clearly
# delimited so the LLM can see where one source ends and the next begins.
# The prompt template (rag_v1.txt) explicitly instructs the model to treat these
# blocks as reference material only — not as instructions (§6B injection-awareness).


def _assemble_context(results: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for r in results:
        section_label = f" §{r.section}" if r.section else ""
        header = f"[SOURCE: {r.doc_id} | p.{r.page}{section_label}]"
        blocks.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(blocks)


def _load_prompt(prompts_dir: str) -> str:
    path = Path(prompts_dir) / _PROMPT_VERSION
    return path.read_text(encoding="utf-8")


# ── Step 3–5: ask() — retrieve → rerank → refuse → context → LLM → AskResult ─


def ask(
    query: str,
    retriever: Retriever | HybridRetriever,
    llm: LLMClient,
    settings: Settings,
    reranker: Reranker | None = None,
) -> AskResult:
    """Retrieve top_k chunks, rerank, apply refuse gate, call LLM, return cited answer.

    reranker is optional — if None, the pipeline skips reranking and refuse gate
    (useful for testing or when rerank is disabled). When provided, it re-scores
    the fused candidate list and filters to rerank_top_n before the LLM call.

    Citations are built from the final reranked (or raw) results — not from LLM output.
    """
    # ── Step 3: retrieve ──────────────────────────────────────────────────────
    # Fetch rrf_fetch_k candidates when reranking, otherwise top_k.
    fetch_k = min(settings.top_k * 3, 30) if reranker is not None else settings.top_k
    results = retriever.search(query, top_k=fetch_k)

    log.info(
        "ask: query=%r fetched=%d reranker=%s",
        query[:80],
        len(results),
        reranker is not None,
    )

    # ── Step 4: rerank + refuse gate ─────────────────────────────────────────
    if reranker is not None:
        reranked = reranker.rerank(query, results)
        top_score = reranked[0].score if reranked else -1.0

        log.info(
            "rerank: top_score=%.4f threshold=%.4f refused=%s",
            top_score,
            settings.refuse_threshold,
            top_score < settings.refuse_threshold,
        )

        if top_score < settings.refuse_threshold:
            return AskResult(
                query=query,
                answer=_REFUSE_MESSAGE,
                citations=[],
                retrieved_chunks=reranked,
                llm_model=settings.llm_model,
                prompt_version=_PROMPT_VERSION,
                refused=True,
            )

        # Keep only rerank_top_n for the LLM context
        results = reranked[: settings.rerank_top_n]

    # ── Step 5: LLM call ─────────────────────────────────────────────────────
    citations = _build_citations(results)
    context = _assemble_context(results)

    system_prompt = _load_prompt(settings.prompts_dir)
    user_message = f"SOURCES:\n\n{context}\n\n---\n\nQUESTION: {query}"

    log.info(
        "llm: query=%r top_k=%d context_chars=%d",
        query[:80],
        len(results),
        len(context),
    )

    answer = llm.complete(system=system_prompt, user=user_message)

    return AskResult(
        query=query,
        answer=answer,
        citations=citations,
        retrieved_chunks=results,
        llm_model=settings.llm_model,
        prompt_version=_PROMPT_VERSION,
    )
