"""ask() — the minimal end-to-end RAG path for spec-2a.

Build order (authoring discipline):
  1. Citation construction — pure data-shaping from chunk payload metadata.
     No LLM involved; crisp and fully testable from retrieval results alone.
  2. Context assembly — format retrieved chunks into source blocks that the LLM
     receives as reference text. Delimited and labeled so the model treats them
     as data, not instructions (§6B injection-awareness).
  3. LLM call — delegated to LLMClient.complete(); ask() assembles around it.

Citations are sourced from retrieved chunk metadata, NOT parsed from LLM output.
The LLM is instructed to produce a SOURCES list as a grounding signal for the
human reader, but we do not trust it for citation accuracy — the retrieval layer
already knows exactly which chunks were surfaced.
"""

from __future__ import annotations

import logging
from pathlib import Path

from claimcontext.config import Settings
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.models import AskResult, Citation, RetrievalResult
from claimcontext.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

_EXCERPT_LEN = 200
_PROMPT_VERSION = "rag_v1.txt"


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


# ── Step 3: ask() — compose retrieve → context → LLM → AskResult ─────────────


def ask(
    query: str,
    retriever: Retriever,
    llm: LLMClient,
    settings: Settings,
) -> AskResult:
    """Retrieve top_k chunks, assemble context, call LLM, return cited answer.

    Citations are built from retrieval metadata before the LLM call —
    they do not depend on what the LLM says about its own sources.
    """
    results = retriever.search(query, top_k=settings.top_k)

    citations = _build_citations(results)
    context = _assemble_context(results)

    system_prompt = _load_prompt(settings.prompts_dir)
    user_message = f"SOURCES:\n\n{context}\n\n---\n\nQUESTION: {query}"

    log.info(
        "ask: query=%r top_k=%d context_chars=%d",
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
