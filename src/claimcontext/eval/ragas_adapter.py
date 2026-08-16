"""RAGAS adapter for spec-4 eval harness.

Scores passing ANSWER entries directly against each RAGAS "collections" metric's
async .ascore(...) and writes per-entry scores back into EvalResult objects.

RAGAS 0.4.3 API (verified against installed package):
  - ragas.metrics.collections.* (ContextPrecision, ContextRecall, Faithfulness,
    AnswerRelevancy) are a "modern component architecture" — they subclass
    SimpleBaseMetric, NOT the legacy ragas.metrics.base.Metric. The legacy
    ragas.evaluate()/EvaluationDataset path hard-checks isinstance(m, Metric) and
    raises TypeError("All metrics must be initialised metric objects...") for any
    collections metric. There is no dataset-batch entry point for this metric
    family in 0.4.3 — each metric is scored per-sample via its own async ascore():
      ContextPrecision.ascore(user_input, reference, retrieved_contexts)
      ContextRecall.ascore(user_input, retrieved_contexts, reference)
      Faithfulness.ascore(user_input, response, retrieved_contexts)
      AnswerRelevancy.ascore(user_input, response)
    Each returns a MetricResult with a `.value: float` attribute.
  - Metrics require InstructorBaseRagasLLM: use InstructorLLM(client, model, provider)
  - AnswerRelevancy also requires HuggingFaceEmbeddings(model=...)

Judge LLM decision:
  Answer LLM: Ollama/llama3.2 (local).
  Judge LLM:  Default: Ollama/mistral (local, different family — no API key needed).
              Alternatives: OpenAI GPT-4o (set eval_ragas_llm_provider=openai + OPENAI_API_KEY)
              or Anthropic (eval_ragas_llm_provider=anthropic + ANTHROPIC_API_KEY).
  Rationale:  self-preference bias is systematic — same-family judge inflates
              faithfulness and relevance for its own model's outputs. Using mistral
              as judge (vs llama3.2 as answer LLM) preserves the different-family
              property locally. If both are llama3.2, scores are self-assessments —
              directional but inflated. The scorecard bias warning prints regardless.

Dependency fix:
  ragas 0.4.3 imports langchain_community.chat_models.vertexai.ChatVertexAI which was
  removed from langchain-community in recent versions. We inject a stub into sys.modules
  before importing ragas. The stub is a no-op type; the isinstance() check in
  ragas.llms.base that references it will never match — no functional impact.
"""

from __future__ import annotations

import logging
import sys
import types

log = logging.getLogger(__name__)

# ── Vertexai stub (must run before any ragas import) ──────────────────────────
_VERTEXAI_MODULE = "langchain_community.chat_models.vertexai"
if _VERTEXAI_MODULE not in sys.modules:
    _stub = types.ModuleType(_VERTEXAI_MODULE)
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
    sys.modules[_VERTEXAI_MODULE] = _stub

import asyncio  # noqa: E402

import instructor  # noqa: E402
from ragas.embeddings import HuggingFaceEmbeddings  # noqa: E402
from ragas.llms import InstructorLLM  # noqa: E402
from ragas.llms.base import InstructorModelArgs  # noqa: E402
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness  # noqa: E402
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy  # noqa: E402

from claimcontext.config import Settings  # noqa: E402
from claimcontext.eval.schema import EvalReport, EvalResult, ExpectedBehavior  # noqa: E402


def _build_judge_llm(settings: Settings) -> InstructorLLM:
    """Build the RAGAS judge LLM. Use a different family from the answer LLM.

    Must use the *async* client variant (AsyncOpenAI / AsyncAnthropic) — scoring
    runs through InstructorLLM.agenerate(), and InstructorLLM raises TypeError at
    call time ("Cannot use agenerate() with a synchronous client") if handed a
    sync client. instructor.from_openai()/from_anthropic() auto-detect sync vs
    async from the client instance and return the matching Instructor variant.
    """
    provider = settings.eval_ragas_llm_provider
    model = settings.eval_ragas_llm_model

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible API — instructor.from_openai() wraps it.
        # Use a different model family from the answer LLM (e.g. mistral vs llama3.2)
        # to preserve the different-family property without a cloud API key.
        # Mode.JSON, not the default Mode.TOOLS: mistral via Ollama's OpenAI-compat
        # endpoint returns plain JSON content rather than a proper tool_call, so
        # instructor's TOOLS-mode parser finds no tool_calls and errors. JSON mode
        # parses the response content directly instead of expecting a function call.
        from openai import AsyncOpenAI

        base_url = settings.eval_ragas_llm_base_url
        ollama_client = AsyncOpenAI(base_url=base_url, api_key="ollama")
        client = instructor.from_openai(ollama_client, mode=instructor.Mode.JSON)
        return InstructorLLM(client=client, model=model, provider="openai")

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("eval_ragas_llm_provider='openai' requires OPENAI_API_KEY.")
        from openai import AsyncOpenAI

        client = instructor.from_openai(AsyncOpenAI(api_key=settings.openai_api_key))
        # max_tokens=4096, not InstructorModelArgs's default 1024: RAGAS's own
        # docstring warns GPT-5/o-series reasoning models need 4096+ — reasoning
        # tokens are drawn from the same budget as the visible output, so the
        # default silently truncates before the structured JSON response is ever
        # emitted (observed: IncompleteOutputException on every call at 1024).
        return InstructorLLM(
            client=client,
            model=model,
            provider="openai",
            model_args=InstructorModelArgs(max_tokens=4096),
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("eval_ragas_llm_provider='anthropic' requires ANTHROPIC_API_KEY.")
        import anthropic

        client = instructor.from_anthropic(
            anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        )
        return InstructorLLM(client=client, model=model, provider="anthropic")

    raise ValueError(f"Unknown eval_ragas_llm_provider: {provider!r}")


def ragas_judge_available(settings: Settings) -> bool:
    """Return True if a RAGAS judge LLM is configured and reachable.

    - ollama: always considered available (Ollama daemon checked at call time)
    - openai: requires OPENAI_API_KEY
    - anthropic: requires ANTHROPIC_API_KEY
    """
    p = settings.eval_ragas_llm_provider
    if p == "ollama":
        return True
    if p == "openai":
        return bool(settings.openai_api_key)
    if p == "anthropic":
        return bool(settings.anthropic_api_key)
    return False


def _build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    # HuggingFaceEmbeddings (capital F) is the concrete class; HuggingfaceEmbeddings
    # (lowercase f) is abstract and cannot be instantiated. Argument is "model=", not
    # "model_name=" — verified against ragas 0.4.3 source.
    return HuggingFaceEmbeddings(model=settings.eval_ragas_embed_model)


def score_with_ragas(report: EvalReport, settings: Settings) -> EvalReport:
    """Populate RAGAS metric scores on all passing ANSWER entries.

    Filters to entries where expected_behavior=ANSWER, passed=True, and
    ground_truth_answer is set (the runner populates this). Returns the report
    with context_precision, context_recall, faithfulness, answer_relevance filled.
    """
    scoreable: list[tuple[int, EvalResult]] = [
        (i, r)
        for i, r in enumerate(report.entries)
        if r.expected_behavior == ExpectedBehavior.ANSWER
        and r.passed
        and r.ground_truth_answer is not None
    ]

    print(
        f"ragas_adapter.py, score_with_ragas, filter scoreable entries executing : "
        f"{len(scoreable)}/{len(report.entries)} entries are ANSWER+passed+has ground_truth"
    )

    if not scoreable:
        log.info("no scoreable ANSWER entries — skipping RAGAS evaluation")
        print(
            "ragas_adapter.py, score_with_ragas, filter scoreable entries executing : "
            "none — skipping"
        )
        return report

    log.info(
        "RAGAS: scoring %d entries (judge=%s/%s embed=%s)",
        len(scoreable),
        settings.eval_ragas_llm_provider,
        settings.eval_ragas_llm_model,
        settings.eval_ragas_embed_model,
    )

    print(
        f"ragas_adapter.py, _build_judge_llm, build judge LLM client executing : "
        f"provider={settings.eval_ragas_llm_provider} model={settings.eval_ragas_llm_model}"
    )
    judge_llm = _build_judge_llm(settings)
    print("ragas_adapter.py, _build_judge_llm, build judge LLM client executing : judge_llm ready")

    print(
        f"ragas_adapter.py, _build_embeddings, build embeddings client executing : "
        f"model={settings.eval_ragas_embed_model}"
    )
    embeddings = _build_embeddings(settings)
    print(
        "ragas_adapter.py, _build_embeddings, build embeddings client executing : embeddings ready"
    )

    cp_m = ContextPrecision(llm=judge_llm)
    cr_m = ContextRecall(llm=judge_llm)
    f_m = Faithfulness(llm=judge_llm)
    # strictness=1, not the RAGAS default of 3: default generates 3 paraphrased
    # questions per answer (3 judge calls) to average out generation noise. On a
    # local judge model this triples answer_relevancy's cost for a marginal
    # precision gain at POC scale (11-entry golden set). Revisit if scores look
    # noisy entry-to-entry.
    ar_m = AnswerRelevancy(llm=judge_llm, embeddings=embeddings, strictness=1)
    print(
        "ragas_adapter.py, score_with_ragas, construct RAGAS metric objects executing : "
        "context_precision, context_recall, faithfulness, answer_relevancy ready"
    )

    log.info("scoring %d entries against RAGAS collections metrics", len(scoreable))
    print(
        f"ragas_adapter.py, score_with_ragas, run all entries concurrently executing : "
        f"scoring {len(scoreable)} entries"
    )
    per_entry_scores = asyncio.run(_score_all(scoreable, cp_m=cp_m, cr_m=cr_m, f_m=f_m, ar_m=ar_m))

    for (report_idx, entry), scores in zip(scoreable, per_entry_scores, strict=True):
        report.entries[report_idx] = entry.model_copy(update=scores)

    log.info("RAGAS scoring complete for %d entries", len(scoreable))
    print(
        f"ragas_adapter.py, score_with_ragas, run all entries concurrently executing : "
        f"complete for {len(scoreable)} entries"
    )
    return report


async def _score_all(
    scoreable: list[tuple[int, EvalResult]],
    cp_m: ContextPrecision,
    cr_m: ContextRecall,
    f_m: Faithfulness,
    ar_m: AnswerRelevancy,
) -> list[dict[str, float | None]]:
    return await asyncio.gather(*(_score_one(r, cp_m, cr_m, f_m, ar_m) for _, r in scoreable))


async def _score_one(
    entry: EvalResult,
    cp_m: ContextPrecision,
    cr_m: ContextRecall,
    f_m: Faithfulness,
    ar_m: AnswerRelevancy,
) -> dict[str, float | None]:
    """Run the four RAGAS metrics for one entry concurrently.

    A single metric raising (judge LLM timeout, malformed response) does not
    fail the whole entry — that metric's field is left None, matching the
    previous evaluate(raise_exceptions=False) behavior.
    """
    chunk_texts = [chunk.text for chunk in entry.ask_result.retrieved_chunks]
    if not chunk_texts:
        chunk_texts = [c.text_excerpt for c in entry.ask_result.citations]

    user_input = entry.question
    response = entry.ask_result.answer
    reference = entry.ground_truth_answer
    assert reference is not None  # guaranteed by the scoreable filter

    print(
        f"ragas_adapter.py, _score_one, score entry={entry.entry_id} executing : "
        f"contexts={len(chunk_texts)} calling judge LLM"
    )
    results = await asyncio.gather(
        cp_m.ascore(user_input=user_input, reference=reference, retrieved_contexts=chunk_texts),
        cr_m.ascore(user_input=user_input, retrieved_contexts=chunk_texts, reference=reference),
        f_m.ascore(user_input=user_input, response=response, retrieved_contexts=chunk_texts),
        ar_m.ascore(user_input=user_input, response=response),
        return_exceptions=True,
    )
    cp_r, cr_r, f_r, ar_r = results

    scores = {
        "context_precision": _safe_result("context_precision", entry.entry_id, cp_r),
        "context_recall": _safe_result("context_recall", entry.entry_id, cr_r),
        "faithfulness": _safe_result("faithfulness", entry.entry_id, f_r),
        "answer_relevance": _safe_result("answer_relevancy", entry.entry_id, ar_r),
    }
    print(f"ragas_adapter.py, _score_one, score entry={entry.entry_id} executing : {scores}")
    return scores


def _safe_result(metric_name: str, entry_id: str, result: object) -> float | None:
    if isinstance(result, BaseException):
        log.warning("RAGAS metric raised: %r", result)
        print(
            f"ragas_adapter.py, _safe_result, {metric_name} for entry={entry_id} executing : "
            f"ERROR={result!r}"
        )
        return None
    value = _safe(getattr(result, "value", None))
    print(
        f"ragas_adapter.py, _safe_result, {metric_name} for entry={entry_id} executing : "
        f"value={value}"
    )
    return value


def _safe(val: object) -> float | None:
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
