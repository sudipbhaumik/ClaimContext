"""RAGAS adapter for spec-4 eval harness.

Converts passing ANSWER entries to a RAGAS EvaluationDataset, calls evaluate(),
and writes per-entry scores back into EvalResult objects in the report.

RAGAS 0.4.3 API (verified against installed package):
  - Dataset type: ragas.EvaluationDataset of ragas.SingleTurnSample
  - Sample fields: user_input, response, retrieved_contexts (list[str]), reference (str)
  - Metrics require InstructorBaseRagasLLM: use InstructorLLM(client, model, provider)
  - AnswerRelevancy also requires HuggingFaceEmbeddings(model=...)
  - evaluate() → EvaluationResult; .scores is list[dict[str, float|None]] in input order
  - Metric keys: "context_precision", "context_recall", "faithfulness", "answer_relevancy"
    (note: "answer_relevancy" with a "y", not "answer_relevance")

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

import instructor  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import HuggingFaceEmbeddings  # noqa: E402
from ragas.llms import InstructorLLM  # noqa: E402
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness  # noqa: E402
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from claimcontext.config import Settings  # noqa: E402
from claimcontext.eval.schema import EvalReport, EvalResult, ExpectedBehavior  # noqa: E402


def _build_judge_llm(settings: Settings) -> InstructorLLM:
    """Build the RAGAS judge LLM. Use a different family from the answer LLM."""
    provider = settings.eval_ragas_llm_provider
    model = settings.eval_ragas_llm_model

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible API — instructor.from_openai() wraps it.
        # Use a different model family from the answer LLM (e.g. mistral vs llama3.2)
        # to preserve the different-family property without a cloud API key.
        from openai import OpenAI

        base_url = settings.eval_ragas_llm_base_url
        ollama_client = OpenAI(base_url=base_url, api_key="ollama")
        client = instructor.from_openai(ollama_client)
        return InstructorLLM(client=client, model=model, provider="openai")

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("eval_ragas_llm_provider='openai' requires OPENAI_API_KEY.")
        from openai import OpenAI

        client = instructor.from_openai(OpenAI(api_key=settings.openai_api_key))
        return InstructorLLM(client=client, model=model, provider="openai")

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("eval_ragas_llm_provider='anthropic' requires ANTHROPIC_API_KEY.")
        import anthropic

        client = instructor.from_anthropic(anthropic.Anthropic(api_key=settings.anthropic_api_key))
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

    if not scoreable:
        log.info("no scoreable ANSWER entries — skipping RAGAS evaluation")
        return report

    log.info(
        "RAGAS: scoring %d entries (judge=%s/%s embed=%s)",
        len(scoreable),
        settings.eval_ragas_llm_provider,
        settings.eval_ragas_llm_model,
        settings.eval_ragas_embed_model,
    )

    judge_llm = _build_judge_llm(settings)
    embeddings = _build_embeddings(settings)
    cp_m = ContextPrecision(llm=judge_llm)
    cr_m = ContextRecall(llm=judge_llm)
    f_m = Faithfulness(llm=judge_llm)
    ar_m = AnswerRelevancy(llm=judge_llm, embeddings=embeddings)

    samples: list[SingleTurnSample] = []
    for _, r in scoreable:
        chunk_texts = [chunk.text for chunk in r.ask_result.retrieved_chunks]
        if not chunk_texts:
            # Fall back to citation excerpts if retrieved_chunks is empty
            chunk_texts = [c.text_excerpt for c in r.ask_result.citations]

        samples.append(
            SingleTurnSample(
                user_input=r.question,
                response=r.ask_result.answer,
                retrieved_contexts=chunk_texts,
                reference=r.ground_truth_answer,
            )
        )

    dataset = EvaluationDataset(samples=samples)  # type: ignore[arg-type]
    run_cfg = RunConfig(timeout=120, max_retries=2)

    log.info("calling RAGAS evaluate() — judge LLM API calls follow")
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[cp_m, cr_m, f_m, ar_m],  # type: ignore[list-item]
        run_config=run_cfg,
        raise_exceptions=False,
        show_progress=True,
    )

    scores_list: list[dict] = ragas_result.scores  # type: ignore[union-attr]
    for list_pos, (report_idx, entry) in enumerate(scoreable):
        if list_pos >= len(scores_list):
            break
        s = scores_list[list_pos]
        report.entries[report_idx] = entry.model_copy(
            update={
                "context_precision": _safe(s.get("context_precision")),
                "context_recall": _safe(s.get("context_recall")),
                "faithfulness": _safe(s.get("faithfulness")),
                # RAGAS key is "answer_relevancy" (with "y"); our field is "answer_relevance"
                "answer_relevance": _safe(s.get("answer_relevancy")),
            }
        )

    log.info("RAGAS scoring complete for %d entries", len(scoreable))
    return report


def _safe(val: object) -> float | None:
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
