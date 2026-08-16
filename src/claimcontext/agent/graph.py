"""LangGraph wiring for the spec-5a agent orchestrator (Claude plumbing, per the
§3 Python-mastery split — routing/decompose decision logic lives in routing.py).

Graph shape:

    START -> router -> (single | multi | refuse) -> compose -> END

router sets state.route by calling routing.classify_route(). The conditional
edge reads that field and dispatches to exactly one of the three path nodes.
"single" and "multi" both populate state.ask_results by calling the real ask()
pipeline (spec-2c/3) — once for "single", once per sub-query for "multi"; each
call stays claim-scoped (see spec-5a's grounding-gap constraint). "refuse"
manufactures one AskResult without calling ask() at all, reusing ask.py's exact
refusal message and audit shape (routing.py). All three paths converge on
compose, which combines answers (never raw retrieved context — see spec-5a) into
state.final_answer.

No checkpointer is wired into compile() in this spec — spec-9a's job. The state
schema (AgentState) is checkpointer-compatible already; adding one later should
not require reshaping state.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from claimcontext.agent.routing import classify_route, decompose_query, manufacture_refusal_audit
from claimcontext.agent.state import AgentState
from claimcontext.auth.models import Principal
from claimcontext.config import Settings
from claimcontext.retrieval.ask import _PROMPT_VERSION, _REFUSE_MESSAGE, ask
from claimcontext.retrieval.errors import LLMError
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.models import AskResult
from claimcontext.retrieval.reranker import Reranker
from claimcontext.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

# Retryable/escalatable failures ONLY — transient LLM and Qdrant-connection
# problems. Deliberately NOT bare Exception (spec-5b known gap #3, promoted to
# fixed): a blanket except swallows genuine bugs (AttributeError, TypeError,
# KeyError from a real defect inside ask()) into a graceful "escalation" message,
# which quietly defeats exactly the kind of regression-discovery discipline this
# project's eval harness was built to enforce elsewhere (spec-4). A real bug
# should crash loudly in tests/eval, not disappear into a polite refusal.
# ConfigurationError/IndexStalenessError are deliberately excluded — retrying a
# missing API key or a stale index doesn't help; those are setup bugs, not
# transient failures, and should surface immediately, not after 2 retries.
_RETRYABLE_EXCEPTIONS = (
    LLMError,
    ResponseHandlingException,
    UnexpectedResponse,
    ConnectionError,
    TimeoutError,
)

# Distinct from _REFUSE_MESSAGE (imported from ask.py) — deliberately (spec-5b
# Decisions): "the system doesn't know" (a normal refusal) and "the system hit an
# internal operational limit" (escalation) must be distinguishable in the returned
# data, not silently collapsed into the same outcome. Still non-disclosing about
# claim existence, same §6B discipline as every other refusal-shaped message.
_ESCALATE_MESSAGE = (
    "This request could not be completed due to a system limit (either the "
    "question required more sub-steps than allowed, or an internal service call "
    "failed repeatedly). Please rephrase your question more narrowly, or escalate "
    "to a supervisor if this persists."
)


def _agent_llm_client(settings: Settings) -> LLMClient:
    """Routing/decomposition may use a different (typically cheaper/faster) model
    than answer generation — settings.agent_model vs settings.llm_model. Reuse the
    same LLMClient when they match; build a second one only when they differ, so
    the common case (agent_model unset, defaults equal) pays no extra cost.
    """
    if settings.agent_model == settings.llm_model:
        return LLMClient(settings)
    return LLMClient(settings.model_copy(update={"llm_model": settings.agent_model}))


def build_agent_graph(
    settings: Settings,
    retriever: Retriever | HybridRetriever,
    llm: LLMClient,
    reranker: Reranker | None = None,
) -> CompiledStateGraph:
    """Construct and compile the spec-5a agent graph.

    llm is the ANSWER-generation client (passed through unchanged to every ask()
    call, exactly as it is today for the bare ask() path). Routing/decomposition
    use a separate client built from settings.agent_model — see _agent_llm_client.
    """
    agent_llm = _agent_llm_client(settings)

    def _escalate(state: AgentState, reason: str) -> AskResult:
        log.warning(
            "agent: escalating query_hash=%s reason=%s",
            state.query[:60],
            reason,
        )
        return AskResult(
            query=state.query,
            answer=_ESCALATE_MESSAGE,
            citations=[],
            retrieved_chunks=[],
            llm_model=settings.llm_model,
            prompt_version=_PROMPT_VERSION,
            refused=True,
            adjuster_id=state.principal.adjuster_id,
        )

    def _ask_with_retry(query: str, principal: Principal) -> AskResult:
        """Retry a single ask() call on a TRANSIENT failure only
        (_RETRYABLE_EXCEPTIONS), bounded by settings.agent_retry_attempts
        (spec-5b hardening). ask() itself already has per-provider timeout/
        fallback behavior (spec-2a/2c); this is an additional layer at the
        agent's own call site, catching failures ask() doesn't itself retry
        (e.g. a transient Qdrant connection error). A non-transient exception
        (a real bug) is NOT caught here — it propagates immediately, on the
        first attempt, so it surfaces as a loud failure in tests/eval rather
        than a graceful escalation.
        """
        last_error: Exception | None = None
        for attempt in Retrying(
            stop=stop_after_attempt(settings.agent_retry_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                try:
                    return ask(
                        query=query,
                        retriever=retriever,
                        llm=llm,
                        settings=settings,
                        reranker=reranker,
                        principal=principal,
                    )
                except _RETRYABLE_EXCEPTIONS as exc:
                    last_error = exc
                    raise
        assert last_error is not None
        raise last_error

    def router_node(state: AgentState) -> dict:
        route = classify_route(state.query, state.principal, agent_llm, settings)
        log.info(
            "agent: router query_hash=%s route=%s",
            state.query[:60],
            route,
        )
        return {"route": route}

    def route_selector(state: AgentState) -> Literal["single", "multi", "refuse"]:
        assert state.route is not None  # router_node always sets it
        return state.route

    def single_node(state: AgentState) -> dict:
        try:
            result = _ask_with_retry(state.query, state.principal)
        except _RETRYABLE_EXCEPTIONS:
            log.exception("agent: single_node ask() failed after retries")
            return {
                "ask_results": [_escalate(state, "tool_failure")],
                "escalation_reason": "tool_failure",
            }
        return {"ask_results": [result]}

    def multi_node(state: AgentState) -> dict:
        sub_queries = decompose_query(state.query, settings, agent_llm)

        # Budget check (spec-5b Proof 4a): decompose_query no longer silently
        # caps its output — a query needing more sub-steps than allowed must
        # escalate visibly, not get silently truncated into an undersized answer.
        if len(sub_queries) > settings.agent_max_sub_queries:
            log.warning(
                "agent: decompose produced %d sub-queries, exceeding agent_max_sub_queries=%d",
                len(sub_queries),
                settings.agent_max_sub_queries,
            )
            return {
                "sub_queries": sub_queries,
                "ask_results": [_escalate(state, "budget_exceeded")],
                "escalation_reason": "budget_exceeded",
            }

        results: list[AskResult] = []
        escalation_reason: str | None = None
        for sub_query in sub_queries:
            try:
                result = _ask_with_retry(sub_query, state.principal)
            except _RETRYABLE_EXCEPTIONS:
                log.exception(
                    "agent: multi_node ask() failed after retries for sub_query=%r", sub_query
                )
                results.append(_escalate(state, "tool_failure"))
                escalation_reason = "tool_failure"
                continue
            results.append(result)
        return {
            "sub_queries": sub_queries,
            "ask_results": results,
            "escalation_reason": escalation_reason,
        }

    def refuse_node(state: AgentState) -> dict:
        manufacture_refusal_audit(state.query, state.principal)
        result = AskResult(
            query=state.query,
            answer=_REFUSE_MESSAGE,
            citations=[],
            retrieved_chunks=[],
            llm_model=settings.llm_model,
            prompt_version=_PROMPT_VERSION,
            refused=True,
            adjuster_id=state.principal.adjuster_id,
        )
        return {"ask_results": [result]}

    def compose_node(state: AgentState) -> dict:
        return {"final_answer": _compose(state)}

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("single", single_node)
    graph.add_node("multi", multi_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("compose", compose_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_selector,
        {"single": "single", "multi": "multi", "refuse": "refuse"},
    )
    graph.add_edge("single", "compose")
    graph.add_edge("multi", "compose")
    graph.add_edge("refuse", "compose")
    graph.add_edge("compose", END)

    return graph.compile()


def _compose(state: AgentState) -> AskResult:
    """Combine state.ask_results into one AskResult.

    Grounding-gap constraint (spec-5a): combines pre-generated ANSWERS, never
    pools retrieved_chunks across calls into a fresh context window — that would
    recreate the q08 failure mode (cross-claim contamination) at this layer,
    bypassing every per-call refuse gate. Each sub-answer's own citations and
    retrieved_chunks are preserved and concatenated (not merged/deduped across
    claims), so provenance stays traceable to which ask() call produced which
    part — see spec-5a Proof 4.

    A single-result state (the "single" and "refuse" paths) short-circuits:
    nothing to compose, the one AskResult passes through unchanged.
    """
    results = state.ask_results
    if len(results) == 1:
        return results[0]

    # Multi-part: build one composed answer text that attributes each part to
    # its own sub-query, and concatenate citations/chunks WITHOUT deduping
    # across claims — a reader (or a later automated check, per Proof 4) must be
    # able to tell which citations came from which sub-answer.
    parts: list[str] = []
    all_citations = []
    all_chunks = []
    any_refused = False
    for sub_query, result in zip(
        state.sub_queries or [r.query for r in results], results, strict=False
    ):
        if result.refused:
            any_refused = True
            parts.append(f'Regarding "{sub_query}": {result.answer}')
        else:
            parts.append(f'Regarding "{sub_query}": {result.answer}')
        all_citations.extend(result.citations)
        all_chunks.extend(result.retrieved_chunks)

    return AskResult(
        query=state.query,
        answer="\n\n".join(parts),
        citations=all_citations,
        retrieved_chunks=all_chunks,
        llm_model=results[0].llm_model,
        prompt_version=results[0].prompt_version,
        refused=any_refused and all(r.refused for r in results),
        adjuster_id=state.principal.adjuster_id,
    )


def run_agent(
    query: str,
    principal: Principal,
    settings: Settings,
    retriever: Retriever | HybridRetriever,
    llm: LLMClient,
    reranker: Reranker | None = None,
) -> AskResult:
    """Entry point: run the agent graph for one query, return its final AskResult.

    Same return type ask() returns — the agent is a drop-in router in front of
    the same output contract; nothing downstream needs to know whether an answer
    came from bare ask() or through the graph.
    """
    compiled = build_agent_graph(settings, retriever, llm, reranker)
    initial_state = AgentState(query=query, principal=principal)
    result_state = compiled.invoke(initial_state)
    final_answer = result_state["final_answer"]
    assert isinstance(final_answer, AskResult)
    return final_answer
