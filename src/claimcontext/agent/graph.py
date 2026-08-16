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

from claimcontext.agent.routing import classify_route, decompose_query, manufacture_refusal_audit
from claimcontext.agent.state import AgentState
from claimcontext.auth.models import Principal
from claimcontext.config import Settings
from claimcontext.retrieval.ask import _PROMPT_VERSION, _REFUSE_MESSAGE, ask
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.models import AskResult
from claimcontext.retrieval.reranker import Reranker
from claimcontext.retrieval.retriever import Retriever

log = logging.getLogger(__name__)


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
        result = ask(
            query=state.query,
            retriever=retriever,
            llm=llm,
            settings=settings,
            reranker=reranker,
            principal=state.principal,
        )
        return {"ask_results": [result]}

    def multi_node(state: AgentState) -> dict:
        sub_queries = decompose_query(state.query, settings, agent_llm)
        results: list[AskResult] = []
        for sub_query in sub_queries:
            result = ask(
                query=sub_query,
                retriever=retriever,
                llm=llm,
                settings=settings,
                reranker=reranker,
                principal=state.principal,
            )
            results.append(result)
        return {"sub_queries": sub_queries, "ask_results": results}

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
