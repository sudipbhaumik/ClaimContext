"""Trajectory capture for the spec-6 agent eval harness.

Additive, not a rewrite: run_agent() (spec-5a) keeps its existing signature and
return type (AskResult) unchanged — anything already calling it is unaffected.
run_agent_with_trajectory() is a parallel entry point that invokes the exact same
compiled graph and reads the FULL final AgentState (route, sub_queries,
ask_results, escalation_reason) instead of discarding everything but
final_answer. No node in graph.py's control flow is changed by this module —
only AgentState gained one additive field (escalation_reason, spec-6) that
single_node/multi_node already populate as part of their existing escalation
paths.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from claimcontext.agent.graph import build_agent_graph
from claimcontext.agent.state import AgentState
from claimcontext.auth.models import Principal
from claimcontext.config import Settings
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.models import AskResult
from claimcontext.retrieval.reranker import Reranker
from claimcontext.retrieval.retriever import Retriever


class TrajectoryStep(BaseModel):
    """One ask() call the graph made."""

    query: str
    refused: bool
    escalated: bool
    citations_doc_ids: list[str]


class AgentTrajectory(BaseModel):
    """The full path the graph took for one query — spec-6's unit of eval,
    as distinct from spec-4's unit of eval (an AskResult's answer content).
    """

    query: str
    route: Literal["single", "multi", "refuse"]
    steps: list[TrajectoryStep]
    escalation_reason: str | None
    final_answer: AskResult

    @property
    def escalated(self) -> bool:
        return self.escalation_reason is not None


def _step_from_result(query: str, result: AskResult, escalated: bool) -> TrajectoryStep:
    return TrajectoryStep(
        query=query,
        refused=result.refused,
        escalated=escalated,
        citations_doc_ids=[c.doc_id for c in result.citations],
    )


def run_agent_with_trajectory(
    query: str,
    principal: Principal,
    settings: Settings,
    retriever: Retriever | HybridRetriever,
    llm: LLMClient,
    reranker: Reranker | None = None,
) -> AgentTrajectory:
    """Run the agent graph and return the full trajectory, not just the answer.

    Same graph, same invocation as run_agent() — this does not change agent
    behavior, only what's read back out of the final state afterward.
    """
    compiled = build_agent_graph(settings, retriever, llm, reranker)
    initial_state = AgentState(query=query, principal=principal)
    result_state = compiled.invoke(initial_state)
    state = AgentState.model_validate(result_state)

    assert state.route is not None
    assert state.final_answer is not None

    escalated = state.escalation_reason is not None
    if state.route == "multi" and state.sub_queries:
        # One step per sub-query, matched positionally with ask_results — same
        # zip pattern _compose() in graph.py already uses for the same purpose.
        steps = [
            _step_from_result(sub_query, result, escalated)
            for sub_query, result in zip(state.sub_queries, state.ask_results, strict=False)
        ]
    else:
        # "single" or "refuse": exactly one ask_results entry, no sub_queries.
        steps = [_step_from_result(query, r, escalated) for r in state.ask_results]

    return AgentTrajectory(
        query=query,
        route=state.route,
        steps=steps,
        escalation_reason=state.escalation_reason,
        final_answer=state.final_answer,
    )
