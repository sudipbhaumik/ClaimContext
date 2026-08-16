"""Agent state model for the spec-5a LangGraph orchestrator.

Kept intentionally flat and typed (Pydantic, per CLAUDE.md §4) so it stays
checkpointer-compatible for spec-9a without a reshape: every field here is either
a primitive, a Pydantic model already used elsewhere (Principal, AskResult), or a
list of one of those — nothing that needs a custom LangGraph reducer to serialize.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from claimcontext.auth.models import Principal
from claimcontext.retrieval.models import AskResult


class AgentState(BaseModel):
    """State threaded through the graph for one query.

    Nodes return partial-dict updates (LangGraph 1.x convention, verified against
    the installed langgraph==1.2.10 package — see spec-5a). Each field below is
    fully owned and set once by exactly one node; none require append-style
    reducers.
    """

    query: str
    principal: Principal

    # Set by the router node. None only transiently before routing runs.
    route: Literal["single", "multi", "refuse"] | None = None

    # Populated by the decompose node on the "multi" path only. Empty on
    # "single"/"refuse" — the single-retrieval node treats [query] as its one
    # sub-query implicitly rather than requiring this to be populated.
    sub_queries: list[str] = []

    # One AskResult per ask() call the graph made — one entry for "single" or
    # "refuse" (a single call or a single manufactured refusal), one per
    # sub-query for "multi". Order matches sub_queries when both are populated.
    ask_results: list[AskResult] = []

    # Set by the compose node (or directly by single/refuse nodes, which have
    # exactly one AskResult and no composition to do). This is what run_agent()
    # returns.
    final_answer: AskResult | None = None

    # Set by single_node/multi_node when hardening escalates (spec-5b) —
    # "budget_exceeded" | "tool_failure" | None. Additive field (spec-6): lets
    # run_agent_with_trajectory() read the escalation reason directly off state
    # instead of parsing _ESCALATE_MESSAGE text. None on every non-escalated path.
    escalation_reason: str | None = None
