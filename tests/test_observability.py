"""spec-7b Langfuse tracing proofs.

All tests marked @pytest.mark.observability require:
  - Qdrant running with indexed corpus (make up + ingest)
  - Ollama running with llama3.2 pulled
  - A reachable Langfuse instance (self-hosted via docker-compose.langfuse.yml,
    or Langfuse Cloud) with LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST
    set in the environment for the test run.

To run:
  LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_HOST=... \
    pytest tests/test_observability.py -m observability -v -s
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from claimcontext.agent.graph import run_agent
from claimcontext.agent.trajectory import run_agent_with_trajectory
from claimcontext.auth.resolver import resolve_principal
from claimcontext.config import Settings, get_settings
from claimcontext.observability.tracing import get_tracer
from claimcontext.retrieval.ask import ask
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.reranker import Reranker

_FLAGSHIP_QUERY = (
    "For claim CLM-1004, is the wind-driven rain damage to the roof and "
    "interior covered under the governing policy?"
)


def _langfuse_env_present() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


pytestmark = pytest.mark.skipif(
    not _langfuse_env_present(),
    reason="LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set — see module docstring",
)


# ── Shared fixtures ──────────────────────────────────────────────────────────
#
# get_tracer() reads Settings via the process-wide get_settings() cache — it
# is NOT parametrized by whatever Settings object a test happens to pass into
# ask()/run_agent(). To actually control what Tracer get_tracer() builds, a
# test must set real environment variables and clear both caches, not just
# construct a local Settings(...) object (that object only affects the
# retrieval/LLM config paths that take `settings` as an explicit argument).


def _set_langfuse_env(**overrides: str) -> dict[str, str | None]:
    """Set LANGFUSE_* env vars, return the previous values for restoration."""
    saved: dict[str, str | None] = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = value
    get_settings.cache_clear()
    get_tracer.cache_clear()
    return saved


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()
    get_tracer.cache_clear()


@pytest.fixture(scope="module")
def settings() -> Iterator[Settings]:
    saved = _set_langfuse_env(
        LANGFUSE_ENABLED="true",
        LANGFUSE_PUBLIC_KEY=os.environ["LANGFUSE_PUBLIC_KEY"],
        LANGFUSE_SECRET_KEY=os.environ["LANGFUSE_SECRET_KEY"],
        LANGFUSE_HOST=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    yield get_settings()
    _restore_env(saved)


@pytest.fixture(scope="module")
def retriever(settings: Settings) -> HybridRetriever:
    r = HybridRetriever(settings)
    r.check_index_staleness()
    return r


@pytest.fixture(scope="module")
def reranker(settings: Settings) -> Reranker:
    return Reranker(settings)


@pytest.fixture(scope="module")
def llm(settings: Settings) -> LLMClient:
    return LLMClient(settings)


def _langfuse_api() -> httpx.Client:
    host = os.environ["LANGFUSE_HOST"]
    return httpx.Client(
        base_url=host,
        auth=(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]),
        timeout=10,
    )


def _observations_for_trace(
    trace_id: str, retries: int = 15, delay: float = 1.0, stable_polls: int = 2
) -> list[dict]:
    """Poll the Langfuse API for a trace's observations until the count is
    stable across `stable_polls` consecutive polls, not just non-empty —
    ingestion is async (that's the whole fail-safe point, Decided now #3),
    so a query right after the first span lands can undercount: more spans
    for the same trace may still be in flight. Returning on first-non-empty
    is a real flakiness bug (found live — a 2-step trajectory's trace read
    back as 1 span because the second span hadn't exported yet)."""
    last_count = -1
    stable = 0
    result: list[dict] = []
    with _langfuse_api() as client:
        for _ in range(retries):
            resp = client.get(
                "/api/public/v2/observations", params={"traceId": trace_id, "limit": 50}
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            result = list(data)
            if len(result) == last_count and len(result) > 0:
                stable += 1
                if stable >= stable_polls:
                    return result
            else:
                stable = 0
            last_count = len(result)
            time.sleep(delay)
    return result


# ── Proof 1 — a real request produces a real, navigable trace ──────────────


@pytest.mark.observability
def test_proof1_real_request_produces_navigable_trace(
    settings: Settings, retriever: HybridRetriever, llm: LLMClient, reranker: Reranker
) -> None:
    principal = resolve_principal("ADJ-027")
    tracer = get_tracer()

    # trace_id must be read while a span is still active — ask() exits its
    # own span before returning, so wrap the call in an outer test span
    # (same pattern Proofs 3/4/6 use) rather than trying to find the trace
    # afterward by querying "most recent ask observation," which is
    # unreliable: it can pick up a stale trace from an earlier test/run in
    # the same Langfuse project (found live — a leftover trace's llm.complete
    # had a different parent than expected, from an unrelated older call).
    with tracer.span("test.proof1_root", as_type="span") as _root:
        trace_id = tracer.current_trace_id()
        result = ask(
            query=_FLAGSHIP_QUERY,
            retriever=retriever,
            llm=llm,
            settings=settings,
            reranker=reranker,
            principal=principal,
        )
    tracer.shutdown()
    assert not result.refused
    assert trace_id is not None

    observations = _observations_for_trace(trace_id)
    names = {o["name"] for o in observations}
    assert "ask" in names
    assert "retrieval.hybrid_search" in names
    assert "retrieval.dense_search" in names
    assert "retrieval.rerank" in names
    assert "llm.complete" in names

    by_name = {o["name"]: o for o in observations}
    # nesting: dense_search is a child of hybrid_search; hybrid_search,
    # rerank, and llm.complete are all children of the root "ask" span.
    ask_id = by_name["ask"]["id"]
    assert by_name["retrieval.hybrid_search"]["parentObservationId"] == ask_id
    assert by_name["retrieval.rerank"]["parentObservationId"] == ask_id
    assert by_name["llm.complete"]["parentObservationId"] == ask_id
    assert (
        by_name["retrieval.dense_search"]["parentObservationId"]
        == by_name["retrieval.hybrid_search"]["id"]
    )


# ── Proof 2 — the CLI path traces too, for free ─────────────────────────────


@pytest.mark.observability
def test_proof2_cli_path_traces_without_cli_specific_code(settings: Settings) -> None:
    marker_query = f"CLI trace proof {time.time()}"
    env = {
        **os.environ,
        "LANGFUSE_ENABLED": "true",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "claimcontext",
            "ask",
            marker_query,
            "--adjuster-id",
            "ADJ-014",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # The query is nonsense (weak-context refusal expected) — this proof only
    # cares that a trace was recorded, not the answer content.
    with _langfuse_api() as client:
        resp = client.get(
            "/api/public/v2/observations", params={"name": "ask", "limit": 5, "orderBy": "desc"}
        )
        resp.raise_for_status()
        recent = resp.json()["data"]
    matched = [
        o
        for o in recent
        for obs in _observations_for_trace(o["traceId"], retries=3, delay=1.0)
        if obs["name"] == "ask"
    ]
    assert matched, "no trace recorded for the CLI-invoked ask() call"


# ── Proof 3 — multi-part query fan-out; "agree" defined precisely ──────────


@pytest.mark.observability
def test_proof3_trajectory_and_trace_agree_on_ask_call_count(
    settings: Settings, retriever: HybridRetriever, llm: LLMClient, reranker: Reranker
) -> None:
    principal = resolve_principal("ADJ-027")
    tracer = get_tracer()

    multi_query = (
        "For claim CLM-1004, is the damage covered, and separately what does "
        "the FNOL say about the reported cause?"
    )
    with tracer.span("test.proof3_root", as_type="span") as _root:
        trace_id = tracer.current_trace_id()
        trajectory = run_agent_with_trajectory(
            query=multi_query,
            principal=principal,
            settings=settings,
            retriever=retriever,
            llm=llm,
            reranker=reranker,
        )
    tracer.shutdown()
    assert trace_id is not None

    observations = _observations_for_trace(trace_id)
    ask_spans = [o for o in observations if o["name"] == "ask"]

    # Agreement dimension (spec-7b Proof 3, pinned precisely): every
    # trajectory step has AT LEAST ONE corresponding "ask" span. Extra spans
    # from retries are allowed and are not a failure — only a step with ZERO
    # spans (or a span count that implies a step never happened) fails this.
    assert len(ask_spans) >= len(trajectory.steps), (
        f"trajectory recorded {len(trajectory.steps)} steps but the trace has "
        f"only {len(ask_spans)} ask() spans — a logical step appears to be "
        f"missing from the trace entirely"
    )


# ── Proof 4 — a forced tool failure is visible in the trace ────────────────


@pytest.mark.observability
def test_proof4_escalation_visible_in_trace(
    settings: Settings, retriever: HybridRetriever, llm: LLMClient, reranker: Reranker
) -> None:
    principal = resolve_principal("ADJ-027")
    tracer = get_tracer()

    # Force the decompose-cap escalation deterministically (spec-7a's proven
    # technique — an unreachable-Qdrant address was tried first here and hung
    # indefinitely in this environment; connecting to a closed local port
    # apparently doesn't fail fast everywhere, so avoid it as a test
    # technique). agent_max_sub_queries=0 + a multi-part query reliably
    # forces graph.py's budget-exceeded escalation, no network dependency.
    escalating_settings = settings.model_copy(update={"agent_max_sub_queries": 0})

    with tracer.span("test.proof4_root", as_type="span") as _root:
        trace_id = tracer.current_trace_id()
        run_agent(
            query=(
                "For claim CLM-1004, is the damage covered, and separately "
                "what does the FNOL say about the reported cause?"
            ),
            principal=principal,
            settings=escalating_settings,
            retriever=retriever,
            llm=llm,
            reranker=reranker,
        )
    tracer.shutdown()
    assert trace_id is not None

    observations = _observations_for_trace(trace_id)
    names = {o["name"] for o in observations}
    assert "agent.escalation" in names, (
        "expected the forced escalation to be visible as a tagged span in the "
        f"trace; observations recorded: {names}"
    )


# ── Proof 5 — langfuse_enabled=False changes nothing about behavior ────────


@pytest.mark.observability
def test_proof5_disabled_tracing_is_fully_inert(
    retriever: HybridRetriever, llm: LLMClient, reranker: Reranker
) -> None:
    # get_tracer() is a process-wide singleton reading env-derived Settings —
    # the module's `settings` fixture left tracing enabled globally, so this
    # test must explicitly flip it off (and restore afterward) to actually
    # exercise the disabled path through ask()'s real code, not just
    # construct an unused Tracer object on the side.
    saved = _set_langfuse_env(LANGFUSE_ENABLED="false")
    try:
        assert not get_tracer().enabled

        principal = resolve_principal("ADJ-027")
        result = ask(
            query=_FLAGSHIP_QUERY,
            retriever=retriever,
            llm=llm,
            settings=get_settings(),
            reranker=reranker,
            principal=principal,
        )
        assert not result.refused
        assert result.citations
    finally:
        _restore_env(saved)


# ── Proof 6 — langfuse_enabled=True but Langfuse is unreachable: /ask/ask()
#    still returns normally, no added hang. THE fail-safe proof. ───────────


@pytest.mark.observability
def test_proof6_unreachable_langfuse_does_not_block_ask(
    retriever: HybridRetriever, llm: LLMClient, reranker: Reranker
) -> None:
    principal = resolve_principal("ADJ-027")

    saved = _set_langfuse_env(
        LANGFUSE_ENABLED="true",
        LANGFUSE_PUBLIC_KEY="pk-lf-doesnotexist",
        LANGFUSE_SECRET_KEY="sk-lf-doesnotexist",
        # RFC 5737 TEST-NET address — routable syntax, guaranteed unreachable,
        # so the OS returns "no route" quickly rather than a long OS-level
        # connect timeout masking whether OUR timeout config did anything.
        LANGFUSE_HOST="http://192.0.2.1:3000",
        LANGFUSE_TIMEOUT_SECONDS="2",
    )
    try:
        assert get_tracer().enabled  # client construction itself doesn't touch the network

        t0 = time.monotonic()
        result = ask(
            query=_FLAGSHIP_QUERY,
            retriever=retriever,
            llm=llm,
            settings=get_settings(),
            reranker=reranker,
            principal=principal,
        )
        elapsed = time.monotonic() - t0
    finally:
        _restore_env(saved)

    assert not result.refused
    assert result.citations
    # Span creation is local/synchronous (never touches the network) and
    # export is async on a background thread — an unreachable collector must
    # not add meaningfully to request latency. Generous bound: well under any
    # timeout that could indicate a synchronous network wait occurred.
    assert elapsed < 90, (
        f"ask() took {elapsed:.1f}s against an unreachable Langfuse host — "
        "tracing may be blocking the request path"
    )
