"""FastAPI serving layer (spec-7a). Wraps run_agent() over HTTP without
weakening any guarantee the agent/entitlement layers already enforce.

Resources (retriever, LLM client, reranker) are built once at process startup
(lifespan), not per request — see AppState.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from claimcontext.agent.graph import run_agent
from claimcontext.api.schemas import (
    PUBLIC_REFUSAL_MESSAGE,
    AskRequest,
    AskResponse,
    to_ask_response,
)
from claimcontext.auth.errors import AuthorizationError
from claimcontext.auth.resolver import resolve_principal
from claimcontext.config import Settings, get_settings
from claimcontext.retrieval.errors import IndexStalenessError
from claimcontext.retrieval.hybrid_retriever import HybridRetriever
from claimcontext.retrieval.llm_client import LLMClient
from claimcontext.retrieval.reranker import Reranker
from claimcontext.retrieval.retriever import Retriever

# uvicorn configures its own loggers but not ours — without this, log.info()
# calls below are silently dropped when run via `uvicorn claimcontext.api.app:app`
# (as opposed to a path that already calls logging.basicConfig, like __main__.py).
# Same format as __main__.py for consistency.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Generic message for the mid-flight-staleness → 503 mapping (spec-7a,
# "Readiness and staleness semantics"). Distinct from the refusal message —
# a 503 says "try again," a refusal says "this won't be answered."
_STALE_INDEX_MESSAGE = "Service temporarily unavailable. Please try again shortly."

_GENERIC_ERROR_MESSAGE = "An unexpected error occurred."


@dataclass
class AppState:
    settings: Settings
    retriever: Retriever | HybridRetriever
    llm: LLMClient
    reranker: Reranker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    retriever: Retriever | HybridRetriever
    if settings.retrieval_mode == "hybrid":
        retriever = HybridRetriever(settings)
    else:
        retriever = Retriever(settings)
    llm = LLMClient(settings)
    reranker = Reranker(settings)

    # Embedder/Reranker construction above is cheap — the actual model
    # weights are lazy-loaded on first use. Without warming them up here,
    # /ready would report "ready" the moment Qdrant responds, while the
    # first real /ask still pays a one-time ~tens-of-seconds weight-load
    # cost /ready never surfaced. Warming up here means "ready" and
    # "actually fast" become the same claim.
    log.info("startup: warming up embedding + reranker models")
    retriever.warm_up()
    reranker.warm_up()

    app.state.claimcontext = AppState(
        settings=settings, retriever=retriever, llm=llm, reranker=reranker
    )
    log.info("startup: resources constructed once (retriever/llm/reranker)")
    yield
    log.info("shutdown")


app = FastAPI(title="ClaimContext", lifespan=lifespan)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Any exception not handled by a route's own try/except lands here — the
    real detail goes to server-side logs only, never the response body."""
    log.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": _GENERIC_ERROR_MESSAGE})


def _refusal_response() -> JSONResponse:
    """The single response shape used for every refusal-equivalent outcome,
    including unknown adjuster_id — see spec-7a Proof 2. 200, not a 4xx: a
    4xx status code alone would make unknown-adjuster distinguishable from
    every other refusal reason even with identical body text."""
    body = AskResponse(answer=PUBLIC_REFUSAL_MESSAGE, citations=[], refused=True)
    return JSONResponse(status_code=200, content=body.model_dump())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> JSONResponse:
    state: AppState = request.app.state.claimcontext
    checks: dict[str, str] = {}
    ok = True

    try:
        state.retriever.check_index_staleness()
        checks["index"] = "ok"
    except IndexStalenessError as exc:
        ok = False
        checks["index"] = exc.reason
    except Exception:
        log.exception("ready: index staleness check failed unexpectedly")
        ok = False
        checks["index"] = "error"

    status_code = 200 if ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if ok else "not_ready", "checks": checks},
    )


@app.post("/ask")
async def ask_route(body: AskRequest, request: Request) -> JSONResponse:
    state: AppState = request.app.state.claimcontext

    try:
        principal = resolve_principal(body.adjuster_id)
    except AuthorizationError:
        log.info("ask: unknown adjuster_id, refusing")
        return _refusal_response()

    log.info(
        "ask: principal resolved adjuster=%r region=%r", principal.adjuster_id, principal.region
    )

    try:
        # run_agent()/ask() never call check_index_staleness() themselves —
        # only /ready and the CLI do. Resources are lifespan-constructed once
        # at startup (see AppState), so without this explicit check here, a
        # reindex that happens after startup but before the next /ready poll
        # would go unnoticed by /ask entirely. This call is what makes the
        # mid-flight-staleness → 503 decision (spec-7a "Readiness and
        # staleness semantics") real rather than merely documented.
        state.retriever.check_index_staleness()
        result = run_agent(
            query=body.query,
            principal=principal,
            settings=state.settings,
            retriever=state.retriever,
            llm=state.llm,
            reranker=state.reranker,
        )
    except IndexStalenessError:
        # IndexStalenessError is deliberately excluded from the agent's own
        # retry/escalation handling (graph.py) and always propagates loudly —
        # here, mid-flight, it maps to a 503, not the generic 500 bucket
        # (see spec-7a "Readiness and staleness semantics").
        log.error("ask: index went stale mid-flight")
        return JSONResponse(status_code=503, content={"detail": _STALE_INDEX_MESSAGE})

    log.info("ask: route complete refused=%s", result.refused)
    return JSONResponse(status_code=200, content=to_ask_response(result).model_dump())
