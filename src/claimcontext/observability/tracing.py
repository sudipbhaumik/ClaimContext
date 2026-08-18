"""Optional Langfuse tracing (spec-7b).

Fail-safe by construction (spec-7b "Decided now #3"), not by convention at
each call site:

- Langfuse's OTEL BatchSpanProcessor exports asynchronously on a background
  thread (verified against the installed SDK — see spec-7b handoff). Starting
  a span is a local, synchronous, in-process operation; it never makes a
  network call itself, so an unreachable/slow Langfuse host cannot block the
  traced request path.
- The only two places tracing *could* still fail — client construction (bad
  config) and span start/close (SDK bug, misconfigured client) — are guarded
  here, once, in Tracer. Any exception from either logs a WARNING and
  degrades to "this operation goes untraced," never propagates into the
  caller. Exceptions raised by the caller's OWN code inside a `with
  tracer.span(...)` block are never caught here — they propagate exactly as
  they would if tracing didn't exist. This module must never turn a real bug
  into a swallowed tracing failure, or the reverse.

Every call site uses get_tracer() — a process-wide singleton — never
constructs a Tracer/Langfuse client directly, so there is exactly one
Langfuse client (and one OTEL TracerProvider registration) per process,
which is what allows spans from different call sites to nest into one trace.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from claimcontext.config import Settings, get_settings

log = logging.getLogger(__name__)


class Tracer:
    def __init__(self, settings: Settings) -> None:
        self._client: Any = None
        if not settings.langfuse_enabled:
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                timeout=settings.langfuse_timeout_seconds,
            )
        except Exception:
            log.warning(
                "tracing: Langfuse client construction failed — tracing disabled",
                exc_info=True,
            )
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def span(self, name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
        """One traced operation. Yields the Langfuse observation (has
        `.update(output=..., metadata=..., level=..., ...)`), or None when
        tracing is disabled/unavailable — every call site must tolerate None.
        """
        if self._client is None:
            yield None
            return

        try:
            cm = self._client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
            obs = cm.__enter__()
        except Exception:
            log.warning(
                "tracing: failed to start span %r — continuing untraced", name, exc_info=True
            )
            yield None
            return

        try:
            yield obs
        except BaseException:
            exc_info = sys.exc_info()
            try:
                cm.__exit__(*exc_info)
            except Exception:
                log.warning("tracing: failed to close span %r", name, exc_info=True)
            raise
        else:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                log.warning("tracing: failed to close span %r", name, exc_info=True)

    def current_trace_id(self) -> str | None:
        if self._client is None:
            return None
        try:
            trace_id: str | None = self._client.get_current_trace_id()
            return trace_id
        except Exception:
            log.warning("tracing: failed to read current trace id", exc_info=True)
            return None

    def shutdown(self) -> None:
        """Flush and close the Langfuse client. Call once at process
        shutdown (API lifespan shutdown, or CLI exit) — never on the request
        path, since flush is a blocking network operation by design."""
        if self._client is None:
            return
        try:
            self._client.shutdown()
        except Exception:
            log.warning("tracing: shutdown failed", exc_info=True)


@lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    """Process-wide singleton, same pattern as config.get_settings()."""
    return Tracer(get_settings())
