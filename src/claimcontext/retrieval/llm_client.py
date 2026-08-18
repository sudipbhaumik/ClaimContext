"""LLM provider abstraction for spec-2a.

Supports three providers selected by settings.llm_provider:
  - "ollama"    — local via Ollama daemon (default)
  - "anthropic" — Anthropic API (cloud eval/demo)
  - "openai"    — OpenAI API (alt provider, 2.x SDK)

ConfigurationError is raised at construction time if a required secret is absent,
so callers discover misconfiguration before issuing a query.

LLMError wraps all provider-level failures (timeout, auth, network, API errors)
so callers can distinguish retrieval errors from LLM failures.
"""

from __future__ import annotations

import logging

from claimcontext.config import Settings
from claimcontext.observability.tracing import get_tracer
from claimcontext.retrieval.errors import ConfigurationError, LLMError

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        provider = settings.llm_provider

        if provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ConfigurationError(
                    "llm_provider='anthropic' requires ANTHROPIC_API_KEY to be set"
                )
        elif provider == "openai":
            if not settings.openai_api_key:
                raise ConfigurationError("llm_provider='openai' requires OPENAI_API_KEY to be set")
        elif provider == "ollama":
            pass  # no key required; connection validated on first call
        else:
            raise ConfigurationError(f"unknown llm_provider: {provider!r}")

        log.debug("LLMClient ready (provider=%s model=%s)", provider, settings.llm_model)

    def complete(self, system: str, user: str) -> str:
        """Send a system + user prompt; return the assistant text.

        Raises LLMError on any provider-level failure.
        Timeout is settings.llm_timeout_seconds.
        """
        provider = self._settings.llm_provider
        tracer = get_tracer()
        with tracer.span(
            "llm.complete",
            as_type="generation",
            input={"system": system, "user": user},
            model=self._settings.llm_model,
            metadata={"provider": provider},
        ) as obs:
            try:
                if provider == "ollama":
                    result = self._complete_ollama(system, user)
                elif provider == "anthropic":
                    result = self._complete_anthropic(system, user)
                else:
                    result = self._complete_openai(system, user)
            except LLMError as exc:
                if obs is not None:
                    obs.update(level="ERROR", status_message=str(exc))
                raise
            except Exception as exc:
                if obs is not None:
                    obs.update(level="ERROR", status_message=str(exc))
                raise LLMError(f"[{provider}] unexpected error: {exc}") from exc
            if obs is not None:
                obs.update(output=result)
            return result

    def _complete_ollama(self, system: str, user: str) -> str:
        import ollama

        timeout = self._settings.llm_timeout_seconds
        try:
            response = ollama.chat(
                model=self._settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"num_predict": 1024},
                # ollama Python SDK does not expose a per-call timeout on chat();
                # the daemon itself enforces keep_alive and request timeouts.
                # For hard timeout enforcement, wrap in threading if needed (spec-5b).
            )
            _ = timeout  # noted above
            return str(response.message.content)
        except ollama.ResponseError as exc:
            raise LLMError(f"[ollama] response error: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"[ollama] {exc}") from exc

    def _complete_anthropic(self, system: str, user: str) -> str:
        import anthropic

        try:
            client = anthropic.Anthropic(
                api_key=self._settings.anthropic_api_key,
                timeout=float(self._settings.llm_timeout_seconds),
            )
            message = client.messages.create(
                model=self._settings.llm_model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return str(message.content[0].text)  # type: ignore[union-attr]
        except anthropic.APIError as exc:
            raise LLMError(f"[anthropic] API error: {exc}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError(
                f"[anthropic] timeout after {self._settings.llm_timeout_seconds}s"
            ) from exc

    def _complete_openai(self, system: str, user: str) -> str:
        from openai import APIError, APITimeoutError, OpenAI

        try:
            client = OpenAI(
                api_key=self._settings.openai_api_key,
                timeout=float(self._settings.llm_timeout_seconds),
            )
            response = client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMError("[openai] received null content in response")
            return content
        except APITimeoutError as exc:
            raise LLMError(f"[openai] timeout after {self._settings.llm_timeout_seconds}s") from exc
        except APIError as exc:
            raise LLMError(f"[openai] API error: {exc}") from exc
