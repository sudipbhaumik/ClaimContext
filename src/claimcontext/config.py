"""
ClaimContext configuration.

Single source of truth for all tunables. No magic numbers in business logic.
All fields sourced from .env (see .env.example). No secrets in code.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = Field(default="claimcontext")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    environment: Literal["development", "staging", "production"] = Field(default="development")

    # ── Paths ─────────────────────────────────────────────────────────────────
    documents_dir: str = Field(default="data/documents")
    prompts_dir: str = Field(default="prompts")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="claimcontext")

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "anthropic", "openai"] = Field(default="ollama")
    llm_model: str = Field(default="llama3.2")
    llm_timeout_seconds: int = Field(default=30)
    llm_base_url: str = Field(default="http://localhost:11434")  # Ollama daemon URL

    # ── Agent orchestrator (spec-5a) ────────────────────────────────────────────
    agent_max_sub_queries: int = Field(default=4)
    # Routing/decomposition/scope decisions are cheaper than answer generation —
    # default to the same model as llm_model, but allow a distinct (e.g. smaller,
    # faster) model to be configured without touching business logic (§2A.2).
    agent_model: str = Field(default="llama3.2")

    # ── Agent hardening (spec-5b) ────────────────────────────────────────────────
    # Distinct from llm_timeout_seconds (ask()'s answer-generation call) — governs
    # the agent's OWN LLM calls (scope check, multi-part check, decompose).
    agent_llm_timeout_seconds: int = Field(default=15)
    # Per-tool-call timeout (metadata-filter's Qdrant query, etc.).
    agent_tool_timeout_seconds: int = Field(default=10)
    # Per-query ceiling on total tool/LLM calls the graph may make — a budget
    # guard against a pathological decomposition or a future tool-calling loop.
    agent_max_tool_calls: int = Field(default=10)
    # tenacity retry count on agent-internal external calls (router LLM calls,
    # tool calls) before escalating.
    agent_retry_attempts: int = Field(default=2)

    # ── Agent eval (spec-6) ──────────────────────────────────────────────────────
    # Same naming convention as eval_golden_set_path (spec-4) — a distinct golden
    # set scoring the agent's PATH, not answer content.
    agent_eval_golden_set_path: str = Field(
        default="data/agent_eval/trajectory_golden_set_v1.jsonl"
    )
    agent_eval_golden_set_version: str = Field(default="v1")

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5")
    chunker_version: str = Field(default="v1")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(default=10)
    rrf_k: int = Field(default=60)
    retrieval_mode: Literal["dense", "hybrid"] = Field(default="hybrid")
    reranker_model: str = Field(default="BAAI/bge-reranker-base")
    rerank_top_n: int = Field(default=5)
    # refuse_threshold: bge-reranker-base outputs sigmoid-scaled [0,1] scores.
    # Relevant matches score ~0.97+; wrong-but-related passages score ~0.001.
    # Off-corpus queries score ~0.5 (model uncertainty floor), so threshold must
    # exceed 0.5 to refuse genuinely unanswerable queries. Default 0.55 is a
    # conservative starting point — spec-4 golden-set eval calibrates the final value.
    refuse_threshold: float = Field(default=0.55)

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ── Ingestion ─────────────────────────────────────────────────────────────
    ingest_source_dir: Path = Field(default=Path("data/documents"))
    ingest_hash_store_path: Path = Field(default=Path("data/ingest/.hash_store.json"))
    embedding_batch_size: int = Field(default=32)
    qdrant_timeout_seconds: int = Field(default=10)
    qdrant_upsert_batch_size: int = Field(default=64)

    # ── Eval (spec-4) ─────────────────────────────────────────────────────────
    eval_golden_set_path: str = Field(default="data/eval/golden_set_v1.jsonl")
    eval_golden_set_version: str = Field(default="v1")
    eval_context_precision_threshold: float = Field(default=0.60)
    eval_context_recall_threshold: float = Field(default=0.60)
    eval_faithfulness_threshold: float = Field(default=0.65)
    eval_answer_relevance_threshold: float = Field(default=0.65)
    eval_refusal_accuracy_threshold: float = Field(default=1.0)
    tier3_refusal_marker: str = Field(default="claims system")
    # Judge LLM for RAGAS — use a different family from the answer LLM to avoid
    # self-preference bias. Default: ollama/mistral (local, different family from
    # llama3.2). Set eval_ragas_llm_provider=openai + OPENAI_API_KEY for GPT-4o.
    eval_ragas_llm_provider: Literal["openai", "anthropic", "ollama"] = Field(default="ollama")
    eval_ragas_llm_model: str = Field(default="mistral")
    eval_ragas_llm_base_url: str = Field(default="http://localhost:11434/v1")
    # Embedding model for RAGAS AnswerRelevancy metric
    eval_ragas_embed_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # ── Secrets (optional; absent → feature disabled, not a crash) ────────────
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
