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

    # ── Secrets (optional; absent → feature disabled, not a crash) ────────────
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
