# spec-0 handoff

## What was built

Full project skeleton for ClaimContext: uv-managed Python 3.11 environment with exact dependency pins, src-layout package with all subpackage namespaces, Pydantic BaseSettings config with all tunables externalized, Qdrant docker-compose service, ruff/black/mypy/pytest all passing clean, health check, `.env.example`, and initial git commit on `master`.

## Interfaces / contracts now available

- `claimcontext.config.Settings` — Pydantic `BaseSettings` with all config groups: App, Paths, Qdrant, LLM, Embedding, Retrieval, Chunking, Secrets (optional).
- `claimcontext.config.get_settings() -> Settings` — `@lru_cache(maxsize=1)` accessor; every later module imports this instead of reading `os.environ` directly.
- All subpackage namespaces importable: `claimcontext.{models,ingestion,retrieval,guardrails,agent,evaluation,api,observability}`.
- `data/documents/{policies,claim-notes,claim-docs}/` exist as the ingestion source tree.

## Config keys added

| Key | Meaning | Default |
|---|---|---|
| `app_name` | Service name | `claimcontext` |
| `log_level` | Logging level | `INFO` |
| `environment` | Runtime environment | `development` |
| `documents_dir` | Root of source documents | `data/documents` |
| `prompts_dir` | Versioned prompt files | `prompts` |
| `qdrant_url` | Qdrant HTTP endpoint | `http://localhost:6333` |
| `qdrant_collection` | Qdrant collection name | `claimcontext` |
| `llm_provider` | LLM backend | `ollama` |
| `llm_model` | Model name | `llama3.2` |
| `llm_timeout_seconds` | LLM call timeout | `30` |
| `embedding_model` | Embedding model name | `BAAI/bge-large-en-v1.5` |
| `chunker_version` | Chunking strategy version | `v1` |
| `top_k` | Dense retrieval top-k | `10` |
| `rrf_k` | RRF constant | `60` |
| `rerank_top_n` | Rerank output size | `5` |
| `refuse_threshold` | Min score to answer | `0.4` |
| `chunk_size` | Chunk size in tokens | `512` |
| `chunk_overlap` | Overlap in tokens | `64` |
| `anthropic_api_key` | Anthropic secret (optional) | `None` |
| `openai_api_key` | OpenAI secret (optional) | `None` |

## Decisions made

- Used Pydantic v2 `model_config = SettingsConfigDict(...)` (not inner `class Config`) — idiomatic v2 style, avoids deprecation warnings.
- `Optional[str]` → `str | None` (ruff UP045 auto-fixed) — enforced by ruff UP rules.
- `Literal` types for `log_level`, `environment`, `llm_provider` — caught misconfiguration at startup rather than at runtime.
- No deviation from CLAUDE.md or spec-0.

## Deliberately deferred

- `qdrant-client`, `sentence-transformers`, `fastapi`, `langgraph`, `ragas` — spec-1b+ installs what it needs.
- Qdrant health verification at startup (`make up` + curl) — user runs manually per spec-0 §G.
- `tests/test_config.py` covers config only; ingestion/retrieval tests belong in their respective specs.
- Dockerizing the app — spec-8a.

## Known gaps / TODOs

- `specs/` directory currently contains the input spec files (CLAUDE.md, spec-0). Move CLAUDE.md to repo root or keep in specs — decision for user.
- `.env` not yet created; user copies from `.env.example`.

## Proof status

1. `make check` — passes with 0 errors (lint + format + typecheck + test).
2. Health check — `python -m claimcontext.health` prints `app_name`, `environment`, `qdrant_url` from defaults; works without a `.env` file.
3. Secret-safety — `git status` after creating a real `.env` shows it as untracked/ignored; cannot be accidentally committed.
4. Qdrant — `make up` starts Qdrant; `make down` stops it. (Qdrant connectivity verified separately by user.)
