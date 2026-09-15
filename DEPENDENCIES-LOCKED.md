# DEPENDENCIES-LOCKED.md

Human-readable reference for all pinned dependencies. Complements `uv.lock` (the
machine-authoritative lockfile) and `pyproject.toml` (the declared constraints).

**uv.lock is authoritative for versions.** All version numbers below are pulled from
`uv.lock` exactly. On any disagreement between this file and `uv.lock`, `uv.lock`
wins and this file is updated to match.

Verification date: **2026-07-29** against `uv.lock` (parsed programmatically).

Packages marked **`[pending]`** are declared for a future spec but not yet added to
`pyproject.toml` / `uv.lock`; the version listed is the target confirmed at spec
authoring time.

---

## Core runtime

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `pydantic` | `2.13.4` | Typed I/O everywhere — all models, tool I/O, config | spec-0 |
| `pydantic-settings` | `2.14.2` | `BaseSettings` — env-var-sourced config, `.env` loading | spec-0 |
| `pydantic-core` | `2.46.4` | Pydantic v2 Rust core (transitive) | spec-0 |
| `python-dotenv` | `1.2.2` | `.env` file loading | spec-0 |
| `annotated-types` | `0.7.0` | Pydantic v2 annotation support (transitive) | spec-0 |
| `typing-extensions` | `4.16.0` | Backports for `typing` (transitive) | spec-0 |
| `typing-inspection` | `0.4.2` | Runtime type inspection (transitive, Pydantic) | spec-0 |

---

## LLM providers

> ⚠️ **`openai` 2.x** — major version with breaking API changes from 1.x. Most online
> examples are 1.x. Verify all call signatures against 2.x docs before writing code;
> do not follow 1.x examples (same trap as LangChain 0.x/1.x below).

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `ollama` | `0.6.2` **`[pending]`** | Ollama Python SDK — default local LLM provider | spec-2a |
| `anthropic` | `0.120.2` **`[pending]`** | Anthropic SDK — cloud eval/demo path | spec-2a |
| `openai` | `2.50.0` **`[pending]`** | OpenAI SDK — optional alt provider (2.x API) | spec-2a |

> ⚠️ **LangChain / LangGraph are on 1.x** — most online tutorials and StackOverflow
> answers are 0.x and will mislead. When these packages are added (spec-5a/5b), verify
> API signatures against current 1.x docs before writing code.

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `langgraph` | `1.2.10` | Orchestrator — `StateGraph`, conditional edges, `compile()` | spec-5a |
| `langchain-core` | `1.5.2` | LangGraph's own runnable/message primitives (transitive requirement) | spec-5a |
| `langchain` | `1.3.14` **`[pending]`** | LangChain utility glue — not used directly in spec-5a (the graph is built on bare `langgraph`, no LangChain chains/agents needed for routing+tool-call); add when a future spec actually needs it | spec-5b+ |

`langgraph`/`langchain-core` added as direct dependencies in spec-5a (`uv add
langgraph langchain-core`), versions confirmed exact match against `uv.lock`.
`langchain` remains transitive-only/`[pending]` — spec-5a's graph never imports it.

> ⚠️ **MCP moves fast — its top-level API changed between major versions.** The
> server class most online examples show (`mcp.server.fastmcp.FastMCP`) does not
> exist in `mcp==2.0.0` — it was renamed/restructured to
> `mcp.server.mcpserver.MCPServer`. Confirmed via `inspect.signature()` against the
> installed package before writing spec-5b's MCP server, not assumed from docs.
> Re-verify against the installed package on any future MCP-touching change.

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `mcp` | `2.0.0` | MCP Python SDK — `mcp.server.mcpserver.MCPServer`, `.tool()`, `.call_tool()`, `.run(transport=...)` | spec-5b |

---

## Retrieval & vector storage

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `qdrant-client` | `1.18.0` | Qdrant Python client — vector search, payload filtering | spec-1b |
| `tenacity` | `9.1.4` | Retry / exponential backoff on Qdrant upsert/delete | spec-1b |
| `grpcio` | `1.83.0` | gRPC transport for Qdrant client (transitive) | spec-1b |
| `protobuf` | `7.35.1` | Protobuf serialization for Qdrant gRPC (transitive) | spec-1b |
| `httpx` | `0.28.1` | Async HTTP for Qdrant REST path (transitive) | spec-1b |
| `httpcore` | `1.0.9` | httpx transport layer (transitive) | spec-1b |
| `h2` | `4.3.0` | HTTP/2 support for httpx (transitive) | spec-1b |
| `anyio` | `4.14.2` | Async concurrency primitives (transitive) | spec-1b |

---

## Ingestion

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `pymupdf` | `1.28.0` | PDF extraction — layout-aware, table detection | spec-1a |
| `tiktoken` | `0.13.0` | Token counting for chunk-size enforcement (`cl100k_base`) | spec-1b |
| `sentence-transformers` | `5.6.0` | Embedding model wrapper (`bge-large-en-v1.5`) | spec-1b |
| `torch` | `2.13.0` | PyTorch — sentence-transformers backend | spec-1b |
| `transformers` | `5.14.1` | HuggingFace model loading (transitive, sentence-transformers) | spec-1b |
| `tokenizers` | `0.22.2` | Fast tokenizers (transitive, transformers) | spec-1b |
| `huggingface-hub` | `1.24.0` | Model download / cache (transitive) | spec-1b |
| `safetensors` | `0.8.0` | Weight serialization format (transitive) | spec-1b |
| `numpy` | `2.5.1` | Numerical arrays — embedding vectors (transitive) | spec-1b |
| `scikit-learn` | `1.9.0` | Similarity utilities (transitive, sentence-transformers) | spec-1b |
| `scipy` | `1.18.0` | Scientific computing (transitive, sentence-transformers) | spec-1b |
| `regex` | `2026.7.19` | Extended regex (transitive, tiktoken/transformers) | spec-1b |
| `requests` | `2.34.2` | HTTP (transitive, huggingface-hub) | spec-1b |
| `tqdm` | `4.69.0` | Progress bars (transitive, transformers) | spec-1b |
| `filelock` | `3.32.0` | Model cache locking (transitive, huggingface-hub) | spec-1b |
| `fsspec` | `2026.6.0` | Filesystem abstraction (transitive) | spec-1b |
| `pyyaml` | `6.0.3` | YAML parsing (transitive, transformers) | spec-1b |
| `packaging` | `26.2` | Version parsing (transitive) | spec-1b |
| `sympy` | `1.14.0` | Symbolic math (transitive, torch) | spec-1b |
| `mpmath` | `1.3.0` | Multiprecision math (transitive, sympy) | spec-1b |
| `networkx` | `3.6.1` | Graph algorithms (transitive, torch) | spec-1b |
| `jinja2` | `3.1.6` | Template engine (transitive, torch) | spec-1b |
| `triton` | `3.7.1` | GPU kernel compilation (transitive, torch — Linux/CUDA only) | spec-1b |

> ⚠️ **`sentence-transformers` pulls the full PyTorch stack (~1.3 GB download on
> first run).** On macOS, torch uses MPS (Apple Silicon GPU) or CPU — no CUDA.
> On Linux CI without GPU, set `PYTORCH_ENABLE_MPS_FALLBACK=1` or use a CPU-only
> torch wheel to avoid CUDA dependency resolution. The `--dry-run` flag on the
> ingest CLI bypasses model loading entirely.

---

## Eval

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `ragas` | `[pending]` | RAG eval — context precision, recall, faithfulness, answer relevance | spec-4 |

---

## Serving

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `fastapi` | `0.141.1` | Async API — validation, health endpoints | spec-7a |
| `uvicorn` | `0.52.3` | ASGI server for FastAPI | spec-7a |
| `httpx` | `0.28.1` | Async test client for the FastAPI app (tests only) | spec-7a |

---

## Observability

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `langfuse` | `4.14.4` | Trace every LLM and vector-store call. OTEL-based SDK (v3/v4 era) — `Langfuse.start_as_current_observation()`, not the older `@observe`-only API most online examples show. Cloud free tier by default; self-hosted stack defined in `docker-compose.langfuse.yml`, not required to run. | spec-7b |

---

## Dev tooling

| Package | Version | Purpose | First spec |
|---|---|---|---|
| `pytest` | `9.1.1` | Test runner | spec-0 |
| `mypy` | `2.3.0` | Static type checking (`strict=false`, `disallow_untyped_defs=true`) | spec-0 |
| `ruff` | `0.15.22` | Lint + import sort (`E`, `F`, `I`, `UP`, `B`, `SIM`) | spec-0 |
| `black` | `26.5.1` | Code formatting (`line-length=100`) | spec-0 |
| `pluggy` | `1.6.0` | pytest plugin system (transitive) | spec-0 |
| `iniconfig` | `2.3.0` | pytest ini parsing (transitive) | spec-0 |
| `pathspec` | `1.1.1` | `.gitignore`-style path matching (transitive, black) | spec-0 |
| `platformdirs` | `4.10.1` | XDG/platform dirs (transitive, black) | spec-0 |
| `click` | `8.4.2` | CLI framework (transitive, black/typer) | spec-0 |
| `rich` | `15.0.0` | Terminal formatting (transitive) | spec-0 |
| `typer` | `0.27.0` | CLI argument parsing (transitive) | spec-0 |
| `mypy-extensions` | `1.1.0` | mypy plugin support (transitive) | spec-0 |

---

## How to update this file

1. Add or upgrade a package in `pyproject.toml`.
2. Run `uv lock` — `uv.lock` is updated.
3. Re-read the version from `uv.lock` and update the row here.
4. Update "Verification date" at the top.

Do not hand-edit version numbers from memory — pull them from `uv.lock`.
