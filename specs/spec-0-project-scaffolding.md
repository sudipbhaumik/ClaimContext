# spec-0 — Project Scaffolding

> Read `CLAUDE.md` before implementing. This spec is the foundation: no RAG logic yet.
> Target machine: **macOS (MacBook Pro)**. Dependency manager: **uv**.

---

## Scope (what this iteration delivers)

A runnable, committed project skeleton for ClaimContext: verified local toolchain, Python 3.11+ environment managed by `uv`, package/folder structure, externalized configuration (no secrets in code), lint/type/test tooling wired and passing, a Qdrant service running via `docker-compose`, a trivial health check proving the app starts, and an initial local git commit ready for the remote you will create.

**Nothing in this spec touches ingestion, retrieval, agents, or models.** It exists so every later spec runs on stable ground.

---

## In scope

**A. Local toolchain verification and install (macOS)**
- **Verify before installing.** For each tool, check whether it already exists and at what version; only install if missing or below the required floor. Report a table of `tool → found version → action taken` before making changes.
- Tools required: `git`, `python` (3.11+), `uv`, `docker` (+ `docker compose`).
- Homebrew is the install path if a tool is missing; verify `brew` itself first.
- Do **not** silently upgrade anything already satisfying the floor.

**B. Python environment**
- `uv` project initialized; Python 3.11+ pinned (`.python-version`).
- Virtual environment created and activated via `uv`.
- `pyproject.toml` with project metadata and dependency groups (`main`, `dev`).
- `uv.lock` committed.

**C. Core dependencies only** (per DEPENDENCIES-LOCKED.md — exact pins, no re-research)
- Main: `pydantic==2.13.4`, `pydantic-settings==2.14.2`, `python-dotenv==1.2.2`
- Dev: `pytest==9.1.1`, `mypy==2.3.0`, `ruff==0.15.22`, `black==26.5.1`
- **Deferred to later specs:** all ML/retrieval/serving deps (`sentence-transformers`, `qdrant-client`, `langgraph`, `fastapi`, `ragas`, …). Each spec installs what it needs.
- Note: `sentence-transformers` (spec-1b) will pull a large torch stack; on macOS this is CPU/MPS, no CUDA.

**D. Folder structure**
```
claimcontext/
├── src/claimcontext/
│   ├── __init__.py
│   ├── config.py            # Pydantic BaseSettings — ALL config lives here
│   ├── models/__init__.py   # Pydantic domain models (populated spec-1a)
│   ├── ingestion/__init__.py
│   ├── retrieval/__init__.py
│   ├── guardrails/__init__.py
│   ├── agent/__init__.py
│   ├── evaluation/__init__.py
│   ├── api/__init__.py
│   └── observability/__init__.py
├── prompts/                 # versioned prompt files (§2A.3) — empty for now
├── data/documents/
│   ├── policies/
│   ├── claim-notes/
│   └── claim-docs/
├── specs/                   # spec + handoff files
├── tests/
│   └── test_config.py
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example             # committed
├── .env                     # gitignored
├── .gitignore
├── Makefile
└── README.md
```

**E. Configuration (§2A.1, §2A.2 — non-negotiable)**
- `config.py` using Pydantic `BaseSettings`, loading from `.env`.
- **No secrets in code. No magic numbers in code.** Every tunable is a config field with a documented default.
- Placeholder settings groups (values unused until later specs, but the shape exists now):
  - App: `app_name`, `log_level`, `environment`
  - Paths: `documents_dir`, `prompts_dir`
  - Qdrant: `qdrant_url`, `qdrant_collection`
  - LLM: `llm_provider` (`ollama` | `anthropic` | `openai`), `llm_model`, `llm_timeout_seconds`
  - Embedding: `embedding_model`, `chunker_version`  ← index versioning per §2A.4
  - Retrieval: `top_k`, `rrf_k`, `rerank_top_n`, `refuse_threshold`
  - Chunking: `chunk_size`, `chunk_overlap`
- Secrets (`anthropic_api_key`, `openai_api_key`) declared as optional and read from env only.
- `.env.example` documents every key with safe placeholder values; real `.env` gitignored.

**F. Tooling config** (in `pyproject.toml` where possible)
- `ruff` (lint + import sort), `black` (format), `mypy` (strict-ish: `disallow_untyped_defs`), `pytest` (testpaths, `src` layout aware).
- All four must run clean on the scaffold.

**G. Docker services**
- `docker-compose.yml` with **Qdrant only** (`qdrant/qdrant`), port-mapped, named volume for persistence.
- App is NOT containerized in this spec (that's spec-8a).
- `docker compose up -d` brings Qdrant up; verify it responds.

**H. Makefile** — thin convenience targets: `install`, `lint`, `format`, `typecheck`, `test`, `check` (all of them), `up`, `down`.

**I. Health check**
- One trivial module-level entry (e.g. `python -m claimcontext.health` or a `health()` function) that loads config and prints app name + environment + resolved Qdrant URL. Proves config loads and the package imports.
- `tests/test_config.py`: asserts config loads with defaults and that a missing optional secret does not crash.

**J. README (initial)**
- What ClaimContext is (2–3 sentences, informs-not-decides).
- Prerequisites, setup steps, how to run checks, how to start/stop Qdrant.
- A "Design decisions" section seeded with: uv over poetry, Qdrant, config-not-constants, core-deps-only-per-spec.

**K. Git**
- `.gitignore` covering `.venv/`, `.env`, `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `data/documents/*` (keep `.gitkeep`), Qdrant volume.
- `git init`, initial commit on a branch.
- **Do NOT create or add a remote** — I will create the GitHub repo and push myself. Leave the working tree clean and tell me the exact commands to add the remote and push.

---

## Out of scope (do NOT build in this iteration)

- Any ingestion, chunking, embedding, retrieval, agent, guardrail, eval, or API logic.
- Installing ML/retrieval/serving dependencies.
- Dockerizing the application (spec-8a).
- CI/CD workflows (spec-8b).
- Creating the GitHub remote or pushing.
- Synthetic document generation (belongs with spec-1a).
- Any prompt content (directory only).

---

## Interfaces / contracts this spec exposes

- `claimcontext.config.Settings` — the single typed config object every later spec imports. No module may read `os.environ` directly.
- `claimcontext.config.get_settings()` — cached accessor.
- Package namespaces (`ingestion`, `retrieval`, `guardrails`, `agent`, `evaluation`, `api`, `observability`, `models`) exist and are importable.
- `data/documents/{policies,claim-notes,claim-docs}/` exist as the ingestion source tree (spec-1a consumes this).

---

## Authoring split (Python mastery mode, §3)

- **I author (core logic):** `config.py` (the `Settings` model — Pydantic `BaseSettings`, field types, validators, defaults) and `tests/test_config.py`. This is my first Pydantic v2 exercise; explain the options first, then let me write it, then critique — do not write it for me.
- **Claude writes (plumbing/infra):** `pyproject.toml`, tool configs, `docker-compose.yml`, `Makefile`, `.gitignore`, `.env.example`, package `__init__.py` files, README skeleton, the toolchain verification script/commands.

---

## Build order within the spec (stop at each checkpoint)

1. **Verify toolchain** — report the `tool → version → action` table. Install only what's missing. **Checkpoint: I confirm before any install.**
2. **uv project init** — `pyproject.toml`, `.python-version`, venv. Explain uv's model briefly first.
3. **Folder structure + `__init__.py` files + `.gitkeep`s.**
4. **Config** — explain Pydantic `BaseSettings` options and tradeoffs → **I write `config.py`** → Claude critiques → I revise.
5. **`.env.example` + `.gitignore`** — verify no secret can be committed.
6. **Tooling config** (ruff/black/mypy/pytest) → run all → clean.
7. **`docker-compose.yml` (Qdrant)** → `docker compose up -d` → verify reachable.
8. **Health check + `test_config.py`** (I write the test) → passes.
9. **Makefile + README.**
10. **`git init` + initial commit**; print the remote/push commands for me to run.

---

## Proof (non-toy — required before this spec is done)

1. **Clean-room check:** `make check` (lint + format-check + typecheck + test) passes with zero errors on a fresh clone.
2. **Config proof:** running the health check prints app name, environment, and Qdrant URL sourced from `.env` — and **removing `.env` still starts** using documented defaults (no crash, no hardcoded secret).
3. **Secret-safety proof:** `git status` after adding a real `.env` shows it as ignored — it cannot be accidentally committed.
4. **Service proof:** `docker compose up -d` → Qdrant responds on its configured port; `make down` stops it.

---

## Definition of done

- [ ] Toolchain verified; only missing tools installed; table reported.
- [ ] `uv` project with Python 3.11+ pinned; `uv.lock` committed.
- [ ] Core deps only, exact pins from DEPENDENCIES-LOCKED.md.
- [ ] Folder structure created; all packages importable.
- [ ] `Settings` loads from `.env`; **no secrets, no magic numbers in code**.
- [ ] `.env.example` committed; `.env` gitignored and proven ignored.
- [ ] ruff + black + mypy + pytest all clean.
- [ ] Qdrant up via docker-compose and reachable.
- [ ] Health check runs; `test_config.py` passes.
- [ ] README + Makefile present.
- [ ] All four proofs demonstrated.
- [ ] Local git commit made; remote/push commands provided (remote NOT added by Claude).
- [ ] **`specs/spec-0-handoff.md` written** per §6A (interfaces, config keys, decisions, deferrals, gaps, proof status).
