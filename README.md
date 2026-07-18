# ClaimContext

ClaimContext is a grounded, cited Q&A assistant for insurance claim adjusters. An adjuster queries across policy documents, claim notes, and claim documents from one place and gets answers with citations to the source text. It **informs, it does not decide**: it retrieves, summarizes, and cites; the human adjuster makes the claim decision. All data is synthetic.

---

## Prerequisites

- macOS (tested on MacBook Pro)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (dependency manager)
- Docker + Docker Compose (for Qdrant)

---

## Setup

```bash
# Install dependencies (main + dev)
make install

# Copy the example env file and edit as needed
cp .env.example .env
```

---

## Running checks

```bash
make lint        # ruff lint
make format      # black + ruff format check
make typecheck   # mypy
make test        # pytest
make check       # all of the above
```

---

## Qdrant (vector store)

```bash
make up    # docker compose up -d (starts Qdrant on :6333)
make down  # docker compose down
```

Qdrant dashboard: http://localhost:6333/dashboard

---

## Health check

```bash
python -m claimcontext.health
```

Prints app name, environment, and resolved Qdrant URL from config.

---

## Design decisions

| Decision | Rationale |
|---|---|
| **uv over poetry** | Faster resolution, simpler lockfile, first-class workspace support. |
| **Qdrant** | Self-hosted via Docker, supports hybrid dense+sparse, named-volume persistence, no cloud cost during dev. |
| **Config-not-constants** | All tunables (chunk size, top-k, model names, timeouts) live in `config.py` via Pydantic `BaseSettings`, sourced from `.env`. No magic numbers in business logic. |
| **Core deps only per spec** | ML/retrieval/serving dependencies are added in the spec that first needs them — keeps the initial environment small and deterministic. |
