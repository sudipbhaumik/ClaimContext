.PHONY: install lint format typecheck test check up down generate

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format --check src tests scripts

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint format typecheck test

generate:
	uv run python scripts/generate_corpus.py

up:
	docker compose up -d

down:
	docker compose down
