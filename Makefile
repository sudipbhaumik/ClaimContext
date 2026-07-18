.PHONY: install lint format typecheck test check up down

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests

format:
	uv run black --check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint format typecheck test

up:
	docker compose up -d

down:
	docker compose down
