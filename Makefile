.PHONY: install lint format typecheck test check up down generate eval eval-calibrate

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

eval:
	uv run pytest tests/test_eval_harness.py -m eval -v -s

eval-calibrate:
	uv run python -m claimcontext.eval.calibration

up:
	docker compose up -d

down:
	docker compose down
