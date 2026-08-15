.PHONY: install lint format typecheck test check up down generate eval eval-smoke eval-calibrate

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

# Fast regression check for every retrieval/generation-touching change.
# 4-entry subset (1 ANSWER, 2 REFUSE, 1 TIER3) — covers all three ExpectedBehavior
# classes plus the flagship endorsement-override case, at ~1/8th the RAGAS judge
# call volume of the full 11-entry set (only 1 ANSWER entry gets scored).
# Run the full `make eval` before merge or nightly, not on every change.
eval-smoke:
	EVAL_GOLDEN_SET_PATH=data/eval/golden_set_smoke.jsonl \
	EVAL_GOLDEN_SET_VERSION=v1-smoke \
	uv run pytest tests/test_eval_harness.py -m eval -v -s

eval-calibrate:
	uv run python -m claimcontext.eval.calibration

up:
	docker compose up -d

down:
	docker compose down
