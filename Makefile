.PHONY: check lint typecheck test build security
export KERAS_BACKEND=jax
check: lint typecheck test security build
lint: contrast
contrast:
	python3 src/charity_model/shell/contrast_check.py src/charity_model/shell/exec-shell.css
	uv run ruff check . && uv run ruff format --check .
typecheck:
	uv run mypy
test:
	uv run pytest --cov --cov-report=term
security:
	uv run bandit -q -r src && uv run pip-audit --skip-editable
build:
	uv run charity-model report --out dist --seed 42
