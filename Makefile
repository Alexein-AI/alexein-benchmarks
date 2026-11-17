.PHONY: check lint types test public readme

check: lint types test public

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy

test:
	uv run pytest -q

# Public text says what, never how; the README's table and charts are the results files'.
public:
	uv run python scripts/no_internals.py
	uv run python scripts/readme_table.py --check
	uv run python scripts/readme_charts.py --check

readme:
	uv run python scripts/readme_table.py
	uv run python scripts/readme_charts.py
