test:
    uv run pytest

lint:
    uv run ruff check --fix .

fmt:
    uv run ruff format .

typecheck:
    uv run ty check src/

quality:
    uv run ruff format .
    uv run ruff check --fix .
    uv run ty check src/

# One interactive HTML page and one SSA listing per model, under outputs/diagrams/.
render:
    uv run python -m reservoir.render --output-dir outputs/diagrams
