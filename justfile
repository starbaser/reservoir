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

# Print the listing, compile the step and the readout with PyNCD's torch_compile, and
# fit the readout on the delay task. `--device cpu` runs it without a GPU.
run *ARGS:
    uv run python -m reservoir.run_echo_state_network {{ARGS}}
