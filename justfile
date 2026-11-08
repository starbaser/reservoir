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

# One standalone HTML page and one SSA listing per model, under outputs/diagrams/.
# The page opens from the disk with no server and no network.
render:
    uv run python -m reservoir.render --delivery page --output-dir outputs/diagrams

# PNG and PDF figures drawn by PyNCD's headless Chromium, under outputs/figures/.
figures:
    uv run python -m reservoir.render --delivery headless-png --output-dir outputs/figures
    uv run python -m reservoir.render --delivery headless-pdf --output-dir outputs/figures

# Send every model to the tsncd page open on tsncd's relay, which redraws it. Needs
# `npm run server` and `npm run dev` in the tsncd checkout.
send:
    uv run python -m reservoir.render --delivery send --output-dir outputs/diagrams
