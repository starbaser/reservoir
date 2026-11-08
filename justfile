set dotenv-load

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

# tsncd's bundle, built under node_modules/tsncd/dist from the tsncd package.json pins.
# PyNCD's page writer and headless figures read it. Needs `npm ci` first.
build-tsncd:
    npm run tsncd:build

# Build the network, print its listing, check the term against the Torch reference and
# fit the readout on a delay task. `--device cpu` runs it without a GPU.
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

# tsncd's relay on localhost:8765 and tsncd's page on http://127.0.0.1:3000/, or on
# the port TSNCD_PAGE_PORT names.
up:
    process-compose up --detached

down:
    process-compose down

# The log of one viewer process, relay or page, from the file process-compose writes.
# process-compose 1.122.0 answers its own `process logs` command with Not Found over the
# unix socket, so the file is read directly.
logs process='relay' *ARGS:
    tail {{ARGS}} outputs/logs/{{process}}.log

# Send every model to the open page, which redraws it.
send:
    uv run python -m reservoir.render --delivery send --output-dir outputs/diagrams

# Capture what the open page draws for every model, as PNG under outputs/figures/.
capture:
    uv run python -m reservoir.render --delivery capture --output-dir outputs/figures

# Drive the whole viewer once: open tsncd's page headless, send the model through the
# relay, capture the page's drawing, and check the image. Needs `just up`.
check-viewer:
    uv run python -m reservoir.check_viewer --output-dir outputs/figures --page-url "http://127.0.0.1:${TSNCD_PAGE_PORT:-3000}/"
