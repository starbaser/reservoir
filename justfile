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

# Run PyNCD's relay and serve the tsncd page its wheel carries on one port. Open
# the printed address in a browser; the page redraws the last model sent.
serve port="3333":
    PORT={{port}} uv run pyncd-server

# Send the echo state network to the relay `just serve` runs, through PyNCD's own
# send route, so the open page draws it.
draw port="3333":
    PORT={{port}} uv run python -c "import asyncio, websocket_transfer.send_morphism as s, websocket_transfer.websockets_transfer as w; from reservoir import echo_state_network as e; asyncio.run(s.send_morphism(e.echo_state_network(), debugBorders=False, legend=True, inspectionBoxes=True, controls=w.PageControls.SHOWN))"
