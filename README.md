# reservoir

Reservoir computing written as neural circuit diagrams with [PyNCD](https://github.com/starbaser/pyncd) and drawn with [tsncd](https://github.com/starbaser/tsncd). The
project holds an echo state network as a PyNCD morphism, an evaluator that runs that term on
numbers, a plain-Torch reference that runs the same equations on the CPU or a GPU and fits the
readout by ridge regression, and the tooling to draw the diagram as a standalone page, as headless
PNG and PDF figures, and in the live tsncd page viewer.

`docs/echo-state-network.md` states the equations, their spelling in the morphism, the listing, and
what PyNCD derives from the term. `AGENTS.md` is the guide for agents, and
`skills/using-ncd/` is an agent skill covering PyNCD and tsncd as a whole: syntax,
rulings, API, drawing routes, integration and the theory from the paper.

## Requirements

- Linux with an NVIDIA driver. Torch is the CUDA 13.2 wheel, and the tests assert that CUDA is
  available and that the wheel covers every installed GPU.
- Network access on the first install, for the PyNCD repository, the PyPI and PyTorch wheels,
  tsncd's npm dependencies and Playwright's Chromium.
- Either Nix with flakes, which provides every tool below, or `git`, `uv`, Node 24 with `npm`, and,
  for the page viewer, `just` and `process-compose`.

## Installing with Nix

```bash
git clone <this repository> reservoir && cd reservoir
nix develop          # or: direnv allow
```

The shell's `python` is the Nix-built virtual environment with this package editable, PyNCD built
from the commit `uv.lock` pins, the tsncd bundle built from the commit `flake.lock` pins and
exported as `TSNCD_DIST`, and Playwright's Chromium from nixpkgs. `uv` is lock plumbing only in this
route: run `uv lock` after changing `pyproject.toml`, then re-enter the shell.

## Installing without Nix

```bash
git clone <this repository> reservoir && cd reservoir
uv sync                                # Python 3.14, Torch, PyNCD from its repository
uv run playwright install chromium     # the browser for headless figures and the checks
python3 tools/build_tsncd_bundle.py    # tsncd at its pinned commit, built with npm
```

`uv` installs Python 3.14 if the machine has none. If Chromium refuses to start for want of system
libraries, `uv run playwright install-deps chromium` installs them. The last step clones tsncd under
`third_party/tsncd/`, builds its bundle, and writes `TSNCD_DIST=third_party/tsncd/dist` into `.env`,
which `process-compose` reads. The commands below run through `uv run`, and the entry points of this
package find that bundle on their own when `TSNCD_DIST` is unset.

## Checks

```bash
just quality     # ruff format, ruff check, ty
just test        # the whole suite, including the GPU runs and the browser-driven checks
```

Without `just`, the recipes are one-liners: `uv run pytest`, `uv run ruff check .`,
`uv run ty check src/`, and the commands the `justfile` lists.

The suite checks the term's structure and listing, evaluates the term on numbers against the Torch
reference, runs the reference on the CPU and on every CUDA device, writes the standalone page and
the headless PNG and PDF, and drives the page in Chromium: it switches the form of the diagram and
opens an inspection box.

## Drawing

```bash
just render      # outputs/diagrams/echo-state-network.html and .txt (the SSA listing)
just figures     # outputs/figures/echo-state-network.png and .pdf, drawn headless
```

The page opens from the disk with no server. Its buttons switch between the three forms of the
diagram and the two themes, and the inspection box of the readout opens on the pointer.

## The live page viewer

```bash
just up             # relay on localhost:8765, page on http://127.0.0.1:8780/
just send           # send the model to the open page, which redraws it
just capture        # write what the open page draws to outputs/figures/
just check-viewer   # drive the whole route once in a headless browser and check it
just logs           # the relay's log; `just logs site` for the page server's
just down
```

The relay is PyNCD's websocket relay, and the page is tsncd's bundle served at the root with the
standalone pages under `/diagrams/`. The bundle dials the relay port it was built with, 8765, so one
relay at a time holds that port. `SITE_PORT` changes the page's port.

## The model in brief

```
h_t = (1 - alpha) h_{t-1} + alpha tanh(W_in u_t + W h_{t-1})      h_{-1} = 0
y_t = W_out h_t
```

`W_in` and `W` are `ops.FixedArray`s, the recurrence is a block repeated once per step whose state
and collected states are two stream slots on the tape, tanh and the leak are `ops.Arithmetic` maps,
and `W_out` is the one learned `ops.Linear`. The reference fits `W_out` by ridge regression, which
has no operator in PyNCD.

The method is from Vincent Abbott and Gioele Zardini, *Weaves, Wires, and Morphisms*,
[arXiv:2604.07242](https://arxiv.org/abs/2604.07242), and the upstream repositories are [mit-zardini-lab/pyncd](https://github.com/mit-zardini-lab/pyncd) and
[mit-zardini-lab/tsncd](https://github.com/mit-zardini-lab/tsncd).
