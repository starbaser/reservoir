# reservoir — Agent Guide

Reservoir computing written as neural circuit diagrams with PyNCD. The project holds an echo state
network as a PyNCD morphism, an evaluator that runs that term on numbers, a plain-Torch reference
that runs the same equations on the CPU or a GPU and fits the readout by ridge regression, and the
tooling to draw the diagram as a standalone page, as headless PNG and PDF figures, and in the live
tsncd page viewer. `README.md` is the entry point for a person; this file is for an agent working in
the repository.

## Entering the project

Read this guide and `README.md` first, then the document the map below names for the question at
hand. Documentation is orientation. Verify an implementation-sensitive claim in the source, the
tests, the lock files or the running program, and correct a living document with the change that
invalidates it.

## Dependencies

- [PyNCD](https://github.com/starbaser/pyncd), the Python algebra of neural circuit diagrams, a git dependency in `uv.lock` pinned to
  [`d056b47`](https://github.com/starbaser/pyncd/tree/d056b4755fad127e48063739126fc2f19192650b). The repository is a fork of [mit-zardini-lab/pyncd](https://github.com/mit-zardini-lab/pyncd) with reviewed fixes and the
  packaging metadata on top. Its root `CLAUDE.md` and the vault under `obsidian/` state how a model
  is written; read the relevant notes before changing how the network is expressed.
- [tsncd](https://github.com/starbaser/tsncd), the TypeScript renderer, a flake input pinned by `flake.lock` to [`b8649a6`](https://github.com/starbaser/tsncd/tree/b8649a6492c15244cf2b0be49e0acf2e2abb32d7), a fork of
  [mit-zardini-lab/tsncd](https://github.com/mit-zardini-lab/tsncd).
- Both are read-only dependencies. Every adaptation lives in this project. The two are a matched
  pair: tsncd constructs every term positionally from PyNCD's field order, so move both pins
  together and rebuild the bundle after any tsncd change.

## Conventions

Python 3.14, `uv` for dependencies and the lock, `ruff` for formatting and linting, `ty` for types,
`pytest` for tests. Every function is typed and called with keyword arguments, internal data is a
frozen dataclass, docstrings are Google style, library code returns data and the command-line entry
points print it. Tests assert values and never skip.

## Documentation map

| Question | Read |
| --- | --- |
| "How is the echo state network written as a morphism, and what can PyNCD derive from it?" | `docs/echo-state-network.md` |
| "How is a recurrence over the steps written in PyNCD?" | PyNCD's [`selective_scan.py`](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/notebooks/caching/mamba/selective_scan.py) and its note [Carrying the State of a Scan Between Passes](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/obsidian/08-caching/Carrying%20the%20State%20of%20a%20Scan%20Between%20Passes.md) |
| "How is the term evaluated on numbers, and why not with `torch_compile`?" | `src/reservoir/evaluate_term.py` and `docs/echo-state-network.md` |
| "How are PyNCD and tsncd pinned?" | the `pyncd` entry of `pyproject.toml` and `uv.lock`, the `tsncd` input of `flake.nix`, and `tools/build_tsncd_bundle.py` |
| "How do I draw, start the page viewer, or check it end to end?" | `README.md`, the `justfile`, `process-compose.yml`, `src/reservoir/render.py` and `src/reservoir/check_viewer.py` |

Full `docs/` index:

- `docs/echo-state-network.md` — the equations, their spelling in the morphism, the listing, and the
  limits of what PyNCD derives from the term.

`skills/using-ncd/` is an agent skill for writing, reading, compiling and drawing models
with PyNCD and tsncd: the construction syntax, the rulings on how a model is written,
the API at the pinned revision, the rendering routes, the integration recipes and the
theory from the paper, with an executable example under `scripts/`. Read its `SKILL.md`
before changing the model. An agent runtime that discovers skills from a directory of
its own can be pointed at, or given a copy of, that folder.

## Environment & Gates

Two installation routes exist and `README.md` states both.

- With Nix: `nix develop`, or `direnv allow`. The shell's Python is the uv2nix-built virtual
  environment, with this package editable and PyNCD built from the pinned commit. `uv` is lock
  plumbing there (`uv add`, `uv lock`) and never the runtime; after a dependency change, `uv lock`
  and re-enter the shell. The shell exports `TSNCD_DIST`, the Nix-built bundle, and
  `PLAYWRIGHT_BROWSERS_PATH`, the nixpkgs browsers of the pinned Playwright.
- Without Nix: `uv sync` installs everything including PyNCD from its repository,
  `uv run playwright install chromium` fetches the browser, and
  `python3 tools/build_tsncd_bundle.py` builds tsncd under `third_party/tsncd/` and writes
  `TSNCD_DIST` to `.env`.
- Test: `just test`. Lint and types: `just quality`. Draw: `just render`, `just figures`. Viewer:
  `just up`, `just send`, `just capture`, `just check-viewer`, `just logs`, `just down`.

Verification is proportional: run the focused tests and the static checks on the files touched while
working, and the whole suite at an integration or landing boundary. Do not rerun passing evidence on
an unchanged tree.

## Project-Specific Rules

- Keep every size symbolic in a model and bind it with `NumericConfig`; `src/reservoir/axes.py`
  declares every axis once, because PyNCD identifies an axis by uid and two axes named alike are two
  axes.
- The reservoir loop is not compiled by `torch_compile`. `src/reservoir/evaluate_term.py` evaluates
  the term on tensors and the tests compare it with `src/reservoir/reference_torch.py`; a claim
  about the term's shape is checked on the term, through its domain, codomain, listing and
  parameters.
- The tsncd bundle dials the relay port compiled into it, 8765, so one relay holds that port at a
  time. `just check-viewer` fails rather than falling back to a headless browser when the page does
  not answer, and `render.Delivery.CAPTURE` goes through the relay alone for the same reason.
- Torch is the CUDA 13.2 wheel from the `pytorch-cu132` index. The flake gives every CUDA wheel the
  driver runpath, so `libcuda.so.1` comes from the host driver. The tests run the reference on the
  CPU and on every CUDA device present and assert that CUDA is available.
- `playwright` is pinned to the version of nixpkgs' `playwright-driver` in the locked nixpkgs, and
  the two move together.
- There is no `noxfile.py`. The isolated sessions it would run add nothing over `just test` in
  either installation route.
