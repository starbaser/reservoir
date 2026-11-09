# reservoir — Agent Guide

Reservoir computing written as neural circuit diagrams with PyNCD. The project holds an
echo state network as a PyNCD morphism, its step and its readout compiled to PyTorch by
PyNCD's `torch_compile` and run on the CPU or a GPU, the ridge-regression fit of the
readout. tsncd draws the diagram. `README.md` is the entry point for a person; this file
is for an agent working in the repository.

## Entering the project

Read this guide and `README.md` first, then the document the map below names for the
question at hand. Documentation is orientation. Verify an implementation-sensitive claim
in the source, the tests, the lock files or the running program, and correct a living
document with the change that invalidates it.

## Dependencies

- [PyNCD](https://github.com/starbaser/pyncd), the Python algebra of neural circuit
  diagrams, a git dependency on its `main` branch, which `uv.lock` resolves. The
  repository is a fork of
  [mit-zardini-lab/pyncd](https://github.com/mit-zardini-lab/pyncd) with reviewed fixes
  and the packaging metadata on top. Its root `CLAUDE.md` and the vault under
  `obsidian/` state how a model is written; read the relevant notes before changing how
  the network is expressed.
- [tsncd](https://github.com/starbaser/tsncd), the TypeScript renderer, a fork of
  [mit-zardini-lab/tsncd](https://github.com/mit-zardini-lab/tsncd). This project holds
  no tsncd checkout. PyNCD's wheel carries tsncd's built page, from the revision PyNCD's
  `pyproject.toml` pins, and `pyncd-server` serves it beside the relay. The flake's
  `tsncd` input is the same revision, built by tsncd's own flake and handed to PyNCD's
  wheel build.
- Both are read-only dependencies. Every adaptation lives in this project. The two are a
  matched pair: tsncd constructs every term positionally from PyNCD's field order, so
  PyNCD pins the tsncd revision, and moving the pair is `uv lock --upgrade-package pyncd`
  followed by `nix flake update tsncd` to the revision PyNCD then pins.

## Conventions

Python 3.14, `uv` for dependencies and the lock, `ruff` for formatting and linting, `ty`
for types, `pytest` for tests. Every function is typed and called with keyword
arguments, internal data is a frozen dataclass, docstrings are Google style, library
code returns data and the command-line entry points print it. Tests assert values and
never skip.

## Documentation map

| Question                                                                                  | Read                                                                                                                                                                                                                                                                                                            |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "How is the echo state network written as a morphism, and what can PyNCD derive from it?" | `docs/echo-state-network.md`                                                                                                                                                                                                                                                                                    |
| "How is a recurrence over the steps written in PyNCD?"                                    | PyNCD's [`selective_scan.py`](https://github.com/starbaser/pyncd/blob/main/notebooks/caching/mamba/selective_scan.py) and its note [Carrying the State of a Scan Between Passes](https://github.com/starbaser/pyncd/blob/main/obsidian/08-caching/Carrying%20the%20State%20of%20a%20Scan%20Between%20Passes.md) |
| "How is the network run in Torch through `torch_compile`?"                                | `src/reservoir/run_in_torch.py` and `docs/echo-state-network.md`                                                                                                                                                                                                                                                |
| "Where do PyNCD and tsncd come from?"                                                     | the `pyncd` entry of `pyproject.toml`, which names the `main` branch, `uv.lock`, and the `tsncd` input of `flake.nix`                                                                                                                                                                                          |
| "How do I see the diagram?"                                                               | `just serve` and `just draw` in the `justfile`, and `README.md`                                                                                                                                                                                                                                                 |
| "How do I run the network?"                                                               | `README.md`, the `justfile`, `src/reservoir/run_echo_state_network.py`                                                                                                                                                                                                                                          |

Full `docs/` index:

- `docs/echo-state-network.md` — the equations, their spelling in the morphism, the
  listing, and the limits of what PyNCD derives from the term.

`skills/using-ncd/` is an agent skill for writing, reading, compiling and drawing models
with PyNCD and tsncd: the construction syntax, the rulings on how a model is written,
the API, the rendering routes, the integration recipes and the theory from the paper,
with an executable example under `scripts/`. Read its `SKILL.md` before changing the
model. An agent runtime that discovers skills from a directory of its own can be pointed
at, or given a copy of, that folder.

## Environment & Gates

Two installation routes exist and `README.md` states both.

- With Nix: `nix develop`, or `direnv allow`. The shell's Python is the uv2nix-built
  virtual environment, with this package editable and PyNCD built from the commit
  `uv.lock` resolved from `main`. `uv` is lock plumbing there (`uv add`, `uv lock`) and
  never the runtime; after a dependency change, `uv lock` and re-enter the shell.
  PyNCD's wheel build in the flake takes the tsncd page from the `tsncd` input, so the
  shell holds no Node.
- Without Nix: `uv sync` installs everything including PyNCD from its repository. PyNCD's
  wheel build clones tsncd and runs npm, so Node 24 is needed once, at install.
- Test: `just test`. Lint and types: `just quality`. Run: `just run`.

Verification is proportional: run the focused tests and the static checks on the files
touched while working, and the whole suite at an integration or landing boundary. Do not
rerun passing evidence on an unchanged tree.

## Project-Specific Rules

- Keep every size symbolic in a model and bind it with `NumericConfig`;
  `src/reservoir/axes.py` declares every axis once, because PyNCD identifies an axis by
  uid and two axes named alike are two axes.
- The network runs in Torch through PyNCD's `torch_compile` alone, by the route of
  PyNCD's tutorial `check_gradients_in_torch.py`, in `src/reservoir/run_in_torch.py`.
  `torch_compile` compiles a repeated block only for an integer repetition and has no
  rule for a tape slot, so the compiled step is called once per step from Python. Write
  no evaluator of the term in this project; a claim about the term's shape is checked on
  the term, through its domain, codomain, listing and parameters.
- tsncd draws the diagram. This project holds no drawing code: no standalone page, no
  headless figure, no browser automation, no wrapper around tsncd or around PyNCD's
  relay. `just serve` runs PyNCD's `pyncd-server` and `just draw` calls PyNCD's
  `send_morphism` on the network, and those two recipes are the whole of it.
- Torch comes from PyPI. On Linux the flake gives every CUDA library the driver runpath,
  so `libcuda.so.1` comes from the host driver. The tests run the compiled step on the
  CPU and on every CUDA device present.
- There is no `noxfile.py`. The isolated sessions it would run add nothing over
  `just test` in either installation route.
