# reservoir — Agent Guide

Reservoir computing written as neural circuit diagrams with PyNCD. The project holds an
echo state network as a PyNCD morphism, a plain-Torch reference that runs the same
equations and fits the readout, and a render module that writes the diagram and the
listing. It depends on the private read-only mirrors of PyNCD and tsncd through its
flake inputs. Nothing depends on it.

## Entering the project and first-party stack

On first entering this project—whether to change it, inspect it as a dependency, or make a
cross-project change—read this guide and the relevant documents below for vocabulary, intended
architecture, workflows, and likely implementation paths.

- Documentation is orientation, not ground truth. Verify implementation-sensitive claims in current
  source, generated artifacts, tests, configuration, or runtime behavior. Correct stale living docs
  with the change.
- Sibling documentation does not auto-load. Before relying on a first-party dependency's API or
  behavior, enter that project through its `AGENTS.md` and relevant `docs/`, then verify its current
  implementation.
- First-party dependencies of this project: `pyncd` · `tsncd`. Both are read-only mirrors of
  upstream: only official upstream commits may be merged there, and every adaptation lives here.
  Read `~/dev/projects/pyncd/AGENTS.md` and the vault under `~/dev/projects/pyncd/obsidian/`
  before changing how a model is written. The `using-ncd` skill routes the reading.
- First-party source is editable; fix a generally correct dependency problem at its owning project.

## Conventions

@../eigenpy/CONVENTIONS.md

## Documentation map

Use this map to orient investigation, then verify current behavior at the implementation boundary.

| Question | Read |
| --- | --- |
| "How is the echo state network written as a morphism, and what can PyNCD derive from it?" | `docs/echo-state-network.md` |
| "How is a recurrence over the steps written in PyNCD?" | `~/dev/projects/pyncd/notebooks/caching/mamba/selective_scan.py` and `~/dev/projects/pyncd/obsidian/08-caching/Carrying the State of a Scan Between Passes.md` |
| "How are PyNCD and tsncd pinned and packaged here?" | the `pyncd` and `tsncd` inputs of `flake.nix`, `third_party/pyncd/pyproject.toml`, and the integration reference of the `using-ncd` skill |

Full `docs/` index:

- `docs/echo-state-network.md` — the equations, their spelling in the morphism, the listing, and the limits of what PyNCD derives from the term.

## Environment & Gates

- Enter: `direnv allow`, or `nix develop`. The shell's Python is the uv2nix-built virtual
  environment, with this package editable and PyNCD built from the pinned mirror. A uv-synced
  `.venv` would install the empty packaging shim in place of PyNCD, so `uv` is lock plumbing here
  (`uv add`, `uv lock`) and never the runtime. After a dependency change, `uv lock` and re-enter the
  shell.
- Build: `nix build` (the locked virtual environment) · `nix build .#tsncd` (the tsncd bundle)
- Test: `just test`
- Lint / typecheck: `just lint` · `just typecheck` · `just quality`
- Draw: `just render`, which writes `outputs/diagrams/<slug>.html` and `<slug>.txt`. The shell
  exports `TSNCD_DIST` to the Nix-built bundle, which the page writer reads.

Verification is proportional by default: use focused behavioral tests and touched-scope static
checks while implementing and before focused commits. Run the combined project gates once at a
meaningful integration, landing, or execution-preflight boundary when the accumulated change or
risk warrants them. Do not rerun passing evidence on an unchanged tree; documentation-only changes
receive document-specific checks rather than application suites.

## Project-Specific Rules

- Keep every size symbolic in a model and bind it with `NumericConfig`; `src/reservoir/axes.py`
  declares every axis once, because PyNCD identifies an axis by uid and two axes named alike are
  two axes.
- The PyNCD and tsncd pins are a matched pair. tsncd constructs every term positionally from
  PyNCD's field order, so update both inputs together and rebuild the bundle after any tsncd
  change.
- The reservoir loop is not compiled by `torch_compile`. Numbers come from
  `src/reservoir/reference_torch.py`, and a claim about the term is checked on the term, through its
  domain, codomain, listing and parameters.
- The eigenpy template's `noxfile.py` is left out. An isolated nox session installs the package
  through uv, which cannot obtain PyNCD without the flake.
- Torch comes from the CPU wheel index. Switch the index in `pyproject.toml` if a GPU build is
  wanted.
