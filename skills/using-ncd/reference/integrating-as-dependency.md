# Adding PyNCD and tsncd to a project

How a project consumes [PyNCD](https://github.com/starbaser/pyncd) and
[tsncd](https://github.com/starbaser/tsncd), the forks of
[mit-zardini-lab/pyncd](https://github.com/mit-zardini-lab/pyncd) and
[mit-zardini-lab/tsncd](https://github.com/mit-zardini-lab/tsncd) that carry a few
reviewed fixes and PyNCD's packaging. Check the current revisions before applying a
recipe; a consumer's lock records the actual source used. The reservoir repository that
holds this skill is the worked example of every route below.

## Contents

- [What the two repositories ship](#what-the-two-repositories-ship)
- [Pins](#pins)
- [Route 1: plain uv](#route-1-plain-uv)
- [Route 2: Nix flake with uv2nix](#route-2-nix-flake-with-uv2nix)
- [The tsncd bundle](#the-tsncd-bundle)
- [Serving pages and the relay](#serving-pages-and-the-relay)
- [Training a model that holds parameter arrays](#training-a-model-that-holds-parameter-arrays)
- [What a consumer does not get from the wheel](#what-a-consumer-does-not-get-from-the-wheel)
- [Caveats to carry into the project guide](#caveats-to-carry-into-the-project-guide)
- [Integration checklist](#integration-checklist)

## What the two repositories ship

- **pyncd** is a flat repository of top-level Python packages (`data_structure`,
  `construction_helpers`, `algebra`, `graphs`, `para`, `torch_compile`,
  `websocket_transfer`, ...). Upstream ships no packaging metadata; the fork adds a
  `pyproject.toml` (hatchling) whose wheel holds the eighteen top-level packages and
  `notebooks/display`, the notebook drawing helper, and leaves the other notebooks, the
  vault and `run_server.py` out. Requirements: Python 3.13 or later (3.13 generics with
  defaults are used throughout; observed working on 3.14). The core needs nothing beyond
  the standard library. The extras are `torch` (`torch`, `einops`, `numpy`) for
  `torch_compile`, `diagrams` (`websockets`, `ipykernel`, `pandas`) for the relay, the
  page writer and `notebooks/display`, and `headless` (`playwright`) for the headless
  capture, which also needs a Chromium: `playwright install chromium`, or on NixOS the
  nixpkgs `playwright-driver.browsers` through `PLAYWRIGHT_BROWSERS_PATH`.
- **tsncd** is an npm project (`package.json` with webpack 5, TypeScript 5.9, `ws`,
  KaTeX, MathJax, `html-to-image`). `npm ci && npm run build` writes `dist/` holding
  `index.html`, `assets/main.*.js` and
  `json_files/deepseek_v41_flash_text_only_quantised.json`. `npm run server` runs
  `node src/run_server.ts` directly, so the Node version must strip types (Node 24
  does).
- The two are a matched pair. tsncd constructs every term positionally from PyNCD's
  field order, so mixing revisions silently swaps members or refuses whole terms.
- Neither repository carries a LICENSE file at these revisions; the papers are CC BY
  4.0. Record the question before redistributing.

## Sources

Both come from the `main` branch of the forks, `https://github.com/starbaser/pyncd.git`
and `https://github.com/starbaser/tsncd.git`, which carry the reviewed fixes and the
packaging metadata over [mit-zardini-lab/pyncd](https://github.com/mit-zardini-lab/pyncd)
and [mit-zardini-lab/tsncd](https://github.com/mit-zardini-lab/tsncd). A consumer's lock
records the commit it resolved; the manifest names the branch.

## Route 1: plain uv

PyNCD is a git dependency pinned to one commit. `uv` builds the wheel from the
repository's own `pyproject.toml`.

```toml
[project]
requires-python = ">= 3.13"
dependencies = ["einops>=0.8.2", "numpy>=2", "pyncd[diagrams,torch]", "torch>=2.5"]

[tool.uv.sources]
pyncd = { git = "https://github.com/starbaser/pyncd.git", branch = "main" }
```

Then `uv lock && uv sync`. For the headless capture add `playwright` and run
`uv run playwright install chromium`. The tsncd bundle is built separately, as the
section below says, and reaches every process that writes a page through `TSNCD_DIST`:
export it from the shell, put `TSNCD_DIST=/path/to/tsncd/dist` in a `.env` file and run
with `uv run --env-file .env python ...`, or pass `dist=` explicitly to
`save_standalone_page`, `save_figures` and `HeadlessRenderer`.

## Route 2: Nix flake with uv2nix

The same `pyproject.toml` and `uv.lock` as Route 1. uv2nix fetches the git dependency
from the lock; a git dependency is built from source, so its build system is named in an
override. The flake also builds the tsncd bundle and exports it to the shell.

```nix
inputs.tsncd = { url = "github:starbaser/tsncd"; flake = false; };

# tsncd's production bundle, read through TSNCD_DIST
mkTsncdBundle = pkgs:
  let manifest = builtins.fromJSON (builtins.readFile "${tsncd}/package.json"); in
  pkgs.buildNpmPackage {
    pname = manifest.name; inherit (manifest) version;
    src = tsncd; nodejs = pkgs.nodejs_24;
    npmDepsHash = "sha256-bH3DY2wG6U1S4T4E2Ok+UYflahRUNQnzGw15A+7+O0E=";
    installPhase = ''
      runHook preInstall
      mkdir -p "$out"; cp -r dist/. "$out/"
      runHook postInstall
    '';
  };

# in the pyproject overlay overrides
pyncd = prev.pyncd.overrideAttrs (old: {
  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem { hatchling = [ ]; };
});

# in the dev shell
env = {
  UV_NO_SYNC = "1";
  UV_PROJECT_ENVIRONMENT = "${devEnv}";
  UV_PYTHON = "${devEnv}/bin/python";
  UV_PYTHON_DOWNLOADS = "never";
  TSNCD_DIST = "${tsncdBundle}";
  PLAYWRIGHT_BROWSERS_PATH = "${pkgs.playwright-driver.browsers}";
  PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = "1";
};
```

Expose `packages.tsncd = mkTsncdBundle pkgs` and `checks.tsncd` so `nix build .#tsncd`
and `nix flake check` build the bundle. A CUDA Torch wheel needs the usual autoPatchelf
and driver-runpath overrides beside this; they are independent of PyNCD. The reservoir
repository's `flake.nix` is a complete instance of this route.

## The tsncd bundle

With Nix, Route 2's `mkTsncdBundle` builds `dist/` reproducibly. Without it:

```bash
git clone https://github.com/starbaser/tsncd.git && cd tsncd
npm ci && npm run build            # writes dist/
export TSNCD_DIST="$PWD/dist"
```

The reservoir repository holds this checkout as the git submodule `tsncd/`.
`websocket_transfer.headless.find_dist` resolves an explicit `dist=` argument, then
`$TSNCD_DIST`, then a `tsncd/dist` or `tsncd-public/dist` sibling of the pyncd package
directory. An installed pyncd has no such sibling, so every consumer sets `TSNCD_DIST`.
Rebuild the bundle after any tsncd change; headless captures and standalone pages render
whatever was last built. The bundle dials the relay port compiled into it at build time,
`PORT` with the default 8765.

## Running the live page viewer

The viewer is tsncd's, run in its checkout as its README says: `npm run server` is the
relay on localhost:8765 and `npm run dev` is the page, which connects to the relay.
`notebook_diagrams.show_diagram` in `BROWSER` mode, or
`websocket_transfer.send_morphism.send_morphism(term)`, sends a term to the relay and
every open page redraws it. pyncd's `run_server.py` is the same relay with the same
messages; one relay at a time holds the port, and the bundle dials the port compiled
into it.

## Training a model that holds parameter arrays

`torch_compile.ConstructedModule.construct` directly compiles a `Linear` with inputs and
holds its weight as an `nn.Parameter`. A zero-input `Linear`, used to author a parameter
array, needs PyNCD's tape pipeline. The
[gradient tutorial](https://github.com/starbaser/pyncd/blob/main/notebooks/website/tutorial/check_gradients_in_torch.py)
uses `grab_parameters`, binds sizes, removes grouping blocks, calls `detape`, then
compiles the resulting pure expression. Follow that pipeline when the caller owns named
weight tensors and supplies them after the model inputs:

```python
import algebra.discovering_broadcasts as discovering_broadcasts
import para.algebra.detape as detape
import para.processing.show_grabbed_parameters as show_grabbed_parameters

grabbed = show_grabbed_parameters.grab_parameters(
    model
)  # every weight becomes a Grab of a named slot
config = generate_config.NumericConfig.template(grabbed)
config.assign_values(**sizes)
pure, tape = detape.detape(
    discovering_broadcasts.remove_grouping_blocks(config.apply_context(grabbed))
)
module = torch_compile.ConstructedModule.construct(
    pure
)  # parametrised Linears contract by position
names = [name for name, is_grab in tape if is_grab]  # slot names, in operand order
weights = {name: torch.randn(shape).requires_grad_() for name, shape in ...}
(y,) = module(*data, *(weights[name] for name in names))  # the weights trail the data
```

The code above is a pipeline sketch: derive `shape` from the sized pure domain, and
match the callable's return structure to the particular expression. The caller hands
these external weight tensors to an optimiser. The tape names and order come from
`detape`.

For a boxed model with an internally published activation, the
[executable tutorial example](../scripts/tutorial_model.py) first opens `ParaWrap` with
`.to_base()` and boxes with their operator's `.expand()`, using PyNCD's `Endofunctor` to
traverse the expression. After removing grouping blocks, `tie_tapes` connects the
internal publisher to its readers; parameter extraction and `detape` then expose the
external weights. The original expression is retained for `show_diagram`. This example
does not establish support for arbitrary nested tape boxes or repeated tape operations.

`detape` reaches top-level tape operations only. Remove ordinary grouping blocks first;
it cannot detape a grab inside a repeated block. A directly compiled model whose
parameters all ride input-bearing `Linear`s may repeat a block, as shown by
`torch_compile.ConstructedBlock`.

Fix `78c7a89` labels the target positions separately, so a grabbed `Linear((m,), (m,))`
contracts as a full square map; the tutorial's `compile_pass` needs no
`linear_expansion.expand_parametrised_linears`. See
[the positional linear validator](https://github.com/starbaser/pyncd/blob/main/para/validate_linear_positions.py).

A consumer can add local numerical constructors for operators that have no PyTorch
constructor. Both `ConstructedModule.operation_registry` and `functions_registry` are
process-wide, and a later registration replaces an earlier one for that operator type.
Check both registries before adding an extension; keep numerical implementation local to
the consumer.

The compiler has no constructor for `ops.BlockOperator`, a zero-input `Linear`, an
affine `View`, a bare Para tape, an `ops.ConstantOp`, an `ops.FixedArray` or an
`aops.CovariantView`, and it runs a repeated block only when its repetition is an
integer. It compiles a `View` through a rearrangement alone; for an affine window the
consumer supplies a precomputed neighbourhood table or registers a local constructor for
the required read. A loop over a tape cannot be compiled at all; the reservoir
repository compiles the body of its loop, one step of the recurrence, and calls it once
per step from Python, carrying the state between the calls. A directly compiled
`Normalize` does not learn its gain as a parameter; use `gain=False` when that matches
the authored model, or grab the gain and use the tape pipeline.

## What a consumer does not get from the wheel

The rest of `notebooks/` (the notebooks, the website tutorial modules such as
`check_gradients_in_torch.py`, the SOTA model packages), `example_notebooks/`,
`validations/`, `validate_repository.py`, `run_server.py`, `obsidian/` and `_guide/`.
The wheel adds `notebooks/display` alone, which is what drawing needs. Read upstream's
notebooks and vault in the repository.

## Caveats to carry into the project guide

- The package names are generic (`display`, `graphs`, `solver`, `utilities`, `algebra`).
  Keep them out of environments that hold other packages of those names.
- Literal mismatched extents may be merged rather than rejected by composition. Keep
  model sizes symbolic, assign them with `NumericConfig`, and assert the sized domain
  and codomain. `NumericConfig.assign_values` does not reject an unknown name.
- `ops.Einops.template('a b -> a b')` does not reliably hold the entire rank-two array
  at this revision. Use `cat.ProdObject((array,)).identity()` for a held wire.
- `cat.Composed` and `cat.ProductOfMorphisms` check nothing. Assert `dom()` and `cod()`.
- Equal axis names are not equal axes; equal `nm.FreeNumeric.named` names are the same
  symbol. Alignment is positional.
- A term with no matching tsncd class cannot be drawn. Check the mirror's term classes
  for every new operator before promising a diagram; compiler support is a separate
  question.
- tsncd's `defaultRenderHandlerSettings.debugBorders` is `true`; pyncd's
  `display_settings` sends `false`. A custom sender must send it too.
- `show_diagram` falls back to its own headless browser when the open page does not
  answer a capture, so a capture says nothing about whether a page is open.
- The relay port 8765 is held by exactly one process; the page port is 3000 under
  `npm run dev`.

## Integration checklist

```text
- [ ] pin PyNCD and tsncd to a matched pair; record both revisions in the project guide
- [ ] uv source: pyncd as a git dependency at one commit; with Nix, the hatchling override for it
- [ ] flake: tsncd input + buildNpmPackage with npmDepsHash; dev shell exports TSNCD_DIST and PLAYWRIGHT_BROWSERS_PATH
- [ ] without Nix: the tsncd checkout at the pinned commit, built with `npm install && npm run build`, named by TSNCD_DIST
- [ ] `python -c "import construction_helpers, websocket_transfer.headless as h; print(h.find_dist())"` succeeds
- [ ] extras: pyncd[torch] for compilation, pyncd[diagrams] for the relay and pages, pyncd[headless] for headless capture
- [ ] a model with parameter arrays follows `grab_parameters`, size binding, grouping-block removal, `detape` and `ConstructedModule.construct`; a local Torch constructor is registered only after checking both registries
- [ ] a render module draws every model with notebook_diagrams.show_diagram (HTML, INTERACTIVE) and writes its listing
- [ ] executable models compare outputs and gradients with a plain-Torch control; a loop the compiler cannot run has its body compiled and is iterated from Python
- [ ] only the render module imports notebooks/display; nothing imports run_server.py
```
