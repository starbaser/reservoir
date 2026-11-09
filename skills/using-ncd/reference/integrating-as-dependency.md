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
  `pyproject.toml` (hatchling) whose wheel holds the eighteen top-level packages,
  `notebooks/display`, the notebook drawing helper, and tsncd's built page under
  `websocket_transfer/tsncd_bundle/`, and leaves the other notebooks and the vault out.
  `hatch_build.py` writes the page at wheel build time from the tsncd revision
  `[tool.pyncd.tsncd]` pins, by cloning it and running `npm ci && npm run build`, or by
  copying the directory `PYNCD_TSNCD_BUNDLE` names. `pyncd-server` is the console
  script: the relay, serving that page on the same port. Requirements: Python 3.13 or
  later (3.13 generics with defaults are used throughout; observed working on 3.14).
  `websockets` is the one base dependency. The extras are `torch` (`torch`, `einops`,
  `numpy`) for `torch_compile`, `diagrams` (`ipykernel`, `pandas`) for
  `notebooks/display`, and `headless` (`playwright`) for the headless capture, which
  also needs a Chromium: `playwright install chromium`, or on NixOS the nixpkgs
  `playwright-driver.browsers` through `PLAYWRIGHT_BROWSERS_PATH`.
- **tsncd** is an npm project (`package.json` with webpack 5, TypeScript 5.9, `ws`,
  KaTeX, MathJax, `html-to-image`) that installs as a package:
  `npm install github:starbaser/tsncd` builds it through `prepare`, `npx tsncd serve`
  runs the relay and serves the page on one port, and `npx tsncd dist-path` prints the
  bundle directory. Its flake exports `packages.tsncd`, the installed package with the
  `tsncd` command, and `packages.bundle`, the page alone. Node 24 or later.
- The two are a matched pair. tsncd constructs every term positionally from PyNCD's
  field order, so mixing revisions silently swaps members or refuses whole terms. PyNCD
  pins the tsncd revision its page is built from, so a consumer moves PyNCD and gets the
  matching page.
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

Then `uv lock && uv sync`. Building PyNCD's wheel clones tsncd and runs npm, so `git`
and Node 24 are on the path at install. For the headless capture add `playwright` and
run `uv run playwright install chromium`. The installed wheel carries the tsncd page,
and every process that writes or serves a page finds it through
`websocket_transfer.bundle.locate_bundle`: an explicit `dist=` argument, then
`TSNCD_DIST`, then the packaged page. `TSNCD_DIST` is for a development checkout of
PyNCD, whose editable install carries no page.

## Route 2: Nix flake with uv2nix

The same `pyproject.toml` and `uv.lock` as Route 1. uv2nix fetches the git dependency
from the lock; a git dependency is built from source, so its build system is named in an
override. The sandbox cannot clone tsncd, so the flake takes tsncd's page from tsncd's
own flake and hands it to PyNCD's wheel build.

```nix
inputs.tsncd = { url = "github:starbaser/tsncd/main"; inputs.nixpkgs.follows = "nixpkgs"; };

# in the pyproject overlay overrides
pyncd = prev.pyncd.overrideAttrs (old: {
  PYNCD_TSNCD_BUNDLE = "${tsncd.packages.${system}.bundle}";
  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem { hatchling = [ ]; };
});

# in the dev shell
env = {
  UV_NO_SYNC = "1";
  UV_PROJECT_ENVIRONMENT = "${devEnv}";
  UV_PYTHON = "${devEnv}/bin/python";
  UV_PYTHON_DOWNLOADS = "never";
  PLAYWRIGHT_BROWSERS_PATH = "${pkgs.playwright-driver.browsers}";
  PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = "1";
};
```

Keep the `tsncd` input at the revision PyNCD's `[tool.pyncd.tsncd]` pins, so the wheel
built with npm and the wheel built by the flake carry the same page. A CUDA Torch wheel
needs the usual autoPatchelf and driver-runpath overrides beside this; they are
independent of PyNCD. The reservoir repository's `flake.nix` is a complete instance of
this route.

## The tsncd page

An installed PyNCD wheel carries the page under `websocket_transfer/tsncd_bundle/`,
with `bundle.json` naming the tsncd repository and revision it was built from.
`websocket_transfer.bundle.locate_bundle` resolves an explicit `dist=` argument, then
`$TSNCD_DIST`, then the packaged page, and every page writer and the headless capture
go through it. A PyNCD checkout installed editable carries no page and names one
through `TSNCD_DIST`, for example `npx tsncd dist-path` of an installed tsncd or the
`dist/` of a tsncd checkout after `npm run build`.

## Running the live page viewer

`pyncd-server` runs the relay and serves the page on one port, `8765` unless `PORT` or
`--port` says otherwise, and the address it prints is the one to open. The served page
connects to the relay at its own origin. `notebook_diagrams.show_diagram` in `BROWSER`
mode, or `websocket_transfer.send_morphism.send_morphism(term)`, sends a term to the
relay, with `PORT` naming the relay's port, and every open page redraws it. tsncd's
`tsncd serve` and `npm run server` are the same relay with the same messages; one relay
at a time holds a port.

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
`validations/`, `validate_repository.py`, `obsidian/` and `_guide/`.
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
- A relay port is held by exactly one process. `pyncd-server` and `tsncd serve` serve
  the page on the relay's port; `npm run dev` serves a development page on 3000.

## Integration checklist

```text
- [ ] pin PyNCD and tsncd to a matched pair; record both revisions in the project guide
- [ ] uv source: pyncd as a git dependency at one commit; with Nix, the hatchling override for it
- [ ] flake: tsncd flake input at the revision PyNCD pins, handed to the wheel build through PYNCD_TSNCD_BUNDLE; dev shell exports PLAYWRIGHT_BROWSERS_PATH for headless capture
- [ ] without Nix: git and Node 24 on the path when `uv sync` builds PyNCD's wheel
- [ ] `python -c "import websocket_transfer.bundle as b; print(b.locate_bundle())"` succeeds
- [ ] extras: pyncd[torch] for compilation, pyncd[diagrams] for the relay and pages, pyncd[headless] for headless capture
- [ ] a model with parameter arrays follows `grab_parameters`, size binding, grouping-block removal, `detape` and `ConstructedModule.construct`; a local Torch constructor is registered only after checking both registries
- [ ] a render module draws every model with notebook_diagrams.show_diagram (HTML, INTERACTIVE) and writes its listing
- [ ] executable models compare outputs and gradients with a plain-Torch control; a loop the compiler cannot run has its body compiled and is iterated from Python
- [ ] only the render module imports notebooks/display
```
