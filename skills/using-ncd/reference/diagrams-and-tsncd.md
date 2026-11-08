# Drawing and inspecting diagrams with tsncd

PyNCD builds the algebra and serializes its term; tsncd renders that term in a browser.
Use the official transport and renderer for the same expression used to derive the
executable model. Algebra, transport and browser layout can each cause a display problem;
inspect the appropriate stage before changing the model.

Source checked on the `main` branches of PyNCD and tsncd. PyNCD and tsncd are dependencies, and an
adaptation lives in the consuming project. The rendering routes below originate in
upstream. Historical timing measurements are not fresh benchmarks.
The [tutorial example](../scripts/tutorial_model.py) exercises the interactive HTML
route with ordinary and tape-aware boxes at these revisions.

## Contents

- [Choose a route](#choose-a-route)
- [Read the term as text first](#read-the-term-as-text-first)
- [Route A: relay and browser page](#route-a-relay-and-browser-page)
- [Route B: standalone HTML page](#route-b-standalone-html-page)
- [Route C: headless images and PDFs](#route-c-headless-images-and-pdfs)
- [Route D: capture the page the relay is showing](#route-d-capture-the-page-the-relay-is-showing)
- [Route E: `notebook_diagrams.show_diagram`, in a notebook or a consuming project](#route-e-notebook_diagramsshow_diagram-in-a-notebook-or-a-consuming-project)
- [Display settings](#display-settings)
- [The three figure forms](#the-three-figure-forms)
- [Wire protocol essentials](#wire-protocol-essentials)
- [The mirroring rule](#the-mirroring-rule)
- [Adding an operator's glyph to tsncd](#adding-an-operators-glyph-to-tsncd)
- [tsncd commands](#tsncd-commands)
- [Troubleshooting](#troubleshooting)

## Choose a route

| Need | Route | Needs |
|---|---|---|
| Check the algebra of a term | `agent_display` listing | nothing |
| Live diagram while iterating | A: relay on port 8765 + page from `npm run dev` | node, a browser tab |
| A file a reader opens anywhere, inspection boxes working | E in `HTML` mode: `notebook_diagrams.show_diagram` | `notebooks/display` installed, a built tsncd `dist/` (`TSNCD_DIST`) |
| The bare page writer under that route | B: `standalone_page.save_standalone_page` | a built tsncd `dist/` (`TSNCD_DIST`) |
| PNG, SVG or PDF figures from a script | C: `headless.save_figures` | `playwright` + chromium + `dist/` |
| Image of what the browser currently shows | D: `npm run capture` in tsncd | relay + open page |
| Notebook cells with embedded figures | E: `notebooks/display/notebook_diagrams.py` | `notebooks/display` installed |

pyncd's rule 18 is to draw through `notebooks/display/notebook_diagrams.py`, so a page for a
reader is route E. Upstream ships no package at all, so the consumer's packaging shim
installs `notebooks/display` beside the other packages (integration reference). Route B is
the writer route E calls; used on its own it draws the figure and the legend, and its
inspection boxes stay empty unless the auxiliary information is built the way route E
builds it.

## Read the term as text first

Read the SSA listing to inspect the algebra, and inspect the rendered page or a screenshot
to judge the visual result:

```python
import agent_display as ad

print(ad.listing(term))  # legend, inputs, outputs, operation counts, body
print(ad.summary(term))  # inputs, outputs, counts only
print(ad.trace(term, "%4"))  # what produces %4 and what consumes it
```

Notation: `%n` is a wire and the same `%n` everywhere is the same wire; `R[q, x]` an array
with its datatype and axes; `{d}` an axis the operation consumes (target); `q` an axis it is
broadcast over (degree); `w|x` an axis carrying an affine guard guided by `x`; `loop N
times { }` a repeated block. Listings are deterministic, so two forms can be diffed.
`display.print_category(term)` draws an ASCII diagram for a person at a terminal; do not
read it as text.

## Route A: relay and browser page

The relay sits between a sending Python process and the page, holds the most recent term,
and re-sends it on refresh. Only one relay may hold port 8765.

```bash
# in the tsncd checkout
npm install && npm run server      # relay on ws://localhost:8765 (binds 127.0.0.1 and ::1)
npm run dev                        # page on http://localhost:3000, reconnects on refresh
# or, the same relay from pyncd:
python run_server.py                                        # in a pyncd checkout
python -c 'import asyncio, websocket_transfer.websockets_transfer as w; asyncio.run(w.DataServer().main())'   # installed pyncd
```

Send from Python:

```python
import websocket_transfer.send_morphism as send_morphism
import websocket_transfer.websockets_transfer as wst

await send_morphism.send_morphism(
    term,
    width=1000,
    darkMode=wst.ColorMode.LIGHT,
    debugBorders=False,
    legend=True,
    inspectionBoxes=True,
    auxiliary=auxiliary_information.auxiliary_information(
        send_morphism.to_morphism(term)
    ),
)
```

`send_morphism` accepts a morphism, a hypergraph (converted) or a `cat.DefinedExpression`.
Pass `recycle=True` to normalise a hand-built morphism through the hypergraph round trip
first. Where the relay holds nothing, the page draws the figure the build carries, a
quantised DeepSeek model, until the first `dataUpdate` arrives.

In a notebook, `websocket_transfer.capture.capture_morphism(term)` renders through the
open page and hands the image back inline (needs IPython and an open tab); pass
`disturb_display=False` to render off-screen.

## Route B: standalone HTML page

The writer route E's `HTML` mode calls. Prefer route E: it normalises the term, then builds
the auxiliary information with `expand_with_parameters.write_out_under(...)`, which gives
every `Linear` an inspection box holding its formula and bias. The call below passes the
default write-out, which writes no `Linear` out, so those boxes are empty (observed on a
GIN: legend and block titles only, no operator expansions).

One file holding the bundle, the term and the auxiliary information. Opens from `file://`
with no server and no network; the legend and inspection boxes answer the pointer. No
browser is started to write it.

```python
import websocket_transfer.auxiliary_information as auxiliary_information
import websocket_transfer.send_morphism as send_morphism
import websocket_transfer.standalone_page as standalone_page
import websocket_transfer.websockets_transfer as wst

morphism = send_morphism.to_morphism(term)  # convert once; auxiliary is keyed to it
settings = send_morphism.display_settings(
    darkMode=wst.ColorMode.DARK,
    width=1400,
    debugBorders=False,
    coreDebug=False,
    legend=True,
    inspectionBoxes=True,
    title="Graph Isomorphism Network",
)
standalone_page.save_standalone_page(
    morphism,
    "outputs/diagrams/gin.html",
    settings=settings,
    auxiliary=auxiliary_information.auxiliary_information(morphism),
)
```

The bundle is located by `headless.find_dist`: an explicit `dist=`, else `$TSNCD_DIST`,
else `tsncd/dist` or `tsncd-public/dist` beside the pyncd package directory. An installed
pyncd has no sibling checkout, so set `TSNCD_DIST` to the built `dist/`.
`save_page_with_variants` and `save_page_folder_with_variants` write one page carrying
several variants of a model with a selector.

## Route C: headless images and PDFs

```bash
uv add playwright
uv run playwright install chromium                         # non-NixOS
```

On NixOS, Playwright's downloaded Chromium does not run; use the nixpkgs pair instead
(observed working, pyncd on `PYTHONPATH` or installed):

```bash
BROWSERS=$(nix build --no-link --print-out-paths nixpkgs#playwright-driver.browsers)
PLAYWRIGHT_BROWSERS_PATH="$BROWSERS" PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true \
TSNCD_DIST=/path/to/tsncd/dist \
nix shell --impure --expr '(import <nixpkgs> {}).python313.withPackages (p: [ p.playwright p.websockets ])' \
  --command python figures.py
```

In a flake, add `python3Packages.playwright` to the Python set and `playwright-driver.browsers`
to the shell, exporting the same two variables.

```python
from websocket_transfer.headless import save_figures, capture_headless, HeadlessRenderer

await save_figures(
    {"attention": attention, "raster/fig1.png": attention},
    "paper/figures",
    format="pdf",
    width=1400,
    debugBorders=False,
)  # one browser for the set
png_bytes = await capture_headless(term, format="png", width=1000, debugBorders=False)
async with HeadlessRenderer() as renderer:  # many figures, per-figure options
    await renderer.save(term, "figures/a.svg", width=1200, debugBorders=False)
```

`debugBorders=False` is part of every call: the headless path sends only the options you
give, and the renderer's own default draws a coloured outline around every element.
PDF is the default because it is vector with embedded text and smaller than the PNG.
`width` is the wrap width in pixels and sets the figure's proportions, not its scale: 750
gives about a 1.43 aspect on the transformer, 1400 about 3.29. A notebook kernel holds the
headless renderer until it exits, so a rebuilt bundle reaches it after a kernel restart or
`notebook_diagrams.close_renderer()` in a checkout.

## Route D: capture the page the relay is showing

```bash
npm run capture -- --output tmp/current.png
npm run capture -- --output tmp/current-light.png --dark-mode false
npm run capture -- --output tmp/current.svg --width 1400 --watch 2000
```

Fetches the held term and requests an off-screen capture from a connected page, leaving
the visible diagram untouched. The page must run the current bundle.

## Route E: `notebook_diagrams.show_diagram`, in a notebook or a consuming project

The route rule 18 names. A consuming project draws a page with it:

```python
import notebooks.display.notebook_diagrams as notebook_diagrams

settings = notebook_diagrams.DiagramSettings(
    mode=notebook_diagrams.DiagramMode.HTML,
    width=1400,
    advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
    page_directory=pathlib.Path("outputs/diagrams"),
)
await notebook_diagrams.show_diagram(
    term, settings=settings, slug="gin", title="Graph Isomorphism Network"
)
```

`INTERACTIVE` sends the legend and the inspection boxes, with every block and every
operator that has a standard expansion; the page carries the form, theme and width
controls. The default `advanced_display` is `OFF`. `show_diagram` recycles the term through
a hypergraph first, so a term that conversion cannot handle raises here. On the fork the
recycling defects of the upstream base are fixed: consecutive blocks passing on each other's
wires and consecutive wiring-only blocks recycle (fixes `fe41ce8`, `9a90fe1`).
`mode=DiagramMode.LISTING` prints the SSA listing instead.

`notebooks/display/notebook_diagrams.py` exposes `DiagramSettings` and `show_diagram`.
Declare one settings object at the top and pass it to every call. Modes: `INLINE`
(captures a PNG and embeds it; 166 ms warm), `BROWSER` (sends to the open page; about 15
ms), `HTML` (writes a standalone page under `outputs/pages/`), `DUMP` (term to JSON),
`LISTING` (prints the SSA listing, no browser), `OFF`. `python
notebooks/execute_notebook.py <nb>` runs a notebook with `PYNCD_DIAGRAMS=listing`, so
every `show_diagram` prints a listing. Kernels must start at the repository root.

## Display settings

`send_morphism.display_settings(...)` builds the partial `settings` dict; omitted keys take
the renderer's defaults on every message, so each send fully determines the display.

| Key | Values | Effect |
|---|---|---|
| `darkMode` | `wst.ColorMode.DARK` / `LIGHT`, or `None` | theme; the renderer's own default is dark |
| `width` | px, default 750 | wrap width: proportions, not scale |
| `debugBorders`, `coreDebug` | bool | element outlines; the renderer defaults `debugBorders` to `true`, Python callers send `False` |
| `legend` | bool | table of the term's axes (and naturals) beside the figure; needs `auxiliary` |
| `inspectionBoxes` | bool | a box with title, formula, description, references and the body's diagram opens over a block or expandable operator; needs `auxiliary` |
| `subBlocks` | bool | draw `BlockOperator` bodies as sub-diagrams beside the figure |
| `drawnBlockTags` | list of tag uids | bodies an earlier send already drew |
| `tapeLabels` | bool | label each tape with its slot |
| `axisHover` | `wst.AxisHover.LEGEND` / `EVERYWHERE` / `OFF` | where an axis answers the pointer |
| `axisLabelFontSize` | em, default 0.8 | size of axis labels on wires |
| `form` | `wst.DiagramForm.ALL_BROADCASTED` / `ARROWS_AND_BROADCASTED` / `ARROWS_AND_BOXES` | see below |
| `controls` | `wst.PageControls.SHOWN` / `HIDDEN` | the row of form, theme and width controls |
| `title`, `heading` | str; `wst.PageHeading.NONE` / `TITLE` | tab name `tsncd - <title>`; heading over the figure |
| `multilineSizing` | `wst.MultilineSizing.DYNAMIC` / `FIXED` | row planning; dynamic is the default |
| `displayMode` | `wst.DisplayMode.FAST` / `SLOW` | rendering mode |

`auxiliary_information.auxiliary_information(morphism, ...)` assembles the legend rows,
block records and operator expansions the legend and boxes show. Build it from the same
converted morphism you send, and pass `write_out=expand_with_parameters.write_out_under(...)`
as route E does, or no `Linear` gets an expansion.

## The three figure forms

- `all-broadcasted` (default): every axis is a wire; every operator draws its glyph, its
  contraction cups, its reindexing node and the broadcast wires routed around the glyph.
  The full form, showing what each operator does with each axis.
- `arrows-and-broadcasted`: each array between operators is one heavier arrow labelled
  with its shape above and datatype below; each operator keeps its glyph on a plate whose
  edges open the arrow into axis wires.
- `arrows-and-boxes`: the same arrows, every operator a named box (`Matmul`, `Sum`,
  `Product`, `Contraction` for an `Einops`), nothing of the broadcasting drawn. A reader
  meets this form first.

A page can switch form and theme without reloading: query parameters `form`, `darkMode`,
`controls`, `displayMode`, `variant` (strictly checked; an unknown parameter refuses the
whole address), the on-page controls, or `window.tsncd.display({...})`.

## Wire protocol essentials

Every message is JSON with a `msgType`; an unknown type raises on the server so version skew
fails loudly. `identify` is first on every connection (`clientType` is `DataClient` for a
notebook or `DiagramClient` for a tab; a tab that identifies while a term is held receives
it at once). `dataUpdate` carries `data` (the term as a JSON *string*, doubly encoded),
`settings` (partial, merged over defaults) and optional `auxiliary`. `renderRequest` and
`renderResult` implement capture. `tsncd/PROTOCOL.md` and pyncd's
`obsidian/05-backends/Diagram Wire Format.md` are one mirrored document; change both
repositories together.

## The mirroring rule

A term reaches tsncd only if tsncd mirrors every class it holds. `data_transfer/term_json.py`
writes each `Term` as its class name and its dataclass fields in declaration order; tsncd's
`src/data_transfer/json.ts` looks the name up in `TermDirectory` and calls the constructor
positionally. A class without a mirror stops the whole transport with `Term type not found
in TermDirectory: <name>`; a reordered field produces a term with two members exchanged and
raises nothing. Only the `data_structure` folders are mirrored (`data_structure/`,
`para/data_structure/`, `quantization/data_structure/`, `advanced_axis_dynamics/data_structure/`,
`caching/data_structure/`, `deepseek/data_structure.py`). A hypergraph is converted to a
morphism before sending. The listing needs no mirror, so `LISTING` works on any term.

## Adding an operator's glyph to tsncd

The following describes upstream extension points. Prefer existing representable
operators and their definitions in a consumer; a new glyph is a change to tsncd itself,
made in its repository.

1. Add the operator in pyncd first, as a `cat.Operator` subclass with a `template`.
2. Mirror the class into `src/data_structure/Operators.ts` (or the feature's folder) with
   `@fd.register_term`, constructor parameters in the Python field order, inherited
   fields first.
3. Write the box in `display/Framework/Operations/additionalOperationBoxes.ts`, extending
   `bb.OperationBox`, decorated `@bb.opsRegistry.registerClass(ops.Yours)`; pass the core
   size to `super()` and draw in `update()` against `this.rectangle()`.
4. Labels are `AnnotationElement`s placed with KaTeX LaTeX.
5. Set `override_display_type = bb.BroadcastDisplayType.NODE` where the reindexing should
   draw as a node rather than route around.
6. Register a plate name (`arrows/plateNames.ts`) and a face (`arrows/operatorFaces.ts`)
   for the two arrow forms where the default is wrong.

The four registries (`TermDirectory`, `opsRegistry`, `datatypesRegistry`, `blocksRegistry`)
match the exact constructor name with no superclass walk, so every concrete class is
registered. A registration exists only if its module was evaluated: `src/index.ts` keeps
modules alive with `establish()` calls. Check with `npm run typecheck`, `npm run build`,
then render a JSON fixture exported by
`data_transfer.term_json.TermJSONConverter.export_to_json` through
`HeadlessRenderer.render_json`.

## tsncd commands

```bash
npm run dev              # webpack-dev-server on :3000
npm run watch            # dev server plus a production build into dist/ once edits settle
npm run build            # dist/ — what headless capture and standalone pages read
npm run server           # the relay on :8765
npm run capture -- --output tmp/current.png
npm run validate         # tests and typechecks reached by the modified files
npm run validate:all     # every test file and both typechecks
npm run typecheck        # tsc --noEmit over the browser code
npm run typecheck:server # over the node entry points
```

`ts-loader` runs with `transpileOnly`, so a type error stops neither `dev` nor `build`;
only the typecheck reports it. Node runs `src/run_server.ts` directly with type stripping,
so use a Node that strips types (24 does).

## Troubleshooting

- **Nothing draws, page shows the boot figure.** No relay holds the term, or two relays
  fought for port 8765. Start exactly one.
- **`Term type not found in TermDirectory: X`.** Identify the missing mirror and inspect
  the listing. A missing mirror class is a change to tsncd itself.
- **A member of an operator looks swapped.** Python field order and the TypeScript
  constructor order differ.
- **`Cannot read properties of undefined (reading 'anchors')` from `link_weaves`.** A
  hand-built weave has fewer `TILED` slots than the degree has positions.
- **Two wires drawn unconnected, axis in braces on both sides of a listing.** A hand-built
  contraction related an operand axis and a result axis by uid alone; rebuild it with
  `einops_simplification.einsum`.
- **A multi-input `Broadcasted` draws with its anchors unlinked.** Its inputs are
  reindexed differently from each other (the `JOIN` display mode, not implemented). Expand
  it with `algebra/node_expansion.py`'s `expand_to_nodes` into one `View` per input.
- **Headless capture or a page shows an old renderer.** `dist/` is stale; `npm run build`,
  reload the tab, restart the kernel or call `close_renderer()`.
- **`No built tsncd bundle found`.** Set `TSNCD_DIST` or pass `dist=`.
- **Figure has outlines around every element.** It was sent with `debugBorders: true`;
  pass `debugBorders=False`.
- **`ValueError: HypergraphObject(...) is not in list` from `hypergraph_to_morphism`.** A
  wire required on the right of a branch is absent on its left: inspect both the graph
  wiring and the branch construction. At the upstream base valid consecutive blocks with
  pass-through wires also raised it; the fork fixes that case (`fe41ce8`).
