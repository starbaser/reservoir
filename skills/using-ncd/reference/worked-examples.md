# Worked examples

Start with [the consumer-owned tutorial model](../scripts/tutorial_model.py): its
construction is the same expression used for compilation and drawing. PyNCD's asserted
[Building a Model notebook](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/notebooks/base_features/BuildingAModel.ipynb)
shows direct identity wires for ordinary intermediate results, compact
`ops.BlockOperator` boxes for ordinary components, and `ParaBlockOperator` for boxes
whose layers really read or write a tape. The first examples below were checked at
upstream `a051f8f2` and the attention core was observed again at fork `f29cb267`.
The graph isomorphism network was historically checked at `20d7040` (three layers,
19 parameters, output shape `(7, 2)`); these measurements have not been rerun at
PyNCD `0eae8e09d7ed53114ae86fc742b4919ee534c7dd`.
Use the examples as constructions, then verify the actual compiler route and numerical
behavior at the revision you consume.

## Contents

- [Tutorial-based executable example](#tutorial-based-executable-example)
- [Attention core from templates](#attention-core-from-templates)
- [Residual feed-forward, lifted and repeated](#residual-feed-forward-lifted-and-repeated)
- [A transformer out of blocks](#a-transformer-out-of-blocks)
- [Convolution as a reindexing and a linear map](#convolution-as-a-reindexing-and-a-linear-map)
- [Declared axes and `einsum`](#declared-axes-and-einsum)
- [Configure, compile and compare with Torch](#configure-compile-and-compare-with-torch)
- [Graph isomorphism network over an adjacency wire](#graph-isomorphism-network-over-an-adjacency-wire)
- [Neighbour-restricted attention](#neighbour-restricted-attention)
- [A neural cellular automaton step](#a-neural-cellular-automaton-step)
- [A render module for a consuming project](#a-render-module-for-a-consuming-project)
- [Patterns a consuming project adds to upstream's notebooks](#patterns-a-consuming-project-adds-to-upstreams-notebooks)

## Tutorial-based executable example

[The tutorial model](../scripts/tutorial_model.py) authors one expression whose input
branches into a hidden value and shared features. The publisher is an ordinary
`ops.BlockOperator`. The hidden value remains on a direct identity wire while a
`Para.Drop` publishes the features to a tape slot. Two reader layers each take that
slot through a `Para.Grab`, combine it with the hidden value through a `Linear` and
addition, and apply SiLU. Because these readers genuinely touch the tape, each is
boxed with `ParaBlockOperator`; ordinary components need no tape-aware box.

Run `python skills/using-ncd/scripts/tutorial_model.py --output-dir /path/to/diagrams`
in a consumer environment with PyNCD, PyTorch and
`TSNCD_DIST`. The script binds sizes, expands boxes through PyNCD's native
`Endofunctor` and operator methods, removes grouping blocks, ties and grabs tape
slots, detapes, then constructs the module. It compares outputs and gradients of the
input and four weights with plain Torch before drawing the **same authored term**
through `notebook_diagrams.show_diagram` as an interactive HTML page. The adapter is
consumer code using existing mechanisms, not a custom exporter or a general-purpose
compiler guarantee. In particular, this small non-looping example says nothing about
arbitrarily nested `ParaWrap` expansion.

## Attention core from templates

```python
import agent_display as ad
import construction_helpers  # noqa: F401 - installs @, * and >>
import data_structure.Category as cat
import data_structure.Operators as ops

qk = ops.Einops.template("q h d, x h d -> h q x")
softmax = ops.SoftMax.template()  # consumes the last axis it is composed after
mask = ops.WeightedTriangularLower.template()  # causal, row-normalised
sv = ops.Einops.template("h q x, x h d -> q h d")
core = qk @ softmax @ mask @ sv
print(ad.listing(core))
```

Observed listing:

```text
legend  %n value   R[..] array   {a} axis consumed   a axis broadcast over

in    %0 : R[q, h, d], %1 : R[x, h, d], %2 : R[x, h, d]
out   %3 : R[q, h, d]
ops   2x Einops, 1x SoftMax, 1x WeightedTriangularLower

%4 = Einops(%0[q, {d}], %1[x, {d}]) : R[h, q, x]
%5 = SoftMax(%4[h, q, {x}]) : R[h, q, {x}]
%6 = WeightedTriangularLower<wtril>(%5[h, {q}, {x}]) : R[h, {q}, {x}]
%3 = Einops(%6[h, q, {x}], %2[{x}, h, d]) : R[q, h, d]
```

Read it: the first `Einops` consumes `d` on both operands and is broadcast over `q`, `h`
and `x`; the softmax's fresh axis became `x` by position; the mask consumes both `q` and
`x`; the second contraction consumes `x`. Two different `d`s exist (the one summed over and
the value width) because they never meet at one position.

## Residual feed-forward, lifted and repeated

```python
feed_forward = (
    ops.Linear.template("m", "f", name="in")
    @ ops.ReLU.template(name="ReLU")
    @ ops.Linear.template("f", "m", name="out")
)
residual = (
    (0, 0) @ feed_forward @ ops.AdditionOp.template()
)  # copy, body beside identity, sum
block = cat.RawAxis.named("x") >> cat.Block.template(
    residual @ ops.Normalize.template(gain=False, epsilon=ops.NO_EPSILON),
    title="\\text{Feed Forward}",
    fill_color="#519F50",
    repetition=2,
)
```

Observed listing:

```text
in    %0 : R[x, m]
out   %1 : R[x, m]
ops   1x AdditionOp, 2x Linear, 1x Normalize, 1x ReLU

loop 2 times {
  %2 = Linear<in>(%0[x, {m}]) : R[x, {f}]
  %3 = ReLU(%2[x, f]) : R[x, f]
  %4 = Linear<out>(%3[x, {f}]) : R[x, {m}]
  %5 = AdditionOp<+>(%4[x, m], %0[x, m]) : R[x, m]
  %1 = Normalize<RMSNorm>(%5[x, {m}]) : R[x, {m}]
}
```

`(0, 0) @ feed_forward` supplies the copy; the composition carries the second copy past
the body as an identity, and the addition reads both. The lift `x >>` put `x` first on
every array, and the repetition prints as a loop.

## A transformer out of blocks

From upstream's `example_notebooks/minimum_working_example.py`:

```python
import data_structure.Numeric as nm
import data_structure.Term as fd


def attention_core():
    return cat.Block.template(
        ops.Einops.template("q h k, x h k -> h q x")
        @ ops.SoftMax.template()
        @ ops.WeightedTriangularLower.template()
        @ ops.Einops.template("h q x, x h k -> q h k"),
        title="Attention Core",
        fill_color="#C5BEDF",
    )


def attention_layer():
    query, key, value = (
        ops.Linear.template(("m",), 2, n) for n in "qkv"
    )  # 2 mints h and k
    output = ops.Linear.template(2, ("m",), "o")
    return (query * key * value) @ attention_core() @ output


def residual(target):
    return cat.Block.template(
        (0, 0) @ target @ ops.AdditionOp.template() @ ops.Normalize.template(),
        title="Add \\& Norm",
        fill_color="#F1F4C1",
    )


def feed_forward():
    return cat.Block.template(
        ops.Linear.template(1, ("d_ff",), "in")
        @ ops.Elementwise.template()
        @ ops.Linear.template(("d_ff",), 1, "out"),
        title="Feed Forward",
        fill_color="#C1E8F7",
    )


def transformer():
    layers = cat.Block.template(
        residual((0, 0, 0) @ attention_layer()) @ residual(feed_forward()),
        title="Transformer Layer",
        fill_color="#F3F3F4",
        repetition=nm.Integer(6),
    )
    vocabulary = fd.DynamicName("v", settings=fd.DynamicNameSettings(overline=True))
    embedding = cat.Block.template(
        ops.Embedding.template(vocabulary), title="Embedding", fill_color="#FCE0E1"
    )
    aggregator = cat.Block.template(
        ops.Linear.template(1, (vocabulary,)) @ ops.SoftMax.template(),
        title="Aggregator",
        fill_color="#DBDFEF",
    )
    return embedding @ layers @ aggregator
```

Composing the rank-1 projections onto the rank-3 attention signature lifts them over the
token axis, so the layer reads `R[x, m]` and writes `R[x, m]`; `(0, 0, 0)` fans one input
to the three projections.

## Convolution as a reindexing and a linear map

```python
convolution_reindexing = cat.StrideMorphism.from_matrix(
    (1, 1), dom_names=("x'", "w"), cod_names=("x",), name="+"
)  # x = x' + w
input_channels = fd.DynamicName("c", fd.DynamicName("in")).capture(cat.RawAxis())
convolution_matrix = (convolution_reindexing * input_channels) >> cat.Reals()
output_channels = fd.DynamicName("c", fd.DynamicName("out")).capture(cat.RawAxis())
convolution = convolution_matrix @ ops.Linear.template(2, output_channels)
```

The stride morphism reads `R[x, c_in]` and writes `R[x', w, c_in]`; the `Linear` consumes
the last two axes (window and input channels) at every output position. This is a
valid circuit construction, not proof that the stock PyTorch compiler implements the
affine `View`. The [upstream numerical tutorial](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/notebooks/website/tutorial/check_gradients_in_torch.py)
registers a strided-read constructor for its own checks; a consuming project may do
likewise, checking values and gradients against an independent implementation.

## Declared axes and `einsum`

Self-attention from one copied input carries one token axis at both score positions.
Build such contractions with `einsum` over `IndexVariable`s, never with a template beside
declared axes:

```python
import algebra.einops_simplification as einops_simplification

R = cat.Reals()
tokens, channels = cat.RawAxis.named("x"), cat.RawAxis.named("d")
query = einops_simplification.IndexVariable("q", tokens)
key = einops_simplification.IndexVariable("k", tokens)
channel = einops_simplification.IndexVariable("c", channels)
scores = einops_simplification.einsum(
    ((query, channel), (key, channel)), (query, key), R
)
```

Observed: `%2 = Einops(%0[x, {d}], %1[x, {d}]) : R[x, x]`, and the two result positions
hold the same axis object.

## Configure, compile and compare with Torch

```python
import torch
import term_utilities.generate_config as generate_config
import torch_compile.torch_compile as torch_compile

config = generate_config.NumericConfig.template(core)
sorted(t.uid._name.body for t in config.terms)  # ['d', 'd', 'h', 'q', 'x']
config.assign_values(q=5, x=7, h=2, d=4)  # binds both d's by name body
sized = config.apply_context(core)
assert not generate_config.NumericConfig.template(sized).terms
module = torch_compile.ConstructedModule.construct(sized)
q, k, v = (torch.randn(*s) for s in ((5, 2, 4), (7, 2, 4), (7, 2, 4)))
(out,) = module(q, k, v)  # one tensor per codomain wire
```

Observed: output shape `(5, 2, 4)`, equal within `1e-6` to the Torch reference
`einsum('hqx,xhd->qhd', tril-normalised softmax(einsum('qhd,xhd->hqx', q, k)), v)`.
Upstream's `example_notebooks/Results.ipynb` does the same for a whole residual attention
block against a handwritten `nn.Module`, copying parameters across and asserting
`torch.equal`.

## Graph isomorphism network over an adjacency wire

The graph is data on a wire, every layer reads it, and each layer returns `(A, H)`.
Weights ride `Linear`s; epsilon is a `Linear` with no inputs, a parameter array.
The historically checked explicit-operand route grabs parameters, removes grouping
blocks, detapes, then compiles. `detape` reaches top-level grabs only, so this example
authors layers one after another, each with its weight names. This compiler limitation
does not prohibit mathematically valid nested blocks or `ParaWrap` expressions; their
full expansion and compilation need independent verification at the consumed revision.

```python
import functools
import operator

import algebra.einops_simplification as einops_simplification
import construction_helpers  # noqa: F401
import construction_helpers.lift as lift
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd

R = cat.Reals()
nodes, features, channels, hidden, outputs = (cat.RawAxis.named(n) for n in "nfcho")
node = einops_simplification.IndexVariable("v", nodes)
neighbour = einops_simplification.IndexVariable("u", nodes)
channel = einops_simplification.IndexVariable("c", channels)
ADJACENCY = cat.Array(R, (nodes, nodes))
STATE = cat.Array(R, (nodes, channels))


def hold(array):  # carry a wire past an operation
    return cat.ProdObject((array,)).identity()


def summed():  # + lifted over (n, c) alone, so it can stand beside a held wire
    return lift.morphism_object_lift(
        ops.AdditionOp.template(), cat.ProdObject((nodes, channels))
    )


def gin_layer(layer):
    epsilon = cat.Broadcasted(  # a parameter array: Linear with no inputs
        operator=ops.Linear(
            name=fd.DynamicName.from_str(f"\\epsilon^{{({layer})}}").reconstruct(
                settings=fd.DynamicNameSettings(bold=True)
            )
        ),
        input_weaves=(),
        output_weaves=(cat.Weave(R, ()),),
        reindexings=(),
    )
    neighbour_sum = einops_simplification.einsum(
        ((node, neighbour), (neighbour, channel)), (node, channel), R
    )
    scaled_self = (epsilon * hold(STATE)) @ einops_simplification.einsum(
        ((), (node, channel)), (node, channel), R
    )
    perceptron = nodes >> (
        ops.Linear.template((channels,), (hidden,), f"W_1^{{({layer})}}", bias=True)
        @ ops.ReLU.template(name="ReLU")
        @ ops.Linear.template((hidden,), (channels,), f"W_2^{{({layer})}}", bias=True)
    )
    body = (
        (0, 0, 1, 1, 1)  # A, A, H, H, H
        @ (hold(ADJACENCY) * neighbour_sum * hold(STATE) * scaled_self)
        @ (hold(ADJACENCY) * summed() * hold(STATE))
        @ (hold(ADJACENCY) * summed())
        @ (hold(ADJACENCY) * perceptron)
    )
    return cat.Block.template(
        body, title=f"\\text{{GIN Layer {layer}}}", fill_color="#2C78BF"
    )


def gin(layers):
    embed = nodes >> ops.Linear.template((features,), (channels,), "W_{in}", bias=True)
    readout = nodes >> ops.Linear.template(
        (channels,), (outputs,), "W_{out}", bias=True
    )
    return (
        (hold(ADJACENCY) * embed)
        @ functools.reduce(
            operator.matmul, (gin_layer(l) for l in range(1, layers + 1))
        )
        @ cat.Rearrangement((1,), (ADJACENCY, STATE))  # drop A, keep H
        @ readout
    )
```

Its historical test grabbed the weights onto the tape, compiled the detaped term,
and compared values and gradients against a plain-Torch control reading the same
weight tensors by slot name. `expand_parametrised_linears` was needed at `a051f8f2`;
it was not needed at `f29cb267`. The route below reflects that later observation,
but this example has not been requalified against the current revision:

```python
grabbed = show_grabbed_parameters.grab_parameters(gin(3))
config = generate_config.NumericConfig.template(grabbed)
config.assign_values(n=7, f=3, c=5, h=11, o=2)
pure, tape = detape.detape(
    discovering_broadcasts.remove_grouping_blocks(config.apply_context(grabbed))
)
module = torch_compile.ConstructedModule.construct(pure)
names = [
    name for name, is_grab in tape if is_grab
]  # 'W{in}', '\\epsilon^{(1)}', 'W1^{(1)}', 'bW1^{(1)}', ...
shapes = [
    tuple(int(a.local_size()._value) for a in array.shape()) for array in pure.dom()
][2:]
weights = {
    name: torch.randn(shape).requires_grad_() for name, shape in zip(names, shapes)
}
(logits,) = module(adjacency, features, *(weights[name] for name in names))
expected = control(
    adjacency, features, weights
)  # plain Torch, reading weights["W1^{(1)}"] etc.
torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-5)
ours = torch.autograd.grad(logits.square().sum(), list(weights.values()))
theirs = torch.autograd.grad(expected.square().sum(), list(weights.values()))

block = gin_layer(1)
assert block.dom() == block.cod() and tuple(block.dom()) == (ADJACENCY, STATE)
```

The explicit-operand route passes the resulting weights to the compiled module in
the slot order given by `detape`. A model with no parameter array, such as the
graph transformer and cellular automaton below, can compile the supported operators
directly with `ConstructedModule.construct(sized)`; verify each operator used.

## Neighbour-restricted attention

A graph transformer restricts each node's softmax to its neighbours with operators pyncd
already has, because $\mathrm{softmax}(s) A / \sum \mathrm{softmax}(s) A = e^s A / \sum
e^s A$:

```python
scores = einops_simplification.einsum(
    ((query, head, width), (key, head, width)), (head, query, key), R
) @ ops.Arithmetic.template(
    nm.x / nm.SquareRoot(head_width.local_size()),
    name=fd.DynamicName("x / \\sqrt{|d|}"),
)
over_neighbours = (
    ((scores @ ops.SoftMax.template()) * hold(ADJACENCY))
    @ einops_simplification.einsum(
        ((head, query, key), (query, key)), (head, query, key), R
    )
    @ ops.L1Norm.template()
)
```

The historical normalisations use
`ops.Normalize.template((channels,), gain=False, epsilon=EPSILON)`. Check a learned
gain's explicit parameter route at the actual compiler revision before training it.

## A neural cellular automaton step

The window is the neighbourhood table `N[x, w, x']` supplied as data and read by
`einsum`. The historical implementation used a table because the stock compiler at
`a051f8f2` and `f29cb267` did not compile a stride window. PyNCD's current numerical
tutorial registers a separate strided-read constructor whose out-of-range values are
zero; the table is an example choice, not a prohibition on consumer-owned registrations.

```python
cells, window, channels, perception, hidden = (cat.RawAxis.named(n) for n in "xwcqm")
cell, source = (
    einops_simplification.IndexVariable("x", cells),
    einops_simplification.IndexVariable("s", cells),
)
offset, channel = (
    einops_simplification.IndexVariable("w", window),
    einops_simplification.IndexVariable("c", channels),
)
NEIGHBOURHOOD, STATE = (
    cat.Array(R, (cells, window, cells)),
    cat.Array(R, (cells, channels)),
)


def perceive():
    windows = einops_simplification.einsum(
        ((cell, offset, source), (source, channel)), (cell, offset, channel), R
    )
    return cat.Block.template(
        windows
        @ (
            cells
            >> ops.Linear.template((window, channels), (perception,), "K", bias=True)
        ),
        title="\\text{Perception}",
        fill_color="#2C78BF",
    )


def cellular_step():
    body = (
        cat.Rearrangement((0, 0, 1, 1), (NEIGHBOURHOOD, STATE))
        @ (hold(NEIGHBOURHOOD) * (perceive() @ update()) * hold(STATE))
        @ (
            hold(NEIGHBOURHOOD)
            * lift.morphism_object_lift(
                ops.AdditionOp.template(), cat.ProdObject((cells, channels))
            )
        )
    )
    return cat.Block.template(
        body, title="\\text{Neural Cellular Automaton Step}", fill_color="#918175"
    )
```

A rollout calls the compiled module repeatedly; the circuit is one step.

## A render module for a consuming project

One page and one listing per model, drawn through `notebook_diagrams.show_diagram` as rule
18 asks, run as `python -m mypackage.render`:

```python
import asyncio, contextlib, io, pathlib
import notebooks.display.notebook_diagrams as notebook_diagrams

MODELS = {"gin": ("Graph Isomorphism Network", lambda: gin(layers=3)), ...}

async def draw(out: pathlib.Path) -> None:
    settings = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.HTML, width=1400,
        advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
        page_directory=out)
    for slug, (title, build) in MODELS.items():
        term = build()
        await notebook_diagrams.show_diagram(term, settings=settings, slug=slug, title=title)
        listing = io.StringIO()
        with contextlib.redirect_stdout(listing):
            await notebook_diagrams.show_diagram(
                term, settings=settings, mode=notebook_diagrams.DiagramMode.LISTING)
        (out / f"{slug}.txt").write_text(listing.getvalue())

asyncio.run(draw(pathlib.Path("outputs/diagrams")))
```

## Patterns a consuming project adds to upstream's notebooks

These patterns describe the historical consuming project. Verify compiler-sensitive
ones against the current source before adopting them:

- **Axes are module-level objects and arrays are named constants** (`ADJACENCY`, `STATE`),
  passed to `cat.Rearrangement(mapping, (arrays...))` explicitly where a bare tuple would
  have to infer its domain.
- **Every contraction over graph data is `einops_simplification.einsum` over
  `IndexVariable`s**, including two index variables over one node axis (`v`, `u`).
  Templates mint axes, which the composition traps forbid beside declared ones.
- **Small local helpers `hold` and `over`** call `ProdObject(...).identity()` and
  `lift.morphism_object_lift` as the construction notebook demonstrates. They are
  consumer conveniences defined near the expression that uses them.
- **Ordinary shared intermediate results remain on wires.** The graph example carries
  adjacency beside state (`(A, H) -> (A, H)`). A true published tape value may use
  `Para.Drop` and `Para.Grab`, but the top-level `detape` route does not imply arbitrary
  nested tape compilation.
- **The example with a parameter array writes layers consecutively** for its
  explicit-operand `grab_parameters` / `detape` route. Do not infer that a repeated
  block with nested parameters compiles at this revision without a numerical check.
- **`Normalize(gain=False)` and windows as data tables** are choices made by the
  historical example. A documented consumer-side numerical constructor can support
  additional operations; compare its values and gradients rather than hiding a limit.
- **The project owns its adaptations**, leaving PyNCD and tsncd unchanged.
- **Every model has a test with an independent Torch control**, comparing outputs and
  gradients, plus the structural assertion that a repeated block's domain equals its
  codomain.
- **Drawing goes through pyncd's own `notebook_diagrams.show_diagram` against the stock tsncd bundle**
  from `TSNCD_DIST`. Nothing renders outside the official path.
