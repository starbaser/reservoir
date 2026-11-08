# pyncd API surface

The construction, reading, sizing and compilation surface at PyNCD
`0eae8e09d7ed53114ae86fc742b4919ee534c7dd`
([the repository](https://github.com/starbaser/pyncd/tree/d056b4755fad127e48063739126fc2f19192650b), whose `d056b47` adds the relay address
configuration, a `torch_compile` fix and the packaging). Its upstream base is
`a051f8f2`; three fixes address positional `Linear` contraction and graph recycling.
Start with [the construction tutorial](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/notebooks/base_features/BuildingAModel.ipynb)
and use [the gradient tutorial](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/notebooks/website/tutorial/check_gradients_in_torch.py)
for the native compilation pipeline. The examples below distinguish expressions that
construct, terms that render, and terms the PyTorch backend actually compiles.

## Contents

- [Imports and aliases](#imports-and-aliases)
- [Axes, sizes, datatypes and arrays](#axes-sizes-datatypes-and-arrays)
- [Seeds: operators and their templates](#seeds-operators-and-their-templates)
- [Composition: the overloads](#composition-the-overloads)
- [Rearrangements, holds and lifts](#rearrangements-holds-and-lifts)
- [Stride morphisms and views](#stride-morphisms-and-views)
- [Contractions over declared axes](#contractions-over-declared-axes)
- [Blocks, boxes and definitions](#blocks-boxes-and-definitions)
- [Tapes: Para](#tapes-para)
- [Reading a term](#reading-a-term)
- [Sizes: NumericConfig](#sizes-numericconfig)
- [Compiling to PyTorch](#compiling-to-pytorch)
- [Hypergraphs and rewrites](#hypergraphs-and-rewrites)
- [Rewrite limits](#rewrite-limits)
- [Checks inside a checkout](#checks-inside-a-checkout)

## Imports and aliases

```python
import construction_helpers as ch  # side effect: installs @, * and >> on cat.Morphism
import construction_helpers.lift as lift
import data_structure.Category as cat  # ProdObject, Morphism, Composed, ProductOfMorphisms,

# Rearrangement, Block, BlockTag, BlockAesthetics,
# DefinedExpression, Axis, RawAxis, StrideMorphism,
# Datatype, Reals, Natural, Array, Weave, WeaveMode,
# Operator, Broadcasted
import data_structure.Operators as ops  # the operator classes and their templates
import data_structure.Numeric as nm  # FreeNumeric, Integer, x, Sigmoid, SquareRoot, Clamp, ...
import data_structure.Term as fd  # Term, UID, DynamicName, DynamicNameSettings, Context
import algebra.einops_simplification as einops_simplification  # IndexVariable, einsum, index_shapes
import term_utilities.generate_config as generate_config  # NumericConfig
import torch_compile.torch_compile as torch_compile  # ConstructedModule
import agent_display as ad  # listing, summary, trace
import deepseek.data_structure as ds  # TopK, Select, IndexSelect, SparseAxis, Complex, Rotary
import para.data_structure.Para as Para  # Grab, Drop, TapeSlot, OuterTapeSlot, new_slot
```

The fixed aliases `fd`, `cat`, `ops`, `nm`, `ch`, `hg`, `util` are the ones upstream uses
in every file; alias every other module by its own basename.

## Axes, sizes, datatypes and arrays

| Thing | Construction | Notes |
|---|---|---|
| axis | `cat.RawAxis.named('x')`, `cat.RawAxis.named('q', code_form='queries')` | each call mints a new UID and a new size symbol `\|x\|`; two calls with one name are two axes |
| axis with a given size | `fd.DynamicName('x').capture(cat.RawAxis(_size=a.local_size() * b.local_size()))` | a product size, or `cat.RawAxis(_size=nm.Integer(3))` for a literal (avoid in models) |
| axis size | `axis.local_size()` | an `nm.Numeric`; `nm.FreeNumeric.named('k')` names a shared symbol (same UID for the same name) |
| datatypes | `cat.Reals()`, `cat.Natural(bound)`, `cat.Natural.template('v')`, `ds.Complex(cat.Reals())` | a `Natural` is an index below `bound`; an embedding's input and a selection's positions are naturals |
| array | `cat.Array(cat.Reals(), (x, m))` | datatype and shape; `array.shape()` is a `ProdObject` |
| object | `cat.ProdObject((array_a, array_b))` | a product of arrays; `.identity()` is the identity morphism on it |
| names | `fd.DynamicName('W', subscript=fd.DynamicName('in'), settings=fd.DynamicNameSettings(bold=True, overline=False))` | `fd.DynamicName.from_str('W_{in}')` parses; `.capture(term)` names a term |

Sizes print between absolute bars, `|x|`; a `code_form` gives the legend a code name.

## Seeds: operators and their templates

A seed is a `cat.Broadcasted(operator, input_weaves, output_weaves, reindexings,
backup_degree=None)`. Templates build one from a signature or a size. Letters in a
signature are fresh axes. In `'q d, x d -> q x'` a letter on both sides is degree
(tiled), a letter on the left only is absorbed (contracted), a letter on the right only is
degree the operands do not name (a repeat). Every output segment must list the degree
axes in the same order.

| Operator | Build | Meaning | Torch |
|---|---|---|---|
| `Einops` | `ops.Einops.template('q h d, x h d -> h q x')` | contraction, outer product, transpose, repeat; one output segment | yes |
| `SoftMax` | `ops.SoftMax.template(contracted=False)` | over the one axis it consumes, which `@` identifies with the last axis of the operand | yes |
| `L1Norm` | `ops.L1Norm.template(epsilon=ops.NO_EPSILON)` | `x / (sum x + eps)` over the consumed axis | yes |
| `L2Norm` | `ops.L2Norm.template((n, m))` | `x / sqrt(sum x^2)` over the shape; no weight, no epsilon | yes |
| `Normalize` | `ops.Normalize.template((m,), gain=True, bias=False, epsilon=ops.SYMBOLIC_EPSILON)` | RMSNorm, `y = gamma * x * (mean(x^2) + epsilon)^(-1/2) + beta` over the shape; direct compilation does not hold its gain as a parameter; use `gain=False` only when the model declares no gain | yes |
| `LayerNorm` | `ops.LayerNorm.template((m,), gain=True, bias=True, epsilon=ops.SYMBOLIC_EPSILON)` | `y = gamma * (x - mean(x)) * (mean((x - mean(x))^2) + epsilon)^(-1/2) + beta`, biased variance as `torch.nn.LayerNorm` | yes |
| `Linear` | `ops.Linear.template(input, output, name=None, bias=False, datatype=..., output_datatype=None)` | contraction against a held weight; `input`/`output` are an int (mints that many fresh axes) or a tuple of names or axes (each name a fresh axis, so `template('m', 'm')` holds two axes); a `Natural` input can describe a selection structurally | with ordinary inputs yes; zero-input parameter arrays need `grab_parameters` and `detape`; index selection has no direct compiler rule here |
| `Embedding` | `ops.Embedding.template('v', output_size, name=None)` | `Nat(v) -> R[m]` | yes |
| `AdditionOp` | `ops.AdditionOp.template()` (signature `',->'`) | two operands to one, elementwise, lifted by composition | yes |
| `Arithmetic` | `ops.Arithmetic.template(formula, base=cat.Reals(), reindexing=(), name=None, output_datatype=None)` | elementwise map given by a formula in `nm.x`; differentiable by the algebra | yes |
| `ReLU` | `ops.ReLU.template(name='ReLU')` | the one named elementwise map that is what it says | yes |
| `Elementwise` | `ops.Elementwise.template(name=...)` | a name with no formula; `torch_compile` evaluates every one as ReLU, so prefer `Arithmetic` | relu |
| `View` | `ops.View.template(base=cat.Reals(), reindexing=(stride, id), name=None)` | an affine read; returns the identity when the reindexing is the identity | only a rearrangement directly; the tutorial registers a separate affine-read constructor |
| `WeightedTriangularLower` | `ops.WeightedTriangularLower.template()` | causal lower-triangular mask, rows renormalised | yes |
| `Maximum`, `Product` | `ops.Product.template(axis)`; `ops.sized(ops.Maximum(), axes)` | reductions | yes |
| `Dropout` | `ops.Dropout().bc_signature('->')` | elementwise stochastic operator; no rate field at this revision | no dedicated constructor |
| `Arrange` | `ops.Arrange.template(axis)` | `Nat(\|x\|)[x]` holding `i_x` at `i_x`; no operands | no constructor |
| `FixedArray`, `ConstantOp`, `Cast`, `Modulo`, `BitwiseXor` | see [operator declarations](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/data_structure/Operators.py) | constants and integer arithmetic that can be authored independently of Torch compilation | no bundled catch-all backend; check the tsncd mirror separately before drawing |
| `GenericOperator` | `ops.GenericOperator.template('name', 'q d -> q', datatype=...)` | an opaque operator with a name and shapes; reserved for a body that cannot be written | none |
| `BlockOperator` | `ops.BlockOperator.template(block, 'Name')` | a block as one bold named box; `expand` writes it out | no constructor for the box; compile the body or expanded term where its operators are supported |
| `ds.TopK` | `ds.TopK.template(axis=n, k=..., name='k/n', form=ds.SelectionForm.WEIGHTS)` | keeps the `k` largest and their positions; forms `WEIGHTS`, `WEIGHTS_SELECT`, `ONLY_WEIGHTS`, `ONLY_SELECTION` | no native constructor at this revision |
| `ds.Select`, `ds.IndexSelect` | `ds.Select.at_positions(k, n)` | a gather at a selection's positions | no native constructor at this revision |

What a `Linear` computes: `Linear : R[m] -> R[f]` holds a weight `R[m, f]` and
computes `y_f = sum_m x_m W_{mf}`, so the compiled parameter has the input axes first
and the output axes after them (observed: `Linear.template('m', 'f')` at `m=8, f=16`
gives a `(8, 16)` parameter, applied as `x @ W`). With `bias=True` a bias over the
output axes follows the weight. `module.parameters()` yields the weights in construction
order with nested module names, so name the operators and keep that order in mind when
copying parameters into a control.

A `Linear` whose input and output share one axis object, `Linear.template((m,), (m,))`, is a
full square map `y_j = sum_i x_i W_{ij}`. Observed at `f29cb267`: compiled directly it holds a
`(3, 3)` weight applied as `x @ W`, and grabbed onto the tape and compiled through the
tutorial's `compile_pass` it computes `x @ W` too; the fork's `para/validate_linear_positions.py`
passes its square, repeated, overlapping and interleaved cases forward, backward and transposed.
At upstream `a051f8f2` the grabbed form read its einsum off axis identity and computed the
diagonal of `W`. The positional fix is present in the stated revision.

The epsilon of a norm: `ops.NO_EPSILON` is `nm.Integer(0)`, exactly zero, and the
addition folds it away. `ops.SYMBOLIC_EPSILON` is `nm.FreeNumeric.named(r'\epsilon')`, a
symbol for a number the model does not state; it appears among a `NumericConfig`'s free
terms and the compiler evaluates a symbolic epsilon as zero. A stated value such as
`nm.Integer(1) / nm.Integer(10**6)` compiles to that number. Observed: a block compiled
with `NO_EPSILON` matched a Torch control dividing by the bare root mean square.

`ops.sized(operator, input_size, output_size=None, datatype=cat.Reals())` builds a seed
for an operator with no template of its own, from an int or a tuple of names. The
generic signature reader (`construction_helpers.signature`) is what `AdditionOp` and
`GenericOperator` use; the einops reader (`construction_helpers.einops`) is what `Einops`
and `bc_signature` use. `ops.with_datatypes(seed, inputs, outputs)` replaces the
datatypes of the weaves in order.

## Composition: the overloads

| Form | Meaning |
|---|---|
| `f @ g` | sequential composition. Aligns `f.cod()` with `g.dom()` by position through `fd.EqualityClass`es; lifts the shorter side over surplus axes (comparing the first objects alone); carries surplus wires past the other side as identities |
| `f * g`, `object * morphism`, `object * object` | parallel product; nested tuples flatten; an object beside a morphism becomes its identity |
| `axes >> f` | broadcast `f` over the axes on every input and output; the axes lead every array; a `Rearrangement` keeps its mapping over the lifted objects |
| `axes >> array`, `axes >> cat.Reals()` | prepend axes to an array's shape, or make an array of that datatype |
| `stride >> array`, `(stride * axis) >> cat.Reals()` | one `View` per array reading it through the stride morphism |
| `(0, 0) @ f`, `f @ (1, 0)` | a bare tuple is a `Rearrangement`; output position `i` copies input `mapping[i]`; the domain is read from the neighbour |

`ch.comp(f, g)` is `construction_helpers.composition.composition` and returns the composed
morphism at this revision. It aligns by position; it does not return a context or guarantee
rejection of conflicting fixed extents. `ch.general_product(...)` and
`ch.dynamic_object_lift(...)` are the other underlying functions. Check the sized
domain and codomain explicitly when their extents matter.
`construction_helpers.simple_helper.make_composed(*ms)` and `make_product(*ms)` flatten nested compositions and drop identities, for rewrites.

## Rearrangements, holds and lifts

```python
hold = lambda array: cat.ProdObject((array,)).identity()  # carry a wire unchanged
over = lambda axes, m: lift.morphism_object_lift(
    m, cat.ProdObject(axes)
)  # lift one operation alone
cat.Rearrangement((0, 0, 1, 1), (A, H))  # explicit mapping with its domain: A, A, H, H
cat.Rearrangement((1,), (A, H))  # delete A, keep H
cat.Rearrangement((0, 0), (x,))  # copy in the stride category too
```

`Rearrangement.apply(seq)` maps domain to codomain; `.invert(seq)` maps back;
`.pairwise()` lists `(domain, codomain)` pairs. A `Rearrangement` with an empty mapping
and empty domain is the identity on the unit and vanishes inside a product.
The author's construction idioms in
[the DeepSeek example](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/notebooks/sota/DeepSeekV41Flash/construction_idioms.py)
are example-local, not `construction_helpers.idioms` library exports. Translate a
helpful idiom using supported upstream primitives within the consuming model.

Lifting (`construction_helpers/lift.py`): `Axes >> (Datatype | Array) -> Array`;
`StrideCategory >> (Datatype | Array) -> Broadcasted`; `StrideCategory >> Broadcasted ->
Broadcasted`; `Axes >> BroadcastedCategory -> BroadcastedCategory`. Lifting a
`Broadcasted` prepends `TILED` slots to every weave and composes the lift into every
reindexing; a 0-input seed carries the new axes in `backup_degree`. A seed class outside
the four constructions needs a rule in `lift.OBJECT_LIFTS` (`register_object_lift`);
`para/registries/object_lift.py` registers the tape seeds.
`algebra.factor_out_lift` is the inverse.

## Stride morphisms and views

```python
cat.StrideMorphism.from_matrix(
    (1, 1), dom_names=("x'", "w"), cod_names=("x",), name="+", shifts=()
)
cat.StrideMorphism(
    _dom=(q, w),
    _cod_stride_shift=((q, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
    name=fd.DynamicName("win"),
)
ops.View.template(
    reindexing=(stride, cat.ProdObject((c,)).identity()), name="win"
)  # factored: stride times identity on c
ops.View.template(
    reindexing=cat.Rearrangement((0, 0, 1), (e, m)), name="diag"
)  # diagonalisation
ops.Einops.template("q -> q v")  # a repeat, which is a View
```

Each row of `_cod_stride_shift` is `(codomain_axis, strides_per_domain_axis, shift)`, so
codomain index `i = sum(stride * dom) + shift`. A codomain index outside `[0, size)`
reads the universal unit; `advanced_axis_dynamics.algebra.mark_sparse_domains.guarded_view`
builds the same view and marks the axis that becomes sparse. A reindexing maps output
indices to input indices.

## Contractions over declared axes

```python
v = einops_simplification.IndexVariable("v", nodes)  # an index over an axis
u = einops_simplification.IndexVariable("u", nodes)  # a second index over the same axis
c = einops_simplification.IndexVariable("c", channels)
einops_simplification.einsum(
    ((v, u), (u, c)), (v, c), cat.Reals()
)  # sum_u A[v,u] h[u,c]
einops_simplification.einsum(((), (v, c)), (v, c), cat.Reals())  # scalar times array
einops_simplification.einsum(
    ((x, g, j, c), (g, j, c, o)), (x, g, o), R
)  # shapes may also be axes
einops_simplification.index_shapes(broadcasted)  # read the variables back
```

`einsum(inputs, output, datatype=cat.Reals(), operator=None)`: a shape is a tuple of
index variables or of axes (each axis its own variable). Every axis of the result is a
degree position. Use it, never `Einops.template`, beside declared axes.

## Blocks, boxes and definitions

```python
cat.Block.template(
    body,
    title="\\text{GIN Layer}",
    description=None,
    fill_color="#2C78BF",
    repetition=layers,
    index_name=None,
    references=None,
    formula=None,
    drawing=None,
)
block.dom(), block.cod(), block.block_tag.repetition, block.aesthetics
ops.BlockOperator.template(
    block, "SWA"
)  # one bold box; body drawn once beside the figure
cat.DefinedExpression.template(
    lhs, rhs
)  # lhs := rhs; raises SidesOfADefinitionDisagree
cat.CodeReference(label, url, path, line, end_line)  # shown in an inspection box
```

`repetition` other than one denotes a loop; it may be an `nm.FreeNumeric`. `drawing=
cat.BlockDrawing.BODY_IN_PLACE` draws the body where the block stands with no enclosure.
`algebra.discovering_broadcasts.discover_broadcast_over_axes(block, (h,), 'Name')` turns a
body written per head into one box broadcast over the heads and confirms it.

## Tapes: Para

```python
slot = fd.DynamicName("kv").capture(
    Para.TapeSlot()
)  # an inner slot: one array per lifted index
outer = Para.OuterTapeSlot()  # a parameter slot: one array however lifted
Para.new_slot()  # a counter-named slot, s0, s1, ...
Para.Drop(tape=slot, size=array)  # array -> (), publishes
Para.Grab(tape=slot, size=array)  # () -> array, reads
```

Both are composition-neutral, so a repeated block that drops and grabs keeps its domain
equal to its codomain. `para.data_structure.ParaBlockOperator.ParaBlockOperator.template(
block, 'G')` boxes a block that reads and writes slots, carrying them at its ports.
`para.processing.show_grabbed_parameters.parameter_arrays(seed)` lists the learned arrays
an operator grabs. `ConstructedModule.construct` runs no tape.
`para.algebra.detape.detape(term)` returns the pure term with every top-level grab an extra
trailing operand and every drop an extra trailing result, plus the slot names in order, which is
how a caller holding the weights compiles a model with parameters; it raises on tape
operations below the top level. Remove grouping blocks before detaping.

## Reading a term

```python
print(ad.listing(term))  # legend + in/out/ops + SSA body; deterministic
print(ad.summary(term))  # in/out/ops only
print(ad.trace(term, "%4"))  # producer and consumers of one wire
ad.listing_without_legend(term)
import display as dpl

dpl.print_category(term)  # ASCII diagram for a person, not for parsing
```

Observed listing of `ops.Einops.template('q d, x d -> q x') @ ops.SoftMax.template()`:
`%3 = Einops(%0[q, {d}], %1[x, {d}]) : R[q, x]` then `%2 = SoftMax(%3[q, {x}]) : R[q, {x}]`.
Braces mark the target, bare letters the degree. Useful structural reads:
`term.dom()`, `term.cod()`, `tuple(array.shape())`, `len(array.shape())`,
`seed.degree()`, `weave.target()`, `weave.select_degree(xs)`, `weave.select_target(xs)`,
`seed.has_empty_domain()`, `term_utilities.term_utilities.type_search(cls, term)`,
`term_utilities.term_utilities.is_identity(m)`. The later
`term_utilities.same_construction` helper is absent at this revision.

## Sizes: NumericConfig

```python
config = generate_config.NumericConfig.template(term)  # collect every free size
[
    t.uid._name.body for t in config.terms
]  # their names (duplicates are separate symbols)
config.assign_values(
    x=16, m=32, h=2, d=8
)  # by name body; one call may bind several symbols
sized = config.apply_context(term)  # or config(term)
generate_config.NumericConfig.template(sized).terms  # what is still free
config.assigned_integers_by_name()  # {'m': 32, ...}, for figure legends
```

`assign_value(name, int)` binds one. Matching is on the body of the name, or on the
bodies run together (`T14` for `T_{14}`), so one-letter bodies avoid collisions. A
size that is a product of symbols (`|a| |b|`) is bound through its factors.
`assign_values` does not reject an unknown name here; inspect `config.terms` and
confirm no required size remains free after applying the context.

## Compiling to PyTorch

```python
module = torch_compile.ConstructedModule.construct(sized_term)  # an nn.Module
outputs = module(*input_tensors)  # inspect the constructed term's result structure
list(module.parameters())  # the weights the Linears hold
```

Dispatch: `Rearrangement`, `ProductOfMorphisms`, `Composed`, `Block` have constructors (a
`Block` whose repetition is an integer above one constructs its body that many times, so each
iteration holds its own weights); a `Broadcasted` is looked up by `type(operator)` in
`ConstructedModule.operation_registry` (classes declared `class X(ConstructedModule,
operation_key=ops.Y)`) and then in `functions_registry` (`ConstructedModule.add_function(ops.Y,
fn, dim=False, semantic=False)`, wrapped by `broadcast_func`)
([compiler dispatch](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/torch_compile/torch_compile.py#L36)).
Registered by importing `torch_compile.torch_compile`: `Einops`, `SoftMax`, `L1Norm`,
`L2Norm`, `Maximum`, `Product`, `AdditionOp`, `View` (rearrangement only), `Elementwise`
(as ReLU), `ReLU`, `Arithmetic`, `WeightedTriangularLower`, `Linear` with inputs,
`para...Transpose`, `Embedding`, `Normalize`, `LayerNorm`. A zero-input parameter-array
`Linear`, a `BlockOperator`, the Para tape seeds and many model-specific operators have no
constructor. An expression may be valid and renderable even when this backend cannot
execute it. The official gradient tutorial adds local `Einops`, `SoftMax`, `View` and
`ReindexTranspose` constructors for sparse/affine operations and uses the upstream
`grab_parameters` → `NumericConfig` → `remove_grouping_blocks` → `detape` → `construct`
pipeline for external weights. Consult the [integration guide](integrating-as-dependency.md#training-a-model-that-holds-parameter-arrays)
for that sequence. Register a consumer's numerical constructor only after checking both
registries; registrations affect the whole process.

## Hypergraphs and rewrites

```python
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m

graph = hg.Multigraph.from_morphism(term)  # the form rewrites work in
back = h2m.hypergraph_to_morphism(graph)  # back to the form terms are read in
normal = h2m.recycle(term)  # both directions; normalises a hand-built term
```

The bootstrap fixes to graph recycling hold pass-through wires only for their producers
and preserve wiring-only blocks, discarded and identity blocks, and loop ports
(`fe41ce8`, `9a90fe1`). They do not add a Torch constructor for a boxed operator.

Rewrites any model may use live in `algebra/` (`operator_expansion.expand_softmax`,
`expand_l1_norms`, `linear_expansion`, `einops_rearrange`, `reindexing_absorption.absorb`,
`remove_identities`, `define_by_expansion`). Feature packages hold the rest: `para/` (the
backward pass), `quantization/`, `caching/`, `advanced_axis_dynamics/`.

### Rewrite limits

At the stated revision, these limits are module variables read when a pass runs.
Use the existing settings before proposing library API changes solely to raise a cap.

| Setting | Default | Applies to |
|---|---|---|
| `algebra.merge_into_consumer.MERGE_LIMIT` | `256` | Each producer/consumer merge search and the outer absorption/merge rounds in `merge_reindexings_and_einops` |
| `para.data_structure.ParaWrap.WRAP_LIMIT` | `1024` | Tape wrapping through `para_wrap_graph`, `to_para_wrap`, and `to_para_wrap_rows` |

Set positive integer limits in the consuming program before invoking those passes:

```python
from algebra import merge_into_consumer
from para.data_structure import ParaWrap

merge_into_consumer.MERGE_LIMIT = 4096
ParaWrap.WRAP_LIMIT = 4096
```

These assignments affect subsequent calls throughout the current Python process;
they do not edit the package. Assign the module attributes: rebinding a local name
obtained with `from ... import MERGE_LIMIT` does not update the module setting.
The values are finite iteration counts; these upstream loops do not accept `None`.

The loops need a final iteration that finds no remaining rewrite. A chain requiring
256 merges therefore raises with `MERGE_LIMIT = 256` and completes with `257`.
Likewise, 1,024 tape reads complete with `WRAP_LIMIT = 1025`. Reaching a limit alone
does not establish that a rule cycles or that the compiler is defective.

Verified on upstream `a051f8f2` and the fork at `988c693d`; the relevant loops are
unchanged from the stated revision. See the [merge search](https://github.com/mit-zardini-lab/pyncd/blob/a051f8f2df12be97bb438a7b19980513fbe11c36/algebra/merge_into_consumer.py#L241-L262),
[outer rounds](https://github.com/mit-zardini-lab/pyncd/blob/a051f8f2df12be97bb438a7b19980513fbe11c36/algebra/einops_rearrange.py#L426-L448),
and [tape wrapping](https://github.com/mit-zardini-lab/pyncd/blob/a051f8f2df12be97bb438a7b19980513fbe11c36/para/data_structure/ParaWrap.py#L171-L210).

## Checks inside a checkout

```bash
python validations/run_validations.py algebra/x.py     # targets reached by the modified files
python validations/run_validations.py --all            # every validator and notebook
python validate_repository.py                           # imports, notebooks, calls, vault links
python notebooks/execute_notebook.py notebooks/base_features/BuildingAModel.ipynb   # listings, no browser
```

These exist in the source checkout only; an installed consumer writes its own checks.
PyNCD has no project flake or `uv.lock` of its own; from `d056b47` it has a `pyproject.toml`,
so a consuming project installs it as a git dependency and provides the environment.
