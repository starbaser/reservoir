# Rulings on how a model is written in PyNCD

Condensed from PyNCD's [model rulings](<https://github.com/starbaser/pyncd/blob/78f465c67ad95672064c6d3c6205c65fb2083ce4/obsidian/06-practice/Representing%20Models.md>)
and asserted [construction tutorial](https://github.com/starbaser/pyncd/blob/78f465c67ad95672064c6d3c6205c65fb2083ce4/notebooks/base_features/BuildingAModel.ipynb),
read at PyNCD `0eae8e09d7ed53114ae86fc742b4919ee534c7dd`.
The note records modelling choices; the notebook demonstrates executable constructions.
For a consumer-owned expression that both draws and compiles, start with
[the tutorial model](../scripts/tutorial_model.py), then check its compilation against
the actual operators used. A rejected form is kept beside its replacement wherever
the pair decides how a construction is written.

## Contents

- [The rules in short](#the-rules-in-short)
- [Composition traps of `@`, `*` and `>>`](#composition-traps-of---and-)
- [Weights and Linears](#weights-and-linears)
- [Sizes and the configuration stage](#sizes-and-the-configuration-stage)
- [Blocks, repetitions and presentation](#blocks-repetitions-and-presentation)
- [Operators and what they stand for](#operators-and-what-they-stand-for)
- [Reversed rulings to recognise in old code](#reversed-rulings-to-recognise-in-old-code)

## The rules in short

In the order a construction meets them.

1. **Declare the axes explicitly.** Predeclare every structural `cat.RawAxis` and pass
   the axis objects rather than strings. Composition aligns by position and never by
   name, so exact identity beats guessing what `@` will align.
2. **Reach for a standard operator first; add a custom one only for new semantics.**
   `Einops` and `Linear` come before `GenericOperator`. An elementwise product is
   `Einops.template('a, a -> a')`. The unit-axis spelling `'1, 1 -> 1'` parses, and
   alignment cannot merge an axis whose size is a literal, so it is not written.
3. **A weight rides a `Linear`, and a per-index weight produces its axis.** A `Linear`
   with no inputs is a parameter array. A mixture-of-experts up-projection is
   `Linear (m,) -> (e, f)` broadcast over the tokens; the down-projection produces the
   expert axis again and is followed by a diagonalisation.
4. **The target of a weight array is what makes a map act per channel.** A tap table
   whose target carries the window, the heads and the channels is a depthwise
   convolution. The same table read as a `Linear` over the window alone, tiled over the
   channels, is one shared filter and draws the same shape.
5. **Selection is `ds.TopK` from `deepseek/data_structure.py`, drawn compressed.** Pass
   `template(axis=)` so the caller's axis is consumed and `template(name=)` so two
   selections in one model are distinguishable.
6. **A selection sits between the router's projection and the softmax.** The softmax
   then normalises over the experts that were chosen.
7. **Factor a reindexing.** A reindexing acting on some axes is the acting
   `StrideMorphism` times the identity on the rest, written
   `View.template(reindexing=(stride, id_c))`.
8. **A stride morphism may name one axis in its domain and in its codomain.** That is the
   statement that a convolution preserves length, and a padded convolution needs it.
9. **Write a window's domain with the offsets and the channels last.** Composition aligns
   a `Linear` against the last axes of what it is handed.
10. **A position-dependent map consumes its axis.** Positional encoding, rotary embeddings
    and a positional bias are each a function of the index, and a function of the index
    cannot be tiled over it.
11. **Put the operand a composition supplies first in a signature.** The surplus wire is
    then the one the composition does not carry: `'e f, e -> e f'` reads the gates as the
    surplus; `'e, e f -> e f'` lifts the projection against them.
12. **Assert on the domain and the codomain of a chain.** `cat.Composed` reads its domain
    off its first entry and its codomain off its last and compares no neighbouring pair,
    so a composition that does not compose raises nothing.
13. **A compression/expansion skip uses a tape slot.** The two paths then carry none
    of each other's wires, as the model rulings prescribe. The construction tutorial's
    short residual addition uses a copied input and an identity wire instead; do not
    turn every intermediate value into a tape slot.
14. **Present compactly with `ops.BlockOperator`.** A block wrapped as a bold named
    operator draws as one box whose body renders once beside the main figure. Reserve
    `GenericOperator` for a body that cannot be written.
15. **A value one layer publishes and later layers read uses a tape slot.** The
    publisher uses `Para.Drop` and each reader uses `Para.Grab`, so a repeated layer
    carries its hidden state alone, as the construction tutorial demonstrates. Carry
    ordinary intermediate results directly. Check `block.dom() == block.cod()` on a
    repeated block and match the dropped and grabbed slots. Compiling a nested tape
    form needs separate verification.
16. **Every block carries a fill colour, and a derivative rule never invents a block.** An
    uncoloured block has no drop shadow and cannot be seen.
17. **Every size is a free symbol, assigned by name in one `NumericConfig`.** No `Integer`
    literal appears in a construction. Keep name bodies to a single letter.
18. **Draw through one `DiagramSettings` declared at the top** (in a pyncd checkout's
    notebooks) and deliver each diagram in the notebook's INLINE output.
19. **Simplify the drawing with the Yoneda and Cartesian tricks.** Draw the smaller form
    and leave the schedule to the cost tooling.
20. **Expand onto the tape, and draw the implicit or the wrapped form.** The `ParaWrap`
    form is the default drawing of grabs and drops.
21. **Every axis of the result of a contraction is a degree position.** An `Einops` writes
    no axis of its own. Build a contraction over declared axes with
    `einops_simplification.einsum`, which places every axis from the shapes.
22. **A negative index is the unit, and causality is the sign of an index.** A causal
    window reads back from the current token, at `i_x - i_w`, built with
    `mark_sparse_domains.guarded_view`. Write no mask operator and no guard field.
23. **Two axes carrying two forms are concatenated, not joined into one axis**
    (`aops.ConcatenateAxes`).
24. **A body computed once per index of an axis is one `ops.BlockOperator` broadcast over
    that axis, confirmed by `discovering_broadcasts`.**
25. **A boxed layer that reads and writes a tape is a `ParaBlockOperator`,** carrying the
    tape at its own ports.
26. **A contraction against a parameter array is a `Linear` that produces the array's
    axes, followed by a diagonalisation.**
27. **A block carries a long title naming the thing in words, and the box over it carries
    a short name.** Prose titles are written with an explicit `\text{}`.
28. **A fixed one-hot is a covariant read of one index with no operands.**
29. **A part the reference implements as one module is one box, and a box may stand
    inside a box.**
30. **A payload read at a selection's positions is a `ds.IndexSelect` broadcast over the
    selection axis.**
31. **An axis cut into the parts that fill it is a deconcatenation**
    (`aops.DeconcatenateAxes`).
32. **Reals that are rotated in pairs are complex numbers** (`dst.PairsAsComplex`,
    `dst.Decomplex`).
33. **An operator whose value depends on an index is a `GenericOperator` with a
    definition over every position,** stated as a `cat.DefinedExpression` whose
    right-hand side gets the index as a value from an `ops.Arrange`.
34. **A value proportional to a position is a product with an `ops.Arrange`, and a
    formula in a position is an `ops.Arithmetic` applied to one.** Two formulas that read
    one index share one `Arrange`, whose result is copied.
35. **A clamp between two bounds is an `nm.Clamp`.**
36. **Write a grouping of the axes produced by a linear map into the linear map.**
    `Linear((m,), (h', g, d))` rather than a `Linear` followed by a view that splits heads.
37. **Display an expression in the CausalSlide form** when it holds causal reads.
38. **Grab a parameter from an outer tape slot, and grab a random sample, a residual or a
    cache from an inner one.**
39. **A square root is an `nm.SquareRoot`, and a scale by its reciprocal is a division:**
    `nm.x / nm.SquareRoot(d.local_size())` prints `x / \sqrt{|d|}`.
40. **Write a repeat that an operation reads as a deletion in that operation's
    reindexing.** A repeat is a view whose reindexing has stride zero over the repeated
    axis; `reindexing_absorption.absorb` moves it into the consumer.

## Composition traps of `@`, `*` and `>>`

- `@` composes in sequence and aligns axes by position. A name plays no part. Two letters
  written in two templates are two axes until a composition puts them at the same
  position.
- A composition supplies what either side is missing. Surplus wires on the left pass
  beside the right side as identities, and the shorter side is lifted over surplus axes.
  A rank mismatch is therefore often accepted silently: `Einops('q d, x d -> q x') @
  Linear(('a', 'b', 'c'), 'o')` lifts the left side over one axis and composes.
- A bare tuple on the left of `@` is a `Rearrangement` whose output position `i` copies
  the input `mapping[i]`. Its domain is read from the morphism it is composed with.
  `(0, 0)` copies, `(1, 0)` swaps, `()` deletes, `(0, 1, 1)` copies the second wire.
- `add_excess_lift` compares the first objects of the two sides alone and lifts the whole
  of the shorter side. A product mixing a rank-deficient operation with held wires lifts
  the held wires too and fails with the observed message:

  ```text
  ValueError: Cannot align axes of different lengths. Tried to align:
  [x d] [x d]
  [x noname] [x x d]
  ```

  Pre-lift the one operation: `lift.morphism_object_lift(op, cat.ProdObject(axes))`.
  The construction notebook defines a small local `over` helper for this call.
  Single-object chains never hit this.
- `*` composes in parallel. A wire is carried past an operation by the identity on its
  array, `cat.ProdObject((array,)).identity()`, which the construction notebook calls
  `hold`. It is the direct carrier for an ordinary intermediate result; do not invent
  an `Einops` operation solely to carry it.
- `>>` broadcasts a morphism over a batch axis on every input and every output, and the
  batch axis becomes the first axis of every array.
- Import `construction_helpers` for its side effects even where the name is unused. The
  `@`, `*` and `>>` methods raise `NotImplementedError` until it is imported.
- A contraction built by hand from declared axes uses `einops_simplification.einsum`
  over the axis objects or over `IndexVariable`s. `Einops.template` mints fresh axes, and
  a minted axis composed onto a declared one can win canonicality and rename the
  declared one throughout the term.
- One declared axis may stand at two positions. Self-attention written from one copied
  input carries the token axis at both positions of the scores, and every rule reads a
  `Broadcasted` by position, never by axis membership. Do not deduplicate a shape, test
  an axis for membership, key a dictionary by an axis or call `shape.index(axis)`.
- An empty left side of an einops signature parses to one empty input segment, so
  `Einops.template('-> q v')` is a repeat of a scalar. A morphism with an empty domain is
  built by hand as a `cat.Broadcasted` with `input_weaves=()`, `reindexings=()` and a
  `backup_degree` where it is broadcast.
- `cat.Composed` and `cat.ProductOfMorphisms` written directly check nothing. Observed:
  `cat.Composed((f, g))` with mismatched sides constructs and reports the domain of `f`
  and the codomain of `g`. Assert on both ends of every chain.
- Keep every size a free symbol and assign it once in a `NumericConfig`. A structural
  assertion and a numerical comparison still matter: positional autoalignment does
  not prove that the intended positions were paired.
- Three projections reading one copy of a feature map cannot name different axes,
  because composition aligns by position and lifts each projection over the axes it is
  missing.

## Weights and Linears

- Every weight rides an `ops.Linear`; no explicit weight array remains on a wire in a
  finished model. A `Linear` with no inputs is a parameter array and draws in the
  weight-box idiom. A zero-input `Linear` is not directly compiled as a module-owned
  parameter. For explicit weight operands, use `show_grabbed_parameters.grab_parameters`,
  bind sizes, remove grouping blocks, `detape.detape`, then
  `torch_compile.ConstructedModule.construct`, as the
  [upstream training check](https://github.com/starbaser/pyncd/blob/78f465c67ad95672064c6d3c6205c65fb2083ce4/notebooks/website/tutorial/check_gradients_in_torch.py)
  does. Do not promise that this route expands an arbitrary nested `ParaWrap`.
- A weight is not lifted over a batch axis. State the batch axis in the einsum beside the
  weight, `'x m, m f -> x f'`.
- A parameter table a per-position operation needs is broadcast at the consumer: give the
  consuming operation a reindexing that drops the degree.
- A per-index weight produces its axis: `ops.Linear.template(('m',), ('e', 'f'),
  name='up')`.
- The down-projection of a mixture produces the expert axis a second time, then a
  diagonalisation `View.template(reindexing=cat.Rearrangement((0, 0, 1), (e, m)))`, then
  the gates as `Einops.template('e, e m -> m')`.
- A scalar head weighting is a `Linear` onto the empty shape:
  `ops.Linear.template(('i',), ())`.
- An integer output size on a `Linear` mints that many fresh axes, which the signature
  that consumes them names by position: `Linear.template('m', 2, name='q')` followed by
  `'q h d, x h d -> h q x'`.
- A `Linear` reads two kinds of input. A real datatype is contracted. A `Natural` index
  selects one of the weights (`ops.selects_weights`), and a selecting `Linear` must not be
  expanded as a contraction.
- A linear map followed by a slice of its results is one `Linear` per part, each reading a
  copy of the input, with no axis to cut and no slice.

## Sizes and the configuration stage

- Every size is a free symbol assigned by name in one `NumericConfig` after the
  construction, selection counts and window widths included.
- Keep a size name's body to a single letter. `DynamicName.from_str` splits on `_` and
  `assign_values` matches on the body, so `k_e` and `k_x` collide. Use `k` and `s`, or
  build a `fd.DynamicName(body=...)` directly.
- A number that is not sourced stays a symbol. Print what the configuration could not
  bind beside what it did; verify the bound term has no remaining free sizes.
- Two letters written in two templates are two axes with two sizes, and `assign_values`
  binds both by name body. Name the value width `v` when it must differ from `d`.
- An axis minted by an integer size takes the name of whichever letter alignment later
  puts at its position.
- A hand-named tape slot must not take the derivation's own `s0, s1, ...` form.
- Two separate chains of `@` can give one axis two size symbols. Compare rows by axis uid.

## Blocks, repetitions and presentation

- `ops.BlockOperator.template(block, name)` presents compactly: one bold named box whose
  body renders once beside the figure, with the definition attached.
- A `Block` with a repetition other than one denotes a loop; the repetition may be a free
  symbol.
- Every block carries a fill colour. `Block.template` defaults to `'white'`; a hand-built
  `BlockAesthetics` states a colour explicitly.
- A block title is LaTeX as given; prose titles carry `\text{}`. One short line per
  title; a table beside the figure carries the rest.
- A derived expression is drawn as the derivation returned it. Do not assemble a figure
  from separately derived pieces or add a box the derivation did not produce.
- The compressed form is the default drawing.

## Operators and what they stand for

- A root-mean-square normalisation is `ops.Normalize` whatever it carries: `gain`, `bias`
  and `epsilon` cover every combination.
  Rejected: `ops.L2Norm.template((n, m))` followed by the root of the number of values.
  Replacement: `ops.Normalize.template((n, m), gain=False, bias=False, epsilon=...)`.
- `ops.L2Norm` stays for a normalisation stated as a division by the root of a sum, whose
  result has length one. It takes no weight and no epsilon.
- A division by a sum plus an epsilon is `ops.L1Norm.template(epsilon=...)`. A
  `SoftMax` is an exponential followed by an `L1Norm` at `ops.NO_EPSILON`.
- Name the axis a normalisation consumes by its position in the shape, not by the axis.
- A gather is an operator of its own (`ds.Select`, `ds.IndexSelect`); neither an
  `Einops` signature nor an affine reindexing states a selection.
- A repeat is a `View` whose reindexing does not name the repeated axis:
  `ops.Einops.template('q -> q v')` writes exactly that `View`. No repeat operator exists.
- The operator that carries a reindexing is `ops.View`. Affine geometry never justifies
  an opaque operator: a window, a convolution and a shift are one factored reindexing at
  three strides.
- An operation the operator set cannot state is defined in full beside the term that
  stands for it, in prose, giving its value at a generic index.
- Addition is `ops.AdditionOp.template()`. `ops.SoftMax.template()` consumes the last axis
  of the array it is composed after.
- `ops.Elementwise.template(name=...)` carries a name and no formula, and
  `torch_compile` evaluates every named `Elementwise` as a ReLU. Write a formula with
  `ops.Arithmetic.template(nm.x * nm.Sigmoid(nm.x))` instead, so the derivative and the
  compiled value both follow from it.
- Every rotary table writes its own name above its circle; an inverse rotation is the
  plain table followed by an elementwise conjugate.
- Block titles and descriptions of one model live in one text module the construction
  modules import.

## Reversed rulings to recognise in old code

- The folded mixture down-projection `Linear (e, f) -> (m,)` with gates multiplied in
  first was accepted on 2026-08-15 and replaced by the drawn form on 2026-08-18.
- `relabel_axis`, reading a key axis under a second name as an affine identity, was
  overruled on 2026-09-25: a copied input composed onto the contraction by position is
  the form, and the scores carry the one token axis at both positions.
- A hard rule against lifting a parameter array was superseded on 2026-08-21, because a
  lifted 0-input `Broadcasted` now carries its degree in `backup_degree`. The modelling
  advice stands: a weight is not lifted over a batch axis.
- `ops.Identity` was renamed `ops.View`; `para...Broadcast` was removed in favour of a
  `View` whose reindexing drops the repeated axis.
