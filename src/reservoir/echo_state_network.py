"""An echo state network as a morphism in PyNCD.

The network drives a reservoir of `n` units with an input sequence of `t` steps and
reads every state out with one learned linear map. The update is

    h_t = (1 - alpha) h_{t-1} + alpha tanh(W_in u_t + W h_{t-1}),    h_{-1} = 0
    y_t = W_out h_t

where `W_in` and `W` are fixed random arrays and `W_out` is the only learned map.

The recurrence reads the state of the step before, so it cannot be an operator
broadcast over the steps. It is a `cat.Block` repeated once per step whose counter
`i_t` is the name its tag carries, following the selective scan of PyNCD's Mamba
notebook, `notebooks/caching/mamba/selective_scan.py` in the PyNCD repository. The
state `h` is a loop variable: a `Para.StreamGrab` of its slot reads the
state the iteration starts from and a `Para.StreamDrop` writes the state the next
iteration starts from. The states `H` over every step are a second loop variable, and
each iteration writes its state at position `i_t` through a covariant view and adds the
write. Both start from zeros that a `Para.Drop` puts on each slot before the loop.

`W_in` and `W` are `ops.FixedArray`s, which the model reads and never learns, so a
derived backward pass gives them no slot. `W_out` is an `ops.Linear`, which holds its
weight and is the one learned array. `docs/echo-state-network.md` states the mapping
from the equations to the morphism and what PyNCD can and cannot derive from it.
"""

from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.einops_simplification as einops_simplification
import construction_helpers  # noqa: F401 - installs the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import para.data_structure.Para as Para

from reservoir.axes import (
    DRIVE,
    INPUT_SEQUENCE,
    STATE,
    STATES,
    STEP_COUNTER_NAME,
    R,
    inputs,
    outputs,
    steps,
    units,
)

INPUT_WEIGHTS_NAME = "W^{in}"
RECURRENT_WEIGHTS_NAME = "W"
READOUT_NAME = "W^{out}"

LEAK = nm.FreeNumeric.named(r"\alpha")
"""The leak rate as a symbol, for a diagram. A check passes a rational instead."""

STATE_SLOT = fd.DynamicName("h").capture(Para.TapeSlot())
STATES_SLOT = fd.DynamicName("H").capture(Para.TapeSlot())

UPDATE_FORMULA = (
    r"h_{t} = (1 - \alpha) h_{t-1} + \alpha \tanh(W^{in} u_{t} + W h_{t-1}), \quad "
    r"H[t] = h_{t}"
)

NETWORK_COLOUR = "#fce8c3"
RESERVOIR_COLOUR = "#fed06e"
STEP_COLOUR = "#68a8e4"

type RealMorphism = cat.Morphism[cat.Array[cat.Reals, cat.RawAxis]]
"""A morphism over real arrays that may hold tape seeds beside broadcast operators."""


def hold(array: cat.Array) -> cat.Morphism:
    """The identity on one array, which is a wire passing straight through."""
    return cat.ProdObject((array,)).identity()


def route(mapping: tuple[int, ...], arrays: tuple[cat.Array, ...]) -> cat.Rearrangement:
    """A rearrangement whose codomain position `j` carries domain array `mapping[j]`."""
    return cat.Rearrangement(tuple(mapping), tuple(arrays))


def row_at(axis: cat.Axis, position: nm.Numeric) -> sc.StrideMorphism:
    """The row with no domain axis that selects `position` of `axis`."""
    return sc.StrideMorphism(
        _dom=(),
        _cod_stride_shift=((axis, (), position),),
        name=fd.DynamicName(position.to_latex()),
    )


def read_at(
    axis: cat.Axis, position: nm.Numeric, rest: tuple[cat.Axis, ...]
) -> cat.Broadcasted:
    """An array `[axis, *rest]` read at `position` of `axis`, returning `[*rest]`."""
    return ops.View.template(
        reindexing=(
            row_at(axis=axis, position=position),
            cat.ProdObject(rest).identity(),
        ),
        name=position.to_latex(),
    )


def write_at(
    axis: cat.Axis, position: nm.Numeric, rest: tuple[cat.Axis, ...]
) -> cat.Broadcasted:
    """An array `[*rest]` written at `position` of `axis`, which returns `[axis, *rest]`
    holding the universal unit at every other position. It is the covariant reading of
    the row `read_at` reads."""
    tiled = (cat.WeaveMode.TILED,) * len(rest)
    return cat.Broadcasted(
        operator=aops.CovariantView(
            reindexing=row_at(axis=axis, position=position),
            name=fd.DynamicName(position.to_latex()),
        ),
        input_weaves=(cat.Weave(R, tiled),),
        output_weaves=(cat.Weave(R, (axis, *tiled)),),
        reindexings=(cat.ProdObject(rest).identity(),),
    )


def zeros_over(shape: tuple[cat.Axis, ...]) -> cat.Broadcasted:
    """The array of zeros over `shape`, which is where a sum starts."""
    return cat.Broadcasted(
        operator=ops.ConstantOp(value=nm.Integer(0)),
        input_weaves=(),
        output_weaves=(cat.Weave(R, (cat.WeaveMode.TILED,) * len(shape)),),
        reindexings=(),
        backup_degree=cat.ProdObject(shape),
    )


def add_over(shape: tuple[cat.Axis, ...]) -> cat.Broadcasted:
    """The sum of two arrays over `shape`, entry by entry."""
    return einops_simplification.einsum((shape, shape), shape, R, ops.AdditionOp())


def pointwise(
    formula: nm.Numeric, shape: tuple[cat.Axis, ...], name: str | None = None
) -> cat.Broadcasted:
    """`formula` in `nm.x` applied to every entry of an array over `shape`."""
    return ops.Arithmetic.template(formula, base=cat.Array(R, shape), name=name)


def drive_from_inputs() -> RealMorphism:
    """`W_in u_t` at every step, which depends on no state and so stands before the
    loop, broadcast over the steps."""
    input_weights = ops.FixedArray.template(INPUT_WEIGHTS_NAME, (inputs, units), R)
    contraction = einops_simplification.einsum(
        ((inputs, units), (steps, inputs)), (steps, units), R
    )
    return (input_weights * hold(INPUT_SEQUENCE)) @ contraction


def drive_from_state() -> RealMorphism:
    """`W h` for one state. Both axes of `W` are the reservoir axis, so the contraction
    is written over two index variables of that axis rather than over the axis."""
    source = einops_simplification.IndexVariable("j", units)
    target = einops_simplification.IndexVariable("k", units)
    recurrent_weights = ops.FixedArray.template(
        RECURRENT_WEIGHTS_NAME, (units, units), R
    )
    contraction = einops_simplification.einsum(
        ((source, target), (source,)), (target,), R
    )
    return (recurrent_weights * hold(STATE)) @ contraction


def leaky_update(leak: nm.Numeric) -> RealMorphism:
    """`(a, h) -> (1 - leak) h + leak tanh(a)` over the reservoir units."""
    activation = pointwise(2 * nm.Sigmoid(2 * nm.x) - 1, (units,), name=r"\tanh")
    return (
        (activation * hold(STATE))
        @ (pointwise(leak * nm.x, (units,)) * pointwise((1 - leak) * nm.x, (units,)))
        @ add_over((units,))
    )


def one_step(leak: nm.Numeric, counter: nm.Numeric) -> RealMorphism:
    """Step `counter` of the loop: its drive, the state the iteration starts from and
    the states written so far, to the states with this step's state written at
    `counter`. The next state and the states are dropped for the next iteration."""
    return (
        (
            read_at(axis=steps, position=counter, rest=(units,))
            * Para.StreamGrab(tape=STATE_SLOT, size=STATE)
            * Para.StreamGrab(tape=STATES_SLOT, size=STATES)
        )
        @ route((0, 1, 1, 2), (STATE, STATE, STATES))
        @ (hold(STATE) * drive_from_state() * hold(STATE) * hold(STATES))
        @ (add_over((units,)) * hold(STATE) * hold(STATES))
        @ (leaky_update(leak) * hold(STATES))
        @ route((0, 0, 1), (STATE, STATES))
        @ (
            hold(STATE)
            * write_at(axis=steps, position=counter, rest=(units,))
            * hold(STATES)
        )
        @ (hold(STATE) * add_over((steps, units)))
        @ (Para.StreamDrop(tape=STATE_SLOT, size=STATE) * route((0, 0), (STATES,)))
        @ (Para.StreamDrop(tape=STATES_SLOT, size=STATES) * hold(STATES))
    )


def start_state_at_zero() -> RealMorphism:
    """The zero state, dropped on its slot, which is where the recurrence starts."""
    return zeros_over((units,)) @ Para.Drop(tape=STATE_SLOT, size=STATE)


def start_states_at_zero() -> RealMorphism:
    """The zero states over every step, dropped on their slot."""
    return zeros_over((steps, units)) @ Para.Drop(tape=STATES_SLOT, size=STATES)


def step_repeated_over_the_steps(leak: nm.Numeric) -> cat.Block:
    """One step repeated once per step of the sequence, with the counter named."""
    counter = nm.FreeNumeric.named(STEP_COUNTER_NAME)
    return cat.Block.template(
        one_step(leak=leak, counter=counter),
        title=r"\text{One Step}",
        fill_color=STEP_COLOUR,
        repetition=steps.local_size(),
        index_name=STEP_COUNTER_NAME,
        formula=UPDATE_FORMULA,
    )


def reservoir_loop(leak: nm.Numeric) -> cat.Block:
    """The reservoir: the state and the states started at zero, then the loop. It reads
    the drive at every step and returns the state after every step."""
    return cat.Block.template(
        (start_state_at_zero() * start_states_at_zero() * hold(DRIVE))
        @ step_repeated_over_the_steps(leak=leak),
        title=r"\text{Reservoir}",
        fill_color=RESERVOIR_COLOUR,
        formula=UPDATE_FORMULA,
    )


def readout() -> cat.BroadcastedCategory:
    """The learned linear map from the state after every step to the outputs."""
    return steps >> ops.Linear.template(
        input_size=(units,), output_size=(outputs,), name=READOUT_NAME
    )


def echo_state_network(leak: nm.Numeric = LEAK) -> cat.Block:
    """The whole network, from the input sequence to the readout at every step."""
    return cat.Block.template(
        drive_from_inputs() @ reservoir_loop(leak=leak) @ readout(),
        title=r"\text{Echo State Network}",
        fill_color=NETWORK_COLOUR,
    )
