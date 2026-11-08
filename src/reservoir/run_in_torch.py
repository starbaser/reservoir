"""Running the network in PyTorch through PyNCD's `torch_compile`.

A morphism is compiled by the route of PyNCD's tutorial,
`notebooks/website/tutorial/check_gradients_in_torch.py` in the PyNCD repository:
`show_grabbed_parameters.grab_parameters` moves every weight onto the tape, a
`NumericConfig` binds every size to an integer, `remove_grouping_blocks` removes the
blocks that only group, `detape` turns each grab into a further operand, and
`ConstructedModule.construct` builds the module. The compiled module takes the
morphism's inputs followed by its weights in the order the tape names them.

`torch_compile` compiles a repeated block only when its repetition is an integer and
has no rule for a tape slot, so the loop over the steps is run here: `one_step` is
compiled once and called once per step, carrying the state from one call to the next,
which is what the loop in the morphism states.
"""

from __future__ import annotations

import fractions
from collections.abc import Mapping
from dataclasses import dataclass

import algebra.discovering_broadcasts as discovering_broadcasts
import data_structure.Category as cat
import data_structure.Numeric as nm
import para.algebra.detape as detape
import para.processing.show_grabbed_parameters as show_grabbed_parameters
import term_utilities.generate_config as generate_config
import torch
import torch_compile.torch_compile as torch_compile

from reservoir import echo_state_network
from reservoir.reservoir_weights import ReservoirWeights


@dataclass(frozen=True)
class Compiled:
    module: torch.nn.Module
    """The compiled morphism, taking its inputs followed by its weights."""

    weight_names: tuple[str, ...]
    """The names of the weights, in the order the module takes them."""

    def __call__(self, *inputs: torch.Tensor, **weights: torch.Tensor) -> torch.Tensor:
        result = self.module(*inputs, *(weights[name] for name in self.weight_names))
        return result[0] if isinstance(result, tuple) else result


def rational(value: float) -> nm.Numeric:
    """`value` as the exact rational a formula in the morphism is written with."""
    fraction = fractions.Fraction(str(value))
    return nm.Integer(fraction.numerator) / nm.Integer(fraction.denominator)


def compile_morphism(morphism: cat.Morphism, sizes: Mapping[str, int]) -> Compiled:
    """`morphism` compiled at `sizes`, with its weights as operands."""
    grabbed = show_grabbed_parameters.grab_parameters(morphism)
    config = generate_config.NumericConfig.template(grabbed)
    config.assign_values(**sizes)
    pure, tape = detape.detape(
        discovering_broadcasts.remove_grouping_blocks(config.apply_context(grabbed))
    )
    return Compiled(
        module=torch_compile.ConstructedModule.construct(pure),
        weight_names=tuple(name for name, is_grab in tape if is_grab),
    )


def compile_step(*, leak: float, inputs: int, units: int) -> Compiled:
    """One step of the network, `(u_t, h_{t-1}) -> h_t`, with `W_in` and `W` as
    operands."""
    return compile_morphism(
        echo_state_network.one_step(rational(leak)), {"m": inputs, "n": units}
    )


def compile_readout(*, steps: int, units: int, outputs: int) -> Compiled:
    """The readout at every step, `H -> Y`, with `W_out` as an operand."""
    return compile_morphism(
        echo_state_network.readout(), {"t": steps, "n": units, "p": outputs}
    )


def run_reservoir(
    *, step: Compiled, weights: ReservoirWeights, input_sequence: torch.Tensor
) -> torch.Tensor:
    """The state after every step of `input_sequence`, over `(steps, units)`, from the
    zero state, by calling the compiled step once per step on the device the weights
    live on."""
    device = weights.recurrent_weights.device
    sequence = input_sequence.to(device=device, dtype=weights.recurrent_weights.dtype)
    state = torch.zeros_like(weights.recurrent_weights[0])
    states = []
    for input_vector in sequence:
        state = step(
            input_vector,
            state,
            **{
                echo_state_network.INPUT_WEIGHTS_NAME: weights.input_weights,
                echo_state_network.RECURRENT_WEIGHTS_NAME: weights.recurrent_weights,
            },
        )
        states.append(state)
    return torch.stack(states)


def apply_readout(
    *, readout: Compiled, states: torch.Tensor, readout_weights: torch.Tensor
) -> torch.Tensor:
    """`Y = H W_out` at every step, through the compiled readout."""
    return readout(states, **{echo_state_network.READOUT_NAME: readout_weights})
