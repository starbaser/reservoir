"""The PyNCD term computes the same numbers as the Torch reference.

The term is evaluated with `evaluate_term`, which runs its loop, its tape and its
reindexings on tensors, and the reference runs the equations directly. Agreement to
float64 tolerance is what ties the diagram to the numbers.
"""

import data_structure.Numeric as nm
import pytest
import torch

from reservoir import echo_state_network, evaluate_term, reference_torch
from reservoir.axes import SMALL_SIZES

LEAK = 0.3
RATIONAL_LEAK = nm.Integer(3) / nm.Integer(10)


def arrays() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The input sequence, `W_in`, `W` and `W_out` at the small sizes, from one seed."""
    generator = torch.Generator().manual_seed(3)
    steps, inputs, units, outputs = (SMALL_SIZES[name] for name in "tmnp")

    def draw(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator, dtype=torch.float64) - 0.5

    return (
        draw(steps, inputs),
        draw(inputs, units),
        draw(units, units),
        draw(units, outputs),
    )


def test_the_term_computes_the_reference_readout() -> None:
    sequence, input_weights, recurrent_weights, readout_weights = arrays()
    (from_term,) = evaluate_term.evaluate(
        term=echo_state_network.echo_state_network(leak=RATIONAL_LEAK),
        inputs=(sequence,),
        sizes=SMALL_SIZES,
        fixed_arrays={
            echo_state_network.INPUT_WEIGHTS_NAME: input_weights,
            echo_state_network.RECURRENT_WEIGHTS_NAME: recurrent_weights,
        },
        linear_weights={echo_state_network.READOUT_NAME: readout_weights},
    )
    weights = reference_torch.ReservoirWeights(
        input_weights=input_weights, recurrent_weights=recurrent_weights, leak=LEAK
    )
    states = reference_torch.run_reservoir(weights=weights, input_sequence=sequence)
    expected = reference_torch.apply_readout(states=states, readout=readout_weights)
    assert tuple(from_term.shape) == (SMALL_SIZES["t"], SMALL_SIZES["p"])
    torch.testing.assert_close(from_term, expected, rtol=1e-12, atol=1e-12)


def test_the_reservoir_block_alone_returns_the_states() -> None:
    sequence, input_weights, recurrent_weights, _ = arrays()
    drive = sequence @ input_weights
    (from_term,) = evaluate_term.evaluate(
        term=echo_state_network.reservoir_loop(leak=RATIONAL_LEAK),
        inputs=(drive,),
        sizes=SMALL_SIZES,
        fixed_arrays={echo_state_network.RECURRENT_WEIGHTS_NAME: recurrent_weights},
        linear_weights={},
    )
    weights = reference_torch.ReservoirWeights(
        input_weights=input_weights, recurrent_weights=recurrent_weights, leak=LEAK
    )
    expected = reference_torch.run_reservoir(weights=weights, input_sequence=sequence)
    torch.testing.assert_close(from_term, expected, rtol=1e-12, atol=1e-12)


def test_a_missing_fixed_array_is_reported_by_name() -> None:
    sequence, _, recurrent_weights, readout_weights = arrays()
    with pytest.raises(evaluate_term.ArrayNotGiven, match=r"FixedArray W\^\{in\}"):
        evaluate_term.evaluate(
            term=echo_state_network.echo_state_network(leak=RATIONAL_LEAK),
            inputs=(sequence,),
            sizes=SMALL_SIZES,
            fixed_arrays={echo_state_network.RECURRENT_WEIGHTS_NAME: recurrent_weights},
            linear_weights={echo_state_network.READOUT_NAME: readout_weights},
        )
