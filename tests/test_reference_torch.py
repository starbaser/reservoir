"""The Torch reference computes the update the morphism states and fits its readout.

The delay task asks the readout to return the input from `delay` steps before. A
reservoir of 100 units at spectral radius 0.9 recovers a delay of one step with a
normalised error of 0.0083 and a delay of ten steps with 0.082, observed with
Torch 2.15.0 on CPU in float64.
"""

import pytest
import torch

from reservoir import reference_torch

WASHOUT = 100
FIT_END = 700
SEQUENCE_LENGTH = 1000


def weights() -> reference_torch.ReservoirWeights:
    return reference_torch.random_reservoir(
        inputs=1, units=100, leak=1.0, spectral_radius=0.9, input_scale=0.5, seed=0
    )


def input_sequence() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return (
        torch.rand(SEQUENCE_LENGTH, 1, generator=generator, dtype=torch.float64) - 0.5
    )


def delayed(*, sequence: torch.Tensor, delay: int) -> torch.Tensor:
    return torch.cat([torch.zeros(delay, 1, dtype=torch.float64), sequence[:-delay]])


def test_the_recurrent_weights_are_scaled_to_the_spectral_radius() -> None:
    radius = torch.linalg.eigvals(weights().recurrent_weights).abs().max()
    assert float(radius) == pytest.approx(0.9, abs=1e-12)
    assert tuple(weights().input_weights.shape) == (1, 100)


def test_the_first_two_states_follow_the_update_from_zero() -> None:
    reservoir = weights()
    sequence = input_sequence()
    states = reference_torch.run_reservoir(weights=reservoir, input_sequence=sequence)
    first = torch.tanh(sequence[0] @ reservoir.input_weights)
    second = torch.tanh(
        sequence[1] @ reservoir.input_weights + first @ reservoir.recurrent_weights
    )
    assert tuple(states.shape) == (SEQUENCE_LENGTH, 100)
    torch.testing.assert_close(states[0], first)
    torch.testing.assert_close(states[1], second)


def test_a_leak_below_one_keeps_part_of_the_previous_state() -> None:
    leaky = reference_torch.random_reservoir(
        inputs=1, units=100, leak=0.3, spectral_radius=0.9, input_scale=0.5, seed=0
    )
    sequence = input_sequence()
    states = reference_torch.run_reservoir(weights=leaky, input_sequence=sequence)
    first = 0.3 * torch.tanh(sequence[0] @ leaky.input_weights)
    second = 0.7 * first + 0.3 * torch.tanh(
        sequence[1] @ leaky.input_weights + first @ leaky.recurrent_weights
    )
    torch.testing.assert_close(states[0], first)
    torch.testing.assert_close(states[1], second)


def test_the_readout_solves_the_normal_equations() -> None:
    states = reference_torch.run_reservoir(
        weights=weights(), input_sequence=input_sequence()
    )[WASHOUT:FIT_END]
    targets = delayed(sequence=input_sequence(), delay=1)[WASHOUT:FIT_END]
    readout = reference_torch.fit_readout(states=states, targets=targets, ridge=1e-8)
    assert tuple(readout.shape) == (100, 1)
    gram = states.T @ states + 1e-8 * torch.eye(100, dtype=torch.float64)
    torch.testing.assert_close(gram @ readout, states.T @ targets)


@pytest.mark.parametrize(("delay", "bound"), [(1, 0.01), (10, 0.1)])
def test_the_readout_recovers_a_delayed_input(delay: int, bound: float) -> None:
    sequence = input_sequence()
    targets = delayed(sequence=sequence, delay=delay)
    states = reference_torch.run_reservoir(weights=weights(), input_sequence=sequence)
    readout = reference_torch.fit_readout(
        states=states[WASHOUT:FIT_END], targets=targets[WASHOUT:FIT_END], ridge=1e-8
    )
    predicted = reference_torch.apply_readout(states=states[FIT_END:], readout=readout)
    error = reference_torch.normalised_error(
        predicted=predicted, targets=targets[FIT_END:]
    )
    assert error < bound


def test_a_ridge_that_is_not_positive_is_rejected() -> None:
    states = torch.zeros(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"ridge is 0.0, and the penalty has to be"):
        reference_torch.fit_readout(states=states, targets=states, ridge=0.0)
