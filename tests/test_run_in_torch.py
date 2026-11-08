"""The step PyNCD compiles computes the update the equations state, on the CPU and on
every CUDA device of the machine, and the readout fit by ridge regression recovers a
delayed input.

The delay task asks the readout to return the input from `delay` steps before. A
reservoir of 100 units at spectral radius 0.9 recovers a delay of one step with a
normalised error of 0.0083 and a delay of ten steps with 0.082, observed with
Torch 2.15.0+cu132 in float64 on the CPU, an RTX A6000 and an RTX PRO 6000.
"""

import pytest
import torch

from reservoir import echo_state_network, reservoir_weights, run_in_torch

UNITS = 100
WASHOUT = 100
FIT_END = 700
SEQUENCE_LENGTH = 1000
DEVICES = [
    torch.device("cpu"),
    *(torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())),
]


def weights(
    device: torch.device = reservoir_weights.CPU, leak: float = 1.0
) -> reservoir_weights.ReservoirWeights:
    return reservoir_weights.random_reservoir(
        inputs=1,
        units=UNITS,
        leak=leak,
        spectral_radius=0.9,
        input_scale=0.5,
        seed=0,
        device=device,
    )


def input_sequence() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return (
        torch.rand(SEQUENCE_LENGTH, 1, generator=generator, dtype=torch.float64) - 0.5
    )


def delayed(*, sequence: torch.Tensor, delay: int) -> torch.Tensor:
    return torch.cat([torch.zeros(delay, 1, dtype=torch.float64), sequence[:-delay]])


def states_of(reservoir: reservoir_weights.ReservoirWeights) -> torch.Tensor:
    step = run_in_torch.compile_step(leak=reservoir.leak, inputs=1, units=UNITS)
    return run_in_torch.run_reservoir(
        step=step, weights=reservoir, input_sequence=input_sequence()
    )


def test_cuda_is_available_and_the_build_covers_every_installed_gpu() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
    supported = torch.cuda.get_arch_list()
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        assert f"sm_{major}{minor}" in supported


def test_the_compiled_step_takes_the_two_weights_by_name() -> None:
    step = run_in_torch.compile_step(leak=0.3, inputs=2, units=5)
    assert set(step.weight_names) == {
        echo_state_network.INPUT_WEIGHTS_NAME,
        echo_state_network.RECURRENT_WEIGHTS_NAME,
    }


@pytest.mark.parametrize("leak", [1.0, 0.3])
def test_the_compiled_step_computes_the_update(leak: float) -> None:
    generator = torch.Generator().manual_seed(2)
    input_vector = torch.rand(2, generator=generator, dtype=torch.float64)
    state = torch.rand(5, generator=generator, dtype=torch.float64)
    input_weights = torch.rand(2, 5, generator=generator, dtype=torch.float64)
    recurrent_weights = torch.rand(5, 5, generator=generator, dtype=torch.float64)
    step = run_in_torch.compile_step(leak=leak, inputs=2, units=5)
    computed = step(
        input_vector,
        state,
        **{
            echo_state_network.INPUT_WEIGHTS_NAME: input_weights,
            echo_state_network.RECURRENT_WEIGHTS_NAME: recurrent_weights,
        },
    )
    expected = (1 - leak) * state + leak * torch.tanh(
        input_vector @ input_weights + state @ recurrent_weights
    )
    torch.testing.assert_close(computed, expected, rtol=1e-12, atol=1e-12)


def test_the_compiled_readout_applies_the_weight_at_every_step() -> None:
    generator = torch.Generator().manual_seed(3)
    states = torch.rand(6, 5, generator=generator, dtype=torch.float64)
    readout_weights = torch.rand(5, 1, generator=generator, dtype=torch.float64)
    readout = run_in_torch.compile_readout(steps=6, units=5, outputs=1)
    computed = run_in_torch.apply_readout(
        readout=readout, states=states, readout_weights=readout_weights
    )
    torch.testing.assert_close(computed, states @ readout_weights)


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_the_recurrent_weights_are_scaled_to_the_spectral_radius(
    device: torch.device,
) -> None:
    reservoir = weights(device)
    radius = torch.linalg.eigvals(reservoir.recurrent_weights).abs().max()
    assert float(radius) == pytest.approx(0.9, abs=1e-12)
    assert tuple(reservoir.input_weights.shape) == (1, UNITS)
    assert reservoir.input_weights.device == device
    assert reservoir.recurrent_weights.device == device


@pytest.mark.parametrize("device", DEVICES[1:], ids=str)
def test_a_gpu_computes_the_same_states_as_the_cpu(device: torch.device) -> None:
    on_cpu = states_of(weights())
    on_gpu = states_of(weights(device))
    assert on_gpu.device == device
    torch.testing.assert_close(on_gpu.cpu(), on_cpu)


def test_the_first_two_states_follow_the_update_from_zero() -> None:
    reservoir = weights()
    sequence = input_sequence()
    states = states_of(reservoir)
    first = torch.tanh(sequence[0] @ reservoir.input_weights)
    second = torch.tanh(
        sequence[1] @ reservoir.input_weights + first @ reservoir.recurrent_weights
    )
    assert tuple(states.shape) == (SEQUENCE_LENGTH, UNITS)
    torch.testing.assert_close(states[0], first)
    torch.testing.assert_close(states[1], second)


def test_the_readout_solves_the_normal_equations() -> None:
    states = states_of(weights())[WASHOUT:FIT_END]
    targets = delayed(sequence=input_sequence(), delay=1)[WASHOUT:FIT_END]
    readout = reservoir_weights.fit_readout(states=states, targets=targets, ridge=1e-8)
    assert tuple(readout.shape) == (UNITS, 1)
    gram = states.T @ states + 1e-8 * torch.eye(UNITS, dtype=torch.float64)
    torch.testing.assert_close(gram @ readout, states.T @ targets)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize(("delay", "bound"), [(1, 0.01), (10, 0.1)])
def test_the_readout_recovers_a_delayed_input(
    delay: int, bound: float, device: torch.device
) -> None:
    sequence = input_sequence()
    targets = delayed(sequence=sequence, delay=delay)
    states = states_of(weights(device))
    assert states.device == device
    readout_weights = reservoir_weights.fit_readout(
        states=states[WASHOUT:FIT_END], targets=targets[WASHOUT:FIT_END], ridge=1e-8
    )
    readout = run_in_torch.compile_readout(
        steps=SEQUENCE_LENGTH - FIT_END, units=UNITS, outputs=1
    )
    predicted = run_in_torch.apply_readout(
        readout=readout, states=states[FIT_END:], readout_weights=readout_weights
    )
    error = reservoir_weights.normalised_error(
        predicted=predicted, targets=targets[FIT_END:]
    )
    assert error < bound


def test_a_ridge_that_is_not_positive_is_rejected() -> None:
    states = torch.zeros(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"ridge is 0.0, and the penalty has to be"):
        reservoir_weights.fit_readout(states=states, targets=states, ridge=0.0)
