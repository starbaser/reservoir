"""A plain-Torch echo state network, the numerical reference for the morphism.

`torch_compile` has no rule for the tape seeds, the constant and the covariant write
that the reservoir loop holds, so the morphism in `echo_state_network` is not compiled.
This module runs the same equations in `torch.float64`: the reservoir loop, the
ridge-regression fit of the readout, which is a closed form the algebra has no operator
for, and the readout itself.

The index conventions follow the morphism. `W_in` is over `(inputs, units)` and
`W` over `(units, units)`, so a drive is `u @ W_in` and the recurrent term is
`h @ W`. `W_out` is over `(units, outputs)`, so the readout is `H @ W_out`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DTYPE = torch.float64


@dataclass(frozen=True)
class ReservoirWeights:
    input_weights: torch.Tensor
    """`W_in` over `(inputs, units)`."""

    recurrent_weights: torch.Tensor
    """`W` over `(units, units)`, scaled to the spectral radius it was built with."""

    leak: float
    """The leak rate `alpha` of the update."""


def random_reservoir(
    *,
    inputs: int,
    units: int,
    leak: float,
    spectral_radius: float,
    input_scale: float,
    seed: int,
) -> ReservoirWeights:
    """Fixed random weights: uniform entries, with `W` scaled so its largest
    eigenvalue has modulus `spectral_radius` and `W_in` scaled by `input_scale`.

    Args:
        inputs: The number of input channels.
        units: The number of reservoir units.
        leak: The leak rate of the update, in `(0, 1]`.
        spectral_radius: The modulus of the largest eigenvalue of `W` after scaling.
        input_scale: The half-width of the uniform entries of `W_in`.
        seed: The seed of the generator the entries are drawn from.
    """
    generator = torch.Generator().manual_seed(seed)
    input_weights = (
        torch.rand(inputs, units, generator=generator, dtype=DTYPE) * 2 - 1
    ) * input_scale
    unscaled = torch.rand(units, units, generator=generator, dtype=DTYPE) * 2 - 1
    radius = torch.linalg.eigvals(unscaled).abs().max()
    return ReservoirWeights(
        input_weights=input_weights,
        recurrent_weights=unscaled * (spectral_radius / radius),
        leak=leak,
    )


def run_reservoir(
    *, weights: ReservoirWeights, input_sequence: torch.Tensor
) -> torch.Tensor:
    """The state after every step of `input_sequence`, over `(steps, units)`, from the
    zero state.

    Args:
        weights: The fixed weights and the leak rate.
        input_sequence: The inputs over `(steps, inputs)`.
    """
    drives = input_sequence.to(DTYPE) @ weights.input_weights
    state = torch.zeros(weights.recurrent_weights.shape[0], dtype=DTYPE)
    states = []
    for drive in drives:
        activation = torch.tanh(drive + state @ weights.recurrent_weights)
        state = (1 - weights.leak) * state + weights.leak * activation
        states.append(state)
    return torch.stack(states)


def fit_readout(
    *, states: torch.Tensor, targets: torch.Tensor, ridge: float
) -> torch.Tensor:
    """`W_out` over `(units, outputs)` minimising `|H W_out - Y|^2 + ridge |W_out|^2`,
    which solves `(H^T H + ridge I) W_out = H^T Y`.

    Args:
        states: The states `H` over `(steps, units)`.
        targets: The targets `Y` over `(steps, outputs)`.
        ridge: The weight of the penalty on `W_out`, which has to be positive.
    """
    if ridge <= 0:
        raise ValueError(f"ridge is {ridge}, and the penalty has to be positive")
    units = states.shape[1]
    gram = states.T @ states + ridge * torch.eye(units, dtype=DTYPE)
    return torch.linalg.solve(gram, states.T @ targets.to(DTYPE))


def apply_readout(*, states: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
    """`H W_out`, the readout at every step, over `(steps, outputs)`."""
    return states @ readout


def normalised_error(*, predicted: torch.Tensor, targets: torch.Tensor) -> float:
    """The root mean square error divided by the standard deviation of the targets."""
    targets = targets.to(DTYPE)
    error = ((predicted - targets) ** 2).mean().sqrt()
    return float(error / targets.std())
