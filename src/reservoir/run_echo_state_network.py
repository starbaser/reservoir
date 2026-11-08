"""Run the echo state network from end to end.

    python -m reservoir.run_echo_state_network --device cuda

The run prints the SSA listing of the PyNCD morphism, compiles its step and its readout
with PyNCD's `torch_compile`, and performs the delay task on `--device`: it draws a
random reservoir, runs the compiled step over a random sequence, fits the readout by
ridge regression to return the input from `--delay` steps before, and reports the
normalised error on the steps after the fit.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import agent_display
import torch

from reservoir import echo_state_network, reservoir_weights, run_in_torch

UNITS = 100
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 0.5
SEED = 0
SEQUENCE_LENGTH = 1000
WASHOUT = 100
"""The steps discarded before the fit, over which the state forgets its zero start."""
FIT_END = 700
RIDGE = 1e-8


@dataclass(frozen=True)
class Run:
    listing: str
    """The SSA listing of the morphism."""

    device: torch.device
    """The device the delay task ran on."""

    delay: int
    """How many steps back the readout was fit to look."""

    error: float
    """The normalised error of the fitted readout on the steps after the fit."""


def random_sequence(*, steps: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(steps, 1, generator=generator, dtype=torch.float64) - 0.5


def delay_task(*, leak: float, delay: int, device: torch.device) -> float:
    """The normalised error of a readout fit to return the input `delay` steps before,
    on the steps after the fit."""
    weights = reservoir_weights.random_reservoir(
        inputs=1,
        units=UNITS,
        leak=leak,
        spectral_radius=SPECTRAL_RADIUS,
        input_scale=INPUT_SCALE,
        seed=SEED,
        device=device,
    )
    step = run_in_torch.compile_step(leak=leak, inputs=1, units=UNITS)
    readout = run_in_torch.compile_readout(
        steps=SEQUENCE_LENGTH - FIT_END, units=UNITS, outputs=1
    )
    sequence = random_sequence(steps=SEQUENCE_LENGTH, seed=1)
    targets = torch.cat(
        [torch.zeros(delay, 1, dtype=torch.float64), sequence[:-delay]]
    ).to(device)
    states = run_in_torch.run_reservoir(
        step=step, weights=weights, input_sequence=sequence
    )
    readout_weights = reservoir_weights.fit_readout(
        states=states[WASHOUT:FIT_END], targets=targets[WASHOUT:FIT_END], ridge=RIDGE
    )
    predicted = run_in_torch.apply_readout(
        readout=readout, states=states[FIT_END:], readout_weights=readout_weights
    )
    return reservoir_weights.normalised_error(
        predicted=predicted, targets=targets[FIT_END:]
    )


def run(*, leak: float, delay: int, device: torch.device) -> Run:
    return Run(
        listing=agent_display.listing(echo_state_network.echo_state_network()),
        device=device,
        delay=delay,
        error=delay_task(leak=leak, delay=delay, device=device),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--leak", type=float, default=1.0)
    parser.add_argument("--delay", type=int, default=1)
    args = parser.parse_args(argv)
    result = run(leak=args.leak, delay=args.delay, device=args.device)
    print(result.listing)
    print(
        f"delay task on {result.device}, {UNITS} units, delay {result.delay}: "
        f"normalised error {result.error:.4f}"
    )


if __name__ == "__main__":
    main()
