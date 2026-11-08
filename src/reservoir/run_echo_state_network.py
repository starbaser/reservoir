"""Run the echo state network from end to end.

    python -m reservoir.run_echo_state_network --device cuda

The run prints the SSA listing of the PyNCD term, checks that the term and the Torch
reference compute the same readout on a short random sequence, and then performs the
delay task with the reference on `--device`: it draws a random reservoir, runs it over
a random sequence, fits the readout by ridge regression to return the input from
`--delay` steps before, and reports the normalised error on the steps after the fit.
"""

from __future__ import annotations

import argparse
import fractions
from collections.abc import Sequence
from dataclasses import dataclass

import agent_display
import data_structure.Numeric as nm
import torch

from reservoir import echo_state_network, evaluate_term, reference_torch

UNITS = 100
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 0.5
SEED = 0
AGREEMENT_STEPS = 20
"""The length of the sequence the term is evaluated on. The evaluator runs the term's
loop one step at a time in Python, so the check is kept short."""
SEQUENCE_LENGTH = 1000
WASHOUT = 100
"""The steps discarded before the fit, over which the state forgets its zero start."""
FIT_END = 700
RIDGE = 1e-8


@dataclass(frozen=True)
class Run:
    listing: str
    """The SSA listing of the term."""

    disagreement: float
    """The largest absolute difference between the term's readout and the
    reference's, on the agreement sequence."""

    device: torch.device
    """The device the delay task ran on."""

    delay: int
    """How many steps back the readout was fit to look."""

    error: float
    """The normalised error of the fitted readout on the steps after the fit."""


def rational(value: float) -> nm.Numeric:
    """`value` as the exact rational the term's leak symbol is bound to."""
    fraction = fractions.Fraction(str(value))
    return nm.Integer(fraction.numerator) / nm.Integer(fraction.denominator)


def random_sequence(*, steps: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(steps, 1, generator=generator, dtype=torch.float64) - 0.5


def disagreement_between_term_and_reference(*, leak: float) -> float:
    weights = reference_torch.random_reservoir(
        inputs=1,
        units=UNITS,
        leak=leak,
        spectral_radius=SPECTRAL_RADIUS,
        input_scale=INPUT_SCALE,
        seed=SEED,
    )
    sequence = random_sequence(steps=AGREEMENT_STEPS, seed=1)
    readout = random_sequence(steps=UNITS, seed=2)
    (from_term,) = evaluate_term.evaluate(
        term=echo_state_network.echo_state_network(leak=rational(leak)),
        inputs=(sequence,),
        sizes={"t": AGREEMENT_STEPS, "m": 1, "n": UNITS, "p": 1},
        fixed_arrays={
            echo_state_network.INPUT_WEIGHTS_NAME: weights.input_weights,
            echo_state_network.RECURRENT_WEIGHTS_NAME: weights.recurrent_weights,
        },
        linear_weights={echo_state_network.READOUT_NAME: readout},
    )
    states = reference_torch.run_reservoir(weights=weights, input_sequence=sequence)
    from_reference = reference_torch.apply_readout(states=states, readout=readout)
    return float((from_term - from_reference).abs().max())


def delay_task(*, leak: float, delay: int, device: torch.device) -> float:
    """The normalised error of a readout fit to return the input `delay` steps before,
    on the steps after the fit."""
    weights = reference_torch.random_reservoir(
        inputs=1,
        units=UNITS,
        leak=leak,
        spectral_radius=SPECTRAL_RADIUS,
        input_scale=INPUT_SCALE,
        seed=SEED,
        device=device,
    )
    sequence = random_sequence(steps=SEQUENCE_LENGTH, seed=1)
    targets = torch.cat(
        [torch.zeros(delay, 1, dtype=torch.float64), sequence[:-delay]]
    ).to(device)
    states = reference_torch.run_reservoir(weights=weights, input_sequence=sequence)
    readout = reference_torch.fit_readout(
        states=states[WASHOUT:FIT_END], targets=targets[WASHOUT:FIT_END], ridge=RIDGE
    )
    predicted = reference_torch.apply_readout(states=states[FIT_END:], readout=readout)
    return reference_torch.normalised_error(
        predicted=predicted, targets=targets[FIT_END:]
    )


def run(*, leak: float, delay: int, device: torch.device) -> Run:
    return Run(
        listing=agent_display.listing(echo_state_network.echo_state_network()),
        disagreement=disagreement_between_term_and_reference(leak=leak),
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
        f"term against reference, {AGREEMENT_STEPS} steps: "
        f"largest difference {result.disagreement:.1e}"
    )
    print(
        f"delay task on {result.device}, {UNITS} units, delay {result.delay}: "
        f"normalised error {result.error:.4f}"
    )


if __name__ == "__main__":
    main()
