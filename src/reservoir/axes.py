"""The axes, the arrays and the loop counter of the echo state network.

Every axis is declared once here and passed to the constructions that read it, because
PyNCD identifies an axis by its uid rather than by its name. No size appears in an
expression. Each axis carries a free symbol, and a `NumericConfig` binds the symbols by
the body of their names, as `SMALL_SIZES` does for the checks.
"""

from __future__ import annotations

import data_structure.Category as cat
import data_structure.Numeric as nm

R = cat.Reals()

steps = cat.RawAxis.named("t")
"""The time steps of the input sequence, which the reservoir loop runs over."""

inputs = cat.RawAxis.named("m")
"""The channels of one input vector."""

units = cat.RawAxis.named("n")
"""The units of the reservoir, which is the size of its state."""

outputs = cat.RawAxis.named("p")
"""The channels of one readout vector."""

STEP_COUNTER_NAME = "i_t"
STEP_COUNTER = nm.FreeNumeric.named(STEP_COUNTER_NAME)
"""The counter of the loop over the steps, naming the step an iteration computes."""

INPUT_SEQUENCE = cat.Array(R, (steps, inputs))
DRIVE = cat.Array(R, (steps, units))
"""`W_in u_t` at every step, computed before the loop."""
STATE = cat.Array(R, (units,))
STATES = cat.Array(R, (steps, units))
"""The state after every step, which the readout reads."""
READOUT = cat.Array(R, (steps, outputs))

SMALL_SIZES: dict[str, int] = {"t": 6, "m": 2, "n": 5, "p": 1}
"""The sizes the network is checked at, by the body of each axis name."""
