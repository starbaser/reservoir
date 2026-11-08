"""The structure of the echo state network as a morphism.

The loop cannot be compiled by `torch_compile`, so these checks read the term: its
domain and codomain, the loop and its loop variables, the operators the listing reports,
the one learned array, and the sizes a configuration binds.
"""

import agent_display
import data_structure.Category as cat
import data_structure.Numeric as nm
import para.data_structure.Para as Para
import para.processing.show_grabbed_parameters as show_grabbed_parameters
import term_utilities.generate_config as generate_config
import term_utilities.term_utilities as term_utilities

from reservoir import echo_state_network
from reservoir.axes import (
    DRIVE,
    INPUT_SEQUENCE,
    READOUT,
    SMALL_SIZES,
    STATE,
    STATES,
    STEP_COUNTER,
    outputs,
    steps,
    units,
)

RATIONAL_LEAK = nm.Integer(3) / nm.Integer(10)


def free_symbols(term: cat.Morphism) -> set[nm.FreeNumeric]:
    return set(generate_config.NumericConfig.template(term).terms)


def test_the_network_maps_the_input_sequence_to_the_readout() -> None:
    network = echo_state_network.echo_state_network()
    assert tuple(network.dom()) == (INPUT_SEQUENCE,)
    assert tuple(network.cod()) == (READOUT,)


def test_the_reservoir_reads_the_drive_and_returns_the_states() -> None:
    reservoir = echo_state_network.reservoir_loop(leak=echo_state_network.LEAK)
    assert tuple(reservoir.dom()) == (DRIVE,)
    assert tuple(reservoir.cod()) == (STATES,)


def test_one_step_is_repeated_once_per_step_with_equal_domain_and_codomain() -> None:
    network = echo_state_network.echo_state_network()
    loops = [
        block
        for block in term_utilities.type_search(cat.Block, network)
        if block.repetition != nm.Integer(1)
    ]
    assert len(loops) == 1
    (loop,) = loops
    assert loop.repetition == steps.local_size()
    assert tuple(loop.dom()) == tuple(loop.cod()) == (STATES,)


def test_the_state_and_the_states_are_the_two_loop_variables() -> None:
    network = echo_state_network.echo_state_network()
    grabs = {
        (grab.tape, grab.size)
        for grab in term_utilities.type_search(Para.StreamGrab, network)
    }
    drops = {
        (drop.tape, drop.size)
        for drop in term_utilities.type_search(Para.StreamDrop, network)
    }
    expected = {
        (echo_state_network.STATE_SLOT, STATE),
        (echo_state_network.STATES_SLOT, STATES),
    }
    assert grabs == expected
    assert drops == expected


def test_the_summary_reports_every_operator_of_the_update() -> None:
    summary = agent_display.summary(echo_state_network.echo_state_network())
    assert "in    %0 : R[t, m]" in summary
    assert "out   %1 : R[t, p]" in summary
    assert (
        "ops   3x AdditionOp, 3x Arithmetic, 2x ConstantOp, 1x CovariantView, "
        "2x Einops, 2x FixedArray, 1x Linear, 1x View"
    ) in summary


def test_the_listing_states_the_loop_and_its_tape() -> None:
    listing = agent_display.listing(echo_state_network.echo_state_network())
    assert "loop |t| times {" in listing
    assert "= StreamGrab<h>() : R[n]" in listing
    assert "= StreamGrab<H>() : R[t, n]" in listing
    assert "= StreamDrop<h>(" in listing
    assert "= StreamDrop<H>(" in listing
    assert "= FixedArray<W^{in}>() : R[{m}, {n}]" in listing
    assert "= FixedArray<W>() : R[{n}, {n}]" in listing
    assert "= Arithmetic<\\tanh>(" in listing
    assert ": R[t, {p}]" in listing


def test_the_readout_is_the_only_learned_array() -> None:
    network = echo_state_network.echo_state_network()
    parameters = [
        (name.to_bodies(), tuple(array.shape()))
        for seed in term_utilities.type_search(cat.Broadcasted, network)
        for name, array in show_grabbed_parameters.parameter_arrays(seed)
    ]
    assert parameters == [(echo_state_network.READOUT_NAME, (units, outputs))]


def test_binding_the_sizes_leaves_the_counter_and_a_symbolic_leak() -> None:
    network = echo_state_network.echo_state_network()
    config = generate_config.NumericConfig.template(network)
    assert {symbol.uid._name.to_bodies() for symbol in config.terms} == {
        "\\alpha",
        "it",
        "m",
        "n",
        "p",
        "t",
    }
    config.assign_values(**SMALL_SIZES)
    assert free_symbols(config.apply_context(network)) == {
        STEP_COUNTER,
        echo_state_network.LEAK,
    }


def test_binding_the_sizes_of_a_network_with_a_rational_leak_leaves_the_counter() -> (
    None
):
    network = echo_state_network.echo_state_network(leak=RATIONAL_LEAK)
    config = generate_config.NumericConfig.template(network)
    config.assign_values(**SMALL_SIZES)
    assert free_symbols(config.apply_context(network)) == {STEP_COUNTER}
