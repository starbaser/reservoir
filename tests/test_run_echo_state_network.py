"""The end-to-end run prints the listing, finds the term and the reference in
agreement, and recovers the delayed input."""

import pytest
import torch

from reservoir import run_echo_state_network


def test_the_run_agrees_and_recovers_the_delay(capsys: pytest.CaptureFixture) -> None:
    run_echo_state_network.main(["--device", "cpu", "--leak", "0.3", "--delay", "1"])
    out = capsys.readouterr().out
    assert "\\text{Echo State Network} {" in out
    assert "largest difference" in out
    assert "normalised error" in out


def test_the_term_and_the_reference_agree_at_a_leak_of_one() -> None:
    disagreement = run_echo_state_network.disagreement_between_term_and_reference(
        leak=1.0
    )
    assert disagreement < 1e-12


def test_the_delay_task_runs_on_a_gpu() -> None:
    error = run_echo_state_network.delay_task(
        leak=1.0, delay=1, device=torch.device("cuda")
    )
    assert error < 0.01
