"""The end-to-end run prints the listing and recovers the delayed input."""

import pytest
import torch

from reservoir import run_echo_state_network


def test_the_run_prints_the_listing_and_the_error(
    capsys: pytest.CaptureFixture,
) -> None:
    run_echo_state_network.main(["--device", "cpu", "--leak", "0.3", "--delay", "1"])
    out = capsys.readouterr().out
    assert "\\text{Echo State Network} {" in out
    assert "normalised error" in out


def test_the_delay_task_runs_on_a_gpu() -> None:
    error = run_echo_state_network.delay_task(
        leak=1.0, delay=1, device=torch.device("cuda")
    )
    assert error < 0.01
