"""The render module draws every figure through PyNCD's own routes, against the tsncd
bundle the dev shell exports as `TSNCD_DIST` and the Chromium Playwright is given."""

import asyncio
import pathlib

import agent_display
import pytest

from reservoir import echo_state_network, render


def listing() -> str:
    return agent_display.listing(echo_state_network.echo_state_network())


def test_the_page_delivery_writes_a_standalone_page_and_the_listing(
    tmp_path: pathlib.Path,
) -> None:
    rendered = asyncio.run(
        render.draw(output_dir=tmp_path, delivery=render.Delivery.PAGE)
    )
    assert [figure.slug for figure in rendered] == ["echo-state-network"]
    (figure,) = rendered
    assert figure.output == tmp_path / "echo-state-network.html"
    page = figure.output.read_text()
    assert "<html" in page
    assert "Echo State Network" in page
    assert figure.listing.read_text() == listing()


@pytest.mark.parametrize(
    ("delivery", "suffix", "magic"),
    [
        (render.Delivery.HEADLESS_PNG, ".png", b"\x89PNG"),
        (render.Delivery.HEADLESS_PDF, ".pdf", b"%PDF"),
    ],
    ids=["png", "pdf"],
)
def test_a_headless_delivery_draws_the_figure_in_its_own_chromium(
    tmp_path: pathlib.Path, delivery: render.Delivery, suffix: str, magic: bytes
) -> None:
    (figure,) = asyncio.run(render.draw(output_dir=tmp_path, delivery=delivery))
    assert figure.output == tmp_path / f"echo-state-network{suffix}"
    image = figure.output.read_bytes()
    assert image[: len(magic)] == magic
    assert len(image) > 10_000
    assert figure.listing.read_text() == listing()
