"""The render module writes one page and one listing per figure through PyNCD's own
drawing route, against the tsncd bundle the dev shell exports as `TSNCD_DIST`."""

import asyncio
import pathlib

import agent_display

from reservoir import echo_state_network, render


def test_draw_writes_a_page_and_the_listing_of_every_figure(
    tmp_path: pathlib.Path,
) -> None:
    rendered = asyncio.run(render.draw(output_dir=tmp_path))
    assert [figure.page.name for figure in rendered] == ["echo-state-network.html"]
    (figure,) = rendered
    page = figure.page.read_text()
    assert "<html" in page
    assert "Echo State Network" in page
    assert figure.listing.read_text() == agent_display.listing(
        echo_state_network.echo_state_network()
    )
