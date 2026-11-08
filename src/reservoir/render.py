"""One interactive HTML page and one SSA listing per model, under an output directory.

Every page is drawn through PyNCD's `notebook_diagrams.show_diagram` against the tsncd
bundle that `TSNCD_DIST` names, which the dev shell exports. A page opens with no
server and no network. The listing is the form an agent reads a morphism in.

    python -m reservoir.render --output-dir outputs/diagrams
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import pathlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import agent_display
import data_structure.Category as cat
from notebooks.display import notebook_diagrams

from reservoir import echo_state_network


@dataclass(frozen=True)
class Figure:
    title: str
    """The heading of the page."""

    build: Callable[[], cat.Morphism]
    """Builds the morphism the page draws."""


@dataclass(frozen=True)
class RenderedFigure:
    page: pathlib.Path
    """The HTML page written."""

    listing: pathlib.Path
    """The SSA listing written beside it."""


FIGURES: Mapping[str, Figure] = {
    "echo-state-network": Figure(
        title="Echo State Network",
        build=lambda: echo_state_network.echo_state_network(),
    ),
}


async def draw(
    *, output_dir: pathlib.Path, figures: Mapping[str, Figure] = FIGURES
) -> tuple[RenderedFigure, ...]:
    """Write every figure's page and listing under `output_dir` and return the paths.

    Args:
        output_dir: The directory the pages and listings are written into.
        figures: The figures to draw, by the slug that names their files.
    """
    settings = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.HTML,
        width=1200,
        advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
        debug_borders=False,
        page_directory=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for slug, figure in figures.items():
        term = figure.build()
        with contextlib.redirect_stdout(io.StringIO()):
            await notebook_diagrams.show_diagram(
                term, settings=settings, slug=slug, title=figure.title
            )
        listing = output_dir / f"{slug}.txt"
        listing.write_text(agent_display.listing(term))
        rendered.append(
            RenderedFigure(page=output_dir / f"{slug}.html", listing=listing)
        )
    return tuple(rendered)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    for figure in asyncio.run(draw(output_dir=args.output_dir)):
        print(figure.page)
        print(figure.listing)


if __name__ == "__main__":
    main()
