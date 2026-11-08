"""Draw every model through PyNCD's `notebook_diagrams.show_diagram`, by one of four
deliveries, and write its SSA listing beside the result.

    python -m reservoir.render --delivery page --output-dir outputs/diagrams

`page` writes one standalone HTML page per model, which opens from the disk with no
server and no network. `headless-png` and `headless-pdf` draw in a Chromium that PyNCD
starts itself through Playwright and write the image. `send` sends each model to the
tsncd page open on tsncd's relay, which redraws it. The first three read the tsncd
bundle `TSNCD_DIST` names, and `send` needs tsncd's `npm run server` and `npm run dev`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import enum
import io
import pathlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import agent_display
import data_structure.Category as cat
from notebooks.display import notebook_diagrams

from reservoir import echo_state_network


class Delivery(enum.Enum):
    PAGE = "page"
    HEADLESS_PNG = "headless-png"
    HEADLESS_PDF = "headless-pdf"
    SEND = "send"


@dataclass(frozen=True)
class Figure:
    title: str
    """The heading of the page."""

    build: Callable[[], cat.Morphism]
    """Builds the morphism the page draws."""


@dataclass(frozen=True)
class RenderedFigure:
    slug: str
    """The name the files are written under."""

    output: pathlib.Path | None
    """The page or the image written, and None for a send, which writes nothing."""

    listing: pathlib.Path
    """The SSA listing written beside it."""


FIGURES: Mapping[str, Figure] = {
    "echo-state-network": Figure(
        title="Echo State Network",
        build=lambda: echo_state_network.echo_state_network(),
    ),
}

WIDTH = 1200


def settings_for(
    *, delivery: Delivery, output_dir: pathlib.Path, slug: str
) -> notebook_diagrams.DiagramSettings:
    """The settings one delivery draws `slug` with."""
    shared = notebook_diagrams.DiagramSettings(
        width=WIDTH,
        advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
        debug_borders=False,
        page_directory=output_dir,
    )
    match delivery:
        case Delivery.PAGE:
            return dataclasses.replace(shared, mode=notebook_diagrams.DiagramMode.HTML)
        case Delivery.SEND:
            return dataclasses.replace(
                shared, mode=notebook_diagrams.DiagramMode.BROWSER
            )
        case Delivery.HEADLESS_PNG | Delivery.HEADLESS_PDF:
            image_format = "pdf" if delivery is Delivery.HEADLESS_PDF else "png"
            return dataclasses.replace(
                shared,
                mode=notebook_diagrams.DiagramMode.INLINE,
                display_format=notebook_diagrams.DisplayFormat(
                    renderer=notebook_diagrams.DiagramRenderer.HEADLESS,
                    format=image_format,
                    save_to=output_dir / f"{slug}.{image_format}",
                ),
            )


def output_of(
    *, delivery: Delivery, output_dir: pathlib.Path, slug: str
) -> pathlib.Path | None:
    match delivery:
        case Delivery.PAGE:
            return output_dir / f"{slug}.html"
        case Delivery.SEND:
            return None
        case Delivery.HEADLESS_PDF:
            return output_dir / f"{slug}.pdf"
        case Delivery.HEADLESS_PNG:
            return output_dir / f"{slug}.png"


class DiagramNotDelivered(RuntimeError):
    """`show_diagram` reported that it could not draw, in the text it prints."""


async def draw(
    *,
    output_dir: pathlib.Path,
    delivery: Delivery = Delivery.PAGE,
    figures: Mapping[str, Figure] = FIGURES,
) -> tuple[RenderedFigure, ...]:
    """Draw every figure by `delivery` under `output_dir`, write each listing, and
    return the paths.

    `show_diagram` reports a failed drawing by printing a line that starts with
    `(no diagram` rather than raising, so that a notebook run continues. Here that
    line is turned back into an error, because a delivery that wrote nothing has
    failed.

    Args:
        output_dir: The directory the pages, images and listings are written into.
        delivery: How each figure is drawn and where it goes.
        figures: The figures to draw, by the slug that names their files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for slug, figure in figures.items():
        term = figure.build()
        settings = settings_for(delivery=delivery, output_dir=output_dir, slug=slug)
        output = output_of(delivery=delivery, output_dir=output_dir, slug=slug)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            await notebook_diagrams.show_diagram(
                term, settings=settings, slug=slug, title=figure.title
            )
        for line in printed.getvalue().splitlines():
            if line.startswith("(no diagram"):
                raise DiagramNotDelivered(f"{slug}: {line}")
        if output is not None and not output.is_file():
            raise DiagramNotDelivered(f"{slug}: {output} was not written")
        listing = output_dir / f"{slug}.txt"
        listing.write_text(agent_display.listing(term))
        rendered.append(RenderedFigure(slug=slug, output=output, listing=listing))
    await notebook_diagrams.close_renderer()
    return tuple(rendered)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--delivery",
        type=Delivery,
        choices=list(Delivery),
        default=Delivery.PAGE,
        metavar="|".join(delivery.value for delivery in Delivery),
    )
    args = parser.parse_args(argv)
    for figure in asyncio.run(draw(output_dir=args.output_dir, delivery=args.delivery)):
        if figure.output is None:
            print(f"sent {figure.slug} to the open page")
        else:
            print(figure.output)
        print(figure.listing)


if __name__ == "__main__":
    main()
