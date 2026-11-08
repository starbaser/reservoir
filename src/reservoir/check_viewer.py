"""Drive the page viewer once, end to end, and write what it drew.

    python -m reservoir.check_viewer --output-dir outputs/figures

With the relay and the site of `just up` running, the check opens the served page in
Playwright's Chromium, which connects to the relay as a tsncd page does, sends every
model through the relay with `render.Delivery.SEND`, waits until the page's own DOM
shows each model's title, asks the page for its drawing with `render.Delivery.CAPTURE`,
which goes through the relay alone and never falls back to a headless browser, and
writes a screenshot of the page beside the captured PNG. It fails if the page does not
connect, if a send is not drawn, or if a capture is not a PNG. `SITE_PORT` names the
site's port, default 8780, and PyNCD's `PORT` names the relay's, default 8765.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

from playwright.async_api import async_playwright

from reservoir import render

PNG_MAGIC = b"\x89PNG"
CONNECTED_MESSAGE = "Connected"
"""What the page logs to the console once its socket to the relay is open."""
CONNECT_TIMEOUT_SECONDS = 15
DRAW_TIMEOUT_SECONDS = 30


class ViewerNotWorking(RuntimeError):
    """The page did not connect, did not draw, or did not answer the capture."""


@dataclass(frozen=True)
class ViewerCheck:
    captures: tuple[pathlib.Path, ...]
    """The PNGs the page drew, one per model."""

    screenshot: pathlib.Path
    """A screenshot of the live page after the last model was drawn."""


def site_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('SITE_PORT', '8780')}/"


async def check(*, output_dir: pathlib.Path) -> ViewerCheck:
    output_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(viewport={"width": 1700, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        connected = asyncio.Event()
        page.on(
            "console",
            lambda message: (
                connected.set() if message.text == CONNECTED_MESSAGE else None
            ),
        )
        await page.goto(site_url(), wait_until="load")
        await page.wait_for_function("window.tsncd !== undefined")
        try:
            await asyncio.wait_for(connected.wait(), CONNECT_TIMEOUT_SECONDS)
        except TimeoutError as error:
            raise ViewerNotWorking(
                f"the page at {site_url()} did not connect to the relay within "
                f"{CONNECT_TIMEOUT_SECONDS} s"
            ) from error
        sent = await render.draw(output_dir=output_dir, delivery=render.Delivery.SEND)
        for figure in sent:
            title = render.FIGURES[figure.slug].title.replace(" ", "\u00a0")
            try:
                await page.wait_for_function(
                    "title => [...document.querySelectorAll('.katex')]"
                    ".some(e => e.textContent === title)",
                    arg=title,
                    timeout=DRAW_TIMEOUT_SECONDS * 1000,
                )
            except Exception as error:
                raise ViewerNotWorking(
                    f"the page did not draw {figure.slug} within "
                    f"{DRAW_TIMEOUT_SECONDS} s"
                ) from error
        captured = await render.draw(
            output_dir=output_dir, delivery=render.Delivery.CAPTURE
        )
        screenshot = output_dir / "viewer-page.png"
        await page.screenshot(path=str(screenshot))
        await browser.close()
    if errors:
        raise ViewerNotWorking(f"the page raised: {errors}")
    captures = []
    for figure in captured:
        if figure.output is None or figure.output.read_bytes()[:4] != PNG_MAGIC:
            raise ViewerNotWorking(f"{figure.slug}: the page returned no PNG")
        captures.append(figure.output)
    if len(sent) != len(captures):
        raise ViewerNotWorking(f"sent {len(sent)} models and captured {len(captures)}")
    return ViewerCheck(captures=tuple(captures), screenshot=screenshot)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(check(output_dir=args.output_dir))
    for capture in result.captures:
        print(f"captured {capture}")
    print(f"screenshot {result.screenshot}")


if __name__ == "__main__":
    main()
