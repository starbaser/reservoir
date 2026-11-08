"""The standalone page draws the network, switches its form and opens inspection boxes.

Playwright drives the Chromium of the dev shell over the page `render` writes. The
titles and names are typeset by KaTeX, so they are read from the `.katex` elements
rather than from the page text, and the SVG backdrop lies over those elements, so the
pointer is moved to a name's position rather than hovered on the element.
"""

import asyncio
import pathlib

from playwright.async_api import Page, async_playwright

from reservoir import render

KATEX_TEXTS = "() => [...document.querySelectorAll('.katex')].map(e => e.textContent)"
CENTRE_OF_KATEX = (
    "text => { const e = [...document.querySelectorAll('#diagram .katex')]"
    ".find(e => e.textContent === text); if (!e) return null;"
    " const r = e.getBoundingClientRect();"
    " return [r.x + r.width / 2, r.y + r.height / 2]; }"
)
INSPECTION_TEXTS = (
    "() => [...document.querySelectorAll('.inspection-box')]"
    ".map(e => e.innerText.replace(/\\s+/g, ' ').trim())"
)
FORM_BUTTONS = ["Arrows and boxes", "Arrows and broadcasted", "All broadcasted"]


async def rendered_texts(page: Page) -> set[str]:
    texts = await page.evaluate(KATEX_TEXTS)
    return {text.replace("\xa0", " ").replace("​", "") for text in texts}


async def exercise(page_path: pathlib.Path) -> dict[str, object]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(viewport={"width": 1700, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(page_path.resolve().as_uri(), wait_until="load")
        await page.wait_for_function("window.tsncd !== undefined")
        await page.wait_for_selector("#diagram svg")
        broadcasted = await rendered_texts(page)
        buttons = await page.eval_on_selector_all(
            "button.display-selector-button",
            "els => els.map(e => e.textContent.trim())",
        )
        legend_rows = await page.eval_on_selector_all(
            "table.axis-legend tr", "els => els.length"
        )
        await page.get_by_role("button", name="Arrows and boxes").click()
        await page.wait_for_function(
            "() => [...document.querySelectorAll('.katex')]"
            ".some(e => e.textContent.includes('Matmul'))"
        )
        boxes = await rendered_texts(page)
        await page.get_by_role("button", name="All broadcasted").click()
        await page.wait_for_function(
            "() => ![...document.querySelectorAll('.katex')]"
            ".some(e => e.textContent.includes('Matmul'))"
        )
        readout_centre = await page.evaluate(CENTRE_OF_KATEX, "Wout")
        assert readout_centre is not None
        await page.mouse.move(*readout_centre)
        await page.wait_for_selector(".inspection-box")
        hovered = await page.evaluate(INSPECTION_TEXTS)
        await page.mouse.click(*readout_centre)
        await page.wait_for_function(
            "() => [...document.querySelectorAll('.inspection-box')]"
            ".some(e => e.innerText.includes('unlock'))"
        )
        locked = await page.evaluate(INSPECTION_TEXTS)
        title = await page.title()
        await browser.close()
    return {
        "title": title,
        "broadcasted": broadcasted,
        "buttons": buttons,
        "legend_rows": legend_rows,
        "boxes": boxes,
        "hovered": hovered,
        "locked": locked,
        "errors": errors,
    }


def test_the_page_draws_switches_form_and_opens_an_inspection_box(
    tmp_path: pathlib.Path,
) -> None:
    (figure,) = asyncio.run(render.draw(output_dir=tmp_path))
    assert figure.output is not None
    result = asyncio.run(exercise(figure.output))
    assert result["errors"] == []
    assert result["title"] == "tsncd - Echo State Network"
    assert result["buttons"][:3] == FORM_BUTTONS
    assert result["legend_rows"] == 5
    for text in ("Echo State Network", "One Step", "tanh", "Win", "Wout", "αx"):
        assert text in result["broadcasted"], text
    assert "Matmul" in result["boxes"]
    assert "[t, m]" in result["boxes"]
    (hovered,) = result["hovered"]
    assert "click to lock" in hovered
    assert "A learned linear" in hovered
    (locked,) = result["locked"]
    assert "click to unlock" in locked
