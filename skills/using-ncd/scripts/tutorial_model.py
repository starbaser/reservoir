"""Compile and draw one tutorial-style model with wires, boxes and a shared tape value.

Run in an environment providing PyNCD, PyTorch and a built TSNCD_DIST:
    python tutorial_model.py --output-dir /path/to/diagrams

Construction follows BuildingAModel sections 9–11. Parameter extraction follows
notebooks/website/tutorial/check_gradients_in_torch.py. The expansion adapter below
uses upstream operations; it is example code, not an additional PyNCD compiler API.
Verified with PyNCD and tsncd on their main branches.
"""

import argparse
import asyncio
from pathlib import Path

import agent_display
import construction_helpers  # noqa: F401 - installs composition syntax
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import para.data_structure.ParaBlockOperator as para_box
import para.data_structure.ParaWrap as para_wrap
import torch
from algebra import discovering_broadcasts
from graphs.processing import hypergraph_functor
from notebooks.display import notebook_diagrams
from para.algebra import detape, tie_tapes
from para.data_structure import Para
from para.processing import show_grabbed_parameters
from term_utilities import generate_config
from torch_compile import torch_compile


def model():
    token, width, latent = (cat.RawAxis.named(name) for name in "xmc")
    hidden = cat.Array(cat.Reals(), (token, width))
    entries = cat.Array(cat.Reals(), (token, latent))
    hold = cat.ProdObject((hidden,)).identity()
    slot = fd.DynamicName("cache").capture(Para.TapeSlot())

    publisher = ops.BlockOperator.template(
        cat.Block.template(
            (0, 0)
            @ (
                (token >> ops.Linear.template((width,), (width,), "p"))
                * (token >> ops.Linear.template((width,), (latent,), "v"))
            ),
            title=r"\text{Publish shared features}",
            fill_color="#C5BEDF",
        ),
        "Pub",
    )

    def reader(name):
        body = (
            (hold * Para.Grab(tape=slot, size=entries))
            @ (hold * (token >> ops.Linear.template((latent,), (width,), name)))
            @ ops.AdditionOp.template()
            @ ops.Arithmetic.template(nm.x * nm.Sigmoid(nm.x))
        )
        return para_box.ParaBlockOperator.template(
            cat.Block.template(
                body,
                title=rf"\text{{Read shared features {name}}}",
                fill_color="#B8D8CE",
            ),
            name,
        )

    term = (
        publisher
        @ (hold * Para.Drop(tape=slot, size=entries))
        @ reader("r")
        @ reader("s")
    )
    assert tuple(term.dom()) == tuple(term.cod()) == (hidden,)
    return term


class OpenBoxes(hypergraph_functor.Endofunctor):
    """Traverse with the upstream functor and use each box's native expansion."""

    def apply_root(self, term):
        if isinstance(term, para_wrap.ParaWrap):
            return self.apply_category(term.to_base())
        if isinstance(term, cat.Broadcasted) and isinstance(
            term.operator, ops.BlockOperator
        ):
            return self.apply_category(term.operator.expand(term))
        return term


def compile_and_check(term):
    config = generate_config.NumericConfig.template(term)
    config.assign_values(x=5, m=4, c=3)
    sized = config.apply_context(term)
    assert not generate_config.NumericConfig.template(sized).terms

    opened = discovering_broadcasts.remove_grouping_blocks(OpenBoxes()(sized))
    wired = tie_tapes.tie_tapes(opened)
    assert tuple(wired.dom()) == tuple(sized.dom())
    assert tuple(wired.cod()) == tuple(sized.cod())
    pure, tape = detape.detape(show_grabbed_parameters.grab_parameters(wired))
    assert all(is_grab for _, is_grab in tape)
    names = [name for name, _ in tape]
    assert set(names) == {"Wp", "Wv", "Wr", "Ws"}, names
    executable = torch_compile.ConstructedModule.construct(pure)

    generator = torch.Generator().manual_seed(7)
    x = torch.randn(5, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = {
        name: torch.nn.Parameter(
            torch.randn(
                *(nm.evaluate_integer(axis.local_size(), {}) for axis in array.shape()),
                generator=generator,
                dtype=torch.float64,
            )
            / 4
        )
        for name, array in zip(names, pure.dom()[1:], strict=True)
    }
    actual = executable(x, *(weights[name] for name in names))[0]
    shared = x @ weights["Wv"]
    expected = torch.nn.functional.silu(
        torch.nn.functional.silu(x @ weights["Wp"] + shared @ weights["Wr"])
        + shared @ weights["Ws"]
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    operands = [x, *weights.values()]
    actual_grad = torch.autograd.grad(actual, operands, cotangent)
    expected_grad = torch.autograd.grad(expected, operands, cotangent)
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(observed, reference, atol=1e-10, rtol=1e-10)
    print(
        f"PASS: output {tuple(actual.shape)}, input gradient, "
        f"{len(weights)} parameter gradients"
    )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    term = model()
    compile_and_check(term)
    print(agent_display.listing(term))
    settings = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.HTML,
        advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
        debug_borders=False,
        width=1000,
        page_directory=args.output_dir,
    )
    await notebook_diagrams.show_diagram(
        term,
        settings=settings,
        slug="tutorial-model",
        title="Wires and shared features",
    )


if __name__ == "__main__":
    asyncio.run(main())
