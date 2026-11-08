# reservoir

An echo state network written as a neural circuit diagram with
[PyNCD](https://github.com/starbaser/pyncd), drawn with
[tsncd](https://github.com/starbaser/tsncd), and run in PyTorch on a GPU. The repository
is a worked example: the network as a PyNCD expression, an evaluator that runs that
expression on numbers, a plain-Torch reference that runs the same equations and fits the
readout, the diagram as a page, as figures, and in tsncd's live viewer, and a test suite
that checks every part.

## The model

An echo state network is a dynamical system driven by an input sequence. At step `t` it
receives an input vector `u_t` with `m` entries, keeps a state vector `h_t` with `n`
entries, and emits an output vector `y_t` with `p` entries:

```
h_t = (1 - alpha) h_{t-1} + alpha tanh(W_in u_t + W h_{t-1})      h_{-1} = 0
y_t = W_out h_t
```

The matrices `W_in` (`m` by `n`) and `W` (`n` by `n`) are drawn at random once and never
changed. `W` is scaled so that its largest eigenvalue has modulus below one, which makes
the state forget its past at a geometric rate, and `alpha` in `(0, 1]` is the leak rate
that sets how fast the state moves. The only matrix that is fit is `W_out` (`n` by `p`),
and it is fit by ridge regression: run the network over a sequence, collect the states
`H` (one row per step), and solve `(H^T H + lambda I) W_out = H^T Y` for the targets
`Y`. No gradient descent is involved. The network is a random nonlinear feature map over
time with a linear least-squares readout.

The neural circuit diagram is the same model drawn as wires and operations. A wire is an
array axis, labelled by its name, a box is an operation, and the repeated block is the
loop over the steps. The same expression that draws the diagram is evaluated on numbers
here, so the picture and the computation cannot drift apart.

## What is in the repository

| Path                                      | Holds                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------- |
| `src/reservoir/axes.py`                   | The four axes `t`, `m`, `n`, `p` and the arrays between the operations.               |
| `src/reservoir/echo_state_network.py`     | The network as a PyNCD expression, one function per part of the equations.            |
| `src/reservoir/evaluate_term.py`          | Runs a PyNCD expression on tensors, including the loop over the steps.                |
| `src/reservoir/reference_torch.py`        | The same equations in plain Torch, the ridge fit, and the error measure.              |
| `src/reservoir/run_echo_state_network.py` | The end-to-end run `just run` performs.                                               |
| `src/reservoir/render.py`                 | Draws the diagram as a page, as PNG and PDF, and through the live viewer.             |
| `src/reservoir/check_viewer.py`           | Drives the live viewer once in a headless browser and checks what it drew.            |
| `package.json`                            | tsncd, pinned to one commit, and the tools its bundle and its dev server need.        |
| `webpack.config.js`                       | tsncd's own webpack configuration, applied to tsncd installed under `node_modules/`.  |
| `tests/`                                  | The checks, run with `just test`.                                                     |
| `docs/echo-state-network.md`              | Each line of the equations beside its spelling in PyNCD, the listing, and the limits. |
| `skills/using-ncd/`                       | A guide to PyNCD and tsncd as a whole, written for an agent and readable by a person. |

## Requirements

- Linux with an NVIDIA driver. Torch is the CUDA 13.2 wheel, and the tests assert that
  CUDA is available and that the wheel covers every installed GPU.
  `just run --device cpu` works without one.
- Network access on the first install, for the PyNCD repository, the PyPI and PyTorch
  wheels, tsncd's npm dependencies and Playwright's Chromium.
- Either Nix with flakes, which provides every tool below, or `git`, `uv`, Node 24 with
  `npm`, `just` and `process-compose`.

## Installing with Nix

```bash
git clone https://github.com/starbaser/reservoir.git && cd reservoir
nix develop                  # or: direnv allow
npm ci                       # tsncd at its pinned commit, under node_modules/
```

The shell's `python` is a Nix-built virtual environment with this package editable,
PyNCD built from the commit `uv.lock` pins, a tsncd bundle built from the commit
`package-lock.json` pins and exported as `TSNCD_DIST`, and Playwright's Chromium from
nixpkgs. `npm ci` installs tsncd itself, which the live viewer runs from.

## Installing without Nix

```bash
git clone https://github.com/starbaser/reservoir.git && cd reservoir
uv sync                              # Python 3.14, Torch, PyNCD from its repository
uv run playwright install chromium   # the browser for the figures and the checks
npm ci                               # tsncd at its pinned commit, under node_modules/
npm run tsncd:build                  # tsncd's bundle, for the page and the figures
```

`uv` installs Python 3.14 if the machine has none. If Chromium refuses to start for want
of system libraries, `uv run playwright install-deps chromium` installs them.
`package.json` pins tsncd to one commit, and `npm ci` installs it under
`node_modules/tsncd/` like any other package. PyNCD reads the bundle from there when
`TSNCD_DIST` is unset. Every `just` recipe runs through `uv run`.

## Running the network

```bash
just run                     # on the first GPU
just run --device cpu        # without a GPU
just run --leak 0.3 --delay 5
```

The run prints three things. First, the SSA listing of the PyNCD expression, which is
the diagram written as text, one operation per line, with the loop indented. Second, the
largest difference between the readout computed by evaluating that expression and the
readout computed by the Torch reference on the same random weights, which is at the
level of float64 rounding. Third, the delay task: a reservoir of 100 units is run over a
random sequence of 1000 steps, `W_out` is fit on steps 100 to 700 to return the input
from `delay` steps before, and the normalised error (root mean square error divided by
the standard deviation of the targets) is reported on steps 700 to 1000. At the defaults
the error is 0.0083.

## Using it from Python

```python
import torch
import agent_display
import data_structure.Numeric as nm
from reservoir import echo_state_network, evaluate_term, reference_torch

network = echo_state_network.echo_state_network()  # the PyNCD expression
print(agent_display.listing(network))  # the SSA listing

weights = reference_torch.random_reservoir(
    inputs=1,
    units=100,
    leak=1.0,
    spectral_radius=0.9,
    input_scale=0.5,
    seed=0,
    device=torch.device("cuda"),
)
sequence = torch.rand(1000, 1, dtype=torch.float64) - 0.5
states = reference_torch.run_reservoir(weights=weights, input_sequence=sequence)
targets = torch.cat([torch.zeros(1, 1, dtype=torch.float64), sequence[:-1]])
readout = reference_torch.fit_readout(
    states=states[100:700], targets=targets[100:700].cuda(), ridge=1e-8
)
predicted = reference_torch.apply_readout(states=states[700:], readout=readout)
print(reference_torch.normalised_error(predicted=predicted, targets=targets[700:]))

(from_term,) = evaluate_term.evaluate(  # the expression itself, on numbers
    term=echo_state_network.echo_state_network(leak=nm.Integer(1)),
    inputs=(sequence[:20],),
    sizes={"t": 20, "m": 1, "n": 100, "p": 1},
    fixed_arrays={
        "W^{in}": weights.input_weights.cpu(),
        "W": weights.recurrent_weights.cpu(),
    },
    linear_weights={"W^{out}": readout.cpu()},
)
```

Every size in the expression is a symbol, and `sizes` binds them when the expression is
evaluated. The leak is a symbol too, `alpha` in the diagram, and the evaluated
expression is built with the rational it takes. PyNCD's own `torch_compile` does not
compile this loop, because it has no rule for the loop's state slots or for the write of
each state at its step, so `evaluate_term` runs the expression directly.
`docs/echo-state-network.md` states what PyNCD derives from the expression and what it
does not.

## Drawing the diagram

```bash
just render      # outputs/diagrams/echo-state-network.html and .txt (the listing)
just figures     # outputs/figures/echo-state-network.png and .pdf
```

The page opens from the disk with no server. Its buttons switch between three forms of
the diagram and two themes, and moving the pointer over `W_out` opens a box with the
readout's formula.

## The live viewer

tsncd's viewer is a relay and a page. A Python process sends a model to the relay, and
every open page redraws it.

```bash
just up               # tsncd's relay on localhost:8765 and page on http://127.0.0.1:3000/
just send             # send the network to the relay; the open page redraws
just capture          # write what the page draws to outputs/figures/
just check-viewer     # the whole route once in a headless browser
just logs             # the relay's log; `just logs page` for the page's
just down
```

Open `http://127.0.0.1:3000/` in a browser after `just up`. The two processes are
tsncd's own relay server and dev server, run from `node_modules/tsncd/` through the
`tsncd:relay` and `tsncd:page` scripts of `package.json` by `process-compose.yml`. The
page takes a few seconds to compile the first time. If port 3000 is taken, put
`TSNCD_PAGE_PORT=3100` in `.env`. The relay's port, 8765, is compiled into the page, so
one relay at a time holds it.

## Checks

```bash
just test        # the whole suite: the expression, the evaluator against the reference,
                 # the reference on the CPU and every GPU, the page, the figures, the run
just quality     # ruff format, ruff check, ty
```

## Reading further

- `docs/echo-state-network.md`, each line of the equations beside its spelling in PyNCD.
- `skills/using-ncd/SKILL.md`, PyNCD and tsncd as a whole: the construction syntax, the
  rules for reading a diagram, the listing, compilation, and the drawing routes.
- Vincent Abbott and Gioele Zardini, _Weaves, Wires, and Morphisms_,
  [arXiv:2604.07242](https://arxiv.org/abs/2604.07242), the method.
- [mit-zardini-lab/pyncd](https://github.com/mit-zardini-lab/pyncd) and
  [mit-zardini-lab/tsncd](https://github.com/mit-zardini-lab/tsncd), the upstream
  repositories. This project pins forks of both.
