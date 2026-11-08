# reservoir

An echo state network written as a neural circuit diagram with
[PyNCD](https://github.com/starbaser/pyncd), drawn with
[tsncd](https://github.com/starbaser/tsncd), and run in PyTorch on a GPU. The repository
is a worked example: the network as a PyNCD expression, its step and its readout
compiled to PyTorch by PyNCD, the ridge fit of the readout, the diagram as a page, as
figures, and in tsncd's live viewer, and a test suite that checks every part.

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
loop over the steps. PyNCD compiles the step of that same expression to PyTorch, so the
picture and the computation cannot drift apart.

## What is in the repository

| Path                                      | Holds                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------- |
| `src/reservoir/axes.py`                   | The four axes `t`, `m`, `n`, `p` and the arrays between the operations.               |
| `src/reservoir/echo_state_network.py`     | The network as a PyNCD expression, one function per part of the equations.            |
| `src/reservoir/run_in_torch.py`           | Compiles the step and the readout with PyNCD's `torch_compile` and runs the loop.     |
| `src/reservoir/reservoir_weights.py`      | The random weights, the ridge fit of the readout, and the error measure.              |
| `src/reservoir/run_echo_state_network.py` | The end-to-end run `just run` performs.                                               |
| `src/reservoir/render.py`                 | Draws the diagram as a page, as PNG and PDF, and through the live viewer.             |
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
  `npm` and `just`.

## Installing with Nix

```bash
git clone https://github.com/starbaser/reservoir.git && cd reservoir
nix develop                  # or: direnv allow
```

The shell's `python` is a Nix-built virtual environment with this package editable,
PyNCD built from the commit `uv.lock` pins, tsncd's bundle built from the commit
`flake.lock` pins and exported as `TSNCD_DIST`, and Playwright's Chromium from nixpkgs.

## Installing without Nix

```bash
git clone https://github.com/starbaser/reservoir.git && cd reservoir
uv sync                              # Python 3.14, Torch, PyNCD from its repository
uv run playwright install chromium   # the browser for the figures and the checks
```

`uv` installs Python 3.14 if the machine has none. If Chromium refuses to start for want
of system libraries, `uv run playwright install-deps chromium` installs them. The pages
and the figures also need tsncd's bundle: clone tsncd beside this repository as its
README says, build it, and name the bundle.

```bash
git clone https://github.com/starbaser/tsncd.git ../tsncd
git -C ../tsncd checkout b8649a6492c15244cf2b0be49e0acf2e2abb32d7
(cd ../tsncd && npm install && npm run build)
export TSNCD_DIST="$PWD/../tsncd/dist"
```

Every `just` recipe runs through `uv run`.

## Running the network

```bash
just run                     # on the first GPU
just run --device cpu        # without a GPU
just run --leak 0.3 --delay 5
```

The run prints two things. First, the SSA listing of the PyNCD expression, which is the
diagram written as text, one operation per line, with the loop indented. Second, the
delay task: PyNCD compiles the step and the readout of that expression to PyTorch, a
reservoir of 100 units is run over a random sequence of 1000 steps by calling the
compiled step once per step, `W_out` is fit on steps 100 to 700 to return the input from
`delay` steps before, and the normalised error (root mean square error divided by the
standard deviation of the targets) is reported on steps 700 to 1000. At the defaults the
error is 0.0083.

## Using it from Python

```python
import torch
import agent_display
from reservoir import echo_state_network, reservoir_weights, run_in_torch

network = echo_state_network.echo_state_network()  # the PyNCD expression
print(agent_display.listing(network))  # the SSA listing

step = run_in_torch.compile_step(leak=1.0, inputs=1, units=100)  # torch_compile
readout = run_in_torch.compile_readout(steps=300, units=100, outputs=1)

weights = reservoir_weights.random_reservoir(
    inputs=1,
    units=100,
    leak=1.0,
    spectral_radius=0.9,
    input_scale=0.5,
    seed=0,
    device=torch.device("cuda"),
)
sequence = torch.rand(1000, 1, dtype=torch.float64) - 0.5
states = run_in_torch.run_reservoir(step=step, weights=weights, input_sequence=sequence)
targets = torch.cat([torch.zeros(1, 1, dtype=torch.float64), sequence[:-1]]).cuda()
readout_weights = reservoir_weights.fit_readout(
    states=states[100:700], targets=targets[100:700], ridge=1e-8
)
predicted = run_in_torch.apply_readout(
    readout=readout, states=states[700:], readout_weights=readout_weights
)
print(reservoir_weights.normalised_error(predicted=predicted, targets=targets[700:]))
```

`compile_step` and `compile_readout` follow the route of PyNCD's own tutorial: every
weight is moved onto the tape, every size is bound to an integer, and `torch_compile`
builds the module, which takes the weights as operands. PyNCD compiles a repeated block
only for an integer repetition and has no rule for a tape slot, so the loop over the
steps is a Python loop that calls the compiled step. `docs/echo-state-network.md` states
what PyNCD derives from the expression and what it does not.

## Drawing the diagram

```bash
just render      # outputs/diagrams/echo-state-network.html and .txt (the listing)
just figures     # outputs/figures/echo-state-network.png and .pdf
```

The page opens from the disk with no server. Its buttons switch between three forms of
the diagram and two themes, and moving the pointer over `W_out` opens a box with the
readout's formula.

## The live viewer

tsncd's viewer is its relay and its page, run in the tsncd checkout exactly as tsncd's
README says, each in a terminal of its own.

```bash
(cd ../tsncd && npm run server)   # the relay on localhost:8765
(cd ../tsncd && npm run dev)      # the page, which opens in the browser and connects
just send                         # send the network to the relay; the page redraws
```

The relay holds the last model it was sent, so a refreshed page redraws it.

## Checks

```bash
just test        # the whole suite: the expression, the compiled step and readout on the
                 # CPU and every GPU, the page, the figures, the run
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
