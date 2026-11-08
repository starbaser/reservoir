# The echo state network as a morphism

An echo state network drives a fixed random recurrent network, the reservoir, with an
input sequence and trains one linear readout on the states it passes through. With input
`u_t` over `m` channels, state `h_t` over `n` units, output `y_t` over `p` channels and
leak rate `alpha`, the network computes

```
h_t = (1 - alpha) h_{t-1} + alpha tanh(W_in u_t + W h_{t-1})      h_{-1} = 0
y_t = W_out h_t
```

`W_in` over `(m, n)` and `W` over `(n, n)` are drawn at random and never changed.
`W_out` over `(n, p)` is the only array that is fit, and it is fit by ridge regression
over the states rather than by gradient descent. `src/reservoir/echo_state_network.py`
writes the network as a PyNCD morphism, `src/reservoir/run_in_torch.py` compiles its
step and its readout with PyNCD's `torch_compile`, and
`src/reservoir/reservoir_weights.py` draws the weights and fits the readout.

## Each part of the equations has one spelling in PyNCD

| In the equations                | In the morphism                                                                                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the sequence `u` over `t` steps | the domain array `R[t, m]`                                                                                                                                              |
| the recurrence over `t`         | a `cat.Block` repeated `\|t\|` times whose tag names the counter `i_t`                                                                                                  |
| `u_t` inside step `i_t`         | an `ops.View` of the sequence whose row has no domain axis and the shift `i_t`                                                                                          |
| `W_in u_t`                      | `ops.Linear` from `m` to `n`                                                                                                                                            |
| `h_{t-1}` and `h_t`             | a loop variable on the tape slot `h`: `Para.StreamGrab` reads the state the iteration starts from and `Para.StreamDrop` writes the state the next iteration starts from |
| `h_{-1} = 0`                    | `ops.ConstantOp` zeros dropped on the slot before the loop                                                                                                              |
| `W h_{t-1}`                     | `ops.Linear` from `n` to `n`                                                                                                                                            |
| `tanh`                          | `ops.Arithmetic` with the formula `2 sigma(2x) - 1`, named `\tanh`                                                                                                      |
| `(1 - alpha) h + alpha a`       | two `ops.Arithmetic` maps in the symbol `alpha` and an `ops.AdditionOp`                                                                                                 |
| the states `H` over every step  | a second loop variable on the slot `H`, to which each step adds its state written at position `i_t` by an `aops.CovariantView`                                          |
| `W_out h_t` at every step       | `ops.Linear` from `n` to `p`, broadcast over `t`, after the loop                                                                                                        |

The SSA listing of the term reads as follows.

```
\text{Echo State Network} {
  \text{Reservoir} {
    %2 = ConstantOp() : R[n]
     = Drop<h>(%2)
    %3 = ConstantOp() : R[t, n]
     = Drop<H>(%3)
    loop |t| times {
      %4 = View<it>(%0[t, m]) : R[m]   via i_t
      %5 = StreamGrab<h>() : R[n]
      %6 = StreamGrab<H>() : R[t, n]
      %7 = Linear<W^{in}>(%4[{m}]) : R[{n}]
      %8 = Linear<W>(%5[{n}]) : R[{n}]
      %9 = AdditionOp<+>(%7[n], %8[n]) : R[n]
      %10 = Arithmetic<\tanh>(%9[n]) : R[n]
      %11 = Arithmetic<\alpha x>(%10[n]) : R[n]
      %12 = Arithmetic<x (1 - \alpha)>(%5[n]) : R[n]
      %13 = AdditionOp<+>(%11[n], %12[n]) : R[n]
      %14 = CovariantView<i_t>(%13[n]) : R[{t}, n]
      %15 = AdditionOp<+>(%14[t, n], %6[t, n]) : R[t, n]
       = StreamDrop<h>(%13)
       = StreamDrop<H>(%15)
    }
  }
  %1 = Linear<W^{out}>(%15[t, {n}]) : R[t, {p}]
}
```

The construction follows the selective scan of PyNCD's Mamba notebook,
[`selective_scan.py`](https://github.com/starbaser/pyncd/blob/main/notebooks/caching/mamba/selective_scan.py),
which is the same shape: a recurrence over the tokens written as a repeated block whose
state is a loop variable. The notes that state the rules are
[Para Category](https://github.com/starbaser/pyncd/blob/main/obsidian/07-para/Para%20Category.md)
and
[Carrying the State of a Scan Between Passes](https://github.com/starbaser/pyncd/blob/main/obsidian/08-caching/Carrying%20the%20State%20of%20a%20Scan%20Between%20Passes.md)
in the PyNCD repository.

## What PyNCD derives from the term, and what it does not

- **The diagram.** Every class the term holds has a tsncd mirror, so tsncd draws it. On
  the page, the three forms and the two themes switch with the buttons, and the
  inspection box of `W^{out}` opens on the pointer with the readout's formula.
- **The parameters.** `show_grabbed_parameters.parameter_arrays` reports the three
  `ops.Linear`s, `W_in` over `(m, n)`, `W` over `(n, n)` and `W_out` over `(n, p)`. The
  model fits `W_out` alone and holds the other two fixed, which is a choice about
  training and not a property of the term.
- **The numbers.** `src/reservoir/run_in_torch.py` compiles `one_step` and `readout`
  with `torch_compile` by the route of PyNCD's tutorial
  [`check_gradients_in_torch.py`](https://github.com/starbaser/pyncd/blob/main/notebooks/website/tutorial/check_gradients_in_torch.py):
  `grab_parameters`, a `NumericConfig` for the sizes, `remove_grouping_blocks`,
  `detape`, `ConstructedModule.construct`. The compiled step takes `u_t`, `h_{t-1}`, `W`
  and `W_in` and returns `h_t`. `torch_compile` compiles a repeated block only when its
  repetition is an integer and has no rule for a tape slot, so the loop over the steps
  is run in Python by calling the compiled step once per step.
  `tests/test_run_in_torch.py` checks the compiled step against the update equation to
  float64 tolerance on the CPU and on every CUDA device.
- **No ridge regression.** The closed-form fit needs a matrix inverse or solve, and
  PyNCD has no such operator. The fit is a forward-pass write to the readout's slot in
  the sense of PyNCD's
  [Training](https://github.com/starbaser/pyncd/blob/main/obsidian/07-para/Training.md)
  note, and this project performs it in Torch.
- **No parallel form over time.** A tanh recurrence is a non-associative operation over
  the step axis, which PyNCD's
  [Design Space](https://github.com/starbaser/pyncd/blob/main/obsidian/06-practice/Design%20Space.md)
  note lists outside the rewrites that split a fold into chunks. The loop is the only
  form, which is the correct form for a reservoir.

## Extending the starter

- A bias on the units is an `ops.Linear` with a bias, or a parameter array over `(n,)`
  added before the tanh.
- Output feedback `W_fb y_{t-1}` is an `ops.Linear` from `p` to `n` applied to the
  readout of the step before, which makes the readout part of the loop.
- A readout over `[h_t; u_t]` is two `ops.Linear`s summed, per PyNCD's ruling that a
  linear map cut into parts is one linear map per part.
- The spectral radius and the sparsity of `W` are properties of the values the weight
  holds. The term records them only in the operator's name or formula.
