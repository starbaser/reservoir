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
`W_out` over `(n, p)` is the only learned array, and it is fit by ridge regression over
the states rather than by gradient descent. `src/reservoir/echo_state_network.py` writes
the network as a PyNCD morphism and `src/reservoir/reference_torch.py` runs the same
equations in Torch.

## Each part of the equations has one spelling in PyNCD

| In the equations                | In the morphism                                                                                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the sequence `u` over `t` steps | the domain array `R[t, m]`                                                                                                                                              |
| `W_in u_t` at every step        | `ops.FixedArray` for `W_in` contracted with the sequence by `einsum`, before the loop, broadcast over `t`                                                               |
| the recurrence over `t`         | a `cat.Block` repeated `                                                                                                                                                | t   | `times whose tag names the counter`i_t` |
| `u_t` inside step `i_t`         | an `ops.View` whose row has no domain axis and the shift `i_t`                                                                                                          |
| `h_{t-1}` and `h_t`             | a loop variable on the tape slot `h`: `Para.StreamGrab` reads the state the iteration starts from and `Para.StreamDrop` writes the state the next iteration starts from |
| `h_{-1} = 0`                    | `ops.ConstantOp` zeros dropped on the slot before the loop                                                                                                              |
| `W h_{t-1}`                     | `ops.FixedArray` for `W` contracted over two index variables of `n`, because both axes of `W` are the reservoir axis                                                    |
| `tanh`                          | `ops.Arithmetic` with the formula `2 sigma(2x) - 1`, named `\tanh`                                                                                                      |
| `(1 - alpha) h + alpha a`       | two `ops.Arithmetic` maps in the symbol `alpha` and an `ops.AdditionOp`                                                                                                 |
| the states `H` over every step  | a second loop variable on the slot `H`, to which each step adds its state written at position `i_t` by an `aops.CovariantView`                                          |
| `W_out h_t` at every step       | `ops.Linear` from `n` to `p`, broadcast over `t`, after the loop                                                                                                        |

The SSA listing of the term reads as follows.

```
\text{Echo State Network} {
  %2 = FixedArray<W^{in}>() : R[{m}, {n}]
  %3 = Einops(%2[{m}, n], %0[t, {m}]) : R[t, n]
  \text{Reservoir} {
    %4 = ConstantOp() : R[n]
     = Drop<h>(%4)
    %5 = ConstantOp() : R[t, n]
     = Drop<H>(%5)
    loop |t| times {
      %6 = View<it>(%3[t, n]) : R[n]   via i_t
      %7 = StreamGrab<h>() : R[n]
      %8 = StreamGrab<H>() : R[t, n]
      %9 = FixedArray<W>() : R[{n}, {n}]
      %10 = Einops(%9[{n}, n], %7[{n}]) : R[n]
      %11 = AdditionOp<+>(%6[n], %10[n]) : R[n]
      %12 = Arithmetic<\tanh>(%11[n]) : R[n]
      %13 = Arithmetic<\alpha x>(%12[n]) : R[n]
      %14 = Arithmetic<x (1 - \alpha)>(%7[n]) : R[n]
      %15 = AdditionOp<+>(%13[n], %14[n]) : R[n]
      %16 = CovariantView<i_t>(%15[n]) : R[{t}, n]
      %17 = AdditionOp<+>(%16[t, n], %8[t, n]) : R[t, n]
       = StreamDrop<h>(%15)
       = StreamDrop<H>(%17)
    }
  }
  %1 = Linear<W^{out}>(%17[t, {n}]) : R[t, {p}]
}
```

The construction follows the selective scan of PyNCD's Mamba notebook,
[`selective_scan.py`](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/notebooks/caching/mamba/selective_scan.py),
which is the same shape: a recurrence over the tokens written as a repeated block whose
state is a loop variable. The notes that state the rules are
[Para Category](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/obsidian/07-para/Para%20Category.md)
and
[Carrying the State of a Scan Between Passes](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/obsidian/08-caching/Carrying%20the%20State%20of%20a%20Scan%20Between%20Passes.md)
in the PyNCD repository.

## What PyNCD derives from the term, and what it does not

- **The diagram.** Every class the term holds has a tsncd mirror. `just render` writes
  the standalone page and the listing under `outputs/diagrams/`, `just figures` draws
  PNG and PDF figures headless, and `just up` with `just send` draws it in the live page
  viewer. On the page, the three forms and the two themes switch with the buttons, and
  the inspection box of `W^{out}` opens on the pointer with the readout's formula. The
  blocks carry their title and formula in the page's records, and this tsncd revision
  opens no box on a block drawn as a backdrop, only on an operator that carries an
  expansion.
- **The parameters.** `show_grabbed_parameters.parameter_arrays` reports `W_out` over
  `(n, p)` and nothing else, so a derived backward pass gives `W_in` and `W` no slot.
  That matches the model, in which only the readout is trained.
- **The numbers.** `torch_compile` has no rule for a tape seed, an `ops.ConstantOp`, an
  `ops.FixedArray` or an `aops.CovariantView`, compiles a repeated block only when its
  repetition is an integer, and reads every `ops.View` as the identity, so the loop is
  not compiled. `src/reservoir/evaluate_term.py` evaluates the term instead, running its
  loop, its tape and its reindexings on tensors, and `tests/test_evaluate_term.py`
  checks that the term and `reference_torch` agree to float64 tolerance on the same
  weights. That agreement is what ties the diagram to the numbers.
- **No ridge regression.** The closed-form fit needs a matrix inverse or solve, and
  PyNCD has no such operator. The fit is a forward-pass write to the readout's slot in
  the sense of PyNCD's
  [Training](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/obsidian/07-para/Training.md)
  note, and this project performs it in Torch.
- **No parallel form over time.** A tanh recurrence is a non-associative operation over
  the step axis, which PyNCD's
  [Design Space](https://github.com/starbaser/pyncd/blob/d056b4755fad127e48063739126fc2f19192650b/obsidian/06-practice/Design%20Space.md)
  note lists outside the rewrites that split a fold into chunks. The loop is the only
  form, which is the correct form for a reservoir.

## Extending the starter

- A bias on the units is one more `ops.FixedArray` over `(n,)` added before the tanh.
- Output feedback `W_fb y_{t-1}` is a fixed array over `(p, n)` contracted with the
  readout of the step before, which makes the readout part of the loop.
- A readout over `[h_t; u_t]` is two `ops.Linear`s summed, per PyNCD's ruling that a
  linear map cut into parts is one linear map per part.
- The spectral radius and the sparsity of `W` are properties of the values the fixed
  array holds. The term records them only in the array's name or formula.
