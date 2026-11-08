"""Evaluating a PyNCD term on numbers, for the operators the echo state network holds.

`torch_compile` cannot run the reservoir loop: it reads every `ops.View` as the
identity, compiles a repeated block only when its repetition is an integer and never
binds a counter, and has no rule for a tape seed, an `ops.ConstantOp`, an
`ops.FixedArray` or an `aops.CovariantView`. This module evaluates exactly those
operators and the ones beside them, in `torch.float64`, following the evaluator PyNCD's
Mamba notebook keeps for its selective scan, `notebooks/caching/mamba/evaluate_numerically.py`
in the PyNCD repository. It exists so that the term and the Torch
reference can be compared on the same numbers.

A `cat.Broadcasted` is evaluated at every index of its degree at once. Each operand is
read through its reindexing at the positions the rows give for every index of the
degree, and a position outside its axis reads the universal unit, which is zero here
because every sum this term takes over such a position is a sum. The operator then acts
on the targets.

A repeated block holding a `Para.StreamGrab` is a loop over the tape. Each iteration
binds the counter the block's tag names, reads every domain wire of the block, and reads
each loop variable at the value its slot held when the iteration started. The values its
`Para.StreamDrop`s write hold from the next iteration on, and the loop's result is the
result of its last iteration.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards
import advanced_axis_dynamics.data_structure.Operators as aops
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import para.data_structure.Para as Para
import term_utilities.generate_config as generate_config
import torch
import torch_compile.torch_compile as torch_compile

type Tensors = tuple[torch.Tensor, ...]
DTYPE = torch.float64
EINSUM_LETTERS = "abcdefghijklmnopqrstuvw"


class OperatorNotEvaluated(NotImplementedError):
    """An operator or construction this module has no evaluation for."""


class ArrayNotGiven(KeyError):
    """A `FixedArray` or a `Linear` whose name has no entry among the arrays given."""


class LoopWithoutACounter(ValueError):
    """A repeated block whose tag carries no name, so no counter can be bound."""


def name_key(name: fd.DynamicName | str | None) -> str:
    """The text an array is looked up by, the same for the string a template was given
    and for the name the operator carries."""
    if name is None:
        raise ArrayNotGiven("an operator with no name cannot be given an array")
    if isinstance(name, str):
        name = fd.DynamicName.from_str(name)
    return name.to_bodies()


def sizes_by_symbol(
    *, term: fd.GeneralTerm, sizes: Mapping[str, int]
) -> dict[nm.Numeric, int]:
    """Every free size of `term` whose name is a key of `sizes`, bound to its value."""
    return {
        symbol: sizes[symbol.uid._name.to_bodies()]
        for symbol in generate_config.NumericConfig.template(term).terms
        if symbol.uid._name is not None and symbol.uid._name.to_bodies() in sizes
    }


@dataclass
class TermEvaluation:
    """One evaluation of a term. The bindings and the tape change as the loop runs,
    which is why the dataclass is mutable."""

    bindings: dict[nm.Numeric, int]
    """Every free size bound to an integer, and the loop counter while a loop runs."""

    fixed_arrays: Mapping[str, torch.Tensor]
    """The value of every `ops.FixedArray`, by `name_key` of its name."""

    linear_weights: Mapping[str, torch.Tensor]
    """The weight of every `ops.Linear`, by `name_key` of its name, over the axes of
    its input target followed by the axes of its output target."""

    tape: dict[fd.UID, torch.Tensor] = field(default_factory=dict)
    written_for_the_next_iteration: dict[fd.UID, torch.Tensor] = field(
        default_factory=dict
    )

    def integer(self, numeric: nm.Numeric) -> int:
        return nm.evaluate_integer(numeric, self.bindings)

    def size(self, axis: cat.Axis) -> int:
        return self.integer(axis.local_size())

    def morphism(self, *, term: cat.Morphism, inputs: Tensors) -> Tensors:
        match term:
            case cat.Composed(content=parts):
                values = inputs
                for part in parts:
                    values = self.morphism(term=part, inputs=values)
                return values
            case cat.ProductOfMorphisms(content=parts):
                results: list[torch.Tensor] = []
                start = 0
                for part in parts:
                    wires_read = len(part.dom())
                    results.extend(
                        self.morphism(
                            term=part, inputs=inputs[start : start + wires_read]
                        )
                    )
                    start += wires_read
                return tuple(results)
            case cat.Rearrangement():
                return term.apply(inputs)
            case cat.Block():
                return self.block(block=term, inputs=inputs)
            case Para.StreamGrab() | Para.Grab():
                return (self.tape[term.tape.uid],)
            case Para.StreamDrop():
                self.written_for_the_next_iteration[term.tape.uid] = inputs[0]
                return ()
            case Para.Drop():
                self.tape[term.tape.uid] = inputs[0]
                return ()
            case cat.Broadcasted():
                return self.broadcasted(target=term, inputs=inputs)
        raise OperatorNotEvaluated(f"no evaluation for {type(term).__name__}")

    def block(self, *, block: cat.Block, inputs: Tensors) -> Tensors:
        repetition = block.block_tag.repetition
        if repetition == nm.Integer(1):
            return self.morphism(term=block.body, inputs=inputs)
        count = self.integer(repetition)
        counter_name = block.block_tag.uid._name
        if counter_name is None:
            raise LoopWithoutACounter(
                f"the block repeated {count} times names no counter; pass index_name"
                " to cat.Block.template"
            )
        counter = nm.FreeNumeric.named(counter_name)
        taped = any(True for _ in stream_grabs(block.body))
        values = inputs
        for iteration in range(count):
            self.bindings[counter] = iteration
            values = self.morphism(term=block.body, inputs=inputs if taped else values)
            self.tape.update(self.written_for_the_next_iteration)
            self.written_for_the_next_iteration = {}
        del self.bindings[counter]
        return values

    def read_through(
        self,
        *,
        operand: torch.Tensor,
        weave: cat.Weave,
        reindexing: sc.StrideCategory,
        grids: Tensors,
        degree_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        """`operand` read at every index of the degree through `reindexing`, with the
        degree first and the target after it."""
        tiled, targets = tiled_and_target_positions(weave)
        moved = operand.permute(*tiled, *targets)
        if not tiled:
            return moved.expand(*degree_sizes, *moved.shape)
        rows = move_reads_backwards.as_stride_morphism(reindexing)._cod_stride_shift
        valid = torch.ones(degree_sizes, dtype=torch.bool)
        indices: list[torch.Tensor] = []
        for (_, strides, shift), position in zip(rows, tiled, strict=True):
            index = torch.full(degree_sizes, self.integer(shift), dtype=torch.long)
            for grid, stride in zip(grids, strides, strict=True):
                index = index + self.integer(stride) * grid
            extent = operand.shape[position]
            valid = valid & (index >= 0) & (index < extent)
            indices.append(index.clamp(0, extent - 1))
        read = moved[tuple(indices)]
        return read * valid.reshape(*degree_sizes, *(1,) * len(targets)).to(DTYPE)

    def broadcasted(self, *, target: cat.Broadcasted, inputs: Tensors) -> Tensors:
        degree_sizes = tuple(self.size(axis) for axis in target.degree())
        grids = (
            tuple(
                torch.meshgrid(
                    *(torch.arange(size) for size in degree_sizes), indexing="ij"
                )
            )
            if degree_sizes
            else ()
        )
        read = tuple(
            self.read_through(
                operand=operand,
                weave=weave,
                reindexing=reindexing,
                grids=grids,
                degree_sizes=degree_sizes,
            )
            for operand, weave, reindexing in zip(
                inputs, target.input_weaves, target.reindexings, strict=True
            )
        )
        result = self.operator(target=target, read=read, degree_sizes=degree_sizes)
        return (placed(result=result, weave=target.output_weaves[0]),)

    def target_shape(self, weave: cat.Weave) -> tuple[int, ...]:
        return tuple(
            self.size(axis)
            for axis in weave._shape
            if not isinstance(axis, cat.WeaveMode)
        )

    def operator(
        self, *, target: cat.Broadcasted, read: Tensors, degree_sizes: tuple[int, ...]
    ) -> torch.Tensor:
        operator = target.operator
        output_target = self.target_shape(target.output_weaves[0])
        match operator:
            case aops.CovariantView():
                return self.covariant_view(
                    operator=operator,
                    operand=read[0],
                    degree_sizes=degree_sizes,
                    output_target=output_target,
                )
            case ops.Arithmetic():
                return torch_compile.numeric_torch(operator.formula, read[0])
            case ops.View():
                return read[0]
            case ops.Einops():
                return einsum_of_groups(signature=operator.signature, read=read)
            case ops.AdditionOp():
                return sum(read[1:], read[0])
            case ops.ConstantOp():
                zeros = torch.zeros((*degree_sizes, *output_target), dtype=DTYPE)
                return torch_compile.numeric_torch(operator.value, zeros)
            case ops.FixedArray():
                array = self.given(
                    arrays=self.fixed_arrays, name=operator.name, kind="FixedArray"
                )
                return array.expand(*degree_sizes, *array.shape)
            case ops.Linear():
                return self.linear(target=target, read=read, degree_sizes=degree_sizes)
        raise OperatorNotEvaluated(f"no evaluation for {type(operator).__name__}")

    def given(
        self,
        *,
        arrays: Mapping[str, torch.Tensor],
        name: fd.DynamicName | None,
        kind: str,
    ) -> torch.Tensor:
        key = name_key(name)
        if key not in arrays:
            raise ArrayNotGiven(f"no array for the {kind} {key}")
        return arrays[key].to(DTYPE)

    def linear(
        self, *, target: cat.Broadcasted, read: Tensors, degree_sizes: tuple[int, ...]
    ) -> torch.Tensor:
        weight = self.given(
            arrays=self.linear_weights, name=target.operator.name, kind="Linear"
        )
        if target.operator.bias:
            raise OperatorNotEvaluated("a Linear with a bias is not evaluated here")
        if not read:
            return weight.expand(*degree_sizes, *weight.shape)
        contracted = len(target.input_weaves[0].target().shape())
        return torch.tensordot(read[0], weight, dims=contracted)

    def covariant_view(
        self,
        *,
        operator: aops.CovariantView,
        operand: torch.Tensor,
        degree_sizes: tuple[int, ...],
        output_target: tuple[int, ...],
    ) -> torch.Tensor:
        """`operand` written at the codomain positions of the reindexing, with zero at
        every position nothing writes."""
        reindexing = operator.reindexing
        written = torch.zeros((*degree_sizes, *output_target), dtype=DTYPE)
        domain_sizes = tuple(self.size(axis) for axis in reindexing._dom)
        for domain_index in every_index(domain_sizes):
            codomain_index = tuple(
                self.integer(shift)
                + sum(
                    self.integer(stride) * position
                    for stride, position in zip(strides, domain_index, strict=True)
                )
                for _, strides, shift in reindexing._cod_stride_shift
            )
            if all(
                0 <= position < extent
                for position, extent in zip(codomain_index, output_target, strict=True)
            ):
                written[(..., *codomain_index)] += operand[(..., *domain_index)]
        return written


def tiled_and_target_positions(weave: cat.Weave) -> tuple[list[int], list[int]]:
    tiled = [
        position
        for position, entry in enumerate(weave._shape)
        if isinstance(entry, cat.WeaveMode)
    ]
    targets = [
        position
        for position, entry in enumerate(weave._shape)
        if not isinstance(entry, cat.WeaveMode)
    ]
    return tiled, targets


def placed(*, result: torch.Tensor, weave: cat.Weave) -> torch.Tensor:
    """`result`, with the degree first and the target after it, in the order of the
    positions of `weave`."""
    tiled, targets = tiled_and_target_positions(weave)
    order = [*tiled, *targets]
    return result.permute(*(order.index(position) for position in range(len(order))))


def einsum_of_groups(
    *, signature: tuple[tuple[int, ...], ...], read: Tensors
) -> torch.Tensor:
    """The product of the operands summed over every contraction group, with the degree
    leading every operand and the result."""
    operands = ",".join(
        "..." + "".join(EINSUM_LETTERS[group] for group in groups)
        for groups in signature
    )
    return torch.einsum(f"{operands}->...", *read)


def every_index(sizes: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    if not sizes:
        yield ()
        return
    for first in range(sizes[0]):
        for rest in every_index(sizes[1:]):
            yield (first, *rest)


def stream_grabs(term: cat.Morphism) -> Iterator[Para.StreamGrab]:
    match term:
        case Para.StreamGrab():
            yield term
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            for part in parts:
                yield from stream_grabs(part)
        case cat.Block(body=body):
            yield from stream_grabs(body)


def evaluate(
    *,
    term: cat.Morphism,
    inputs: Tensors,
    sizes: Mapping[str, int],
    fixed_arrays: Mapping[str, torch.Tensor],
    linear_weights: Mapping[str, torch.Tensor],
) -> Tensors:
    """`term` evaluated on `inputs`, one tensor per domain array, returning one tensor
    per codomain array.

    Args:
        term: The morphism to evaluate.
        inputs: One `float64` tensor per array of the domain, in order.
        sizes: Every free size of the term, by the body of its name.
        fixed_arrays: The value of every `ops.FixedArray`, by the name it was built
            with.
        linear_weights: The weight of every `ops.Linear`, by the name it was built
            with.
    """
    evaluation = TermEvaluation(
        bindings=sizes_by_symbol(term=term, sizes=sizes),
        fixed_arrays={name_key(name): value for name, value in fixed_arrays.items()},
        linear_weights={
            name_key(name): value for name, value in linear_weights.items()
        },
    )
    return evaluation.morphism(term=term, inputs=inputs)
