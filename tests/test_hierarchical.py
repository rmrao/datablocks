import typing as T
from dataclasses import dataclass, field
from functools import partial

import pytest
import torch

from datablocks import Datablock


@dataclass(frozen=True, repr=False)
class _StringTensorDimNeg1(Datablock):
    string_item: str = field(default="abcde", metadata={"dim": -1})
    tensor_item: torch.Tensor = field(
        default=torch.arange(10).view(2, 5), metadata={"dim": -1}
    )


@dataclass(frozen=True, repr=False)
class _HierarchicalStringTensor(Datablock):
    block: _StringTensorDimNeg1 = field(default_factory=_StringTensorDimNeg1)
    string_item: str = field(default="abcde", metadata={"dim": -1})
    tensor_item: torch.Tensor = field(
        default=torch.arange(10).view(2, 5), metadata={"dim": -1}
    )


def _collate(flag: bool, n: int, value):
    if not flag:
        return value
    if isinstance(value, str):
        return [value for _ in range(n)]
    elif isinstance(value, torch.Tensor):
        return value.unsqueeze(0).repeat(n, *([1] * value.dim()))
    else:
        raise TypeError(type(value))


@pytest.mark.parametrize("collate", [False, True])
def test_hierarchical_slice_negdim(collate: bool):
    data = _HierarchicalStringTensor()
    if collate:
        data = _HierarchicalStringTensor.collate([data, data])
    sliced = data[:3]
    col = partial(_collate, collate, 2)
    for block in (sliced.block, sliced):
        assert block.string_item == col("abc")
        assert torch.all(block.tensor_item == col(torch.tensor([[0, 1, 2], [5, 6, 7]])))

        block = data[2]
        assert block.string_item == col("c")
        assert torch.all(block.tensor_item == col(torch.tensor([[2], [7]])))

        block = data[[0, 1, 3]]
        assert block.string_item == col("abd")
        assert torch.all(block.tensor_item == col(torch.tensor([[0, 1, 3], [5, 6, 8]])))

        block = data[-2:]
        assert block.string_item == col("de")
        assert torch.all(block.tensor_item == col(torch.tensor([[3, 4], [8, 9]])))
