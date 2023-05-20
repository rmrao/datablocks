import typing as T
from dataclasses import dataclass, field
from functools import partial

import pytest
import torch

from datablocks import Datablock


@dataclass(frozen=True, repr=False)
class _StringTensor(Datablock):
    string_item: str = "test"
    tensor_item: torch.Tensor = torch.zeros([3])


@dataclass(frozen=True, repr=False)
class _StringTensorPad1(Datablock):
    string_item: str = "test"
    tensor_item: torch.Tensor = field(default=torch.zeros([3]), metadata={"pad": 1})


@dataclass(frozen=True, repr=False)
class _StringTensorDim0(Datablock):
    string_item: str = field(default="abcde", metadata={"dim": 0})
    tensor_item: torch.Tensor = field(default=torch.arange(5), metadata={"dim": 0})


@dataclass(frozen=True, repr=False)
class _StringTensorDimNeg1(Datablock):
    string_item: str = field(default="abcde", metadata={"dim": -1})
    tensor_item: torch.Tensor = field(
        default=torch.arange(10).view(2, 5), metadata={"dim": -1}
    )


@dataclass(frozen=True, repr=False)
class _StringTensorMultidim(Datablock):
    string_item: str = field(default="abcde", metadata={"dim": -1})
    tensor_item: torch.Tensor = field(
        default=torch.arange(50).view(2, 5, 5), metadata={"dim": (-1, -2)}
    )


blocks = {
    "none": _StringTensor,
    "pad1": _StringTensorPad1,
    "dim0": _StringTensorDim0,
    "dimneg1": _StringTensorDimNeg1,
    "multdim": _StringTensorMultidim,
}

indices = [
    slice(None, 3),
    2,
    [0, 1, 3],
    slice(-2, None),
]


ITEMS = {
    "none": [
        _StringTensor(),
        _StringTensor.collate([_StringTensor(), _StringTensor()]),
    ],
}


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
def test_slice_none(collate: bool):
    data = _StringTensor()
    if collate:
        data = _StringTensor.collate([data, data])
    sliced = data[:1]
    assert sliced.string_item == data.string_item
    assert torch.all(sliced.tensor_item == data.tensor_item)


@pytest.mark.parametrize("collate", [False, True])
def test_slice_posdim(collate: bool):
    data = _StringTensorDim0()
    if collate:
        data = _StringTensorDim0.collate([data, data])
    sliced = data[:3]

    col = partial(_collate, collate, 2)
    assert sliced.string_item == col("abc")
    assert torch.all(sliced.tensor_item == col(torch.tensor([0, 1, 2])))

    sliced = data[2]
    assert sliced.string_item == col("c")
    assert torch.all(sliced.tensor_item == col(torch.tensor([2])))

    sliced = data[[0, 1, 3]]
    assert sliced.string_item == col("abd")
    assert torch.all(sliced.tensor_item == col(torch.tensor([0, 1, 3])))

    sliced = data[-2:]
    assert sliced.string_item == col("de")
    assert torch.all(sliced.tensor_item == col(torch.tensor([3, 4])))


@pytest.mark.parametrize("collate", [False, True])
def test_slice_negdim(collate: bool):
    data = _StringTensorDimNeg1()
    if collate:
        data = _StringTensorDimNeg1.collate([data, data])
    sliced = data[:3]
    col = partial(_collate, collate, 2)
    assert sliced.string_item == col("abc")
    assert torch.all(sliced.tensor_item == col(torch.tensor([[0, 1, 2], [5, 6, 7]])))

    sliced = data[2]
    assert sliced.string_item == col("c")
    assert torch.all(sliced.tensor_item == col(torch.tensor([[2], [7]])))

    sliced = data[[0, 1, 3]]
    assert sliced.string_item == col("abd")
    assert torch.all(sliced.tensor_item == col(torch.tensor([[0, 1, 3], [5, 6, 8]])))

    sliced = data[-2:]
    assert sliced.string_item == col("de")
    assert torch.all(sliced.tensor_item == col(torch.tensor([[3, 4], [8, 9]])))


@pytest.mark.parametrize("collate", [False, True])
def test_slice_multdim(collate: bool):
    data = _StringTensorMultidim()
    if collate:
        data = _StringTensorMultidim.collate([data, data])
    sliced = data[:3]
    col = partial(_collate, collate, 2)
    assert sliced.string_item == col("abc")
    assert torch.all(
        sliced.tensor_item
        == col(
            torch.tensor(
                [
                    [[0, 1, 2], [5, 6, 7], [10, 11, 12]],
                    [
                        [25, 26, 27],
                        [30, 31, 32],
                        [35, 36, 37],
                    ],
                ]
            )
        )
    )

    sliced = data[2]
    assert sliced.string_item == col("c")
    assert torch.all(
        sliced.tensor_item
        == col(
            torch.tensor(
                [
                    [[12]],
                    [
                        [37],
                    ],
                ]
            )
        )
    )

    sliced = data[[0, 1, 3]]
    assert sliced.string_item == col("abd")
    assert torch.all(
        sliced.tensor_item
        == col(
            torch.tensor(
                [
                    [[0, 1, 3], [5, 6, 8], [15, 16, 18]],
                    [
                        [25, 26, 28],
                        [30, 31, 33],
                        [40, 41, 43],
                    ],
                ]
            )
        )
    )

    sliced = data[-2:]
    assert sliced.string_item == col("de")
    assert torch.all(
        sliced.tensor_item
        == col(
            torch.tensor(
                [
                    [
                        [18, 19],
                        [23, 24],
                    ],
                    [
                        [43, 44],
                        [48, 49],
                    ],
                ]
            )
        )
    )
