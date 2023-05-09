from dataclasses import dataclass, field

import pytest
import torch

from datablocks import datablock

"""
TODO:
    Capabilities:
        - Slicing
            - Datatypes
                - str
                - list (not implemented)
                - np.ndarray
                - torch.Tensor
                - error on other
            - forward indexing
            - backwards indexing
            - multi-indexing
        - Collating
            - re-test all operations after collating
            - test error if collating existing example
            - test error if different dtypes / devices
"""


@datablock
@dataclass(frozen=True)
class _StringTensor:
    string_item: str = "test"
    tensor_item: torch.Tensor = torch.zeros([3])


@datablock
@dataclass(frozen=True)
class _StringTensorPad1:
    string_item: str = "test"
    tensor_item: torch.Tensor = field(default=torch.zeros([3]), metadata={"pad": 1})


@datablock
@dataclass(frozen=True)
class _StringTensorDim0:
    string_item: str = field(default="abcde", metadata={"dim": 0})
    tensor_item: torch.Tensor = field(default=torch.arange(5), metadata={"dim": 0})


@datablock
@dataclass(frozen=True)
class _StringTensorDimNeg1:
    string_item: str = field(default="abcde", metadata={"dim": -1})
    tensor_item: torch.Tensor = field(
        default=torch.arange(10).view(2, 5), metadata={"dim": -1}
    )


@datablock
@dataclass(frozen=True)
class _StringTensorMultidim:
    string_item: str = field(default="abcde", metadata={"dim": -1})
    tensor_item: torch.Tensor = field(
        default=torch.arange(50).view(2, 5, 5), metadata={"dim": (-1, -2)}
    )


def test_shift_dtype():
    data = _StringTensor()

    # float16
    half = data.half()
    assert type(half.string_item) == str
    assert half.string_item == data.string_item

    assert half.tensor_item.dtype == torch.float16
    assert data.tensor_item.dtype == torch.float32
    assert torch.all(half.tensor_item == data.tensor_item.half())

    # float64
    double = data.double()
    assert type(double.string_item) == str
    assert double.string_item == data.string_item

    assert double.tensor_item.dtype == torch.float64
    assert data.tensor_item.dtype == torch.float32
    assert torch.all(double.tensor_item == data.tensor_item.double())

    # bfloat16
    bf16 = data.bfloat16()
    assert type(bf16.string_item) == str
    assert bf16.string_item == data.string_item

    assert bf16.tensor_item.dtype == torch.bfloat16
    assert data.tensor_item.dtype == torch.float32
    assert torch.all(bf16.tensor_item == data.tensor_item.bfloat16())

    # float32
    flt = half.float()
    assert type(flt.string_item) == str
    assert flt.string_item == data.string_item

    assert flt.tensor_item.dtype == torch.float32
    assert half.tensor_item.dtype == torch.float16
    assert torch.all(flt.tensor_item == half.tensor_item.float())


def test_shift_device():
    if not torch.cuda.is_available():
        pytest.skip("Skipping cuda test because cuda is not available")
    data = _StringTensor()

    # cuda
    cuda = data.cuda()
    assert type(cuda.string_item) == str
    assert cuda.string_item == data.string_item

    assert cuda.device.type == "cuda"
    assert data.device.type == "cpu"
    assert cuda.tensor_item.device.type == "cuda"
    assert data.tensor_item.device.type == "cpu"
    assert torch.all(cuda.tensor_item == data.tensor_item.cuda())

    # cpu
    cpu = cuda.cpu()
    assert type(cpu.string_item) == str
    assert cpu.string_item == cuda.string_item

    assert cpu.device.type == "cpu"
    assert cuda.device.type == "cuda"
    assert cpu.tensor_item.device.type == "cpu"
    assert cuda.tensor_item.device.type == "cuda"
    assert torch.all(cpu.tensor_item == cuda.tensor_item.cpu())


def test_collate():
    a = _StringTensor("hello", torch.tensor([0, 1]))
    b = _StringTensor("world", torch.tensor([2, 3, 4]))

    test = _StringTensor.collate([a, b])
    assert test.string_item == ["hello", "world"]
    assert torch.all(test.tensor_item == torch.tensor([[0, 1, 0], [2, 3, 4]]))


def test_collate_pad1():
    a = _StringTensorPad1("hello", torch.tensor([0, 1]))
    b = _StringTensorPad1("world", torch.tensor([2, 3, 4]))

    test = _StringTensorPad1.collate([a, b])
    assert test.string_item == ["hello", "world"]
    # Should now be padded with 1 instead of 0 because of field metadata
    assert torch.all(test.tensor_item == torch.tensor([[0, 1, 1], [2, 3, 4]]))


def test_slice_none():
    data = _StringTensor()
    sliced = data[:1]
    assert sliced.string_item == data.string_item
    assert torch.all(sliced.tensor_item == data.tensor_item)


def test_slice_posdim():
    data = _StringTensorDim0()
    sliced = data[:3]
    assert sliced.string_item == "abc"
    assert torch.all(sliced.tensor_item == torch.tensor([0, 1, 2]))

    sliced = data[2]
    assert sliced.string_item == "c"
    assert torch.all(sliced.tensor_item == torch.tensor([2]))

    sliced = data[[0, 1, 3]]
    assert sliced.string_item == "abd"
    assert torch.all(sliced.tensor_item == torch.tensor([0, 1, 3]))

    sliced = data[-2:]
    assert sliced.string_item == "de"
    assert torch.all(sliced.tensor_item == torch.tensor([3, 4]))


def test_slice_negdim():
    data = _StringTensorDimNeg1()
    sliced = data[:3]
    assert sliced.string_item == "abc"
    assert torch.all(sliced.tensor_item == torch.tensor([[0, 1, 2], [5, 6, 7]]))

    sliced = data[2]
    assert sliced.string_item == "c"
    assert torch.all(sliced.tensor_item == torch.tensor([[2], [7]]))

    sliced = data[[0, 1, 3]]
    assert sliced.string_item == "abd"
    assert torch.all(sliced.tensor_item == torch.tensor([[0, 1, 3], [5, 6, 8]]))

    sliced = data[-2:]
    assert sliced.string_item == "de"
    assert torch.all(sliced.tensor_item == torch.tensor([[3, 4], [8, 9]]))


def test_slice_multdim():
    data = _StringTensorMultidim()
    sliced = data[:3]
    assert sliced.string_item == "abc"
    assert torch.all(
        sliced.tensor_item
        == torch.tensor(
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

    sliced = data[2]
    assert sliced.string_item == "c"
    assert torch.all(
        sliced.tensor_item
        == torch.tensor(
            [
                [[12]],
                [
                    [37],
                ],
            ]
        )
    )

    sliced = data[[0, 1, 3]]
    assert sliced.string_item == "abd"
    assert torch.all(
        sliced.tensor_item
        == torch.tensor(
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

    sliced = data[-2:]
    assert sliced.string_item == "de"
    assert torch.all(
        sliced.tensor_item
        == torch.tensor(
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
