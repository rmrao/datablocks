import typing as T
from dataclasses import dataclass, field

import pytest
import torch

from datablocks import datablock

"""
TODO:
    Capabilities:
        - Slicing
            - Datatypes
                - list (not implemented)
                - error on other
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


DATA = _StringTensor()
BATCH = _StringTensor.collate([DATA, DATA])


def _assert_type_batched(data, value: T.Any, base: type):
    if data.batch_size == 0:
        assert isinstance(value, base)
    else:
        assert isinstance(value, list)
        assert all(isinstance(v, base) for v in value)


@pytest.mark.parametrize("data", [DATA, BATCH])
def test_shift_dtype(data: _StringTensor):
    # float16
    half = data.half()
    _assert_type_batched(half, half.string_item, str)
    assert half.string_item == data.string_item

    assert half.tensor_item.dtype == torch.float16
    assert data.tensor_item.dtype == torch.float32
    assert torch.all(half.tensor_item == data.tensor_item.half())

    # float64
    double = data.double()
    _assert_type_batched(double, double.string_item, str)
    assert double.string_item == data.string_item

    assert double.tensor_item.dtype == torch.float64
    assert data.tensor_item.dtype == torch.float32
    assert torch.all(double.tensor_item == data.tensor_item.double())

    # bfloat16
    bf16 = data.bfloat16()
    _assert_type_batched(bf16, bf16.string_item, str)
    assert bf16.string_item == data.string_item

    assert bf16.tensor_item.dtype == torch.bfloat16
    assert data.tensor_item.dtype == torch.float32
    assert torch.all(bf16.tensor_item == data.tensor_item.bfloat16())

    # float32
    flt = half.float()
    _assert_type_batched(flt, flt.string_item, str)
    assert flt.string_item == data.string_item

    assert flt.tensor_item.dtype == torch.float32
    assert half.tensor_item.dtype == torch.float16
    assert torch.all(flt.tensor_item == half.tensor_item.float())


@pytest.mark.parametrize("data", [DATA, BATCH])
def test_shift_device(data):
    if not torch.cuda.is_available():
        pytest.skip("Skipping cuda test because cuda is not available")

    # cuda
    cuda = data.cuda()
    _assert_type_batched(cuda, cuda.string_item, str)
    assert cuda.string_item == data.string_item

    assert cuda.device.type == "cuda"
    assert data.device.type == "cpu"
    assert cuda.tensor_item.device.type == "cuda"
    assert data.tensor_item.device.type == "cpu"
    assert torch.all(cuda.tensor_item == data.tensor_item.cuda())

    # cpu
    cpu = cuda.cpu()
    _assert_type_batched(cpu, cpu.string_item, str)
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
