import dataclasses
import typing as T

import torch

FLOAT_DTYPES: T.Set[torch.dtype] = {
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
}
V = T.TypeVar("V")


@dataclasses.dataclass(frozen=True, kw_only=True)
class TensorlikeDatablockMixin:
    _device: T.Optional[torch.device] = None
    _dtype: T.Optional[torch.dtype] = None

    @property
    def dtype(self) -> torch.dtype:
        if self._dtype is None:
            for value in vars(self).values():
                if isinstance(value, torch.Tensor) and value.dtype in FLOAT_DTYPES:
                    return value.dtype
            return torch.float32
        return self._dtype

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for value in vars(self).values():
                if isinstance(value, torch.Tensor):
                    return value.device
            return torch.device("cpu")
        return self._device

    def to(
        self,
        device: T.Optional[T.Union[str, torch.device]] = None,
        dtype: T.Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        memory_format=torch.preserve_format,
    ):
        if device is None:
            device = self.device
        device = torch.device(device)
        if device == self.device and dtype == self.dtype:
            return self
        kwargs = {
            "device": device,
            "non_blocking": non_blocking,
            "copy": copy,
            "memory_format": memory_format,
        }

        def move_tensor(tensor: V) -> V:
            if isinstance(tensor, TensorlikeDatablockMixin):
                return tensor.to(**kwargs, dtype=dtype)
            if not isinstance(tensor, torch.Tensor):
                return tensor
            elif tensor.dtype in FLOAT_DTYPES:
                return tensor.to(**kwargs, dtype=dtype)
            else:
                return tensor.to(**kwargs)

        state = vars(self)
        state = {key: move_tensor(tensor) for key, tensor in state.items()}
        state["_device"] = device
        state["_dtype"] = dtype
        return dataclasses.replace(self, **state)

    def cuda(
        self,
        device: T.Union[int, str, torch.device] = "cuda",
        non_blocking: bool = False,
        memory_format=torch.preserve_format,
    ):
        device = torch.device(device)
        if device.type == "cpu":
            raise ValueError("Received device 'cpu' for in method .cuda()")
        return self.to(
            device=device, non_blocking=non_blocking, memory_format=memory_format
        )

    def cpu(self, memory_format=torch.preserve_format):
        device = torch.device("cpu")
        return self.to(device=device, memory_format=memory_format)

    def float(self, memory_format=torch.preserve_format):
        return self.to(dtype=torch.float32, memory_format=memory_format)

    def half(self, memory_format=torch.preserve_format):
        return self.to(dtype=torch.float16, memory_format=memory_format)

    def double(self, memory_format=torch.preserve_format):
        return self.to(dtype=torch.float64, memory_format=memory_format)

    def bfloat16(self, memory_format=torch.preserve_format):
        return self.to(dtype=torch.bfloat16, memory_format=memory_format)
