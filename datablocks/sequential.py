import contextlib
import dataclasses
import typing as T

import numpy as np
import torch

V = T.TypeVar("V")
Index = T.Union[int, slice, T.Sequence[int], np.ndarray, torch.Tensor]


def _get_ndim(obj: T.Any) -> int:
    with contextlib.suppress(AttributeError):
        return obj.ndim
    if isinstance(obj, str):
        return 1
    elif isinstance(obj, list):
        return 1 + _get_ndim(obj[0])
    else:
        raise ValueError(f"Failed to compute number of dimensions for object: {obj}")


def _index_object(obj: V, dims: T.Union[int, T.Tuple[int, ...]], index: Index) -> V:
    if isinstance(dims, int):
        dims = (dims,)

    if isinstance(obj, SequentialDatablockMixin):
        return obj[index]
    elif isinstance(obj, str):
        assert len(dims) == 1, dims
        if isinstance(index, (int, slice)):
            return obj[index]
        else:
            return "".join(obj[i] for i in index)
    elif isinstance(obj, list):
        raise NotImplementedError(
            "Sequential slicing not implemented for list datatypes"
        )
    else:
        if isinstance(index, int):
            index = [
                index
            ]  # this ensures the dimension stays and is set to 1 rather than reducing the dimension
        ndim = _get_ndim(obj)
        for d in dims:
            # Note (rmrao): This construction is strange, but I think required. First, build
            # placeholder indices as a list, b/c lists are modifiable. Then replace the specified
            # dim with the index. Doing this directly in the list construction is difficult b/c
            # of negative indexing. However, numpy/torch treat indexing with a list differently
            # from indexing with a tuple - the input must be a tuple for fancy slicing to work
            # as expected. So we convert to a tuple afterwards.
            indices: T.List[Index] = [slice(None) for _ in range(ndim)]
            indices[d] = index
            obj = obj[tuple(indices)]
        return obj


@dataclasses.dataclass(frozen=True)
class SequentialDatablockMixin:
    def __getitem__(self, index: Index):
        state = vars(self)
        dims: T.Dict[str, T.Union[int, T.Tuple[int, ...]]] = {
            f.name: f.metadata.get("dim", None) for f in dataclasses.fields(self)
        }
        newstate = {}
        for key, value in state.items():
            if dims[key] is None:
                continue
            newstate[key] = _index_object(value, dims[key], index)
        return dataclasses.replace(self, **newstate)
