import dataclasses
import typing as T

import numpy as np
import torch


def collate_dense_tensors(
    sequences: T.Sequence[torch.Tensor], constant_value: T.Union[int, float] = 0, dtype=None
) -> torch.Tensor:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    array = torch.full(shape, constant_value, dtype=dtype)
    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


@dataclasses.dataclass(frozen=True)
class CollateDatablockMixin:
    batch_size: int = 0

    @classmethod
    def collate(cls, samples: T.Sequence["CollateDatablockMixin"]):
        for sample in samples:
            if sample.batch_size != 0:
                raise ValueError(
                    "Received sample with batch size != 0, cannot batch already batched items"
                )

        states = [vars(sample) for sample in samples]
        batch_size = len(states)

        new_state = {
            key: cls.merge(states, key) for key in states[0].keys() if key != "batch_size"
        }

        return cls(**new_state, batch_size=batch_size)

    @classmethod
    def collate_objects(cls, objects: T.Sequence, pad: T.Union[int, float] = 0):
        first = objects[0]
        if isinstance(first, CollateDatablockMixin):
            return first.collate(objects)
        elif isinstance(first, torch.Tensor):
            return collate_dense_tensors(objects, constant_value=pad)
        else:
            return list(objects)

    @classmethod
    def _get_padding(cls, name: str) -> T.Union[int, float]:
        for field in dataclasses.fields(cls):
            if field.name == name:
                return field.metadata.get("pad", 0)
        raise RuntimeError("Should never reach here, mismatch in field names")

    @classmethod
    def merge(cls, samples: T.Sequence[dict[str, T.Any]], name: str):
        objects = [sample[name] for sample in samples]
        first = objects[0]
        if name in ("_dtype", "_device"):
            for obj in objects[1:]:
                if obj != first:
                    raise ValueError(
                        f"Expected all objects to have the same {name[1:]}, received '{first}' and '{obj}'"
                    )
            return first
        pad = cls._get_padding(name)
        return cls.collate_objects(objects, pad=pad)
