import dataclasses

from .collate import CollateDatablockMixin
from .sequential import SequentialDatablockMixin
from .tensorlike import TensorlikeDatablockMixin


def datablock(cls):
    if not dataclasses.is_dataclass(cls):
        raise TypeError(
            "Trying to wrap a non-dataclass as a datablock, this is very likely to go wrong"
        )

    try:

        @dataclasses.dataclass(frozen=True)
        class _Datablock(
            TensorlikeDatablockMixin,
            SequentialDatablockMixin,
            CollateDatablockMixin,
            cls,
        ):
            def __repr__(self):
                info = ",".join(
                    f"{key}={value}"
                    for key, value in vars(self).items()
                    if key not in ("_device", "_dtype")
                )
                return f"{self.__class__.__name__}({info}, dtype={self.dtype}, device={self.device})"

        _Datablock.__module__ = cls.__module__
        _Datablock.__name__ = cls.__name__
        for method in _Datablock.__dict__.values():
            if callable(method):
                method.__qualname__ = method.__qualname__.replace(
                    _Datablock.__qualname__, cls.__qualname__
                )
        _Datablock.__qualname__ = cls.__qualname__
        _Datablock.__annotations__ = cls.__annotations__
        _Datablock.__doc__ = cls.__doc__

    except TypeError as e:
        if e.args[0] == "cannot inherit frozen dataclass from a non-frozen one":
            raise TypeError("Cannot wrap a non-frozen dataclass as a datablock")
        raise e

    return _Datablock
