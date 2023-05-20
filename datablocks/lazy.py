import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable


def lazyproperty(fget: Callable[[Any], Any]):
    def lazy_fget(obj):
        if fget.__name__ not in obj._lazy:
            obj._lazy[fget.__name__] = fget(obj)
        return obj._lazy[fget.__name__]

    return property(lazy_fget)


@dataclass(frozen=True)
class LazyDatablockMixin:
    _lazy: dict = field(default_factory=dict, init=False, repr=False)
