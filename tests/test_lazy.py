from dataclasses import dataclass
from timeit import default_timer as timer
import functools

from datablocks import Datablock


@dataclass(frozen=True, repr=False)
class Foo(Datablock):
    bar: str = "ABCDE"

    @functools.cached_property
    def baz(self) -> float:
        # Just picking a function that should be different on subsequent calls
        return timer()


def test_lazy():
    foo = Foo()
    first_time = foo.baz
    assert foo.baz == first_time, "If lazy property was saved, these should be the same"

    sliced = foo[:3]

    sliced_time = sliced.baz
    assert sliced_time != first_time, "Slicing should not carry over the lazy property"
    assert (
        sliced_time == sliced.baz
    ), "If lazy property was saved, these should be the same"
