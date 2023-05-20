# Simple, Composable, Torch-Friendly Data bars

## Idea

PyTorch does a great job of constructing an object-oriented interface for neural networks. Methods like

```python
net.cuda()
net.cpu()
net.half()
net.float()
net.to(dtype, device)
```

allow the user to think of a network as a single object which can be altered, rather than as a series of smaller objects
that must be moved individually.

However, a similar interface does not exist for datatypes. We often want to keep track of multiple related pieces of information about a single example, such as an class of an image, bounding boxes, or the unencoded string text paired with the encoded + tokenized int64 indices.

There are also a number of operations that must take into account all subcomponents, including:

- Moving to/from devices
- Changing dtype
- Slicing
- Collating

A common approach is to treat these data objects as dictionaries, and use some python treemap function to apply an operation to all subcomponents.
However, this breaks multiple coding principles. Dictionaries are not typed, so can cause errors when the variables in the dictionary are unclear.
Some operations, such as slicing, are complicated and may require different indexing for different components. More broadly, this breaks the object-
oriented approach of other PyTorch componenents.

## Design Goals

We want a method that supports all of:

- Moving to/from devices
- Changing dtype
- Slicing
- Collating
- Lazy properties

in an intuitive, user-friendly manner. Additionally, like PyTorch modules, datatypes should be composable.

## Datablocks

This module defines the `Datablock` class, which can turn any frozen dataclass into a composable, torch-friendly databar. Example:

```python
@dataclass(frozen=True, repr=False)
class Foo(Datablock):
    id: str
    sequence: str = field(metadata={"dim": -1})  # indicate the dimension along which slicing should occur
    tensor: torch.Tensor = field(metadata={"pad": -1, "dim": -1})  # indicate the dimension for slicing and the pad value for collating

    @classmethod
    def from_sequence(cls, id: str, sequence: str):
        tensor = torch.tensor(encode(sequence), dtype=torch.int64)
        return cls(id=id, sequence=sequence, tensor=tensor)
```

Now we can create a Foo and operate on it in various ways

```python
>>> import string
>>> encode = lambda x: torch.tensor([string.ascii_uppercase.index(tok] for tok in x], dtype=torch.int64)
>>> header = "test"
>>> seq = "ABCDE"
>>> foo = Foo.from_sequence(id, seq)
>>> foo
Foo(
    id=test,
    sequence=ABCDE,
    tensor=tensor([0, 1, 2, 3, 4]),
    batch_size=0,
    dtype=torch.float32,
    device=cpu,
)
```

#### Slicing

```python
>>> foo[:2]
Foo(
    id=test,
    sequence=AB,
    tensor=tensor([0, 1]),
    batch_size=0,
    dtype=torch.float32,
    device=cpu,
)

# Lists will work, along with numpy arrays, torch tensors, negative indexing, etc.
>>> foo[[0, 1, 3]]
Foo(
    id=test,
    sequence=ABD,
    tensor=tensor([0, 1, 3]),
    batch_size=0,
    dtype=torch.float32,
    device=cpu,
)
```

#### Shift to cuda

```python
>>> foo.cuda()
Foo(
    id=test,
    sequence=ABCDE,
    tensor=tensor([0, 1, 2, 3, 4], device="cuda:0"),
    batch_size=0,
    dtype=torch.float32,
    device=cuda,
)
```

#### Collate

```python
# Moving to/from cuda and slicing will still work after collating
>>> Foo.collate([foo[:2], foo[2:]])
Foo(
    id=["test", "test"],
    sequence=["AB", "CDE"],
    tensor=tensor(
        [[0, 1, -1]  # pad is automatically -1 b/c of metadata specified in declaration
         [2, 3, 4]]
    ),
    batch_size=2,
    dtype=torch.float32,
    device=cpu,
)
```

#### Composability

It's also possible to compose objects in a straightforward manner

```python
@dataclass(frozen=True, repr=False)
class Bar(Datablock):
    foo: Foo
    baz: torch.Tensor = field(metadata={"dim": -1})


>>> bar = Bar(
    foo,
    baz=torch.arange(5),
)

# All methods (slicing, move to/from device/dtype) will still work.
>>> bar[:2]
Bar(
    foo=Foo(
        id=test,
        sequence=AB,
        tensor=tensor([0, 1]),
        batch_size=0,
        dtype=torch.float32,
        device=cpu,
    ),
    baz=torch.tensor([0, 1]),
    batch_size=0,
    dtype=torch.float32,
    device=cpu,
)

```

### Lazy Properties

Lazy properties allow the datablock to cache the computation of the property in the instance, so subsequent accesses are 
inexpensive. The implementation is designed to throw away any computed lazy properties when a transform is applied (e.g.
slicing, move to cuda, datatype shift, collating). This is to prevent cases where the computed property will no longer be valid
for the new object. If you want support for these transformations, consider making the attribute a standard member of the dataclass.

Lazy properties are useful if you have a property that meets the following conditions:

- Can be derived from the other attributes in the dataclass
- Does not need to be computed for every instance of the dataclass
- Relatively expensive to compute
- Does not need to benefit from other datablock transformations!

To define a lazy property, use the `lazyproperty` decorator:

```python
import time
import torch
from dataclasses import dataclass
from datablocks import Datablock, lazyproperty

@dataclass(frozen=True, repr=False)
class Lazy(Datablock):
    a: torch.Tensor = torch.randn(3, 5)
    @lazyproperty
    def hello(self) -> str:
        time.sleep(10)  # to mimic a long computation
        return "world"
```

Now, accessing this property multiple times will show the speedup:
```python
>>> lazy = Lazy()
>>> lazy.hello
"world"  # 10 s
>>> lazy.hello
"world"  # <0.1 µs
>>> lazy = lazy.cuda()
>>> lazy.hello
"world"  # 10 s, shift to cuda removes cache
```
