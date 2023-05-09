# Simple, Composable, Torch-Friendly Data Structures

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

However, a similar interface does not exist for datatypes. We often want to keep track of multiple related pieces of information about a single
example. Even for the simple case of a protein sequence, we may want to keep track of the `id` (string data), `sequence` (string data), and
`tensor` (int64 tensor data corresponding to encoded sequence).

More complicated objects, such as protein structures, require more than a dozen related sub-components. There are also a number of operations that
must take into account all subcomponents, including:

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

in an intuitive, user-friendly manner. Additionally, like PyTorch modules, datatypes should be composable.

## Datablocks

This module defines the `datablock` decorator, which can turn any frozen dataclass into a composable, torch-friendly datastructure. Example:

```python
@datablock
@dataclass(frozen=True)
class ProteinSequence:
    id: str
    sequence: str = field(metadata={"dim": -1})  # indicate the dimension along which slicing should occur
    tensor: torch.Tensor = field(metadata={"pad": 1, "dim": -1})  # indicate the dimension for slicing and the pad value for collating

    @classmethod
    def from_sequence(cls, sequence: str, alphabet: Alphabet, id: T.Optional[str] = None):
        tensor = torch.tensor(alphabet.encode(sequence), dtype=torch.int64)
        if id is None:
            id = ""
        return cls(id=id, sequence=sequence, tensor=tensor)
```

Now we can create a sequence and operate on it in various ways

```python
>>> header = "test"
>>> seq = "MANLFKLGAE"
>>> sequence = ProteinSequence.from_sequence(seq, alphabet, id=header)
>>> sequence
ProteinSequence(
    id=test,
    sequence=MANLFKLGAE,
    tensor=tensor([20, 5, 17,  4, 18, 15, 4, 6, 5, 9]),
    batch_size=0,
    dtype=torch.float32,
    device=cpu,
)
```

#### Slicing
```python
>>> sequence[:2]
ProteinSequence(
    id=test,
    sequence=MA,
    tensor=tensor([20, 5]),
    batch_size=0,
    dtype=torch.float32,
    device=cpu,
)

# Lists will work, along with numpy arrays, torch tensors, negative indexing, etc.
>>> sequence[[0, 1, 8, 9]]
ProteinSequence(
    id=test,
    sequence=MAAE,
    tensor=tensor([20, 5, 5, 9]),
    batch_size=0,
    dtype=torch.float32,
    device=cpu,
)
```
#### Shift to cuda
```python
>>> sequence.cuda()
ProteinSequence(
    id=test,
    sequence=MANLFKLGAE,
    tensor=tensor([20, 5, 17,  4, 18, 15, 4, 6, 5, 9], device="cuda:0"),
    batch_size=0,
    dtype=torch.float32,
    device=cuda,
)
```
#### Collate
```python
# Moving to/from cuda will still work after collating. Slicing is not fully implemented yet.
>>> ProteinSequence.collate([sequence[:2], sequence[2:]])
ProteinSequence(
    id=["test", "test"],
    sequence=["MA", "NLFKLGAE"],
    tensor=tensor(
        [[20, 5, 1, 1, 1, 1, 1, 1]  # pad is automatically 1 b/c of metadata specified in declaration
         [17, 4, 18, 15, 4, 6, 5, 9]]
    ),
    batch_size=2,
    dtype=torch.float32,
    device=cpu,
)
```

#### Composability
It's also possible to compose objects in a straightforward manner

```python
@datablock
@dataclass(frozen=True)
class ProteinStructure:
    sequence: ProteinSequence
    coords: torch.Tensor = field(metadata={"dim": -3})


>>> structure = ProteinStructure(
    sequence,
    coords=torch.zeros([10, 3, 3]),  # e.g. to represent backbone coordinates
)

# All methods (slicing, move to/from device/dtype) will still work.
>>> structure[:2]
ProteinStructure(
    sequence=ProteinSequence(
        id=test,
        sequence=MA,
        tensor=tensor([20, 5]),
        batch_size=0,
        dtype=torch.float32,
        device=cpu,
    ),
    coords=torch.tensor([
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    ]),
    batch_size=0,
    dtype=torch.float32,
    device=cpu,
)

```
