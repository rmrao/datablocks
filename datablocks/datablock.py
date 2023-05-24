from .collate import CollateDatablockMixin
from .sequential import SequentialDatablockMixin
from .tensorlike import TensorlikeDatablockMixin

from .utils import fields_dict


class Datablock(
    TensorlikeDatablockMixin,
    SequentialDatablockMixin,
    CollateDatablockMixin,
):
    def __repr__(self):
        info = ", ".join(
            f"{key}={value}" for key, value in fields_dict(self, repr=False).items()
        )
        if self.batch_size == 0:
            return f"{self.__class__.__name__}({info}, dtype={self.dtype}, device={self.device})"
        else:
            return f"{self.__class__.__name__}({info}, batch_size={self.batch_size}, dtype={self.dtype}, device={self.device})"
