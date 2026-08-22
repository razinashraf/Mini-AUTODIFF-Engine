from __future__ import annotations

import numpy as np


class Tensor:
    """
    A node in the computational graph.

    A Tensor stores:
    - numerical data
    - accumulated gradients
    - graph connections
    - the operation that created it
    """

    def __init__(
        self,
        data,
        *, #after * must be written as keyword argument.Tensor(5,False)->(5,requires_grad= False)
        requires_grad: bool = True,
    ):
        # Store numerical data as a NumPy array.
        self.data = np.asarray(data, dtype=np.float32)

        # Gradients always have the same shape as the data.
        self.grad = np.zeros_like(self.data)

        # Parent tensors (inputs that produced this tensor).
        self.parents = ()

        # Operation that created this tensor.
        self.creator = None

        # Should this tensor participate in autograd?
        self.requires_grad = requires_grad

    @property #does not need () when calling
    def is_leaf(self) -> bool:
        return self.creator is None


    def backward(self):
        from .engine import AutogradEngine

        AutogradEngine.backward(self)

    def __add__(self, other):
        from .operations import Add

        return Add().forward(self, other)


    def __mul__(self, other):
        from .operations import Multiply

        return Multiply().forward(self, other)