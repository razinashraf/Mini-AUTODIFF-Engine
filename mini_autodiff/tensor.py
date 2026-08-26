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

    def __matmul__(self, other):
        from .operations import MatMul

        return MatMul().forward(self, other)

    def relu(self):
        from .operations import ReLU

        return ReLU().forward(self)

    def sigmoid(self):
        from .operations import Sigmoid

        return Sigmoid().forward(self)

    def tanh(self):
        from .operations import Tanh

        return Tanh().forward(self)

    def __sub__(self, other):
        from .operations import Subtract

        return Subtract().forward(self, other)

    def abs(self):
        from .operations import Abs

        return Abs().forward(self)

    def mean(self):
        from .operations import Mean

        return Mean().forward(self)

    def square(self):
        from .operations import Square

        return Square().forward(self)

    def log(self):
        from .operations import Log

        return Log().forward(self)
    
    def __neg__(self):
        from .operations import Negate

        return Negate().forward(self)

    def __rsub__(self, other):
        return Tensor(other) - self

    def exp(self):
        from .operations import Exp

        return Exp().forward(self)

    def sum(self):
        from .operations import Sum

        return Sum().forward(self)

    def max(self):
        from .operations import Max

        return Max().forward(self)

    def __truediv__(self, other):
        from .operations import Divide

        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        return Divide().forward(self, other)

    def __rtruediv__(self, other):
        from .operations import Divide

        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        return Divide().forward(other, self)

    def softmax(self):
        from .activations import softmax

        return softmax(self)

    def logsumexp(self):
            from .activations import logsumexp
    
            return logsumexp(self)

    def select(self, index):
        from .operations import Select

        return Select().forward(self, index)

    def cross_entropy(self, target):
        from .losses import cross_entropy

        return cross_entropy(self, target)