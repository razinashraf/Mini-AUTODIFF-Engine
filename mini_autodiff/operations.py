from __future__ import annotations
from .tensor import Tensor
from .utils import unbroadcast


class Operation:
    """
    Base class for all differentiable operations.
    """

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError



class Add(Operation):
    def forward(self, left: Tensor, right: Tensor) -> Tensor:

        self.left = left
        self.right = right
        
        out = Tensor(
            left.data + right.data,
            requires_grad=(
                left.requires_grad or right.requires_grad
            ),
        )

        out.parents = (left, right)
        out.creator = self

        return out

    def backward(self, grad_output):
        grad_left = unbroadcast(
            grad_output,
            self.left.data.shape,
        )

        grad_right = unbroadcast(
            grad_output,
            self.right.data.shape,
        )

        return grad_left, grad_right


class Multiply(Operation):
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        out = Tensor(
            left.data * right.data,
            requires_grad=(
                left.requires_grad or right.requires_grad
            ),
        )

        out.parents = (left, right)
        out.creator = self

        # Multiplication backward needs the original inputs.
        self.left = left
        self.right = right

        return out

    def backward(self, grad_output):
        grad_left = grad_output * self.right.data
        grad_right = grad_output * self.left.data

        grad_left = unbroadcast(
            grad_left,
            self.left.data.shape,
        )

        grad_right = unbroadcast(
            grad_right,
            self.right.data.shape,
        )

        return grad_left, grad_right


class MatMul(Operation):
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        out = Tensor(
            left.data @ right.data,
            requires_grad=(
                left.requires_grad or right.requires_grad
            ),
        )

        out.parents = (left, right)
        out.creator = self

        self.left = left
        self.right = right

        return out

    def backward(self, grad_output):
        grad_left = grad_output @ self.right.data.T
        grad_right = self.left.data.T @ grad_output

        return grad_left, grad_right