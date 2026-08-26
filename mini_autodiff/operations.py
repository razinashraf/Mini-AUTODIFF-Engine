from __future__ import annotations
from .tensor import Tensor
from .utils import unbroadcast

import numpy as np

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

class ReLU(Operation):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(
            np.maximum(0, x.data),
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.x = x # storing input.

        return out

    def backward(self, grad_output):
        grad_input = grad_output * (self.x.data > 0) #[False, True, False, True]=[0, 1, 0, 1](mult = chainrule)

        return (grad_input, )
        """
        parents:       (x,)
        gradients:     ([0, 20],)
                 ↑
             ONE gradient

        x → [0, 20]
        """


class Sigmoid(Operation):
    def forward(self, x: Tensor) -> Tensor:
        data = 1.0 / (1.0 + np.exp(-x.data))

        out = Tensor(
            data,
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.output = out

        return out

    def backward(self, grad_output):
        sigmoid_derivative = (
            self.output.data *
            (1.0 - self.output.data)
        )

        grad_input = grad_output * sigmoid_derivative

        return (grad_input,)


class Tanh(Operation):
    def forward(self, x: Tensor) -> Tensor:
        data = np.tanh(x.data)

        out = Tensor(
            data,
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.output = out

        return out

    def backward(self, grad_output):
        tanh_derivative = (
            1.0 - self.output.data ** 2
        )

        grad_input = (
            grad_output * tanh_derivative
        )

        return (grad_input,)


class Subtract(Operation):
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        out = Tensor(
            left.data - right.data,
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
        grad_left = unbroadcast(
            grad_output,
            self.left.data.shape,
        )

        grad_right = unbroadcast(
            -grad_output,
            self.right.data.shape,
        )
        
        return grad_left, grad_right


class Abs(Operation):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(
            np.abs(x.data),
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.x = x

        return out

    def backward(self, grad_output):
        grad_input = (
            grad_output * np.sign(self.x.data)
        )

        return (grad_input,)


class Mean(Operation):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(
            np.mean(x.data),
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.input_shape = x.data.shape
        self.input_size = x.data.size

        return out

    def backward(self, grad_output):
        grad_input = (
            np.ones(self.input_shape, dtype=np.float32)
            * grad_output
            / self.input_size
        )

        return (grad_input,)

class Square(Operation):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(
            x.data ** 2,
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.x = x

        return out

    def backward(self, grad_output):
        grad_input = (
            grad_output * 2.0 * self.x.data
        )

        return (grad_input,)

class Log(Operation):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(
            np.log(x.data),
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.x = x

        return out

    def backward(self, grad_output):
        grad_input = grad_output / self.x.data

        return (grad_input,)


class Negate(Operation):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(
            -x.data,
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        return out

    def backward(self, grad_output):
        return (-grad_output,)

class Exp(Operation):
    def forward(self, x: Tensor) -> Tensor:
        out_data = np.exp(x.data)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.output_data = out_data

        return out

    def backward(self, grad_output):
        grad_input = grad_output * self.output_data

        return (grad_input,)


class Sum(Operation):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(
            np.sum(x.data),
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.input_shape = x.data.shape

        return out

    def backward(self, grad_output):
        grad_input = np.ones(
            self.input_shape,
            dtype=np.float32,
        ) * grad_output

        return (grad_input,)

class Max(Operation):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(
            np.max(x.data),
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.x = x
        self.max_index = np.argmax(x.data)

        return out

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.x.data)

        grad_input.flat[self.max_index] = grad_output

        return (grad_input,)

class Divide(Operation):
    def forward(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        out = Tensor(
            numerator.data / denominator.data,
            requires_grad=(
                numerator.requires_grad
                or denominator.requires_grad
            ),
        )

        out.parents = (numerator, denominator)
        out.creator = self

        self.numerator = numerator
        self.denominator = denominator

        return out

    def backward(self, grad_output):
        numerator = self.numerator.data
        denominator = self.denominator.data

        grad_numerator = grad_output / denominator

        grad_denominator = (
            -grad_output
            * numerator
            / (denominator ** 2)
        )

        grad_numerator = unbroadcast(
            grad_numerator,
            self.numerator.data.shape,
        )

        grad_denominator = unbroadcast(
            grad_denominator,
            self.denominator.data.shape,
        )

        return grad_numerator, grad_denominator


class Select(Operation):
    def forward(self, x: Tensor, index: int) -> Tensor:
        out = Tensor(
            x.data[index],
            requires_grad=x.requires_grad,
        )

        out.parents = (x,)
        out.creator = self

        self.x = x
        self.index = index

        return out

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.x.data)

        grad_input[self.index] = grad_output

        return (grad_input,)