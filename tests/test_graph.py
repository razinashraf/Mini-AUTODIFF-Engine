import numpy as np

from mini_autodiff.tensor import Tensor
from mini_autodiff.operations import Add, Multiply
from mini_autodiff.engine import AutogradEngine


def test_multi_operation_graph():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    c = Tensor([10.0, 20.0])

    multiply = Multiply()
    x = multiply.forward(a, b)

    add = Add()
    y = add.forward(x, c)

    AutogradEngine.backward(y)

    np.testing.assert_array_equal(
        y.data,
        np.array([18.0, 35.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        a.grad,
        np.array([4.0, 5.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        b.grad,
        np.array([2.0, 3.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        c.grad,
        np.array([1.0, 1.0], dtype=np.float32),
    )




def test_gradient_accumulation():
    x = Tensor([2.0, 3.0])

    mul1 = Multiply()
    a = mul1.forward(x, x)

    mul2 = Multiply()
    y = mul2.forward(a, x)

    AutogradEngine.backward(y)

    np.testing.assert_array_equal(
        x.grad,
        np.array([12.0, 27.0], dtype=np.float32),
    )


def test_leaf_tensors():
    a = Tensor([2.0])
    b = Tensor([3.0])

    x = Multiply().forward(a, b)
    y = Add().forward(x, a)

    assert a.is_leaf
    assert b.is_leaf

    assert not x.is_leaf
    assert not y.is_leaf



def test_tensor_api():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    c = Tensor([10.0, 20.0])

    y = a * b + c

    y.backward()

    np.testing.assert_array_equal(
        y.data,
        np.array([18.0, 35.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        a.grad,
        np.array([4.0, 5.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        b.grad,
        np.array([2.0, 3.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        c.grad,
        np.array([1.0, 1.0], dtype=np.float32),
    )