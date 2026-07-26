import numpy as np

from mini_autodiff.tensor import Tensor
from mini_autodiff.operations import Add
from mini_autodiff.engine import AutogradEngine


def test_add_backward():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])

    add = Add()
    c = add.forward(a, b)

    AutogradEngine.backward(c)

    np.testing.assert_array_equal(
        c.data,
        np.array([5.0, 7.0, 9.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        a.grad,
        np.ones(3, dtype=np.float32),
    )

    np.testing.assert_array_equal(
        b.grad,
        np.ones(3, dtype=np.float32),
    )