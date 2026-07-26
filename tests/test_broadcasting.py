import numpy as np

from mini_autodiff.tensor import Tensor

def test_add_broadcasting():
    x = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    b = Tensor([10.0, 20.0, 30.0])

    y = x + b
    y.backward()

    np.testing.assert_array_equal(
        b.grad,
        np.array([2.0, 2.0, 2.0], dtype=np.float32),
    )

    assert b.grad.shape == (3,)

def test_multiply_broadcasting():
    x = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    b = Tensor([10.0, 20.0, 30.0])

    y = x * b
    y.backward()

    np.testing.assert_array_equal(
        b.grad,
        np.array([5.0, 7.0, 9.0], dtype=np.float32),
    )

    assert b.grad.shape == (3,)