import numpy as np

from mini_autodiff import Tensor
from mini_autodiff.gradient_check import numerical_gradient, relative_error, gradient_check, check_parameter_gradients


def test_numerical_gradient():
    w = Tensor(np.array(3.0))

    def loss_fn():
        return w * w

    numerical = numerical_gradient(loss_fn, w)

    assert np.isclose(numerical, 6.0, atol=1e-3)

def test_relative_error():
    error = relative_error(2.0001, 2.0000)

    assert error < 1e-4

def test_gradient_check():
    w = Tensor(np.array(3.0))

    def loss_fn():
        return w * w

    assert gradient_check(loss_fn, w)

def test_all_indices():
    W = Tensor(np.array([
        [2.0, 3.0],
        [4.0, 5.0]
    ]))

    indices = list(np.ndindex(W.data.shape))

    assert indices == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]

def test_check_multiple_parameters():
    W = Tensor(np.array([
        [2.0, 3.0],
        [4.0, 5.0]
    ]))

    b = Tensor(np.array([
        1.0,
        2.0
    ]))

    def loss_fn():
        return (W * W).sum() + (b * b).sum()

    assert check_parameter_gradients(
        loss_fn,
        [W, b]
    )