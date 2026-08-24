import numpy as np

from mini_autodiff.tensor import Tensor


def test_relu_forward():
    x = Tensor([-2.0, 3.0, -1.0, 5.0])

    y = x.relu()

    np.testing.assert_array_equal(
        y.data,
        np.array([0.0, 3.0, 0.0, 5.0], dtype=np.float32),
    )


def test_relu_backward():
    x = Tensor([-2.0, 3.0, -1.0, 5.0])

    y = x.relu()

    y.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )

import numpy as np

from mini_autodiff.tensor import Tensor


def test_sigmoid_forward():
    x = Tensor([-2.0, 0.0, 2.0])

    y = x.sigmoid()

    expected = np.array(
        [
            1.0 / (1.0 + np.exp(2.0)),
            0.5,
            1.0 / (1.0 + np.exp(-2.0)),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        y.data,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_sigmoid_backward():
    x = Tensor([-2.0, 0.0, 2.0])

    y = x.sigmoid()

    y.backward()

    expected = y.data * (1.0 - y.data)

    np.testing.assert_allclose(
        x.grad,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

import numpy as np

from mini_autodiff.tensor import Tensor


def test_tanh_forward():
    x = Tensor([-2.0, 0.0, 2.0])

    y = x.tanh()

    expected = np.tanh(
        np.array(
            [-2.0, 0.0, 2.0],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(
        y.data,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_tanh_backward():
    x = Tensor([-2.0, 0.0, 2.0])

    y = x.tanh()

    y.backward()

    expected = 1.0 - y.data ** 2

    np.testing.assert_allclose(
        x.grad,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )