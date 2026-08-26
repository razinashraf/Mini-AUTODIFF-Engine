import numpy as np

from mini_autodiff.tensor import Tensor

def test_subtract():
    x = Tensor([5.0, 7.0])
    y = Tensor([2.0, 3.0])

    z = x - y

    np.testing.assert_array_equal(
        z.data,
        np.array([3.0, 4.0], dtype=np.float32),
    )

    z.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([1.0, 1.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        y.grad,
        np.array([-1.0, -1.0], dtype=np.float32),
    )


def test_abs_forward():
    x = Tensor([-3.0, 2.0, -5.0, 0.0])

    y = x.abs()

    np.testing.assert_array_equal(
        y.data,
        np.array([3.0, 2.0, 5.0, 0.0], dtype=np.float32),
    )


def test_abs_backward():
    x = Tensor([-3.0, 2.0, -5.0, 0.0])

    y = x.abs()

    y.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([-1.0, 1.0, -1.0, 0.0], dtype=np.float32),
    )

import numpy as np

from mini_autodiff.tensor import Tensor


def test_mean_forward():
    x = Tensor([2.0, 4.0, 6.0, 8.0])

    y = x.mean()

    np.testing.assert_allclose(
        y.data,
        5.0,
    )


def test_mean_backward():
    x = Tensor([2.0, 4.0, 6.0, 8.0])

    y = x.mean()

    y.backward()

    np.testing.assert_allclose(
        x.grad,
        np.array(
            [0.25, 0.25, 0.25, 0.25],
            dtype=np.float32,
        ),
    )


import numpy as np

from mini_autodiff.tensor import Tensor
from mini_autodiff.losses import mae
from mini_autodiff.losses import mse


def test_mae_forward():
    prediction = Tensor([2.0, 5.0, 1.0])
    target = Tensor([1.0, 3.0, 4.0])

    loss = mae(prediction, target)

    np.testing.assert_allclose(
        loss.data,
        2.0,
    )


def test_mae_backward():
    prediction = Tensor([2.0, 5.0, 1.0])
    target = Tensor([1.0, 3.0, 4.0])

    loss = mae(prediction, target)

    loss.backward()

    np.testing.assert_allclose(
        prediction.grad,
        np.array(
            [1 / 3, 1 / 3, -1 / 3],
            dtype=np.float32,
        ),
    )



def test_square_forward():
    x = Tensor([-2.0, 3.0, -4.0])

    y = x.square()

    np.testing.assert_array_equal(
        y.data,
        np.array([4.0, 9.0, 16.0], dtype=np.float32),
    )


def test_square_backward():
    x = Tensor([-2.0, 3.0, -4.0])

    y = x.square()

    y.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([-4.0, 6.0, -8.0], dtype=np.float32),
    )

def test_mse_forward():
    prediction = Tensor([2.0, 5.0, 1.0])
    target = Tensor([1.0, 3.0, 4.0])

    loss = mse(prediction, target)

    np.testing.assert_allclose(
        loss.data,
        14.0 / 3.0,
    )


def test_mse_backward():
    prediction = Tensor([2.0, 5.0, 1.0])
    target = Tensor([1.0, 3.0, 4.0])

    loss = mse(prediction, target)

    loss.backward()

    expected = np.array(
        [
            2.0 / 3.0,
            4.0 / 3.0,
            -2.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        prediction.grad,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

def test_log_forward():
    x = Tensor([1.0, np.e, np.e ** 2])

    y = x.log()

    np.testing.assert_allclose(
        y.data,
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )


def test_log_backward():
    x = Tensor([1.0, 2.0, 4.0])

    y = x.log()

    y.backward()

    np.testing.assert_allclose(
        x.grad,
        np.array([1.0, 0.5, 0.25], dtype=np.float32),
    )


import numpy as np

from mini_autodiff.tensor import Tensor
from mini_autodiff.losses import mae, mse, binary_cross_entropy


def test_mae_forward():
    prediction = Tensor([2.0, 5.0, 1.0])
    target = Tensor([1.0, 3.0, 4.0])

    loss = mae(prediction, target)

    np.testing.assert_allclose(
        loss.data,
        2.0,
    )


def test_mae_backward():
    prediction = Tensor([2.0, 5.0, 1.0])
    target = Tensor([1.0, 3.0, 4.0])

    loss = mae(prediction, target)

    loss.backward()

    expected = np.array(
        [1 / 3, 1 / 3, -1 / 3],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        prediction.grad,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_mse_forward():
    prediction = Tensor([2.0, 5.0, 1.0])
    target = Tensor([1.0, 3.0, 4.0])

    loss = mse(prediction, target)

    np.testing.assert_allclose(
        loss.data,
        14.0 / 3.0,
    )


def test_mse_backward():
    prediction = Tensor([2.0, 5.0, 1.0])
    target = Tensor([1.0, 3.0, 4.0])

    loss = mse(prediction, target)

    loss.backward()

    expected = np.array(
        [
            2.0 / 3.0,
            4.0 / 3.0,
            -2.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        prediction.grad,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_bce_forward():
    prediction = Tensor([0.9, 0.8, 0.2])
    target = Tensor([1.0, 1.0, 0.0])

    loss = binary_cross_entropy(prediction, target)

    expected = (
        -np.log(0.9)
        - np.log(0.8)
        - np.log(0.8)
    ) / 3

    np.testing.assert_allclose(
        loss.data,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_bce_backward():
    prediction = Tensor([0.9, 0.8, 0.2])
    target = Tensor([1.0, 1.0, 0.0])

    loss = binary_cross_entropy(prediction, target)

    loss.backward()

    expected = np.array(
        [
            -1.0 / 0.9,
            -1.0 / 0.8,
            1.0 / 0.8,
        ],
        dtype=np.float32,
    ) / 3.0

    np.testing.assert_allclose(
        prediction.grad,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )