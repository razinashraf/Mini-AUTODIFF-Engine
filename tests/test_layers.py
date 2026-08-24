import numpy as np

from mini_autodiff.tensor import Tensor
from mini_autodiff.operations import MatMul
from mini_autodiff.layers import Linear
from mini_autodiff.parameter import Parameter


def test_matmul_forward():
    x = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    w = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    y = x @ w

    expected = np.array([
        [22.0, 28.0],
        [49.0, 64.0],
    ], dtype=np.float32)

    np.testing.assert_array_equal(y.data, expected)

    assert y.data.shape == (2, 2)


def test_matmul_backward():
    x = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    w = Tensor([
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    y = x @ w

    y.backward()

    expected_x_grad = np.array([
        [11.0, 15.0],
        [11.0, 15.0],
    ], dtype=np.float32)

    expected_w_grad = np.array([
        [4.0, 4.0],
        [6.0, 6.0],
    ], dtype=np.float32)

    np.testing.assert_array_equal(
        x.grad,
        expected_x_grad,
    )

    np.testing.assert_array_equal(
        w.grad,
        expected_w_grad,
    )


def test_linear_shapes():
    layer = Linear(128, 64)

    x = Tensor(
        np.random.default_rng(42).random(
            (32, 128),
            dtype=np.float32,
        )
    )

    y = layer(x)

    assert y.data.shape == (32, 64)

    assert isinstance(layer.weight, Parameter)
    assert isinstance(layer.bias, Parameter)

    assert layer.weight.data.shape == (128, 64)
    assert layer.bias.data.shape == (64,)

def test_linear_backward():
    layer = Linear(3, 2)

    x = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    y = layer(x)

    y.backward()

    assert x.grad.shape == (2, 3)
    assert layer.weight.grad.shape == (3, 2)
    assert layer.bias.grad.shape == (2,)



    """
             x
             │
             ▼
          MatMul
          ▲    ▲
          │    │
          │   weight
          │
          ▼
       matmul_output
             │
             ▼
            Add
           ▲   ▲
           │   │
           │  bias
           │
           ▼
            y
    
    """

    def test_linear_gradients():
        layer = Linear(2, 2)

        # Use known values instead of random initialization.
        layer.weight.data[:] = np.array(
            [[1.0, 2.0],
            [3.0, 4.0]],
            dtype=np.float32,
        )

        layer.bias.data[:] = np.array(
            [1.0, 2.0],
            dtype=np.float32,
        )

        x = Tensor([
            [5.0, 6.0],
            [7.0, 8.0],
        ])

        y = layer(x)

        y.backward()

        expected_x_grad = np.array([
            [3.0, 7.0],
            [3.0, 7.0],
        ], dtype=np.float32)

        expected_weight_grad = np.array([
            [12.0, 12.0],
            [14.0, 14.0],
        ], dtype=np.float32)

        expected_bias_grad = np.array(
            [2.0, 2.0],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(
            x.grad,
            expected_x_grad,
        )

        np.testing.assert_array_equal(
            layer.weight.grad,
            expected_weight_grad,
        )

        np.testing.assert_array_equal(
            layer.bias.grad,
            expected_bias_grad,
        )