import numpy as np
from mini_autodiff.tensor import Tensor
from mini_autodiff.optimizers import gradient_descent_step
from mini_autodiff.optimizers import SGD

def test_gradient_descent_step():
    parameter = Tensor(np.array(10.0))
    parameter.grad = np.array(-3.0)

    gradient_descent_step(parameter, learning_rate=0.2)

    assert np.isclose(parameter.data, 10.6)

def test_gradient_descent_converges_visual():
    w = Tensor(np.array(4.0))

    losses = []

    for _ in range(20):
        loss = w * w
        losses.append(loss.data)

        loss.backward()

        gradient_descent_step(
            w,
            learning_rate=0.1
        )

        w.grad = np.array(0.0)

    print("Losses:", losses)

    assert abs(w.data) < 0.1



def test_sgd_step():
    parameter = Tensor(np.array(10.0))
    parameter.grad = np.array(-3.0)

    optimizer = SGD(
        [parameter],
        learning_rate=0.2
    )

    optimizer.step()

    assert np.isclose(parameter.data, 10.6)


def test_sgd_zero_grad():
    parameter = Tensor(np.array(10.0))
    parameter.grad = np.array(-3.0)

    optimizer = SGD(
        [parameter],
        learning_rate=0.2
    )

    optimizer.zero_grad()

    assert np.isclose(parameter.grad, 0.0)

def test_sgd_multiple_parameters():
    w = Tensor(np.array(4.0))
    b = Tensor(np.array(3.0))

    optimizer = SGD(
        [w, b],
        learning_rate=0.1
    )

    for _ in range(20):
        optimizer.zero_grad()

        loss = (w * w) + (b * b)

        loss.backward()

        optimizer.step()

    assert abs(w.data) < 0.1
    assert abs(b.data) < 0.1
