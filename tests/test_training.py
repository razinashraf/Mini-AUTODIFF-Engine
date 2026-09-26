import numpy as np

from mini_autodiff.tensor import Tensor
from mini_autodiff.optimizers import SGD
from mini_autodiff.training import train_step, train

def test_train_step():
    w = Tensor(np.array(4.0))
    b = Tensor(np.array(3.0))

    optimizer = SGD(
        [w, b],
        learning_rate=0.1
    )

    def loss_fn():
        return (w * w) + (b * b)

    initial_loss = loss_fn().data

    train_step(
        loss_fn,
        optimizer
    )

    new_loss = loss_fn().data
    assert new_loss < initial_loss


def test_train():
    w = Tensor(np.array(4.0))
    b = Tensor(np.array(3.0))

    optimizer = SGD(
        [w, b],
        learning_rate=0.1
    )

    def loss_fn():
        return (w * w) + (b * b)

    losses = train(
        loss_fn,
        optimizer,
        steps=20
    )

    assert len(losses) == 20
    assert losses[-1] < losses[0]
    assert abs(w.data) < 0.1
    assert abs(b.data) < 0.1