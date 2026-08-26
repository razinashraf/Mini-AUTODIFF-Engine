import numpy as np

from mini_autodiff.tensor import Tensor

def test_softmax_forward():
    x = Tensor([2.0, 1.0, 0.0])

    y = x.softmax()

    expected = np.array(
        [0.66524096, 0.24472847, 0.09003057],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        y.data,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

def test_softmax_sum():
    x = Tensor([2.0, 1.0, 0.0])

    y = x.softmax()

    np.testing.assert_allclose(
        y.data.sum(),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    )


def test_softmax_stability():
    x = Tensor([1000.0, 999.0, 997.0])

    y = x.softmax()

    assert np.all(np.isfinite(y.data))

    np.testing.assert_allclose(
        y.data.sum(),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    )

def test_softmax_backward():
    x = Tensor([2.0, 1.0, 0.0])

    y = x.softmax()

    y.backward()

    # y.backward() means the incoming gradient is [1, 1, 1].
    #
    # Because softmax outputs always sum to 1,
    # the total derivative with respect to every input
    # should cancel to zero.

    np.testing.assert_allclose(
        x.grad,
        np.zeros(3, dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )

def test_logsumexp_forward():
    x = Tensor([2.0, 1.0, 0.0])

    y = x.logsumexp()

    expected = np.log(
        np.exp(2.0)
        + np.exp(1.0)
        + np.exp(0.0)
    )

    np.testing.assert_allclose(
        y.data,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

def test_logsumexp_stability():
    x = Tensor([1000.0, 999.0, 997.0])

    y = x.logsumexp()

    assert np.isfinite(y.data)

def test_cross_entropy_forward():
    logits = Tensor([2.0, 1.0, 0.0])

    loss = logits.cross_entropy(0)

    expected = np.log(
        np.exp(2.0)
        + np.exp(1.0)
        + np.exp(0.0)
    ) - 2.0

    np.testing.assert_allclose(
        loss.data,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_cross_entropy_stability():
    logits = Tensor([1000.0, 999.0, 997.0])

    loss = logits.cross_entropy(0)

    assert np.isfinite(loss.data)

def test_cross_entropy_backward():
    logits = Tensor([2.0, 1.0, 0.0])

    loss = logits.cross_entropy(0)

    loss.backward()

    expected = np.array(
        [
            0.66524096 - 1.0,
            0.24472847,
            0.09003057,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        logits.grad,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )