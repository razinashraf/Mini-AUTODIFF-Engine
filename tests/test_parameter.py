import numpy as np
import pytest

from mini_autodiff.parameter import Parameter
from mini_autodiff.tensor import Tensor
from mini_autodiff.init import zeros,ones,rand
from mini_autodiff.init import bias


def test_parameter():
    p = Parameter([1.0, 2.0, 3.0])

    assert isinstance(p, Tensor)
    assert p.requires_grad
    assert p.is_leaf

    np.testing.assert_array_equal(
        p.data,
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_zeros():
    x = zeros((2, 3))

    assert x.data.shape == (2, 3)
    assert np.all(x.data == 0)


def test_ones():
    x = ones((2, 3))

    assert x.data.shape == (2, 3)
    assert np.all(x.data == 1)


def test_random_seed():
    a = rand((2, 3), seed=42)
    b = rand((2, 3), seed=42)

    np.testing.assert_array_equal(a.data, b.data)


def test_invalid_shape():
    with pytest.raises(ValueError):
        zeros((-2, 3))



from mini_autodiff.init import xavier, he
from mini_autodiff.parameter import Parameter


def test_xavier():
    w = xavier((256, 128), seed=42)

    assert isinstance(w, Parameter)
    assert w.data.shape == (256, 128)
    assert w.data.dtype == np.float32
    assert w.requires_grad


def test_he():
    w = he((256, 128), seed=42)

    assert isinstance(w, Parameter)
    assert w.data.shape == (256, 128)
    assert w.data.dtype == np.float32
    assert w.requires_grad


def test_initialization_reproducible():
    a = xavier((256, 128), seed=42)
    b = xavier((256, 128), seed=42)

    np.testing.assert_array_equal(a.data, b.data)


def test_initialization_shapes():
    x = xavier((128, 64), seed=42)
    h = he((128, 64), seed=42)

    assert x.data.shape == (128, 64)
    assert h.data.shape == (128, 64)


def test_initialization_requires_2d():
    with np.testing.assert_raises(ValueError):
        xavier((128,))

    with np.testing.assert_raises(ValueError):
        he((128,))

def test_bias():
    b = bias((128,))

    assert isinstance(b, Parameter)
    assert b.data.shape == (128,)
    assert b.requires_grad

    np.testing.assert_array_equal(
        b.data,
        np.zeros(128, dtype=np.float32),
    )