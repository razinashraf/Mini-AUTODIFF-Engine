import numpy as np

from .tensor import Tensor
from .utils import validate_shape
from .parameter import Parameter

def zeros(shape, *, requires_grad=False):
    shape = validate_shape(shape)
    data = np.zeros(shape, dtype=np.float32)
    return Tensor(data, requires_grad=requires_grad)


def ones(shape, *, requires_grad=False):
    shape = validate_shape(shape)
    data = np.ones(shape, dtype=np.float32)
    return Tensor(data, requires_grad=requires_grad)


def rand(shape, *, requires_grad=False, seed=None):
    shape = validate_shape(shape)
    rng = np.random.default_rng(seed)
    data = rng.random(shape, dtype=np.float32)

    return Tensor(data, requires_grad=requires_grad)

def xavier(shape, *, seed=None):
    shape = validate_shape(shape)

    if len(shape) != 2:
        raise ValueError("Xavier initialization requires a 2D shape") # (input,output)

    fan_in, fan_out = shape

    rng = np.random.default_rng(seed)

    std = np.sqrt(2.0 / (fan_in + fan_out))
    data = rng.normal(0.0, std, size=shape).astype(np.float32)

    return Parameter(data)


def he(shape, *, seed=None):
    shape = validate_shape(shape)

    if len(shape) != 2:
        raise ValueError("He initialization requires a 2D shape")

    fan_in, _ = shape

    rng = np.random.default_rng(seed)

    std = np.sqrt(2.0 / fan_in)
    data = rng.normal(0.0, std, size=shape).astype(np.float32)

    return Parameter(data)

def bias(shape):
    shape = validate_shape(shape)

    data = np.zeros(shape, dtype=np.float32)

    return Parameter(data)