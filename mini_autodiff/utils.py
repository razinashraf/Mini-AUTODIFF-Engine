import numpy as np


def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Reduce a gradient back to the shape of the tensor
    that was broadcast during the forward pass.
    """

    # Remove extra leading dimensions.
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Sum dimensions that were broadcast from size 1.
    for axis, size in enumerate(target_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


def validate_shape(shape):
    if isinstance(shape, int): # turn 3 into (3,)
        shape = (shape,)

    if not isinstance(shape, tuple):
        raise TypeError("shape must be an int or tuple of integers")

    if any(not isinstance(dim, int) for dim in shape):
        raise TypeError("all dimensions must be integers")

    """
    2    → not int? → False
    3.5  → not int? → True   ← problem!
    4    → not int? → False
    """
    
    if any(dim < 0 for dim in shape):
        raise ValueError("dimensions cannot be negative")


    return shape