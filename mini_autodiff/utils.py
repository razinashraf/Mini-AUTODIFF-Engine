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