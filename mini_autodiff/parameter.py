from .tensor import Tensor


class Parameter(Tensor):
    """
    A Tensor intended to be optimized during training.
    """

    def __init__(self, data):
        super().__init__(
            data,
            requires_grad=True,
        )