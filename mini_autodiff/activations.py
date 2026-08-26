from .tensor import Tensor


def softmax(x: Tensor) -> Tensor:
    max_value = x.max()
    shifted = x - max_value
    exp_values = shifted.exp()
    total = exp_values.sum()

    return exp_values / total

def logsumexp(x: Tensor) -> Tensor:
    max_value = x.max()

    shifted = x - max_value

    exp_values = shifted.exp()

    summed = exp_values.sum()

    return summed.log() + max_value