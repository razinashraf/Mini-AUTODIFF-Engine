import numpy as np

from mini_autodiff.tensor import Tensor
from mini_autodiff.layers import Linear, Sequential


def test_sequential_forward():
    linear1 = Linear(4, 8)
    linear2 = Linear(8, 2)

    class ReLULayer:
        def __call__(self, x):
            return x.relu()

    network = Sequential(
        linear1,
        ReLULayer(),
        linear2
    )

    x = Tensor(np.random.randn(5, 4))

    output = network(x)

    assert output.data.shape == (5, 2)

import numpy as np

from mini_autodiff.tensor import Tensor
from mini_autodiff.layers import Linear, Sequential


def test_batch_forward():
    linear1 = Linear(4, 8)
    linear2 = Linear(8, 2)

    class ReLULayer:
        def __call__(self, x):
            return x.relu()

    network = Sequential(
        linear1,
        ReLULayer(),
        linear2
    )

    x = Tensor(np.random.randn(5, 4))

    output = network(x)

    assert output.data.shape == (5, 2)


import numpy as np

from mini_autodiff.tensor import Tensor
from mini_autodiff.layers import Linear, Sequential


def test_prediction_pipeline():
    linear1 = Linear(4, 8)
    linear2 = Linear(8, 2)

    class ReLULayer:
        def __call__(self, x):
            return x.relu()

    network = Sequential(
        linear1,
        ReLULayer(),
        linear2
    )

    x = Tensor(np.random.randn(5, 4))

    logits = network(x)

    assert logits.data.shape == (5, 2)

    probabilities = logits.softmax()

    assert probabilities.data.shape == (5, 2)

    predictions = np.argmax(probabilities.data, axis=1)

    assert predictions.shape == (5,)