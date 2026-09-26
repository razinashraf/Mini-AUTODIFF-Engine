from mini_autodiff.tensor import Tensor
from mini_autodiff.layers import Linear


def test_mlp_activation_chain():
    layer1 = Linear(3, 4)
    layer2 = Linear(4, 1)

    x = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    hidden = layer1(x)
    hidden = hidden.relu()

    output = layer2(hidden)
    output = output.sigmoid()

    output.backward()

    assert hidden.data.shape == (2, 4)
    assert output.data.shape == (2, 1)

    assert x.grad.shape == (2, 3)

    assert layer1.weight.grad.shape == (3, 4)
    assert layer1.bias.grad.shape == (4,)

    assert layer2.weight.grad.shape == (4, 1)
    assert layer2.bias.grad.shape == (1,)




import numpy as np

from mini_autodiff.layers import MLP


def test_mlp_output_shape():
    model = MLP(
        in_features=2,
        layer_sizes=[8, 4, 2]
    )

    x = Tensor(np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]))

    output = model(x)

    assert output.data.shape == (3, 2)

def test_mlp_parameters():
    model = MLP(
        in_features=2,
        layer_sizes=[8, 4, 2]
    )

    parameters = model.parameters()

    assert len(parameters) == 6

    assert parameters[0].data.shape == (2, 8)
    assert parameters[1].data.shape == (8,)

    assert parameters[2].data.shape == (8, 4)
    assert parameters[3].data.shape == (4,)

    assert parameters[4].data.shape == (4, 2)
    assert parameters[5].data.shape == (2,)



from mini_autodiff.optimizers import SGD
from mini_autodiff.training import train_step


def test_mlp_can_learn():
    X = Tensor(np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]))

    y = np.array([0, 1, 1, 1])

    model = MLP(
        in_features=2,
        layer_sizes=[8, 2]
    )

    optimizer = SGD(
        model.parameters(),
        learning_rate=0.1
    )

    def loss_fn():
        logits = model(X)

        losses = []

        for i in range(len(y)):
            losses.append(
                logits.select(i).cross_entropy(y[i])
            )

        total = losses[0]

        for loss in losses[1:]:
            total = total + loss

        return total / len(y)

    initial_loss = loss_fn().data

    for _ in range(100):
        train_step(loss_fn, optimizer)

    final_loss = loss_fn().data

    assert final_loss < initial_loss