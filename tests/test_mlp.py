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


