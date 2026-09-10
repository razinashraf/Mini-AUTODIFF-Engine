from .init import xavier, bias


class Linear:

    def __init__(self, in_features, out_features):
        self.weight = xavier(
            (in_features, out_features)
        )

        self.bias = bias(
            (out_features,)
        )

    def __call__(self, x):
        return x @ self.weight + self.bias


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def __call__(self, x):  #model(x) no forward
        return self.forward(x)