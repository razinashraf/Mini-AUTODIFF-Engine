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