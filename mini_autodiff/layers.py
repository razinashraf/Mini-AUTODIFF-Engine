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

    def parameters(self):
        return [self.weight, self.bias]


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def __call__(self, x):  #model(x) no forward call needed
        return self.forward(x)


    def parameters(self): # we want Sequential.parameters (where sequential contains lot of Linear()and relu)
        params = []

        for layer in self.layers:
            if hasattr(layer, "parameters"): #If this layer has trainable parameters, collect them. Otherwise skip it "parameter" is the method name
                params.extend(layer.parameters()) 

        return params

class MLP:
        def __init__(self, in_features, layer_sizes):
            layers = []

            input_size = in_features

            for i, output_size in enumerate(layer_sizes):
                layers.append(
                    Linear(input_size, output_size)
                )

                if i < len(layer_sizes) - 1: # we dont want relu in the last
                    layers.append(
                        lambda x: x.relu()
                    )

                input_size = output_size

            self.network = Sequential(*layers) # *unpack the list and create a Sequential

        def __call__(self, x):
            return self.network(x)

        def parameters(self):
            return self.network.parameters()

class ReLULayer:
    def __call__(self, x):
        return x.relu()