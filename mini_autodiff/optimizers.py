import numpy as np

def gradient_descent_step(parameter, learning_rate):
    parameter.data -= learning_rate * parameter.grad

class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:                             
            parameter.data -= self.learning_rate * parameter.grad

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)
