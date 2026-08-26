import numpy as np

from .tensor import Tensor


class AutogradEngine:

    @staticmethod #AutogradEngine.backward(output) no need for object
    def backward(output: Tensor):
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)

                for parent in tensor.parents:
                    build_topo(parent)

                topo.append(tensor)

        build_topo(output)

        # d(output) / d(output) = 1
        output.grad = np.ones_like(output.data)

        for tensor in reversed(topo):
            if tensor.is_leaf:
                continue

            input_grads = tensor.creator.backward(tensor.grad)

            for parent, grad in zip(tensor.parents, input_grads):  #(c, 1) (a, 1) 
                
                    if parent.requires_grad:
                        parent.grad += grad

        