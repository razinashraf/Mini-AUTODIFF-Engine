# Mini-AUTODIFF-Engine

A tiny reverse-mode automatic differentiation and neural network engine built completely from scratch in Python and NumPy.

The goal of this project is to understand how modern deep learning frameworks such as PyTorch work internally by rebuilding their core components from first principles.

The project evolves progressively from scalar automatic differentiation into tensor-based autodiff and eventually into a small neural network framework.

---

## Features

* Computational Graph construction
* Directed Acyclic Graphs (DAGs)
* Reverse-mode Automatic Differentiation
* Backpropagation
* Topological Sorting
* Automatic graph traversal
* Gradient Accumulation
* Tensor-based Autodiff
* NumPy-backed Tensors
* Operation abstraction
* Operator Overloading
* NumPy Broadcasting
* Broadcast-aware gradients
* Trainable Parameters
* Tensor initialization utilities
* Xavier initialization
* He initialization
* Bias initialization
* Unit testing

---

# Current Progress

## ✅ Version 1.0 — Scalar Autodiff

The original scalar automatic differentiation engine.

Implemented:

* [x] Scalar `Value` class
* [x] Computational graph construction
* [x] Directed Acyclic Graph representation
* [x] Reverse-mode automatic differentiation
* [x] Topological sorting
* [x] Reverse graph traversal
* [x] Backpropagation
* [x] Chain rule
* [x] Gradient accumulation
* [x] Addition
* [x] Multiplication
* [x] Operator overloading

The Version 1 implementation is preserved as the foundation from which the tensor engine evolved.

---

# ✅ Version 2.0 — Tensor Autodiff

The scalar engine was redesigned into a NumPy-based tensor automatic differentiation system.

Implemented:

* [x] Tensor object
* [x] NumPy-backed tensor data
* [x] Gradient storage
* [x] Parent tensor tracking
* [x] Creator operation tracking
* [x] Operation abstraction
* [x] Addition
* [x] Multiplication
* [x] Autograd engine
* [x] Topological graph traversal
* [x] Reverse-mode backpropagation
* [x] Gradient accumulation
* [x] Leaf tensors
* [x] Intermediate tensors
* [x] `Tensor.backward()`
* [x] Operator overloading
* [x] Broadcasting support
* [x] Broadcast-aware gradients
* [x] Unit tests

The public API now supports expressions such as:

```python
y = x * b + c
y.backward()
```

while the computational graph and gradient propagation are handled internally by the autograd engine.

---

# ✅ Version 2.1 — Tensor Library

The Tensor system was extended with the basic infrastructure required by neural network components.

Implemented:

* [x] `Parameter`
* [x] Shape validation
* [x] Random tensor initialization
* [x] Zero initialization
* [x] One initialization
* [x] Xavier initialization
* [x] He initialization
* [x] Bias initialization
* [x] Reproducible random initialization
* [x] Initialization tests

### Parameter

`Parameter` extends `Tensor` and represents a trainable value belonging to a neural network.

```python
weight = Parameter(...)
```

Parameters automatically require gradients and are intended to be updated by optimizers during training.

### Initialization

The library provides initialization strategies designed to maintain appropriate activation and gradient scales.

Xavier initialization:

[
\sigma =
\sqrt{\frac{2}{fan_{in}+fan_{out}}}
]

He initialization:

[
\sigma =
\sqrt{\frac{2}{fan_{in}}}
]

Biases are initialized to zero.

---

# 🚧 Version 2.2 — Dense Layers

The next stage is to build the first neural network component on top of the existing Tensor and Autograd systems.

## Section 9 — Dense Layers

Planned implementation:

* [ ] Matrix multiplication operation
* [ ] Affine transformation
* [ ] Dense / Linear layer
* [ ] Trainable weight parameter
* [ ] Trainable bias parameter
* [ ] Batch processing
* [ ] Shape validation
* [ ] Forward pass
* [ ] Backward pass through matrix multiplication
* [ ] Dense layer tests

The fundamental transformation is:

[
Y = XW + b
]

For a batch of inputs:

```text
X → (batch_size, input_features)

W → (input_features, output_features)

b → (output_features,)

Y → (batch_size, output_features)
```

For example:

```text
X → (32, 128)
W → (128, 64)
b → (64,)

XW → (32, 64)

XW + b → (32, 64)
```

This stage introduces matrix multiplication and the first neural-network layer built directly on top of the autodiff engine.

---

# 📅 Version 3.0 — Neural Network Framework

After the dense layer is complete, the engine will evolve into a small neural network framework.

Planned:

* [ ] ReLU
* [ ] Sigmoid
* [ ] Tanh
* [ ] Linear / Dense layers
* [ ] Neuron abstraction
* [ ] Layer abstraction
* [ ] Sequential models
* [ ] Loss functions
* [ ] Optimizers
* [ ] SGD
* [ ] Adam
* [ ] Training loop
* [ ] Model parameters
* [ ] Model serialization
* [ ] MNIST training

---

# Architecture

The framework separates numerical storage, mathematical operations, and graph execution.

```text
                       Public API
                           │
                           ▼
                        Tensor
                   data / grad / graph
                           │
                    created by
                           ▼
                       Operation
                  forward / backward
                           │
                           ▼
                    Autograd Engine
             graph traversal / propagation
                           │
                           ▼
                       Parameter
                  trainable Tensor values
```

The responsibilities are intentionally separated.

### Tensor

Responsible for:

* Numerical data
* Gradient storage
* Graph relationships
* Creator operation
* Gradient requirements
* Public tensor operations
* `backward()`

### Operation

Responsible for:

* Forward computation
* Local derivative rules
* Connecting inputs to outputs
* Returning gradients for input tensors

Current operations include:

```text
Add
Multiply
```

Future operations will include:

```text
MatMul
ReLU
Sigmoid
Tanh
...
```

### Autograd Engine

Responsible for:

* Building the computational graph traversal order
* Topological sorting
* Reverse graph traversal
* Calling operation backward functions
* Propagating gradients
* Accumulating gradients
* Handling multiple gradient paths

### Parameter

Responsible for representing trainable tensors.

Parameters are intended to be consumed by future neural network layers and optimizers.

---

# How Automatic Differentiation Works

Consider:

```python
y = x * b + c
```

The forward pass constructs a computational graph:

```text
x ──┐
    │
    ▼
 Multiply ──→ intermediate ──┐
    ▲                        │
    │                        ▼
b ──┘                       Add ──→ y
                              ▲
                              │
                              c
```

Calling:

```python
y.backward()
```

starts reverse-mode automatic differentiation.

The engine:

1. Builds a topological ordering of the graph.
2. Initializes the output gradient.
3. Traverses the graph in reverse order.
4. Calls each operation's local backward rule.
5. Receives gradients for the operation's inputs.
6. Accumulates those gradients into the parent tensors.
7. Continues until the leaf tensors are reached.

This implements the chain rule over the computational graph.

---

# Gradient Accumulation

A tensor can influence an output through multiple paths.

For example:

```text
        x
       / \
      /   \
     ×     ×
      \   /
       \ /
        loss
```

The gradient contributions from every path must be added together.

The engine therefore uses gradient accumulation:

```python
parent.grad += gradient
```

rather than replacing the existing gradient.

This is essential for correct reverse-mode automatic differentiation.

---

# Broadcasting

The Tensor engine supports NumPy-style broadcasting for elementwise operations.

For example:

```python
x.shape == (32, 128)
b.shape == (128,)

y = x + b
```

Forward broadcasting produces:

```text
(32, 128)
+
(128,)
     ↓
(32, 128)
```

During backpropagation, the gradient must be reduced back to the original shape of `b`.

```text
gradient
(32, 128)
     ↓
sum across broadcast dimension
     ↓
(128,)
```

This is handled by the reusable:

```python
unbroadcast()
```

utility.

Broadcast-aware gradients are currently implemented for the elementwise operations supported by the engine.

---

# Tensor Initialization

The Tensor Library provides initialization utilities for creating tensors and trainable parameters.

Examples:

```python
x = zeros((32, 128))

x = ones((32, 128))

x = rand((32, 128), seed=42)
```

Trainable parameters can be created using:

```python
W = xavier((128, 64), seed=42)

W = he((128, 64), seed=42)

b = bias((64,))
```

Initialization functions return `Parameter` objects when the tensor is intended to be trainable.

---

# Project Structure

```text
Mini-AUTODIFF-Engine/
│
├── mini_autodiff/
│   ├── __init__.py
│   ├── tensor.py
│   ├── parameter.py
│   ├── operations.py
│   ├── engine.py
│   ├── utils.py
│   └── init.py
│
├── tests/
│   ├── test_add.py
│   ├── test_graph.py
│   ├── test_parameter.py
│   └── test_init.py
│
├── examples/
│
├── scalar_engine.py
├── README.md
├── requirements.txt
└── .gitignore
```

### `tensor.py`

Defines the `Tensor` object and its public API.

### `parameter.py`

Defines the trainable `Parameter` object.

### `operations.py`

Contains differentiable operations and their local backward rules.

### `engine.py`

Contains the reverse-mode automatic differentiation engine.

### `utils.py`

Contains reusable tensor utilities such as shape validation and broadcast-gradient reduction.

### `init.py`

Contains tensor and parameter initialization utilities including:

* `zeros`
* `ones`
* `rand`
* `xavier`
* `he`
* `bias`

### `scalar_engine.py`

Preserves the original Version 1 scalar autodiff implementation.

### `tests/`

Contains tests for forward computation, graph construction, backward propagation, gradient accumulation, broadcasting, parameters, and initialization.

---

# Example

The current Tensor API allows:

```python
from mini_autodiff.tensor import Tensor

x = Tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

b = Tensor([10.0, 20.0, 30.0])

y = x * b + b

y.backward()

print(y.data)
print(x.grad)
print(b.grad)
```

The user interacts with a simple API:

```python
y = x * b + b
y.backward()
```

while the framework internally handles:

```text
Tensor creation
      ↓
Computational graph
      ↓
Operation tracking
      ↓
Topological sorting
      ↓
Reverse traversal
      ↓
Local derivatives
      ↓
Gradient accumulation
      ↓
Broadcast reduction
```

---

# Educational Goal

Mini-AUTODIFF-Engine is an educational implementation designed to understand the internal mechanisms behind automatic differentiation and neural network frameworks.

The goal is not to compete with production frameworks such as PyTorch.

Instead, the project focuses on understanding **why** these systems work and **how** their core components can be implemented from first principles.

The project intentionally progresses through increasingly complex abstractions:

```text
Scalar Autodiff
       ↓
Tensor Autodiff
       ↓
Tensor Library
       ↓
Dense Layers
       ↓
Neural Network Components
       ↓
Training
```

Each stage builds directly on the previous one.

---

# Learning Philosophy

The project follows a first-principles approach.

Instead of treating:

```python
loss.backward()
```

as a black box, the implementation exposes the mechanisms underneath:

```text
Computational Graph
        ↓
DAG Dependencies
        ↓
Topological Ordering
        ↓
Reverse Traversal
        ↓
Chain Rule
        ↓
Gradient Accumulation
```

Similarly, neural network components will be constructed from the lower-level primitives rather than imported from an existing deep learning framework.

The objective is to understand the architecture deeply enough that a high-level framework becomes understandable rather than mysterious.

---

# Roadmap

```text
                         Mini-AUTODIFF-Engine

                                  │
                                  ▼

                       Version 1 — Scalar
                    Reverse-Mode Autodiff
                                  │
                                  ▼

                       Version 2 — Tensor
                    Reverse-Mode Autodiff
                                  │
                                  ▼

                    Version 2.1 — Tensor Library
               Parameters + Initialization Utilities
                                  │
                                  ▼

                    Version 2.2 — Dense Layers
                  MatMul + Affine Transformations
                                  │
                                  ▼

                       Version 3 — Neural Nets
                Activations + Layers + Loss Functions
                                  │
                                  ▼

                         Training Framework
                     Optimizers + Training Loop
```

---

# Inspiration

This project is inspired by the idea of learning deep learning systems from first principles.

Rather than treating automatic differentiation, tensor operations, initialization, and neural network layers as black boxes, the project rebuilds their core mechanisms manually.

The long-term goal is to progress from:

```text
"What does PyTorch do?"
```

to:

```text
"I understand why PyTorch is designed this way."
```

and eventually:

```text
"I can build a simplified version myself."
```
