# Mini-AUTODIFF-Engine

A tiny reverse-mode automatic differentiation engine built completely from scratch in Python.

The goal of this project is to understand how modern deep learning frameworks such as PyTorch compute gradients internally, by implementing every component from first principles.

---

## Features

* Computational Graph construction
* Reverse-mode Automatic Differentiation
* Backpropagation
* Topological Sorting
* Operator Overloading
* Gradient Accumulation
* Automatic graph traversal

---

## Current Progress

### ✅ Version 1.0 — Scalar Autodiff (Completed)

Implemented:

* Scalar `Value` class
* Computational graph construction
* Reverse-mode automatic differentiation
* Topological sorting
* Backward propagation
* Gradient accumulation
* Addition (`+`)
* Multiplication (`*`)

---

### ✅ Version 2.0 — Tensor Autodiff (Completed)

- [x] Tensor object
- [x] Tensor operations
- [x] Gradient storage
- [x] Operation abstraction
- [x] Autograd engine
- [x] Reverse-mode backpropagation
- [x] Gradient accumulation
- [x] Leaf and intermediate tensors
- [x] Operator overloading
- [x] Broadcasting support
- [x] Broadcast-aware gradients
- [x] Unit tests

---

### 📅 Version 3.0 — Neural Network Framework (Planned)

Future roadmap:

* ReLU
* Sigmoid
* Tanh
* Linear layer
* Neuron abstraction
* Layer abstraction
* Multi-Layer Perceptron (MLP)
* Loss functions
* Optimizers (SGD, Adam)
* Training loop
* Model serialization

---

## Example

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

The public API stays simple:

```python
y = x * b + b
y.backward()
```

while the framework builds and differentiates the computational graph internally.

---

## How It Works

Mini-AUTODIFF-Engine uses reverse-mode automatic differentiation.

During the **forward pass**, tensor operations compute numerical results while dynamically constructing a computational graph.

For example:

```python
y = x * b + c
```

conceptually creates:

```text
x ──┐
    Multiply ──→ intermediate ──┐
b ──┘                           Add ──→ y
                               /
c ─────────────────────────────┘
```

The framework separates the system into three main components:

### Tensor

A `Tensor` stores:

- Numerical data
- Accumulated gradients
- Parent tensors
- The operation that created it
- Whether gradients are required

It also provides the public API for operations and backpropagation.

### Operation

Operations such as `Add` and `Multiply` are responsible for:

- Computing the forward result
- Defining the local backward rule
- Connecting output tensors to their inputs

Each operation returns gradients for its inputs during the backward pass.

### Autograd Engine

The `AutogradEngine` coordinates reverse-mode automatic differentiation.

Calling:

```python
y.backward()
```

delegates the actual backward computation to the engine.

The engine:

1. Builds a topological ordering of the computational graph.
2. Initializes the output gradient.
3. Traverses the graph in reverse topological order.
4. Calls each operation's backward rule.
5. Propagates gradients to parent tensors.
6. Accumulates gradient contributions from multiple paths.
7. Reduces gradients when broadcasting occurred during the forward pass.

---

## Architecture

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
              AutogradEngine
        traversal / propagation
```

The responsibilities are intentionally separated:

```text
Tensor
→ stores state and exposes the public API

Operation
→ defines forward computation and local derivatives

AutogradEngine
→ controls graph traversal and gradient propagation
```

This keeps the framework extensible as more tensor operations are added.

---

## Broadcasting

Version 2 supports NumPy-style broadcasting for the current elementwise operations.

For example:

```python
x.shape == (32, 128)
b.shape == (128,)

y = x + b
```

NumPy handles broadcasting during the forward pass:

```text
(32, 128)
+
(128,)
↓
(32, 128)
```

During backpropagation, the engine must reduce the broadcast gradient back to the original input shape:

```text
(32, 128)
↓ sum across broadcast axis
(128,)
```

This is handled by the reusable `unbroadcast()` utility.

---

## Project Structure

```text
Mini-AUTODIFF-Engine/
│
├── mini_autodiff/
│   ├── __init__.py
│   ├── tensor.py
│   ├── operations.py
│   ├── engine.py
│   └── utils.py
│
├── tests/
│   ├── test_add.py
│   └── test_graph.py
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

### `operations.py`

Contains differentiable operations such as `Add` and `Multiply`.

### `engine.py`

Contains the autograd engine responsible for topological traversal and gradient propagation.

### `utils.py`

Contains reusable utilities such as broadcast-gradient reduction.

### `scalar_engine.py`

Preserves the original Version 1 scalar autodiff implementation.

### `tests/`

Tests forward computation, backward propagation, gradient accumulation, leaf tensors, the public Tensor API, and broadcasting behaviour.

---

## Educational Goal

Mini-AUTODIFF-Engine is an educational implementation of the core ideas behind automatic differentiation systems used by modern deep learning frameworks.

The goal is not to compete with production frameworks.

Instead, the project explores how features such as:

```python
y = x * b + c
y.backward()
```

can be implemented from first principles.

The project intentionally evolves in stages:

```text
Scalar Autodiff
      ↓
Tensor Autodiff
      ↓
Neural Network Components
```

This makes it possible to understand both the mathematics of reverse-mode automatic differentiation and the software architecture required to build an extensible autograd system.

---

## Inspiration

This project is inspired by learning deep learning systems from first principles: instead of treating automatic differentiation as a black box, the engine rebuilds its core mechanisms to understand how computational graphs, reverse-mode differentiation, tensor operations, and gradient propagation work internally.
