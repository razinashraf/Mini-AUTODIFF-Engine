Mini-AUTODIFF-Engine

A tiny reverse-mode automatic differentiation and neural network engine built completely from scratch in Python and NumPy.

The goal of this project is to understand how modern deep learning frameworks such as PyTorch work internally by rebuilding their core components from first principles.

The project evolves progressively from scalar automatic differentiation into tensor-based autodiff, neural network components, and eventually a small trainable neural network framework.

Features

Computational Graph construction

Directed Acyclic Graphs (DAGs)

Reverse-mode Automatic Differentiation

Backpropagation

Topological Sorting

Automatic graph traversal

Chain Rule

Gradient Accumulation

Tensor-based Autodiff

NumPy-backed Tensors

Operation abstraction

Operator Overloading

NumPy Broadcasting

Broadcast-aware gradients

Trainable Parameters

Tensor initialization utilities

Xavier initialization

He initialization

Bias initialization

Matrix multiplication

Dense / Linear layers

ReLU

Sigmoid

Tanh

Exponential

Logarithm

Sum

Max

Division

Stable Softmax

Log-Sum-Exp

MSE

MAE

Binary Cross Entropy

Multiclass Cross Entropy

Numerical-stability techniques

Unit testing

Current Progress

✅ Version 1.0 — Scalar Autodiff

The original scalar automatic differentiation engine.

Implemented:

Scalar Value class

Computational graph construction

Directed Acyclic Graph representation

Reverse-mode automatic differentiation

Topological sorting

Reverse graph traversal

Backpropagation

Chain rule

Gradient accumulation

Addition

Multiplication

Operator overloading

The Version 1 implementation is preserved as the foundation from which the tensor engine evolved.

✅ Version 2.0 — Tensor Autodiff

The scalar engine was redesigned into a NumPy-based tensor automatic differentiation system.

Implemented:

Tensor object

NumPy-backed tensor data

Gradient storage

Parent tensor tracking

Creator operation tracking

Operation abstraction

Addition

Multiplication

Subtraction

Division

Autograd engine

Topological graph traversal

Reverse-mode backpropagation

Gradient accumulation

Leaf tensors

Intermediate tensors

Tensor.backward()

Operator overloading

Broadcasting support

Broadcast-aware gradients

Unit tests

The public API supports expressions such as:

y = x * b + c
y.backward()

while the computational graph and gradient propagation are handled internally by the autograd engine.

✅ Version 2.1 — Tensor Library

The Tensor system was extended with the basic infrastructure required by neural network components.

Implemented:

Parameter

Shape validation

Random tensor initialization

Zero initialization

One initialization

Xavier initialization

He initialization

Bias initialization

Reproducible random initialization

Initialization tests

Parameter

Parameter extends Tensor and represents a trainable value belonging to a neural network.

weight = Parameter(...)

Parameters automatically require gradients and are intended to be updated by future optimizers during training.

Parameters are deliberately separated from ordinary tensors so that future optimizers and model abstractions can identify which tensors should be updated during training.

Initialization

The library provides initialization strategies designed to maintain appropriate activation and gradient scales.

Xavier Initialization

[
\sigma =
\sqrt{\frac{2}{fan_{in}+fan_{out}}}
]

He Initialization

[
\sigma =
\sqrt{\frac{2}{fan_{in}}}
]

Biases are initialized to zero.

✅ Version 2.2 — Dense Layers

The engine can now construct trainable affine transformations.

Implemented:

Matrix multiplication operation

Matrix multiplication gradients

Affine transformation

Dense / Linear layer

Trainable weight parameter

Trainable bias parameter

Batch processing

Shape analysis

Forward pass

Backward pass

Dense layer tests

The fundamental transformation is:

[
Y = XW + b
]

For a batch of inputs:

X → (batch_size, input_features)

W → (input_features, output_features)

b → (output_features,)

XW → (batch_size, output_features)

XW + b → (batch_size, output_features)

For example:

X → (32, 128)
W → (128, 64)
b → (64,)

XW → (32, 64)

XW + b → (32, 64)

This stage introduced matrix multiplication and the first neural-network layer built directly on top of the autodiff engine.

✅ Version 2.3 — Activation Functions

The engine now supports nonlinear activation functions required to construct neural networks.

Implemented:

ReLU

ReLU backward propagation

Sigmoid

Sigmoid backward propagation

Tanh

Tanh backward propagation

Activation unit tests

Multi-layer nonlinear computational graphs

ReLU

[
ReLU(x)=\max(0,x)
]

Derivative:

[
ReLU'(x)=
\begin{cases}
0 & x < 0 \
1 & x > 0
\end{cases}
]

ReLU acts as a nonlinear gradient gate.

Negative inputs produce zero output and block the gradient, while positive inputs pass the gradient through.

Sigmoid

[
\sigma(x)=\frac{1}{1+e^{-x}}
]

Derivative:

[
\sigma'(x)=\sigma(x)(1-\sigma(x))
]

Sigmoid maps values into:

[
(0,1)
]

Tanh

[
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
]

Derivative:

[
\tanh'(x)=1-\tanh^2(x)
]

Tanh maps values into:

[
(-1,1)
]

The activation functions are implemented as operations so their local derivatives integrate directly with the existing autograd engine.

✅ Version 2.4 — Loss Functions

The engine now contains the fundamental loss functions used in regression and classification.

Implemented:

Mean Squared Error

Mean Absolute Error

Binary Cross Entropy

Exponential operation

Logarithm operation

Sum operation

Max operation

Stable division gradients

Stable Softmax

Log-Sum-Exp

Differentiable class selection

Multiclass Cross Entropy

Loss backward propagation

Loss gradient tests

Numerical-stability tests

Mean Squared Error

[
MSE =
\frac{1}{n}
\sum_i
(\hat{y}_i-y_i)^2
]

It heavily penalizes large errors.

Mean Absolute Error

[
MAE =
\frac{1}{n}
\sum_i
|\hat{y}_i-y_i|
]

Unlike MSE, MAE does not square the error.

Binary Cross Entropy

[
BCE =
-\frac{1}{n}
\sum_i
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
]

Its derivative with respect to the predicted probability is:

\frac{p-y}{p(1-p)}
]

Stable Softmax

Softmax converts logits into a probability distribution.

[
p_i=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
]

The output probabilities satisfy:

[
\sum_i p_i=1
]

For example:

logits:

[2.0, 1.0, 0.0]

        ↓ Softmax

[0.6652, 0.2447, 0.0900]

Numerical Stability

Directly computing:

np.exp(x)

can overflow for large logits.

The implementation therefore uses:

softmax(z-\max(z))
]

The maximum value is subtracted before exponentiation.

Log-Sum-Exp

The Log-Sum-Exp operation is:

\log
\left(
\sum_i e^{z_i}
\right)
]

A numerically stable form is:

m+
\log
\left(
\sum_i e^{z_i-m}
\right)
}
]

where:

[
m=\max(z)
]

This prevents exponential overflow for large logits.

The engine builds Log-Sum-Exp from differentiable primitive operations:

logits
   ↓
Max
   ↓
Subtract
   ↓
Exp
   ↓
Sum
   ↓
Log
   ↓
Add Max
   ↓
Log-Sum-Exp

An important mathematical relationship is:

[
\boxed{
\nabla LSE(z)=Softmax(z)
}
]

Multiclass Cross Entropy

For classification with logits (z), the stable formulation is:

[
\boxed{
L =
LSE(z)-z_k
}
]

where (k) is the correct class.

The public API is:

logits = Tensor([2.0, 1.0, 0.0])

loss = logits.cross_entropy(0)

loss.backward()

The loss can therefore be constructed as:

Logits
   │
   ├───────────────┐
   ▼               ▼
Log-Sum-Exp    Correct Logit
   │               │
   └───────┬───────┘
           ▼
       Subtract
           │
           ▼
          Loss

The engine does not need to explicitly calculate Softmax during the stable Cross Entropy forward calculation.

Cross Entropy Gradient

For:

[
L=LSE(z)-z_k
]

the gradient with respect to the logits simplifies to:

[
\boxed{
\nabla_z L=p-y
}
]

where:

(p) is the Softmax probability vector

(y) is the one-hot target vector

Example:

prediction:

[0.6652, 0.2447, 0.0900]

target:

[1, 0, 0]

gradient:

[-0.3348, 0.2447, 0.0900]

The engine verifies this gradient through the actual computational graph rather than hardcoding p - y into the loss function.

Broadcasting

The Tensor engine supports NumPy-style broadcasting for elementwise operations.

For example:

x.shape == (32, 128)
b.shape == (128,)

y = x + b

Forward broadcasting produces:

(32, 128)
+
(128,)
     ↓
(32, 128)

During backpropagation, gradients must be reduced back to the original shapes.

Forward:

small tensor
     ↓
broadcast
     ↓
larger tensor


Backward:

larger gradient
     ↓
unbroadcast
     ↓
original tensor shape

This is handled by the reusable:

unbroadcast()

utility.

If one original value is broadcast to multiple locations, the corresponding gradient contributions are summed back into that original value.

How Automatic Differentiation Works

Consider:

y = x * b + c

The forward pass constructs a computational graph:

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

Calling:

y.backward()

starts reverse-mode automatic differentiation.

The engine:

Builds a topological ordering of the graph.

Initializes the output gradient.

Traverses the graph in reverse order.

Calls each operation's local backward rule.

Receives gradients for the operation's inputs.

Accumulates those gradients into the parent tensors.

Continues until the leaf tensors are reached.

This implements the chain rule over the computational graph.

Gradient Accumulation

A tensor can influence an output through multiple paths.

For example:

        x
       / \
      /   \
     ×     ×
      \   /
       \ /
        loss

The gradient contributions from every path must be added together.

The engine therefore uses gradient accumulation:

parent.grad += gradient

rather than replacing the existing gradient.

This is essential for correct reverse-mode automatic differentiation.

Architecture

The framework separates numerical storage, mathematical operations, and graph execution.

                         Public API
                             │
                             ▼
                          Tensor
                    data / grad / graph
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          Operation       Parameter      Losses
       forward/backward  trainable Tensor
              │
              └──────────────┬──────────────┘
                             ▼
                      Autograd Engine
                  graph traversal / propagation
                             │
                             ▼
                         Gradients

Tensor

Responsible for:

Numerical data

Gradient storage

Graph relationships

Creator operation

Gradient requirements

Public tensor operations

backward()

Operation

Responsible for:

Forward computation

Local derivative rules

Connecting inputs to outputs

Returning gradients for input tensors

Current operations include:

Add
Multiply
Subtract
Divide
MatMul
Exp
Log
Sum
Max
ReLU
Sigmoid
Tanh

Autograd Engine

Responsible for:

Building computational graph traversal order

Topological sorting

Reverse graph traversal

Calling operation backward functions

Propagating gradients

Accumulating gradients

Handling multiple gradient paths

Parameter

Responsible for representing trainable tensors.

Parameters are intended to be consumed by neural network layers and future optimizers.

Loss Functions

Responsible for constructing differentiable objective functions from Tensor operations.

Current losses include:

MSE
MAE
Binary Cross Entropy
Cross Entropy

Tensor Initialization

The Tensor Library provides initialization utilities for creating tensors and trainable parameters.

Examples:

x = zeros((32, 128))

x = ones((32, 128))

x = rand((32, 128), seed=42)

Trainable parameters can be created using:

W = xavier((128, 64), seed=42)

W = he((128, 64), seed=42)

b = bias((64,))

Initialization functions return Parameter objects when the tensor is intended to be trainable.

Project Structure

Mini-AUTODIFF-Engine/
│
├── mini_autodiff/
│   ├── __init__.py
│   ├── tensor.py
│   ├── parameter.py
│   ├── operations.py
│   ├── engine.py
│   ├── utils.py
│   ├── init.py
│   ├── layers.py
│   ├── activations.py
│   └── losses.py
│
├── tests/
│   ├── test_add.py
│   ├── test_graph.py
│   ├── test_broadcasting.py
│   ├── test_parameter.py
│   ├── test_layers.py
│   ├── test_relu.py
│   ├── test_activations.py
│   ├── test_error.py
│   └── test_mlp.py
│
├── examples/
│
├── scalar_engine.py
├── README.md
├── requirements.txt
└── .gitignore

tensor.py

Defines the Tensor object and its public API.

parameter.py

Defines the trainable Parameter object.

operations.py

Contains differentiable operations and their local backward rules.

engine.py

Contains the reverse-mode automatic differentiation engine.

utils.py

Contains reusable tensor utilities such as shape validation and broadcast-gradient reduction.

init.py

Contains tensor and parameter initialization utilities including:

zeros

ones

rand

xavier

he

bias

layers.py

Contains neural network layers such as:

Linear

activations.py

Contains higher-level activation functions such as:

softmax

losses.py

Contains loss functions and numerical-stability utilities including:

MSE

MAE

Binary Cross Entropy

Log-Sum-Exp

Cross Entropy

scalar_engine.py

Preserves the original Version 1 scalar autodiff implementation.

tests/

Contains tests for forward computation, graph construction, backward propagation, gradient accumulation, broadcasting, parameters, initialization, layers, activation functions, losses, and numerical stability.

Example

The Tensor API allows:

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

The user interacts with a simple API:

y = x * b + b
y.backward()

while the framework internally handles:

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

Neural Network Computation

The current components can be combined to construct nonlinear neural network computations:

layer1 = Linear(3, 4)
layer2 = Linear(4, 1)

hidden = layer1(x)
hidden = hidden.relu()

output = layer2(hidden)
output = output.sigmoid()

output.backward()

Conceptually:

Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Sigmoid
  ↓
Output

The same computational graph machinery is used regardless of whether the graph contains simple tensor operations or neural network components.

Classification Pipeline

The engine can now express the core mathematical pipeline used for multiclass classification:

Input
  ↓
Linear Layer
  ↓
Logits
  ↓
Stable Softmax
  ↓
Probabilities
  ↓
Cross Entropy
  ↓
Loss
  ↓
Backward
  ↓
Gradients

The numerically stable Cross Entropy implementation can bypass explicit Softmax during loss calculation:

Logits
   │
   ├───────────────┐
   ▼               ▼
Log-Sum-Exp    Correct Logit
   │               │
   └───────┬───────┘
           ▼
       Subtract
           │
           ▼
          Loss

Its gradient is:

[
\boxed{
\nabla_z L=p-y
}
]

This connects the forward probability model, the loss function, and reverse-mode automatic differentiation into one complete computational pipeline.

Testing

The project is developed test-first around the mathematical behavior of each component.

The test suite verifies:

Forward computation

Backward computation

Chain rule

Gradient accumulation

Computational graph construction

Broadcasting

Broadcast-aware gradients

Parameters

Initialization

Matrix multiplication

Dense layers

Activation functions

Stable Softmax

Log-Sum-Exp

MSE

MAE

Binary Cross Entropy

Cross Entropy

Numerical stability

Run the complete test suite with:

python -m pytest

Educational Goal

Mini-AUTODIFF-Engine is an educational implementation designed to understand the internal mechanisms behind automatic differentiation and neural network frameworks.

The goal is not to compete with production frameworks such as PyTorch.

Instead, the project focuses on understanding why these systems work and how their core components can be implemented from first principles.

The project intentionally progresses through increasingly complex abstractions:

Scalar Autodiff
       ↓
Tensor Autodiff
       ↓
Tensor Library
       ↓
Dense Layers
       ↓
Activation Functions
       ↓
Loss Functions
       ↓
Optimizers
       ↓
Training

Each stage builds directly on the previous one.

Learning Philosophy

The project follows a first-principles approach.

Instead of treating:

loss.backward()

as a black box, the implementation exposes the mechanisms underneath:

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

Similarly, neural network components are constructed from lower-level primitives rather than imported from an existing deep learning framework.

The objective is to understand the architecture deeply enough that a high-level framework becomes understandable rather than mysterious.

Roadmap

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
                    Parameters + Initialization
                                  │
                                  ▼
                      Version 2.2 — Dense Layers
                     MatMul + Affine Transformations
                                  │
                                  ▼
                 Version 2.3 — Activation Functions
                       ReLU + Sigmoid + Tanh
                                  │
                                  ▼
                    Version 2.4 — Loss Functions
                  MSE + MAE + BCE + Cross Entropy
                                  │
                                  ▼
                  Version 3 — Neural Network Framework
                    Layers + Model Abstractions
                                  │
                                  ▼
                       Optimizers + Training
                                  │
                                  ▼
                        Complete Training Loop
                                  │
                                  ▼
                             MNIST

Upcoming

Module / Layer abstraction

Parameter registration

Sequential models

Improved MLP abstraction

Optimizers

SGD

Adam

Zeroing gradients

Training loop

Mini-batch training

Model evaluation

Model serialization

MNIST training

Numerical gradient checking

Additional tensor operations

Higher-dimensional tensor support

Inspiration

This project is inspired by the idea of learning deep learning systems from first principles.

Rather than treating automatic differentiation, tensor operations, initialization, neural network layers, Softmax, and loss functions as black boxes, the project rebuilds their core mechanisms manually.

The long-term goal is to progress from:

"What does PyTorch do?"

to:

"I understand why PyTorch is designed this way."

and eventually:

"I can build a simplified version myself."

Current Milestone

The engine has progressed from a scalar autodiff experiment into a functional educational neural-network foundation.

The current pipeline is:

                    Tensor
                      │
                      ▼
              Computational Graph
                      │
                      ▼
                  Autograd
                      │
                      ▼
                Broadcasting
                      │
                      ▼
                  Parameter
                      │
                      ▼
               Initialization
                      │
                      ▼
                   MatMul
                      │
                      ▼
                Linear Layer
                      │
                      ▼
                Activations
                      │
                      ▼
                    Loss
                      │
                      ▼
                 Backprop
                      │
                      ▼
                  Gradients

The next major step is to turn these individual components into a proper trainable neural network framework with model abstractions, parameter management, optimizers, and training loops.