from .tensor import Tensor


def mae(prediction: Tensor, target: Tensor) -> Tensor:
    error = prediction - target
    absolute_error = error.abs()

    return absolute_error.mean()

def mse(prediction: Tensor, target: Tensor) -> Tensor:
    error = prediction - target
    squared_error = error.square()

    return squared_error.mean()

def binary_cross_entropy(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:

    positive = target * prediction.log()

    negative = (1.0 - target) * (1.0 - prediction).log()

    return -(positive + negative).mean()


def cross_entropy(logits: Tensor, target: int) -> Tensor:
    correct_logit = logits.select(target)

    return logits.logsumexp() - correct_logit

"""
 FORWARD
                         
Input
  │
  ▼
Linear
  │
  ▼
Logits z
[2, 1, 0]
  │
  │
  ├───────────────┐
  │               │
  ▼               │
subtract max      │
  │               │
  ▼               │
[0,-1,-2]         │
  │               │
  ▼               │
exp               │
  │               │                     
  ▼               │
[1,.368,.135]     │
  │               │
  ▼               │
normalize         │
  │               │
  ▼               │
probabilities p   │
[.665,.245,.090]  │
  │               │
  ▼               │
Cross Entropy     │
  │               │
  ▼               │
Loss              │

for stable we do this

FORWARD

Input
  │
  ▼
Linear
  │
  ▼
Logits z
  │
  ▼
Stable Log-Sum-Exp
  │
  ├─────────────── correct-class logit
  │                         │
  └────────────┬────────────┘
               ▼
             Loss

             
BACKWARD

Loss
 │
 ▼
Softmax + Cross Entropy
 │
 │
 ▼
p - y
 │
 ▼
Linear.backward()
 │
 ├────────→ weight.grad
 │
 ├────────→ bias.grad
 │
 └────────→ x.grad

 

p - y is the derivative of the Softmax + Cross-Entropy loss with respect to the logits z.

More precisely:

doL/doZ = p - y
	

So the chain is:

z (logits)
 ↓
Softmax
 ↓
p
 ↓
Cross Entropy
 ↓
L

Backward:

L
 ↓
          ∂L
          ── = p - y
          ∂z
 ↓
z

"""

