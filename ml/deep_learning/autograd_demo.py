"""Backpropagation from first principles, on logistic regression.

For one example with features x (a vector), weights w and bias b:

    z = w·x + b                 (forward: a dot product)
    p = σ(z) = 1 / (1 + e^-z)   (forward: squash to a probability)
    L = -[y log p + (1-y) log(1-p)]   (binary cross-entropy loss)

Chain rule, one link at a time:

    ∂L/∂p = (p - y) / (p (1-p))
    ∂p/∂z = p (1-p)                    (derivative of the sigmoid)
    ∂L/∂z = ∂L/∂p · ∂p/∂z = p - y      (the two p(1-p) cancel)
    ∂L/∂w = ∂L/∂z · ∂z/∂w = (p - y) x
    ∂L/∂b = p - y

Averaged over n examples, in matrix form: ∇w = Xᵀ(p - y) / n,  ∇b = mean(p - y).

``manual_gradients`` computes exactly that; ``autograd_gradients`` lets PyTorch build the
computation graph during the forward pass and walk it backwards (reverse-mode automatic
differentiation). The tests check that both agree. ``gradient_descent`` then uses the
gradient to learn: w ← w - learning_rate · ∇w.
"""
from __future__ import annotations

import torch


def logistic_loss(X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    z = X @ w + b                                 # (n,) logits
    # binary_cross_entropy_with_logits uses log-sum-exp internally: stable for large |z|,
    # unlike computing sigmoid then log, which overflows / gives log(0).
    return torch.nn.functional.binary_cross_entropy_with_logits(z, y)


def manual_gradients(X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    p = torch.sigmoid(X @ w + b)
    residual = p - y                              # ∂L/∂z for every example
    return X.T @ residual / X.shape[0], residual.mean()


def autograd_gradients(X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    w = w.detach().clone().requires_grad_(True)   # leaf tensors PyTorch should track
    b = b.detach().clone().requires_grad_(True)
    loss = logistic_loss(X, y, w, b)              # forward pass records the graph
    loss.backward()                               # backward pass fills .grad via the chain rule
    return w.grad, b.grad


def gradient_descent(
    X: torch.Tensor, y: torch.Tensor, learning_rate: float = 0.5, steps: int = 500
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Plain full-batch gradient descent, no optimiser object: the bare idea."""
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    history = []
    for _ in range(steps):
        loss = logistic_loss(X, y, w, b)
        loss.backward()
        with torch.no_grad():                     # the update itself must not be tracked
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        w.grad.zero_()                            # gradients accumulate unless reset
        b.grad.zero_()
        history.append(loss.item())
    return w.detach(), b.detach(), history
