from mlp import Tensor
import numpy as np
from typing import Iterable, List

def mse_loss(pred: "Tensor", target) -> "Tensor":
    y = pred._coerce(target)
    return ((pred - y) ** 2).mean()


def mae_loss(pred: "Tensor", target) -> "Tensor":
    y = pred._coerce(target)
    return ((pred - y).relu() + (y - pred).relu()).mean()

def log_softmax_single(logits: "Tensor") -> "Tensor":
    shift = logits - logits.data.max()
    lse = shift.exp().sum().log()
    return shift - lse


def softmax_single(logits: "Tensor") -> "Tensor":
    return log_softmax_single(logits).exp()


def cross_entropy_loss_single(logits: "Tensor", target) -> "Tensor":
    if isinstance(target, int):
        y = np.zeros_like(logits.data)
        y[target] = 1
    else: y = target

    lp = log_softmax_single(logits)
    return -(lp * y).sum()


def log_softmax(logits: "Tensor") -> "Tensor":
    m = logits.data.max(axis=1, keepdims=True)
    shift = logits - m
    lse = shift.exp().sum(axis=1, keepdims=True).log()
    return shift - lse


def softmax(logits: "Tensor") -> "Tensor":
    return log_softmax(logits).exp()


def cross_entropy_loss(logits: "Tensor", targets: np.ndarray) -> "Tensor":
    if targets.ndim == 1:
        y = np.zeros_like(logits.data)
        y[np.arange(logits.data.shape[0]), targets] = 1
    else: y = targets

    lp = log_softmax(logits)
    return -(lp * y).sum(axis=1).mean()


def l2_penalty(params: Iterable, lam: float = 1.0) -> "Tensor":
    sum = 0
    for p in params:
        sum += (p ** 2).sum()

    return 0.5 * lam * sum


def l1_penalty(params: Iterable, lam: float = 1.0) -> "Tensor":
    """L1 (lasso) penalty: returns scalar Tensor lam * sum(|theta|), using _abs
    so the sub-gradient lam*sign(theta) flows through backward(). lam==0 -> 0 Tensor."""
    # TODO: implement the L1 penalty as a scalar Tensor via _abs
    sum = 0
    for p in params:
        sum += p._abs().sum()

    return lam * sum


def regularized_loss(
    loss: "Tensor",
    params: Iterable,
    *,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "Tensor":
    if not isinstance(loss, Tensor):
        loss = Tensor(loss)

    return loss + l1_penalty(params, l1) + l2_penalty(params, l2)


def l2_grad_equals_weight_decay(
    params: Iterable, lam: float
) -> List[np.ndarray]:
    
    out = []
    for p in params:
        out.append(lam * p.data)

    return out