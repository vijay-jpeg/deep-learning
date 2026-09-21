from mlp import Tensor
import numpy as np

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