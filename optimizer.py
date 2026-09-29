from mlp import Tensor
import numpy as np
from typing import Iterable, List

class Optimizer:
    """Base class for optimizers: owns a list of parameter Tensors and updates them in place."""

    def __init__(self, params: Iterable["Tensor"]) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self) -> None:
        """Apply one optimization update to ``self.params`` (subclass-defined)."""
        raise NotImplementedError("Optimizer.step is implemented by a subclass")


class SGD(Optimizer):
    """Plain stochastic gradient descent: p.data -= lr * p.grad."""

    def __init__(self, params: Iterable["Tensor"], lr: float = 0.01) -> None:
        assert lr > 0
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None: p.data -= self.lr * p.grad

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, n_params={len(self.params)})"



class SGDMomentum(SGD):
    """SGD with momentum (heavy-ball) and an optional Nesterov variant."""

    def __init__(
        self,
        params: Iterable["Tensor"],
        lr: float,
        beta: float = 0.9,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params, lr)
        assert 0 <= beta < 1
        self.beta = beta
        self.nesterov = nesterov
        self.velocities = np.zeros_like(params)

    def step(self) -> None:
        for i in range(len(self.params)):
            p = self.params[i]
            g = p.grad
            if g is None: pass
            v = self.velocities[i]
            self.velocities[i] = self.beta * v + g

            if self.nesterov:
                p.data -= self.lr * (g + self.beta * v)
            else: p.data -= self.lr * self.velocities[i]


    def reset(self) -> None:
        self.velocities = np.zeros_like(self.params)

    def __repr__(self) -> str:
        return f'SGDMomentum(lr={self.lr}, beta={self.beta}, nesterov={self.nesterov})'


# backwards-compatible public alias used by tests.
Momentum = SGDMomentum