import numpy as np
from typing import Iterable

from mlp import Tensor
from optimizer import SGD


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