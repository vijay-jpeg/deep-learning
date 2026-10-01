from mlp import Tensor
import numpy as np
from typing import Iterable, Tuple

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
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i in range(len(self.params)):
            p = self.params[i]
            g = p.grad
            if g is None: pass
            v = self.velocities[i]
            self.velocities[i] = v = self.beta * v + g

            if self.nesterov:
                p.data -= self.lr * (g + self.beta * v)
            else: p.data -= self.lr * v


    def reset(self) -> None:
        self.velocities = np.zeros_like(self.params)

    def __repr__(self) -> str:
        return f'SGDMomentum(lr={self.lr}, beta={self.beta}, nesterov={self.nesterov})'


# backwards-compatible public alias used by tests.
Momentum = SGDMomentum


class RMSProp(Optimizer):
    """RMSProp: per-parameter step scaled by an EMA of the squared gradient."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        beta: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i in range(len(self.params)):
            p = self.params[i]
            g = p.grad + self.weight_decay * p.data
            v = self.v[i]
            self.v[i] = v = self.beta * v + (1-self.beta) * g**2

            p.data -= self.lr * g / (np.sqrt(v) + self.eps)



class Adam(Optimizer):
    """Adam: 1st-moment EMA + 2nd-moment EMA + bias correction.
    Coupled weight decay (folded into g);"""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def _effective_grad(self, p: "Tensor"):
        return p.grad + self.weight_decay * p.data

    def step(self) -> None:
        self.t += 1

        for i in range(len(self.params)):
            p = self.params[i]
            g = self._effective_grad(p)
            v = self.v[i]
            m = self.m[i]
            self.m[i] = m = self.beta1 * m + (1-self.beta1) * g
            self.v[i] = v = self.beta2 * v + (1-self.beta2) * g**2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay: decay applied to p.data directly, not folded into g."""

    def _effective_grad(self, p):
        return p.grad

    def step(self) -> None:
        super().step()
        for i in range(len(self.params)):
            p = self.params[i]
            p.data *= self.weight_decay
