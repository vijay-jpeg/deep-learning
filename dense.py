from typing import List, Optional
import numpy as np
from tensor import Tensor

class Dense:
    """Fully-connected (linear) layer ``Z = X @ W + b`` built on stage_08 Tensor."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.W = Tensor([np.random.default_rng(seed).uniform(-1, 1, n_out)] * n_in)
        if bias:
            self.b = Tensor([0] * n_out)
        else: self.b = None
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias

    def __call__(self, x: "Tensor") -> "Tensor":
        assert isinstance(x, Tensor)
        is_1d = x.data.ndim == 1
        if is_1d:
            x = x.reshape(1, x.shape[0])

        out = x @ self.W

        if self.bias:
            ones = Tensor(np.ones((out.shape[0], 1)))
            b = self.b.reshape(1, self.b.shape[0])
            bias_matrix = ones @ b
            out = out + bias_matrix

        if is_1d == 1:
            return out.reshape(out.shape[1])

        return out

    def parameters(self) -> List["Tensor"]:
        return [self.W, self.b] if self.bias else [self.W]

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad.fill(0)

    def __repr__(self) -> str:
        return f"Dense(n_in={self.n_in}, n_out={self.n_out}, bias={self.bias})"