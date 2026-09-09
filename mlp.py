from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from tensor import Tensor as Stage8_Tensor
from dense import Dense as Stage10_Dense

Operand = Union["Tensor", float, int, np.ndarray, list]

class Tensor(Stage8_Tensor):
    def _coerce(self, other: Operand) -> "Tensor":
        if isinstance(other, Tensor): return other
        return self._make_tensor(data=other)

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:        
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for i in range(len(shape)):
            if shape[i] == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad.reshape(shape)

    def __add__(self, other: "Operand") -> "Tensor":
        other = self._coerce(other)
        out = self._make_tensor(self.data + other.data, _prev=(self, other), _op="+")

        def _backward():
            Tensor._accumulate(self, self._unbroadcast(out.grad, self.shape))
            Tensor._accumulate(other, self._unbroadcast(out.grad, other.shape))
        out._backward = _backward

        return out

    def __mul__(self, other: "Operand") -> "Tensor":
        other = self._coerce(other)
        out = self._make_tensor(self.data * other.data, _prev=(self, other), _op="*")

        def _backward():
            Tensor._accumulate(self,  self._unbroadcast(out.grad * other.data, self.shape))
            Tensor._accumulate(other, self._unbroadcast(out.grad * self.data, other.shape))
        out._backward = _backward

        return out

class Dense(Stage10_Dense):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        seed: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        self.W = Tensor([np.random.default_rng(seed).uniform(-1, 1, n_out)] * n_in)
        if bias:
            self.b = Tensor([0] * n_out)
        else: self.b = None
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias

    def __call__(self, x) -> "Tensor":
        assert isinstance(x, Tensor)
        is_1d = x.data.ndim == 1
        if is_1d:
            x = x.reshape(1, x.shape[0])

        z = x @ self.W

        if self.bias: z = z + self.b

        if is_1d == 1:
            return z.reshape(z.shape[1])

        return z


class MLP:

    def __init__(
        self,
        sizes: Sequence[int],
        activation: str = "tanh",
        out_activation: str = "none",
        seed: Optional[int] = None,
    ) -> None:
        self.activation = activation
        self.out_activation = out_activation
        self.sizes = sizes
        self.layers = [None] * (len(sizes) - 1)

        for i in range(len(sizes) - 1):
            self.layers[i] = Dense(sizes[i], sizes[i+1], seed + i)

    @staticmethod
    def _apply_activation(z: "Stage8_Tensor", name: str) -> "Stage8_Tensor":
        if name == "tanh":
            return z.tanh()
        elif name == "relu":
            return z.relu()
        elif name == "none":
            return z
        else:
            raise ValueError

    def forward(self, x: "Stage8_Tensor") -> "Stage8_Tensor":
        z = x
        for layer in self.layers:
            z = layer(z)
            z = self._apply_activation(z, self.activation)

        return z

    def __call__(self, x: "Stage8_Tensor") -> "Stage8_Tensor":
        return self.forward(x)

    def parameters(self) -> List["Stage8_Tensor"]:
        out = []
        for layer in self.layers:
            out.append(layer.W)
            if layer.bias: out.append(layer.b)

        return out

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.W.zero_grad()
            if layer.bias: layer.b.zero_grad()

    def __repr__(self) -> str:
        return f"MLP({self.sizes}, activation={self.activation}, out_activation={self.out_activation})"