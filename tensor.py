from micrograd.value import Value
import numpy as np
from typing import Tuple, Union, Sequence

Operand = Union["Tensor", float, int, np.ndarray, list]

class Tensor:
    def __init__(
        self,
        data: Operand,
        _prev: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data = np.array(data, dtype=np.float64)
        self._prev = _prev
        self._op = _op
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None

    def _make_tensor(self, data, _prev = (), _op = "") -> "Tensor":
        return type(self)(data, _prev, _op)

    def _coerce(self, other: Operand) -> "Tensor":
        if isinstance(other, Tensor): return other
        return self._make_tensor(data=other)

    @classmethod
    def from_value(cls, v: Value) -> "Tensor":
        return cls(data=v.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def reshape(self, *shape: int) -> "Tensor":
        if (len(shape) == 1) and (isinstance(shape[0], Sequence)):
            shape = shape[0]
        out = self._make_tensor(self.data.reshape(shape), _prev=(self,), _op="rs")
        def _backward():
            Tensor._accumulate(self, out.grad.reshape(self.shape))
        out._backward = _backward
        return out

    def transpose(self) -> "Tensor":
        out = self._make_tensor(self.data.T, _prev=(self,), _op="T")
        def _backward():
            Tensor._accumulate(self, out.grad.T)
        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    @staticmethod
    def _accumulate(grad_into: "Tensor", incoming: np.ndarray) -> None:
        if grad_into.shape == ():
            incoming = incoming.sum()
        else: assert grad_into.shape == incoming.shape
        grad_into.grad += incoming

    def __add__(self, other: Operand) -> "Tensor":
        other = self._coerce(other)
        out = self._make_tensor(self.data + other.data, _prev=(self, other), _op="+")

        def _backward():
            Tensor._accumulate(self,  out.grad)
            Tensor._accumulate(other, out.grad)
        out._backward = _backward

        return out

    def __mul__(self, other: Operand) -> "Tensor":
        other = self._coerce(other)
        out = self._make_tensor(self.data * other.data, _prev=(self, other), _op="*")

        def _backward():
            Tensor._accumulate(self,  out.grad * other.data)
            Tensor._accumulate(other, out.grad * self.data)
        out._backward = _backward

        return out

    def __pow__(self, c: Union[int, float]) -> "Tensor":
        out = self._make_tensor(self.data ** c, _prev=(self,), _op="**")

        def _backward():
            Tensor._accumulate(self,  (c * self.data ** (c - 1)) * out.grad)
        out._backward = _backward

        return out

    def relu(self) -> "Tensor":
        out = self._make_tensor(np.maximum(self.data, 0), _prev=(self,), _op="relu")

        def _backward():
            Tensor._accumulate(self, out.grad * (out.data > 0))
        out._backward = _backward

        return out

    def tanh(self) -> "Tensor":
        out = self._make_tensor(np.tanh(self.data), _prev=(self,), _op="tanh")
        def _backward():
            Tensor._accumulate(self, (1 - out.data ** 2) * out.grad)
        out._backward = _backward

        return out

    def exp(self) -> "Tensor":
        out = self._make_tensor(np.exp(self.data), _prev=(self,), _op="exp")

        def _backward():
            Tensor._accumulate(self, out.data * out.grad)
        out._backward = _backward

        return out

    def log(self) -> "Tensor":
        out = self._make_tensor(np.log(self.data), _prev=(self,), _op="log")

        def _backward():
            Tensor._accumulate(self, out.grad / self.data)
        out._backward = _backward

        return out

    def __matmul__(self, other: Operand) -> "Tensor":

        other = self._coerce(other)
        out = self._make_tensor(self.data @ other.data, _prev=(self, other), _op="@")

        def _backward():
            A = self.data
            B = other.data
            G = out.grad

            if A.ndim == 1:
                A = A.reshape(1, A.size)
            if B.ndim == 1:
                B = B.reshape(B.size, 1)
            if G.ndim == 0:
                G = G.reshape(1,1)
            elif self.data.ndim == 1:
                G = G.reshape(1, G.size)
            elif other.data.ndim == 1:
                G = G.reshape(G.size, 1)

            grad_A = G @ B.T
            grad_B = A.T @ G

            if self.data.ndim == 1:
                grad_A = grad_A.reshape(self.data.shape)

            if other.data.ndim == 1:
                grad_B = grad_B.reshape(other.data.shape)

            Tensor._accumulate(self, grad_A)
            Tensor._accumulate(other, grad_B)
        out._backward = _backward

        return out

    def __neg__(self) -> "Tensor":
        return self * -1

    def __sub__(self, other: Operand) -> "Tensor":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Operand) -> "Tensor":
        return self._coerce(other) - self

    def __truediv__(self, other: Operand) -> "Tensor":
        return self * (self._coerce(other) ** -1)

    def __rtruediv__(self, other: Operand) -> "Tensor":
        return self._coerce(other) * (self ** -1)

    def __radd__(self, other: Operand) -> "Tensor":
        return self + other

    def __rmul__(self, other: Operand) -> "Tensor":
        return self * other

    def backward(self):
        visited = set()
        topo = []
        def topo_sort(root):
            if root not in visited:
                visited.add(root)

                for child in root._prev:
                    topo_sort(child)

                topo.append(root)

        topo_sort(self)

        self.grad.fill(1)

        for v in reversed(topo): v._backward()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def __repr__(self) -> str:
        return f'Tensor(data={self.data}, grad={self.grad})'