from tensor import Tensor
import numpy as np

class Neuron:

    def __init__(self, n_in: int, activation: str = "tanh", seed: int | None = None):
        self.w = Tensor(np.random.default_rng(seed).uniform(-1, 1, n_in))
        self.b = Tensor(0)
        activations = {
            "tanh": Tensor.tanh,
            "relu": Tensor.relu,
            "none": lambda t: t,
        }
        self.activation = activations[activation]

    def __call__(self, x) -> Tensor:
        x = self.w._coerce(x)
        out = x @ self.w + self.b
        return self.activation(out)

    def parameters(self) -> list:
        return [self.w, self.b]

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad.fill(0)

    def __repr__(self) -> str:
        return f"Neuron(n_in={self.w.shape[0]}, activation={self.activation})"
    