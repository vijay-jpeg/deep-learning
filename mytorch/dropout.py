from typing import List, Optional, Sequence
import numpy as np

from .mlp import Tensor, MLP


class Dropout:
    """Inverted-dropout layer with train/eval modes; forward is ``x * Tensor(m / p_keep)``.
    p_keep in (0, 1] (1.0 is identity); seed drives the private mask RNG."""

    def __init__(self, p_keep: float = 0.5, *, seed: Optional[int] = None) -> None:
        assert 0 < p_keep <= 1

        self.p_keep = p_keep
        self._rng = np.random.default_rng(seed)
        self.training = True
        self.mask = None

    def __call__(self, x) -> "Tensor":
        """Forward: train -> ``x * Tensor(m / p_keep)``;
        eval -> identity."""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        if self.training:
            m = self._rng.binomial(1, self.p_keep, size=x.shape)
            s = m / self.p_keep
            self.mask = s
            return x * Tensor(s)
        
        else:
            self.mask = None
            return x


    def forward(self, x) -> "Tensor":
        return self(x)

    def train(self, mode: bool = True) -> "Dropout":
        """Set ``self.training = mode`` (True -> sample + scale, False -> eval).
        Returns ``self``."""
        self.training = mode
        return self

    def eval(self) -> "Dropout":
        """Switch to eval mode (identity). Returns ``self``."""
        return self.train(False)

    def parameters(self) -> List["Tensor"]:
        return []

    def zero_grad(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"Dropout(p_keep={self.p_keep}, training={self.training}, mask={self.mask})"


class MLPDropout(MLP):
    """An ``MLP`` with a ``Dropout`` after each hidden activation (none
    after output). Adds hidden dropouts + a training flag flipping all of them.
    activation/out_activation in ``{"tanh", "relu", "none"}``."""

    def __init__(
        self,
        sizes: Sequence[int],
        *,
        p_keep: float = 0.5,
        activation: str = "tanh",
        out_activation: str = "none",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(sizes, activation, out_activation, seed)
        self.dropouts = [None] * (len(sizes) - 2)
        self.p_keep = p_keep
        self.training = True
        for i in range(len(sizes) - 2):
            self.dropouts[i] = Dropout(p_keep, seed=seed)
            if seed is not None: seed += 1

    def forward(self, x) -> "Tensor":
        z = x

        for layer, dropout in zip(
            self.layers[:-1],
            self.dropouts,
        ):
            z = layer(z)
            z = self._apply_activation(z, self.activation)
            z = dropout(z)

        z = self.layers[-1](z)
        z = self._apply_activation(z, self.out_activation)

        return z

    def __call__(self, x) -> "Tensor":
        return self.forward(x)

    def train(self, mode: bool = True) -> "MLPDropout":
        for dp in self.dropouts:
            dp.training = mode
        self.training = mode

        return self

    def eval(self) -> "MLPDropout":
        return self.train(False)

    # parameters() and zero_grad() are inherited from MLP

    def __repr__(self) -> str:
        # TODO: short repr with sizes, p_keep, activation, out_activation.
        return f"MLPDropout(sizes={self.sizes}, p_keep={self.p_keep}, activation{self.activation}, out_activation={self.out_activation})"
