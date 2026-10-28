from typing import List, Optional, Sequence
import numpy as np

from mlp import Tensor, MLP


class Dropout:
    """Inverted-dropout layer with train/eval modes; forward is ``x * Tensor(m / p_keep)``.
    p_keep in (0, 1] (1.0 is identity); seed drives the private mask RNG."""

    def __init__(self, p_keep: float = 0.5, *, seed: Optional[int] = None) -> None:
        # TODO: validate 0 < p_keep <= 1; store p_keep, _rng, training=True, mask=None.
        raise NotImplementedError("Dropout.__init__")

    def __call__(self, x) -> "Tensor":
        """Forward: train -> ``x * Tensor(m / p_keep)`` (store scale in self.mask);
        eval -> identity."""
        # TODO: implement the train/eval forward described above.
        raise NotImplementedError("Dropout.__call__")

    def forward(self, x) -> "Tensor":
        """Alias for :meth:`__call__`."""
        # TODO: delegate to __call__.
        raise NotImplementedError("Dropout.forward")

    def train(self, mode: bool = True) -> "Dropout":
        """Set ``self.training = mode`` (True -> sample + scale, False -> eval).
        Returns ``self``."""
        # TODO: set self.training = mode.
        raise NotImplementedError("Dropout.train")

    def eval(self) -> "Dropout":
        """Switch to eval mode (identity). Returns ``self``."""
        # TODO: set eval mode.
        raise NotImplementedError("Dropout.eval")

    def parameters(self) -> List["Tensor"]:
        """Dropout has no learnable parameters (returns [])."""
        # TODO: implement
        raise NotImplementedError("Dropout.parameters")

    def zero_grad(self) -> None:
        """No parameters -> nothing to clear."""
        # TODO: implement
        raise NotImplementedError("Dropout.zero_grad")

    def __repr__(self) -> str:
        # TODO: short repr with p_keep and training.
        raise NotImplementedError("Dropout.__repr__")


class MLPDropout(MLP):
    """An ``MLP`` (stage_11) with a ``Dropout`` after each hidden activation (none
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
        # Defer Dense-stack build + validation + size/activation storage to MLP ctor
        # (positional: sizes, activation, out_activation, seed).
        super().__init__(sizes, activation, out_activation, seed)
        # TODO: store p_keep/training; build self.dropouts, one per hidden layer
        #   (distinct derived seed each).
        raise NotImplementedError("MLPDropout.__init__")

    def forward(self, x) -> "Tensor":
        """Per hidden layer: dense -> activation -> dropout; output: dense ->
        out_activation (no dropout)."""
        # TODO: implement the layered forward described above (no gradients here).
        raise NotImplementedError("MLPDropout.forward")

    def __call__(self, x) -> "Tensor":
        """Alias for :meth:`forward`."""
        # TODO: delegate to forward.
        raise NotImplementedError("MLPDropout.__call__")

    def train(self, mode: bool = True) -> "MLPDropout":
        """Set ``self.training = mode`` on the model AND every owned Dropout.
        Returns self."""
        # TODO: set self.training = mode on self and all dropouts.
        raise NotImplementedError("MLPDropout.train")

    def eval(self) -> "MLPDropout":
        """Put the model AND every owned Dropout in eval mode. Returns self."""
        # TODO: set eval mode on self and all dropouts.
        raise NotImplementedError("MLPDropout.eval")

    # parameters() and zero_grad() are inherited from MLP

    def __repr__(self) -> str:
        # TODO: short repr with sizes, p_keep, activation, out_activation.
        raise NotImplementedError("MLPDropout.__repr__")
