from typing import Dict, List, Optional, Union
import numpy as np
from mlp import Tensor, MLP
from loss_function import mse_loss
from optimizer import SGD

# --------------------------------------------------------------------------- #
# accuracy metric (off-graph, read-only)
# --------------------------------------------------------------------------- #
def accuracy(pred: Union["Tensor", np.ndarray], y: Union["Tensor", np.ndarray]) -> float:
    pred = pred.data.ravel() if isinstance(pred, Tensor) else pred.ravel()
    y = y.data.ravel() if isinstance(y, Tensor) else y.ravel()

    if pred.size != y.size: raise ValueError(pred.size, y.size)

    count = 0
    for i in range(y.size):
        if np.sign(y[i]) == np.sign(pred[i]): count += 1

    return float(count / y.size)

# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def train(
    model: "MLP",
    X: "Tensor",
    y: "Tensor",
    *,
    lr: float = 0.1,
    epochs: int = 200,
    optimizer: Optional["SGD"] = None,
) -> Dict[str, List[float]]:
    
    if not isinstance(X, Tensor) or not isinstance(y, Tensor): 
        raise TypeError

    if X.data.ndim != 2: raise ValueError
    if X.shape[0] != y.shape[0]: raise ValueError
    if y.data.ndim == 2:
        if y.shape[1] != 1: raise ValueError
    else:
        y = Tensor(y.data.reshape(-1,1))

    if optimizer is None: optimizer = SGD(model.parameters(), lr)

    loss_log = [None] * epochs
    accuracy_log = [None] * epochs
    for i in range(epochs):
        pred = model.forward(X)
        L = mse_loss(pred, y)
        L.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_log[i] = float(L.data)
        accuracy_log[i] = accuracy(pred, y)

    return {"loss": loss_log, "accuracy": accuracy_log}

# --------------------------------------------------------------------------- #
# plot helper
# --------------------------------------------------------------------------- #
def plot_history(history: Dict[str, List[float]], path: Optional[str] = None):
    """Plot the loss and accuracy curves from a ``train`` history dict.

    Saves to ``path`` if given, otherwise shows the figure.  Returns the Figure.
    """
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    epochs_axis = range(1, len(history["loss"]) + 1)

    ax_loss.plot(epochs_axis, history["loss"])
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training loss")

    ax_acc.plot(epochs_axis, history["accuracy"])
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.set_title("Training accuracy")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    else:
        plt.show()
    return fig
