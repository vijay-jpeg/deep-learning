from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from mlp import Tensor, MLP
from optimizer import Optimizer, SGD, Adam
from train import accuracy, train
from loss import mse_loss

def iterate_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    
    assert X.shape[0] == y.shape[0]
    assert 0 < batch_size <= y.size

    N = X.shape[0]

    if shuffle:
        indices = np.random.default_rng(seed).permutation(N)
    else:
        indices = np.arange(N)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        if end - start < batch_size and drop_last:
            break

        batch_indices = indices[start:end]

        yield X[batch_indices], y[batch_indices]


def train_minibatch(
    model: "MLP",
    X: "Tensor",
    y: "Tensor",
    *,
    lr: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
    optimizer: Optional["Optimizer"] = None,
    drop_last: bool = False,
) -> Dict[str, object]:
    
    if optimizer is None:
        optimizer = SGD(model.parameters(), lr)

    batch_loss, epoch_loss, steps = [], [], 0

    if y.data.ndim == 1:
        y = y.reshape(-1, 1)
    
    for _ in range(epochs):
        batches = iterate_minibatches(X.data, y.data, batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last)
        batch_loss_sum = 0.0
        sample_count = 0
        for X_b, y_b in batches:
            pred = model(Tensor(X_b))
            loss = mse_loss(pred, y_b)

            current_loss = float(loss.data)
            batch_n = y_b.size
            batch_loss.append(current_loss)
            batch_loss_sum += current_loss * batch_n
            sample_count +=batch_n

            loss.backward()
            optimizer.step()
            steps += 1

            optimizer.zero_grad()

        epoch_loss.append(batch_loss_sum / sample_count)

    return {"batch_loss": batch_loss, "epoch_loss": epoch_loss, "steps": steps}
            
            


def gradient_noise(
    model: "MLP",
    X: "Tensor",
    y: "Tensor",
    batch_size: int,
    *,
    n_batches: int,
    seed: Optional[int] = None,
) -> float:
    
    gradient_vectors = []

    for _ in range(n_batches):
        X_b = Tensor(np.random.default_rng(seed).choice(X.data, size=batch_size, replace=False))
        y_b = np.random.default_rng(seed).choice(y.data, size=batch_size, replace=False)

        model.zero_grad()

        pred = model(X_b)
        loss = mse_loss(pred, y_b)

        loss.backward()

        grad_vector = np.concatenate( [p.grad.ravel() for p in model.parameters()] )

        gradient_vectors.append(grad_vector.copy())

    gradient_vectors = np.stack(gradient_vectors)

    coordinate_variances = np.var(gradient_vectors, axis=0)

    return float(np.mean(coordinate_variances))


def epochs_to_threshold(history: Sequence[float], threshold: float) -> int:
    if threshold >= history[0]: return 1

    for i in range(1, len(history)):
        if history[i-1] >= threshold >= history[i]:
            return i + 1
        
    return -1

