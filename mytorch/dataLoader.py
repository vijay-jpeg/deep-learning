from typing import Dict, Iterator, Optional, Tuple
import numpy as np

from batch import Tensor, MLP, mse_loss, SGD, train_minibatch



class Dataset:

    def __init__(self, X, y) -> None:
        assert X.shape[0] == y.shape[0]
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)

    def __len__(self) -> int:
        return self.y.size

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class DataLoader:

    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        assert batch_size >= 1
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        N = len(self.dataset)
        
        if self.drop_last:
            return N // self.batch_size
        
        return (N + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple["Tensor", "Tensor"]]:
        N = len(self.dataset)

        if self.shuffle:
            indices = self.rng.permutation(N)
        else:
            indices = np.arange(N)

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)

            if end - start < self.batch_size and self.drop_last:
                break

            batch_indices = indices[start:end]
            X_b, y_b = self.dataset[batch_indices]

            yield Tensor(X_b), Tensor(y_b)


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float,
    *,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert 0 <= val_frac < 1
    assert X.shape[0] == y.shape[0]

    N = X.shape[0]
    indices = np.random.default_rng(seed).permutation(N)
    disjoint = round(val_frac * N)
    train_indices = indices[disjoint:]
    val_indices = indices[:disjoint]

    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def train_with_loader(
    model: "MLP",
    loader: "DataLoader",
    *,
    lr: float = 0.1,
    epochs: int = 50,
    optimizer: Optional["SGD"] = None,
) -> Dict[str, object]:
    if optimizer is None:
        optimizer = SGD(model.parameters(), lr)

    batch_loss, epoch_loss, steps = [], [], 0
    
    for _ in range(epochs):
        batch_loss_sum = 0.0
        sample_count = 0
        for X_b, y_b in loader:
            if y_b.data.ndim == 1: y_b.reshape(-1, 1)
            pred = model(X_b)
            loss = mse_loss(pred, y_b)

            current_loss = float(loss.data)
            batch_n = y_b.size
            batch_loss.append(current_loss)
            batch_loss_sum += current_loss * batch_n
            sample_count += batch_n

            loss.backward()
            optimizer.step()
            steps += 1
            optimizer.zero_grad()

        epoch_loss.append(batch_loss_sum / sample_count)

    return {"batch_loss": batch_loss, "epoch_loss": epoch_loss, "steps": steps}
