from typing import Callable, Dict, List, Optional, Sequence
import numpy as np

from tensor import Tensor
from dense import Dense

def xavier_uniform(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    a = np.sqrt(6 / (n_in+n_out)) * gain
    return np.array(np.random.default_rng(seed).uniform(-a, a, size=(n_in,n_out)))

def xavier_normal(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    std = np.sqrt(2 / (n_in+n_out)) * gain
    return np.random.default_rng(seed).normal(0, std, size=(n_in, n_out))

def he_normal(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    std = np.sqrt(2 / n_in)
    return np.random.default_rng(seed).normal(0.0, std, size=(n_in, n_out))

def he_uniform(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    a = np.sqrt(6 / n_in)
    return np.random.default_rng(seed).uniform(-a, a, (n_in,n_out))


def init_dense(layer: "Dense", W: np.ndarray, b: Optional[np.ndarray] = None) -> None:
    if layer.W.shape == W.shape:
        layer.W.data = W.copy()
    else: raise ValueError

    if layer.bias:
        if (b is not None) and (layer.b.shape == b.shape):
            layer.b.data = b.copy()
        else: raise ValueError

def forward_activation_stats(
    sizes: Sequence[int],
    init_fn: Callable[[int, int], np.ndarray],
    activation: str,
    *,
    n_samples: int = 512,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    
    layers = [None] * (len(sizes) - 1)
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i+1]
        layers[i] = Dense(n_in, n_out, bias=True)
        init_dense(layers[i], W=init_fn(n_in, n_out), b=np.zeros(n_out, np.float64))

    z = Tensor(np.random.default_rng(seed).standard_normal((n_samples, sizes[0])))
    stats = []

    for layer in layers:
        z = layer(z)
        if activation == "tanh":
            z = z.tanh()
        elif activation == "relu":
            z = z.relu()
        
        stats.append({"mean": np.mean(z.data), "std": np.std(z.data), "saturated": np.mean(np.abs(z.data) > 0.98), "dead": np.mean(z.data == 0.0)})
    
    print(len(layers))
    print(len(stats))
    
    return stats

forward_activation_stats([32,32,32,32], he_normal, "tanh")