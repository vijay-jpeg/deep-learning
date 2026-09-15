from mlp import Tensor

def mse_loss(pred: "Tensor", target) -> "Tensor":
    y = pred._coerce(target)
    return ((pred - y) ** 2).mean()


def mae_loss(pred: "Tensor", target) -> "Tensor":
    y = pred._coerce(target)
    return ((pred - y).relu() + (y - pred).relu()).mean()


def log_softmax(logits: "Tensor") -> "Tensor":
    shift = logits - logits.data.max()
    lse = shift.exp().sum().log()
    return shift - lse

def softmax(logits: "Tensor") -> "Tensor":
    return log_softmax(logits).exp()


def cross_entropy_loss(logits: "Tensor", target) -> "Tensor":
    if isinstance(target, int):
        target = [i == target for i in range(logits.data.size)]
    
    y = logits._coerce(target)

    lp = log_softmax(logits)
    return -(lp * y).sum()