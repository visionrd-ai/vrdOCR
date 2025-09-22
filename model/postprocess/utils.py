from typing import Tuple
import numpy as np

try:
    import paddle
except Exception:
    paddle = None

try:
    import torch
except Exception:
    torch = None


def tensor_to_numpy(x):
    """Return a numpy array for paddle.Tensor, torch.Tensor, or numpy array/list."""
    if paddle is not None and isinstance(x, getattr(paddle, "Tensor", ())):
        return x.numpy()
    if torch is not None and isinstance(x, getattr(torch, "Tensor", ())):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    # generic sequence -> np
    return np.array(x)


def safe_argmax_max(arr: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """np.argmax + np.max in one place with dtype handling."""
    idx = arr.argmax(axis=axis)
    val = arr.max(axis=axis)
    return idx, val
