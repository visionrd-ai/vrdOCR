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



def tensor_to_numpy(x, prefer_float32: bool = True):
    """
    Convert Paddle/Torch tensors, NumPy arrays, or Python sequences to a NumPy array.
    - If prefer_float32=True, upcasts float16/bfloat16 to float32 to avoid unsupported dtype errors.
    """
    # Paddle tensor
    if paddle is not None and "Tensor" in paddle.__dict__ and isinstance(x, paddle.Tensor):
        t = x
        if prefer_float32:
            # Paddle dtypes have .name, e.g. 'float32', 'float16', 'bfloat16'
            dt_name = getattr(getattr(t, "dtype", None), "name", "")
            if dt_name in ("float16", "bfloat16"):
                t = paddle.cast(t, "float32")
        return t.numpy()

    # Torch tensor
    if torch is not None and isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if prefer_float32:
            # Safely upcast half/bfloat16 to float32
            try:
                if t.dtype in (getattr(torch, "float16", None),
                               getattr(torch, "bfloat16", None)):
                    t = t.to(torch.float32)
            except Exception:
                # Fallback: if dtype attribute is odd, just try float32
                t = t.to(torch.float32)
        return t.numpy()

    # NumPy array (already fine)
    if isinstance(x, np.ndarray):
        if prefer_float32 and x.dtype in (np.float16, getattr(np, "bfloat16", type("x",(object,),{})())):
            return x.astype(np.float32, copy=False)
        return x

    # Python scalars / sequences
    arr = np.array(x)
    if prefer_float32 and arr.dtype in (np.float16, getattr(np, "bfloat16", type("x",(object,),{})())):
        arr = arr.astype(np.float32, copy=False)
    return arr


def safe_argmax_max(arr: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """np.argmax + np.max in one place with dtype handling."""
    idx = arr.argmax(axis=axis)
    val = arr.max(axis=axis)
    return idx, val
