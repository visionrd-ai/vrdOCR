import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Iterable

from model.registeries import LOSSES, build_loss


def _normalize_loss_items(loss_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize:
      - {"name": "CTCLoss", "weight": 1.0, ...}
      - {"CTCLoss": None} / {"CTCLoss": {"arg": 1}} (legacy)
    -> {"name": "CTCLoss", "weight": <opt>, ...params}
    """
    normed: List[Dict[str, Any]] = []
    for entry in loss_items:
        if "name" in entry:
            d = dict(entry)
            name = d.pop("name")
            normed.append({"name": name, **d})
        else:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"[MultiLoss] Invalid loss entry: {entry}")
            name, params = next(iter(entry.items()))
            params = params or {}
            if not isinstance(params, dict):
                raise ValueError(f"[MultiLoss] Params for '{name}' must be dict/None, got {type(params)}")
            normed.append({"name": name, **params})
    return normed


def _infer_device_from_predicts(predicts: Dict[str, Any]) -> torch.device:
    """Try to grab a device from any tensor inside predicts dict/tuple/list."""
    def _iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _iter_tensors(v)
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _iter_tensors(v)
    for t in _iter_tensors(predicts):
        return t.device
    return torch.device("cpu")


def _to_device(x: Any, device: torch.device) -> Any:
    """Move tensors (recursively) to device; leave non-tensors as-is."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        typ = type(x)
        return typ(_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


@LOSSES.register(name="MultiLoss")
class MultiLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # required list
        raw_list = kwargs.pop("loss_config_list")
        loss_items = _normalize_loss_items(raw_list)

        self.loss_funcs = nn.ModuleDict()
        self.loss_weights: Dict[str, float] = {}

        for item in loss_items:
            item = dict(item)   # shallow copy
            name = item.pop("name")
            weight = float(item.pop("weight", 1.0))  # optional per-loss weight

            # build via registry (ensure CTCLoss / NRTRLoss are registered elsewhere)
            self.loss_funcs[name] = build_loss({"name": name, **item})
            self.loss_weights[name] = weight

        self.total_loss: Dict[str, torch.Tensor] = {}

    def forward(self, predicts: Dict[str, Any], batch, device: Optional[torch.device] = None):
        """
        predicts: dict from your MultiHead, e.g. {"ctc": ..., "gtc": ...}
        batch:    original batch from DataLoader; we'll slice like your legacy impl
        device:   optional; if None we infer it from 'predicts'
        """
        self.total_loss = {}

        # pick a device if not provided
        if device is None:
            device = _infer_device_from_predicts(predicts)

        # accumulate on correct device
        total = torch.tensor(0.0, device=device)

        # ensure any tensors we pass from batch are on the same device
        # (your caller passes the original `batch`, so we fix it here)
        # NOTE: we **do not** deep-copy `predicts`; only move batch slices.
        for name, loss_func in self.loss_funcs.items():
            if name in ["CTCLoss", "CTCLossV2"]:
                # expects (image, label_ctc, length, valid_ratio) after predicts['ctc']
                # batch layout comment from your code: [image, label_ctc, label_sar, length, valid_ratio]
                args = batch[:2] + batch[3:]
                args = _to_device(args, device)
                loss_val = loss_func(predicts["ctc"], args)["loss"]

            elif name == "NRTRLoss":
                # # expects (image, label_sar, length, valid_ratio) paired with predicts['gtc']
                args = (batch[:1] + batch[2:])
                args = _to_device(args, device)
                loss_val = loss_func(predicts["gtc"], args)["loss"]

            else:
                raise NotImplementedError(f"{name} is not supported in MultiLoss yet")

            weighted = loss_val * self.loss_weights.get(name, 1.0)
            self.total_loss[name] = weighted
            total = total + weighted

        self.total_loss["loss"] = total
        return self.total_loss
