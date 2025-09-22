# ETESVS/losses/multi_loss.py
import torch
import torch.nn as nn
from typing import Any, Dict, List

from model.registeries import LOSSES, build_loss


def _normalize_loss_items(loss_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Support BOTH styles:

    YAML style (recommended):
      loss_config_list:
        - name: CTCLoss
        - name: NRTRLoss
        # or pass params:
        # - name: CTCLoss
        #   blank: 0
        #   reduction: mean

    Legacy style (your python dict example):
      loss_config_list: [{'CTCLoss': None}, {'NRTRLoss': None}]
      # or with params:
      # [{'CTCLoss': {'blank': 0}}, {'NRTRLoss': {'alpha': 0.5}}]

    Returns a normalized list where each item has at least {"name": <str>, ...params}
    """
    normed = []
    for entry in loss_items:
        if "name" in entry:
            # new style
            d = dict(entry)
            name = d.pop("name")
            normed.append({"name": name, **d})
        else:
            # legacy one-key dict like {"CTCLoss": None or {...}}
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"[MultiLoss] Invalid loss entry: {entry}")
            name, params = next(iter(entry.items()))
            params = params or {}
            if not isinstance(params, dict):
                raise ValueError(f"[MultiLoss] Params for '{name}' must be a dict or None, got {type(params)}")
            normed.append({"name": name, **params})
    return normed


class MultiLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # --- keep your original knobs ---
        self.weight_1 = kwargs.get("weight_1", 1.0)
        self.weight_2 = kwargs.get("weight_2", 1.0)

        # required list
        raw_list = kwargs.pop("loss_config_list")
        loss_items = _normalize_loss_items(raw_list)

        # build losses from registry
        self.loss_funcs = nn.ModuleDict()
        for item in loss_items:
            item = dict(item)  # shallow copy
            name = item.pop("name")
            # build via registry (so CTCLoss / NRTRLoss must be registered in LOSSES)
            loss = build_loss({"name": name, **item})
            self.loss_funcs[name] = loss

        # storage for last call (kept for parity)
        self.total_loss: Dict[str, torch.Tensor] = {}

    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0

        # NOTE: This routing is exactly like your original code.
        # batch layout (per your comment): [image, label_ctc, label_sar, length, valid_ratio]
        # SAR path removed; NRTR path uses (batch[:1] + batch[2:]) in your code — preserved below.

        for name, loss_func in self.loss_funcs.items():
            if name == "CTCLoss":
                # pass predicts['ctc'] and (image, label_ctc, length, valid_ratio)
                loss_val = loss_func(predicts["ctc"], batch[:2] + batch[3:])["loss"] * self.weight_1

            elif name == "NRTRLoss":
                # pass predicts['gtc'] and (image, label_sar, length, valid_ratio) per your original call
                # (yes, odd in naming; you said you'll adjust later — keeping intact)
                loss_val = loss_func(predicts["gtc"].cuda(), (batch[:1] + batch[2:]))["loss"] * self.weight_2

            else:
                raise NotImplementedError(f"{name} is not supported in MultiLoss yet")

            self.total_loss[name] = loss_val
            total_loss = total_loss + loss_val  # keep dtype by accumulating tensor

        self.total_loss["loss"] = total_loss
        return self.total_loss


# register MultiLoss in the LOSSES registry
LOSSES.register(name="MultiLoss")(MultiLoss)
