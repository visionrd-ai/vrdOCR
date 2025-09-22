# ETESVS/heads/ctc_head.py
from tkinter import NE
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union

from model.registeries import HEADS, NECKS, build_neck  # use your registry/builder


class CTCHeadCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_decay: float = 4e-4,
        mid_channels: Optional[int] = None,
        return_feats: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.mid_channels = mid_channels
        self.return_feats = return_feats

        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels)
            self.fc2 = nn.Linear(mid_channels, out_channels)

        # optional: weight decay handled by the optimizer; fc_decay kept for API parity
        self.fc_decay = fc_decay

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # x: (B, W, C)
        if self.mid_channels is None:
            logits = self.fc(x)              # (B, W, out_classes)
            feats_for_return = x
        else:
            feats = self.fc1(x)              # (B, W, mid)
            logits = self.fc2(feats)         # (B, W, out_classes)
            feats_for_return = feats

        if not self.training:
            # probability for export/eval
            probs = F.softmax(logits, dim=2)
            return probs

        if self.return_feats:
            return feats_for_return, logits
        return logits

@HEADS.register(name="CTCHead")
class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_key: Optional[str] = None,
        neck: Optional[Dict[str, Any]] = None,
        head: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self.decoder_key = decoder_key

        if neck is None:
            neck = {"name": "reshape"}  
            
        if isinstance(neck, dict):
            cfg = dict(neck)  # shallow copy
            name = cfg.get("name")
            if name is None:
                raise KeyError("neck must contain the key 'name'")
            if name not in NECKS:
                raise KeyError(f"Unrecognized neck '{name}'")

        self.neck = build_neck(neck)

        head_cfg = head.copy() if head else {}
        self.classifier = CTCHeadCore(
            in_channels=getattr(self.neck, "out_channels", in_channels),
            out_channels=out_channels,
            **head_cfg,
        )

    @property
    def out_channels(self) -> int:
        # number of classes (useful if someone queries)
        # try to infer from final linear if available
        if hasattr(self.classifier, "fc"):
            return self.classifier.fc.out_features
        if hasattr(self.classifier, "fc2"):
            return self.classifier.fc2.out_features
        return None

    def forward(self, feats: Union[torch.Tensor, list], targets: Optional[torch.Tensor] = None):
        """
        feats: backbone features. Usually tensor (B, C, H, W).
               If a list/tuple is provided, we take the last item by default.
        returns: (B, W, num_classes) logits during training, or probs during eval
        """
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]  # pick the highest-resolution stream if caller passes a list

        # neck will output sequence features (B, W, C_neck)
        seq = self.neck(feats)
        out = self.classifier(seq, targets=targets)
        return out
