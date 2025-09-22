# ETESVS/architectures/vrdocr.py
import imp
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from model.registeries import (
    ARCHITECTURE,
    build_backbone,
    build_head
)

@ARCHITECTURE.register(name="vrdOCR")
class vrdOCR(nn.Module):
    """
    Registry-ready OCR architecture.

    Expected top-level cfg (your YAML under MODEL):
      architecture: vrdOCR
      backbone: {...}
      head: {...}
      loss: {...}        # optional for pure inference builds

    We build each submodule via its registry.

    Notes:
      - If head.in_channels is omitted, we will try to infer it from backbone.out_channels.
      - backbone.freeze_backbone: if True, set requires_grad=False on all backbone params.
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        loss: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        # ---- Build backbone -------------------------------------------------
        # keep a copy so we can read special flags (e.g., freeze_backbone)
        bb_cfg = dict(backbone)
        freeze_backbone = bool(bb_cfg.pop("freeze_backbone", False))
        self.backbone = build_backbone(bb_cfg)

        if freeze_backbone:
            print("[vrdOCR] Freezing the backbone parameters.")
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Try to expose backbone out_channels for downstream convenience
        self.backbone_out = getattr(self.backbone, "out_channels", None)

        # ---- Build head -----------------------------------------------------
        head_cfg = dict(head)

        # auto-inject in_channels from backbone if not provided
        if "in_channels" not in head_cfg or head_cfg.get("in_channels") in (None, 0):
            if self.backbone_out is None:
                raise ValueError(
                    "[vrdOCR] head.in_channels not provided and backbone does not expose 'out_channels'. "
                    "Please set 'head.in_channels' in the YAML."
                )
            head_cfg["in_channels"] = self.backbone_out

        self.head = build_head(head_cfg)

    def forward(self, images: torch.Tensor, labels=None):
        # 1) features from backbone
        feats = self.backbone(images)

        # 2) predictions from head
        if labels is not None:
            preds = self.head(feats, labels)
        else:
            preds = self.head(feats)

        # 3) if a loss is attached and labels are provided, compute it here
        if self.training and self.loss_fn is not None and labels is not None:
            # MultiLoss expects (predicts, batch). You can adapt here if needed.
            return self.loss_fn(preds, labels)

        return preds
