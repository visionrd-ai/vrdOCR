from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from model.registeries import (
    ARCHITECTURE, build_backbone, build_neck, build_head
)

@ARCHITECTURE.register(name="vrdOCRV2")
class VRDOCRV2(nn.Module):
    """
    Minimal OCR architecture:
      backbone -> neck (SVTR-like) -> CTC head
    - Does NOT build/use Loss or PostProcess internally.
    - Returns logits and seq_lens (so caller can compute CTC loss and decode separately).
    """
    def __init__(
        self,
        # High-level model glue (for convenience)
        in_channels: int = 480,                     # <-- feed to neck if not set there
        out_channels_list: Optional[Dict[str,int]] = None,  # e.g. {"CTCLabelDecode": 97}
        # Components (these go to registries)
        backbone: Dict[str, Any] = None,
        neck: Dict[str, Any] = None,
        head: Dict[str, Any] = None,
        loss: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        if out_channels_list is None or "CTCLabelDecode" not in out_channels_list:
            raise ValueError("Provide out_channels_list with key 'CTCLabelDecode' (e.g., 97).")

        # Build backbone (prebuilt in your codebase)
        self.backbone = build_backbone(backbone)

        # Ensure neck has in_channels (fallback to top-level in_channels)
        neck_cfg = dict(neck or {})
        neck_cfg.setdefault("in_channels", in_channels)
        self.neck = build_neck(neck_cfg)

        # Ensure head has num_classes from out_channels_list
        head_cfg = dict(head or {})
        head_cfg.setdefault("num_classes", out_channels_list["CTCLabelDecode"])
        self.head = build_head(head_cfg)

    def forward(self, images: torch.Tensor):
        """
        images: (B, 3, H, W)
        Returns:
          {"logits": (T, B, num_classes), "seq_lens": (B,)}
        """
        feats = self.backbone(images)          # (B, C, H', W')
        seq, seq_lens = self.neck(feats)       # (T, B, D), (B,)
        logits = self.head(seq)                # (T, B, num_classes)
        return {"logits": logits, "seq_lens": seq_lens}
