# model/head/ctc_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from model.registeries import HEADS, build_neck

class CTCHeadCore(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, seq_feats: torch.Tensor) -> torch.Tensor:
        """
        seq_feats: (N, T, C_in)  ->  logits: (N, T, C_out)
        """
        return self.fc(seq_feats)

@HEADS.register(name="CTCHeadV2")
class CTCHeadV2(nn.Module):
    """
    Registry-ready CTC head.

    YAML (inside head_list entry):
      - name: CTCHead
        decoder_key: CTCLabelDecode
        neck:
          name: SequenceEncoderNeck       # <— or "svtr" via your existing SequenceEncoderNeck mapping
          encoder_type: svtr
          dims: 120
          depth: 2
          hidden_dims: 120
          kernel_size: [1, 3]
          use_guide: true
        head:
          return_feats: false
          apply_softmax_in_eval: true

    Top-level head config must provide:
      in_channels: <int>               # from backbone (or auto-injected by vrdOCR)
      out_channels_list: { CTCLabelDecode: <vocab_size> }
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neck: Optional[Dict[str, Any]] = None,
        head: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        head = head or {}
        neck = neck or {"name": "SequenceEncoderNeck", "encoder_type": "reshape"}

        # build neck from registry; guarantees (N,T,C_seq)
        self.neck = build_neck({"name": neck.pop("name"), "in_channels": in_channels, **neck})
        seq_dim = getattr(self.neck, "out_channels", None)
        if seq_dim is None:
            raise ValueError("[CTCHead] neck must expose 'out_channels' (feature dim per time step)")

        # classifier
        self.classifier = CTCHeadCore(in_channels=seq_dim, out_channels=out_channels)

        # options
        self.return_feats: bool = bool(head.get("return_feats", False))
        self.apply_softmax_in_eval: bool = bool(head.get("apply_softmax_in_eval", True))

    @property
    def out_channels(self) -> int:
        return self.classifier.fc.out_features

    def forward(self, x: torch.Tensor, targets=None):
        """
        x:   (N, C, H, W)
        out: training   -> (N, T, V) logits
             eval       -> (N, T, V) probs (softmax)  [if apply_softmax_in_eval=True]
             if return_feats=True (train): (seq_feats, logits)
        """
        # (N,T,C_seq)
        seq_feats = self.neck(x)

        # safety: shape must be (N,T,C)
        assert seq_feats.dim() == 3, f"[CTCHead] neck must return (N,T,C). Got {seq_feats.shape}"

        logits = self.classifier(seq_feats)  # (N,T,V)

        if self.training:
            if self.return_feats:
                return seq_feats, logits
            return logits

        # EVAL mode: return probabilities for decoder/postprocess (keeps your old behavior)
        if self.apply_softmax_in_eval:
            return F.softmax(logits, dim=2)
        return logits
