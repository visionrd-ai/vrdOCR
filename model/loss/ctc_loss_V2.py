# model/loss/ctc_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Sequence, Tuple

from model.registeries import LOSSES

@LOSSES.register(name="CTCLossV2")
class CTCLoss(nn.Module):
    """
    Registry-ready CTCLoss wrapper.

    Expects:
      pred:  (N, T, V)  raw logits (not log-softmax)
      batch: tuple/list like (images, labels, target_lengths, valid_ratio?) or
             (labels, target_lengths) if you pass only label info here.

    Notes:
      - blank_index defaults to 0 (typical when your decoder dict is shifted by +1).
      - If valid_ratio is given: input_lengths[i] = floor(valid_ratio[i] * T)
        else: input_lengths[:] = T
    """
    def __init__(self, blank_index: int = 0, zero_infinity: bool = True, **kwargs):
        super().__init__()
        self.blank_index = int(blank_index)
        self.ctc = nn.CTCLoss(blank=self.blank_index, reduction="mean", zero_infinity=zero_infinity)

    def _unpack_batch(self, batch: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Try to accept both:
          (images, labels, target_lengths, valid_ratio) or
          (labels, target_lengths, valid_ratio) or
          (labels, target_lengths)
        """
        if len(batch) >= 4:
            _, labels, target_lengths, valid_ratio = batch[0], batch[1], batch[2], batch[3]
        elif len(batch) == 3:
            labels, target_lengths, valid_ratio = batch[0], batch[1], batch[2]
        elif len(batch) == 2:
            labels, target_lengths = batch
            valid_ratio = None
        else:
            raise ValueError(f"[CTCLoss] Unexpected batch tuple: lengths={len(batch)}")

        return labels, target_lengths, valid_ratio

    def forward(self, pred: torch.Tensor, batch: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """
        pred: (N,T,V) logits
        """
        device = pred.device
        N, T, V = pred.shape

        # targets and lengths
        labels, target_lengths, valid_ratio = self._unpack_batch(batch)
        labels = labels.to(device)                    # (N, L_max) padded with blank or PAD index (NOT -1)
        target_lengths = target_lengths.to(device)    # (N,)

        # input lengths
        if valid_ratio is not None:
            valid_ratio = valid_ratio.to(device).clamp(min=0.0, max=1.0)
            input_lengths = torch.floor(valid_ratio * T).to(torch.long)
            input_lengths = torch.clamp(input_lengths, min=1, max=T)
        else:
            input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

        # fuse targets as 1D for CTCLoss
        # We assume labels already contain the *target symbols* (no special EOS), padded anywhere >=0
        # Use target_lengths to splice:
        flat_targets = torch.cat([labels[i, :target_lengths[i]] for i in range(N)], dim=0)

        # convert logits -> log_probs and permute to (T,N,V)
        log_probs = F.log_softmax(pred, dim=2).transpose(0, 1)

        loss = self.ctc(log_probs, flat_targets, input_lengths, target_lengths)
        return {"loss": loss}
