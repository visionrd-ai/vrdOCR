import torch
import torch.nn as nn
from model.registeries import LOSSES

@LOSSES.register(name="CTCLossV3")
class CTCLossWrapper(nn.Module):
    """
    Thin wrapper around torch.nn.CTCLoss.
    Expects logits shape (T, B, C) BEFORE log_softmax.
    """
    def __init__(self, blank_idx: int = 0, reduction: str = "mean", zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor, input_lengths: torch.Tensor):
        log_probs = logits.log_softmax(2)     # (T, B, C)
        return {"loss": self.ctc(log_probs, targets, input_lengths, target_lengths)}
