import torch
import torch.nn as nn
import torch.nn.functional as F
from model.registeries import LOSSES

@LOSSES.register(name="CTCLoss")
class CTCLoss(nn.Module):
    def __init__(self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = True, **kwargs):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, pred, batch):
        """
        pred: (B, T, C) logits
        batch: [images, labels_ctc, target_lengths?, valid_ratio?]
        """
        labels = batch[1]
        tgt_lengths = batch[2] if len(batch) > 2 else None

        B, T, C = pred.shape

        # compute CTC in fp32 to avoid bf16 pitfalls
        with torch.cuda.amp.autocast(enabled=False):
            logp = F.log_softmax(pred.float(), dim=2)      # (B, T, C)
            logp = logp.permute(1, 0, 2).contiguous()      # (T, B, C)

            input_lengths = torch.full(
                size=(B,), fill_value=T, dtype=torch.long, device=pred.device
            )

            if labels.dim() == 2:
                PAD = 0
                tgt_lengths = (labels != PAD).sum(dim=1).to(torch.long) if tgt_lengths is None else tgt_lengths.to(torch.long)
                targets = labels[labels != PAD].to(torch.long)
            else:
                targets = labels.to(torch.long)
                if tgt_lengths is None:
                    raise ValueError("[CTCLoss] Packed targets require target_lengths.")

            loss = self.ctc(logp, targets, input_lengths, tgt_lengths)
        return {"loss": loss}
