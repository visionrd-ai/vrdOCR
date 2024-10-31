import torch
import torch.nn as nn
import torch.nn.functional as F


class NRTRLoss(nn.Module):
    def __init__(self, smoothing=True, ignore_index=0, **kwargs):
        super(NRTRLoss, self).__init__()
        if ignore_index >= 0 and not smoothing:
            self.loss_func = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )
        self.smoothing = smoothing

    def forward(self, pred, batch):
        pred=pred.cuda()
        batch = [b.cuda() for b in batch]
        max_len = batch[2].max()
        tgt = batch[1][:, 1 : 2 + max_len]

        pred = pred.view(-1, pred.shape[2])  # Reshape in PyTorch
        tgt = tgt.contiguous().view(-1)  # Reshape target in PyTorch

        if self.smoothing:
            eps = 0.1
            n_class = pred.shape[1]
            one_hot = F.one_hot(tgt, num_classes=n_class).float()  # Convert to float for further operations
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)  # PyTorch uses `dim` instead of `axis`
            non_pad_mask = tgt.ne(0)  # Check for non-zero elements, PyTorch equivalent of `not_equal`
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).mean()  # Apply mask and compute mean loss
        else:
            loss = self.loss_func(pred, tgt)

        return {"loss": loss}
