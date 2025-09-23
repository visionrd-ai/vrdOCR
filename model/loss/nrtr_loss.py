# model/loss/nrtr_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.registeries import LOSSES

@LOSSES.register(name="NRTRLoss")
class NRTRLoss(nn.Module):
    def __init__(self, smoothing: bool = True, ignore_index: int = 0, **kwargs):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        if ignore_index >= 0 and not smoothing:
            self.ce = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    def forward(self, pred, batch):
        """
        pred: (B, T_pred, V) or (B*T_pred, V)
        batch: tuple/list like (images, label_seq, length, valid_ratio) — we only use label_seq here.
               label_seq is assumed to be tokenized with [SOS] at index 0 and PAD=ignore_index.
        """
        # --- ensure expected shape/dtype/device ---
        if pred.dim() == 2:
            # if caller already flattened to (B*T, V), we can't infer T_pred safely.
            # Better to reshape before calling loss. But we can still proceed by
            # deriving T_pred from labels; however the safer route is to keep pred 3D.
            raise ValueError("[NRTRLoss] Expected pred to be (B, T, V). Got flatten (B*T, V). Keep it 3D.")

        B, T_pred, V = pred.shape
        device = pred.device
        dtype = pred.dtype
        if dtype != torch.float32:
            pred = pred.float()  # avoid bf16 precision quirks for CE/log_softmax

        # labels
        # batch is typically (images, label_seq, length, valid_ratio)
        label_seq = batch[1].to(device)

        # We align target tokens to exactly T_pred steps and drop the SOS (teacher-forcing shift):
        #   decoder usually predicts y_1..y_L given inputs [SOS, y_1..y_{L-1}]
        # So we compare logits at positions 0..T_pred-1 to target tokens y_1..y_{T_pred}
        tgt = label_seq[:, 1: 1 + T_pred]  # shape (B, T_pred)

        # flatten for token-level CE
        pred_flat = pred.reshape(B * T_pred, V)        # (B*T_pred, V)
        tgt_flat  = tgt.reshape(B * T_pred)            # (B*T_pred,)

        if self.smoothing:
            eps = 0.1
            n_class = V
            # one-hot with label smoothing
            one_hot = F.one_hot(tgt_flat.clamp_min(0), num_classes=n_class).float()
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

            log_prb = F.log_softmax(pred_flat, dim=1)

            if self.ignore_index >= 0:
                non_pad_mask = tgt_flat.ne(self.ignore_index)
                loss_vec = -(one_hot * log_prb).sum(dim=1)
                loss = loss_vec.masked_select(non_pad_mask).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.ce(pred_flat, tgt_flat)

        return {"loss": loss}
