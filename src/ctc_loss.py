from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import paddle.nn as pNN
import torch.nn.functional as F
import paddle 


class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = pNN.CTCLoss(blank=0, reduction="none")
        self.loss_torch = nn.CTCLoss(blank=0, reduction="none")
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.permute(1, 0, 2)  # Transpose to (T, N, C) for PyTorch
        N, B, _ = predicts.shape
        preds_lengths = torch.full(size=(B,), fill_value=N, dtype=torch.int64)
        labels = batch[1].to(torch.int32)  # Convert labels to int32
        label_lengths = batch[2].to(torch.int64)  # Convert label lengths to int64
        labels = labels[labels!=0]
        loss =self.loss_torch(predicts.log_softmax(2),labels,preds_lengths,label_lengths)
        # loss = torch.tensor(self.loss_func(paddle.Tensor(predicts.cpu().detach().numpy()), paddle.Tensor(labels.cpu().detach().numpy()), paddle.Tensor(preds_lengths.cpu().detach().numpy()), paddle.Tensor(label_lengths.cpu().detach().numpy())).numpy())
        # loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)

        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = 1.0 - weight
            weight = weight ** 2
            loss = loss * weight
        
        loss = loss.mean()
        return {"loss": loss}
