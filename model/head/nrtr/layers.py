import math
import torch
import torch.nn as nn
from model.neck.svtrnet_backbone import trunc_normal_

class FCTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        if self.only_transpose:
            return x.transpose(1, 2)
        else:
            return self.fc(x.transpose(1, 2))


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding; returns (B, N, C) + PE(N,C).
    """
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)               # (1, max_len, dim)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        N = x.size(1)
        x = x + self.pe[:, :N, :]
        return self.dropout(x)


class AddPos(nn.Module):
    def __init__(self, dim, w):
        super().__init__()
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, w, dim))
        trunc_normal_(self.dec_pos_embed)

    def forward(self, x):
        x = x + self.dec_pos_embed[:, :x.shape[1], :]
        return x
