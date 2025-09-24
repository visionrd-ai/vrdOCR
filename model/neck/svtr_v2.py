import math
import torch
import torch.nn as nn
from model.registeries import NECKS

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)              # (T, C)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term  = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        return self.dropout(x + self.pe[:T, :].unsqueeze(1))  # (T, B, C)

@NECKS.register(name="SVTRNeckV2")
class SVTRNeckV2(nn.Module):
    """
    SVTR-style neck:
      - Conv refine: in_channels -> d_model
      - Height squeeze to 1
      - Permute to sequence (T=W, B, d_model)
      - TransformerEncoder (num_layers, nhead, FFN size)
    Returns:
      seq: (T, B, d_model)
      seq_lens: (B,)
    """
    def __init__(
        self,
        in_channels: int = 480,    # <-- your request
        d_model: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pool: str = "avg",         # or "max"
    ):
        super().__init__()
        self.d_model = d_model
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.squeeze = nn.AdaptiveAvgPool2d((1, None)) if pool == "avg" else nn.AdaptiveMaxPool2d((1, None))
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C=in_channels, H, W)
        """
        b = x.size(0)
        x = self.refine(x)            # (B, d_model, H, W)
        x = self.squeeze(x)           # (B, d_model, 1, W)
        x = x.squeeze(2)              # (B, d_model, W)
        x = x.permute(2, 0, 1).contiguous()  # (T=W, B, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)           # (T, B, d_model)
        T = x.size(0)
        seq_lens = torch.full((b,), T, dtype=torch.long, device=x.device)
        return x, seq_lens
