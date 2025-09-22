import torch
import torch.nn as nn

from model.head.nrtr.layers import FCTranspose, AddPos
from model.head.nrtr.transformer import Transformer
from model.registeries import HEADS

@HEADS.register(name="NRTRHead")
class NRTRHead(nn.Module):
    """
    Registry-facing NRTR head (no neck). Builds:
      before_gtc = [Flatten(2) -> FCTranspose(in_channels -> nrtr_dim) -> (optional) AddPos]
      gtc_head   = Transformer(d_model=nrtr_dim, ..., out_channels=<NRTR vocab size>)
    Config (matches your YAML):
      - in_channels: from backbone
      - nrtr_dim: 384 (default)
      - max_text_length: int (default 25..150 per your YAML)
      - num_decoder_layers: int (default 4)
      - nhead: optional (default: nrtr_dim // 32, at least 1)
      - use_pos: bool (default True)
      - pos_len: int (default 80)  # matches your AddPos(nrtr_dim, 80)
      - out_channels: vocab size from out_channels_list['NRTRLabelDecode']
      - decoder_key: e.g., "NRTRLabelDecode" (kept for routing)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_key: str = "NRTRLabelDecode",
        nrtr_dim: int = 384,
        max_text_length: int = 150,
        num_decoder_layers: int = 4,
        nhead: int = None,
        use_pos: bool = True,
        pos_len: int = 80,
        residual_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        dim_feedforward: int = None,
        **kwargs,
    ):
        super().__init__()
        self.decoder_key = decoder_key
        if nhead is None:
            nhead = max(1, nrtr_dim // 32)
        if dim_feedforward is None:
            dim_feedforward = nrtr_dim * 4

        pre = [nn.Flatten(2), FCTranspose(in_channels, nrtr_dim)]
        if use_pos:
            pre.append(AddPos(nrtr_dim, 80))
        self.before_gtc = nn.Sequential(*pre)

        self.gtc_head = Transformer(
            d_model=nrtr_dim,
            nhead=nhead,
            num_encoder_layers=-1,      # as in your code
            beam_size=-1,               # greedy by default
            num_decoder_layers=num_decoder_layers,
            max_len=max_text_length,
            dim_feedforward=dim_feedforward,
            attention_dropout_rate=attention_dropout_rate,
            residual_dropout_rate=residual_dropout_rate,
            out_channels=out_channels,  # vocab size (NRTRLabelDecode)
        )

    @property
    def out_channels(self):
        # The transformer ultimately projects to vocab size (out_channels + 1 in module),
        # but for interface we expose the logical vocab size provided.
        return self.gtc_head.out_channels - 1

    def forward(self, feats: torch.Tensor, targets=None):
        """
        feats: (B, C, H, W) from backbone
        targets: training tokens or (tokens, lengths)
        returns: logits during training (B, T, vocab), or decoded ids during eval (B, <=max_len)
        """
        # before_gtc produces (B, N, nrtr_dim)
        src = self.before_gtc(feats)
        return self.gtc_head(src, targets=targets)

