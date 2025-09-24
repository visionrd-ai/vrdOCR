import torch
import torch.nn as nn
from model.registeries import HEADS

@HEADS.register(name="CTCHeadAuto")
class CTCHeadAuto(nn.Module):
    """
    CTC projection head with lazy init:
      - If in_channels is None, infer on first forward from seq.size(-1)
      - num_classes must be provided (e.g., out_channels_list['CTCLabelDecode'])
    Input:  seq (T, B, C_in)
    Output: logits (T, B, num_classes)
    """
    def __init__(self, in_channels: int = None, num_classes: int = None, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        if num_classes is None:
            raise ValueError("CTCHeadAuto requires num_classes (e.g., from out_channels_list['CTCLabelDecode']).")
        self._cfg_in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.proj = None  # lazy

    def _lazy_build(self, in_ch: int, device, dtype):
        self.proj = nn.Linear(in_ch, self.num_classes, bias=True).to(device=device, dtype=dtype)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (T, B, C_in)
        if self.proj is None:
            in_ch = self._cfg_in_channels if self._cfg_in_channels is not None else seq.size(-1)
            self._lazy_build(in_ch, device=seq.device, dtype=seq.dtype)
        return self.proj(self.dropout(seq))
