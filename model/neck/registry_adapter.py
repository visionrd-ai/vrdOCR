import torch.nn as nn
import torch
from functools import partial
from typing import Dict, Any
from model.neck.encoders import SequenceEncoder
from model.registeries import NECKS

class SequenceEncoderNeck(nn.Module):
    """
    Registry adapter that wraps SequenceEncoder.

    Works with either:
      - name: SequenceEncoder
        encoder_type: svtr
        ...
      - name: svtr  # or rnn/fc/cascadernn/reshape
        ...         # (encoder_type is pre-bound by the alias registration)

    Also plays nicely with your direct path where you do:
      encoder_type = neck_args.pop("name")
      SequenceEncoder(in_channels=..., encoder_type=encoder_type, **neck_args)
    """
    def __init__(self, in_channels=480, encoder_type='svtr', **kwargs):
        super().__init__()
        # Pass through exactly what your head constructs
        self.core = SequenceEncoder(
            in_channels=in_channels,
            encoder_type=encoder_type,
            **kwargs
        )
        # expose output channels for heads to size their linear layers
        self.out_channels = getattr(self.core, "out_channels", in_channels)

    def forward(self, x):
        return self.core(x)
      

class SequenceEncoderNeckV2(nn.Module):
    """
    Wraps your SequenceEncoder so heads get a clean (N, T, C) sequence.
    Config example:
      neck:
        name: SequenceEncoderNeck
        encoder_type: svtr
        dims: 120
        depth: 2
        hidden_dims: 120
        kernel_size: [1,3]
        use_guide: true
    """
    def __init__(self, in_channels: int, **kwargs: Dict[str, Any]):
        super().__init__()
        # Pass through exactly what your SequenceEncoder expects
        self.core = SequenceEncoder(in_channels=in_channels, **kwargs)
        self.out_channels = self.core.out_channels  # channel dim at sequence step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W) -> seq: (N, T, C_out)
        """
        seq = self.core(x)   # your SequenceEncoder already returns (N, T, C)
        return seq



def register_default_necks():
    # Primary, explicit entry: lets you write
    #   neck:
    #     name: SequenceEncoder
    #     encoder_type: svtr
    #     dims: 120
    #     depth: 2
    NECKS.register(name="SequenceEncoder")(SequenceEncoderNeck)
    
    NECKS.register(name="SequenceEncoderNeckV2")(SequenceEncoderNeckV2)

    # Aliases that pre-bind `encoder_type` so you can also write:
    #   neck:
    #     name: svtr  # or rnn/fc/cascadernn/reshape
    #     ...
    NECKS.register(name="svtr")(partial(SequenceEncoderNeck, encoder_type='svtr'))
    NECKS.register(name="rnn")(partial(SequenceEncoderNeck,  encoder_type='rnn'))
    NECKS.register(name="fc")(partial(SequenceEncoderNeck,   encoder_type='fc'))
    NECKS.register(name="cascadernn")(partial(SequenceEncoderNeck, encoder_type='cascadernn'))
    NECKS.register(name="reshape")(partial(SequenceEncoderNeck, encoder_type='reshape'))
