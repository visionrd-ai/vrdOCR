import torch
import torch.nn as nn
import torch.nn.functional as F
from src.svtrnet_backbone import (
    Block,
    ConvBNLayer,
    trunc_normal_,
    zeros_,
    ones_,
)



class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Im2Seq, self).__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)  # Transpose to (batch, width, channels)
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers=2, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=None,
        num_layers=1,
        dropout=0,
        bidirectional=False,
        with_linear=False,
    ):
        super(BidirectionalLSTM, self).__init__()
        self.with_linear = with_linear
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                           dropout=dropout, bidirectional=bidirectional)
        if self.with_linear:
            self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_feature):
        recurrent, _ = self.rnn(input_feature)
        if self.with_linear:
            output = self.linear(recurrent)
            return output
        return recurrent


class EncoderWithCascadeRNN(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, num_layers=2, with_linear=False):
        super(EncoderWithCascadeRNN, self).__init__()
        self.out_channels = out_channels[-1]
        self.encoder = nn.ModuleList([
            BidirectionalLSTM(
                in_channels if i == 0 else out_channels[i - 1],
                hidden_size,
                output_size=out_channels[i],
                num_layers=1,
                bidirectional=True,
                with_linear=with_linear,
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(in_channels, hidden_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class EncoderWithSVTR(nn.Module):
    def __init__(self, in_channels, dims=64, depth=2, hidden_dims=120, 
                 use_guide=False, num_heads=8, mlp_ratio=2.0, drop_rate=0.1, 
                 attn_drop_rate=0.1, drop_path=0.0, kernel_size=[3, 3], qk_scale=None):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(in_channels, in_channels // 8, kernel_size=kernel_size, padding=[kernel_size[0] // 2, kernel_size[1] // 2])
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1)
        
        self.svtr_block = nn.ModuleList([
            Block(hidden_dims, num_heads, mlp_ratio=mlp_ratio, qk_scale=qk_scale, 
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_dims)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1)
        self.conv4 = ConvBNLayer(2 * in_channels, in_channels // 8, kernel_size=kernel_size, padding=[kernel_size[0] // 2, kernel_size[1] // 2])
        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1)
        self.out_channels = dims
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):

        z = x.clone() if self.use_guide else x
        h = z
        z = self.conv1(z)
        z = self.conv2(z)
        
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)
        
        for blk in self.svtr_block:
            z = blk(z)
            
        z = self.norm(z)
        z = z.reshape(B, H, W, C).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        
        return z


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels=480, encoder_type='svtr', hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        
        if encoder_type == "reshape":
            self.only_reshape = True
        else:
            support_encoder_dict = {
                "reshape": Im2Seq,
                "fc": EncoderWithFC,
                "rnn": EncoderWithRNN,
                "svtr": EncoderWithSVTR,
                "cascadernn": EncoderWithCascadeRNN,
            }
            assert encoder_type in support_encoder_dict, f"{encoder_type} must be in {support_encoder_dict.keys()}"
            
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, **kwargs)
            elif encoder_type == "cascadernn":
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, hidden_size, **kwargs)
            else:
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, hidden_size)
            
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != "svtr":
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
        return x
