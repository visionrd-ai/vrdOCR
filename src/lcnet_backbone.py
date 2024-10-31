import torch
import torch.nn as nn
import torch.nn.functional as F

def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0, lr_mult=1.0, lab_lr=0.1):
        super(LearnableAffineBlock, self).__init__()
        self.scale = nn.Parameter(torch.full((1,), scale_value) * lr_mult * lab_lr)
        self.bias = nn.Parameter(torch.full((1,), bias_value) * lr_mult * lab_lr)

    def forward(self, x):
        return self.scale * x + self.bias

class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, lr_mult=1.0):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding=(kernel_size - 1) // 2, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Act(nn.Module):
    def __init__(self, act="hswish", lr_mult=1.0, lab_lr=0.1):
        super(Act, self).__init__()
        if act == "hswish":
            self.act = nn.Hardswish()
        else:
            self.act = nn.ReLU()
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        return self.lab(self.act(x))

class LearnableRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, num_conv_branches=1, lr_mult=1.0, lab_lr=0.1):
        super(LearnableRepLayer, self).__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.conv_kxk = nn.ModuleList([
            ConvBNLayer(in_channels, out_channels, kernel_size, stride, groups, lr_mult)
            for _ in range(self.num_conv_branches)
        ])

        self.conv_1x1 = ConvBNLayer(in_channels, out_channels, 1, stride, groups, lr_mult) if kernel_size > 1 else None
        self.identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)
        self.act = Act(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        out = 0
        if self.identity is not None:
            out += self.identity(x)

        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)

        for conv in self.conv_kxk:
            out += conv(x)

        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return identity * x

class LCNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dw_size, use_se=False, conv_kxk_num=4, lr_mult=1.0, lab_lr=0.1):
        super(LCNetV3Block, self).__init__()
        self.use_se = use_se

        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels, out_channels=in_channels, kernel_size=dw_size, stride=stride, groups=in_channels, num_conv_branches=conv_kxk_num, lr_mult=lr_mult, lab_lr=lab_lr
        )
        if use_se:
            self.se = SELayer(in_channels, lr_mult=lr_mult)
        self.pw_conv = LearnableRepLayer(
            in_channels, out_channels, 1, 1, num_conv_branches=conv_kxk_num, lr_mult=lr_mult, lab_lr=lab_lr
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x

class LCNet(nn.Module):
    def __init__(self, scale=1.0, conv_kxk_num=4, lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], lab_lr=0.1, det=False, freeze_backbone=None):
        super(LCNet, self).__init__()
        self.scale = scale
        self.det = det
        self.lr_mult_list = lr_mult_list
        self.net_config = self.get_net_config(det)
        
        self.conv1 = ConvBNLayer(3, make_divisible(16 * scale), 3, 2, lr_mult=lr_mult_list[0])
        self.blocks2 = self.create_blocks("blocks2", scale, conv_kxk_num, lr_mult_list[1], lab_lr)
        self.blocks3 = self.create_blocks("blocks3", scale, conv_kxk_num, lr_mult_list[2], lab_lr)
        self.blocks4 = self.create_blocks("blocks4", scale, conv_kxk_num, lr_mult_list[3], lab_lr)
        self.blocks5 = self.create_blocks("blocks5", scale, conv_kxk_num, lr_mult_list[4], lab_lr)
        self.blocks6 = self.create_blocks("blocks6", scale, conv_kxk_num, lr_mult_list[5], lab_lr)
        self.out_channels = make_divisible(512 * scale)

    def get_net_config(self, det):
        return {
            "blocks2": [[3, 16, 32, 1, False]],
            "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
            "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
            "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False]],
            "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
        } if det else {
            "blocks2": [[3, 16, 32, 1, False]],
            "blocks3": [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
            "blocks4": [[3, 64, 128, (2, 1), False], [3, 128, 128, 1, False]],
            "blocks5": [[3, 128, 256, (1, 2), False], [5, 256, 256, 1, False]],
            "blocks6": [[5, 256, 512, (2, 1), True], [5, 512, 512, 1, True]]
        }

    def create_blocks(self, block_name, scale, conv_kxk_num, lr_mult, lab_lr):
        return nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=lr_mult,
                    lab_lr=lab_lr,
                )
                for k, in_c, out_c, s, se in self.net_config[block_name]
            ]
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        x = F.adaptive_avg_pool2d(x, [1, 150])

        return x

