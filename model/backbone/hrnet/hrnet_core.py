import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.hrnet.ops import HighResolutionModule, blocks_dict

class HighResolutionNet(nn.Module):
    def __init__(self, cfg, norm_layer=None):
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        # stem
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = self.norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = self.norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # stage1
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # stage2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        # stage3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # stage4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = int(np.sum(pre_stage_channels))
        self.last_layer = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, 1, 1, 0),
            self.norm_layer(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, 480, 1, 1, 0)
        )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inch = num_channels_pre_layer[-1]
                    outc = num_channels_cur_layer[i] if j == i - num_branches_pre else inch
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inch, outc, 3, 2, 1, bias=False),
                        self.norm_layer(outc),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )
        layers = [block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules  = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks   = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block        = blocks_dict[layer_config['BLOCK']]
        fuse_method  = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            reset_mso = multi_scale_output or (i != num_modules - 1)
            modules.append(
                HighResolutionModule(num_branches, block, num_blocks, num_inchannels,
                                     num_channels, fuse_method, reset_mso, norm_layer=self.norm_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            x_list.append(self.transition1[i](x) if self.transition1[i] is not None else x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[i if i < self.stage2_cfg['NUM_BRANCHES'] else -1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[i if i < self.stage3_cfg['NUM_BRANCHES'] else -1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # fuse + project to 480 channels and pool to sequence length
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x = [x[0]] + [F.interpolate(t, size=(x0_h, x0_w), mode='bilinear', align_corners=True) for t in x[1:]]
        x = nn.functional.concat(x, dim=1) if hasattr(nn.functional, 'concat') else nn.functional.pad  # safety
        # fallback for older PyTorch:
        import torch
        x = torch.cat([x[0]] + x[1:], dim=1)

        x = self.last_layer(x)
        # keep OCR-friendly output: B x 480 x 1 x 150 (sequence length 150)
        x = F.adaptive_avg_pool2d(x, (1, 150))
        return x
