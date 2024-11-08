import torch
import torch.nn as nn
from src.lcnet_backbone import LCNet
from src.multi_head import MultiHead, CTCHead
# from src.clip_backbone import CLIPEncoder
from utils.postprocess import CTCLabelDecode
from src.resnet_backbone import ResNet, Bottleneck
from src.hrnet_backbone import hrnet32
from src.dla34_backbone import dla34, dla34_fpn
from src.FPN import FPN

class vrdOCR(nn.Module):
    def __init__(self, backbone_config, head_config):
        super(vrdOCR, self).__init__()
        self.device = 'cuda'
        self.backbone = LCNet(**backbone_config).cuda()
        # self.backbone = hrnet32(pretrained=False, progress=False)
        # self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_channels=3)
        # self.backbone = dla34(pretrained=None, return_levels=True)
        # self.backbone = dla34_fpn()
        # self.backbone = self.backbone['forward']
        # self.fpn = FPN(in_channels=[128, 256, 512],out_channels=512,num_outs=3,attention=False).to(self.device)
        

        if backbone_config['freeze_backbone']:
            print("Freezing the backbone")
            for param in self.backbone.parameters():
                param.requires_grad = False 

        self.head = MultiHead(**head_config).cuda()
        # self.head = CTCHead(**head_config)
        # self.decoder = CTCLabelDecode(character_dict_path='utils/en_dict.txt', use_space_char=True)

    def forward(self, images, labels=None):
        # Pass images through the backbone to extract features

        feats = self.backbone(images)
        #[(4, 16, 256, 640), (4, 32, 128, 320), (4, 64, 64, 160), (4, 128, 32, 80), (4, 256, 16, 40), (4, 512, 8, 20)]
        
        # Pass features and labels through the head
        if labels is not None:
            outs = self.head(feats, labels)
        else:
            outs = self.head(feats)
        return outs 