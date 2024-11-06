import torch
import torch.nn as nn
from src.lcnet_backbone import LCNet
from src.multi_head import MultiHead, CTCHead
from src.clip_backbone import CLIPEncoder
from utils.postprocess import CTCLabelDecode
from src.resnet_backbone import ResNet, Bottleneck
from src.hrnet_backbone import hrnet32
class vrdOCR(nn.Module):
    def __init__(self, backbone_config, head_config):
        super(vrdOCR, self).__init__()
        # self.backbone = LCNet(**backbone_config).cuda()
        # self.backbone = hrnet32(pretrained=False, progress=False)
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_channels=3)
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
        # Pass features and labels through the head
        if labels is not None:
            outs = self.head(feats, labels)
        else:
            outs = self.head(feats)
        return outs 