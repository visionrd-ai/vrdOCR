import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 1  # Keep output channels limited
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(stride, 1), padding=1)  # Only downsample height
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        
        x += identity
        x = self.relu(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=None, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial Convolution without downsampling
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=(1, 1), padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Remove MaxPool and continue with layers
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        

    def forward(self, x):

        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 150))
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        i_downsample = None
        layers = []

        # Adjust downsampling to happen only along the height dimension
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=(stride, 1)),  # Downsample height
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )
        
        layers.append(ResBlock(self.in_channels, planes, i_downsample=i_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion
        
        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))
        
        return nn.Sequential(*layers)

# Reduce block counts and restrict output channels to 512
def ResNetLite(num_classes=None, channels=3):
    return ResNet(Bottleneck, [2, 2, 2, 2], num_classes, channels)

# # Testing the network
# model = ResNetLite()
# output = model(torch.randn((1, 3, 256, 256)))
# print(output.shape)
