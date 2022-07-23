import torch
import torchvision.models

import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.functional import softmax, sigmoid

from timm import create_model

from segmentation_model.utils import *

class Wide_Resnet(nn.Module):
    def __init__(self, num_classes=6, img_size=None):
        super(Wide_Resnet, self).__init__()
        self.mobile = torchvision.models.wide_resnet50_2(pretrained=True)
        for param in self.mobile.parameters():
            param.requires_grad = True
        self.low_conv = conv3x3(256, 256)
        self.middle_conv = conv1x1(1024, 256)

        self.attention_middle = attention(1024, 256, 256)
        self.attention_high = attention(2048, 256, 256)

        self.upsampling1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.concate_conv = conv1x1(768, 256)
        self.upsampling2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv0 = conv3x3(512, 256)
        self.conv1 = conv3x3(256, 64)
        self.classification = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.mobile.conv1(x)
        x = self.mobile.bn1(x)
        x = self.mobile.relu(x)
        x = self.mobile.maxpool(x)
        x = self.mobile.layer1(x)
        low_level = x
        x = self.mobile.layer2(x)
        x = self.mobile.layer3(x)
        mid_level = x
        x = self.mobile.layer4(x)
        high_level = x

        low_level = self.low_conv(low_level)
        mid_attention = self.attention_middle(mid_level)  # mid and low is same channel
        mid_conv = self.middle_conv(mid_level)
        high_level = self.attention_high(high_level)  # 320->128 for pretty channel less than half
        high_level = self.upsampling1(high_level)
        outputs = torch.cat([mid_attention, mid_conv, high_level], dim=1)  # 256
        outputs = self.concate_conv(outputs)  # 192
        outputs = self.upsampling2(outputs)
        outputs = torch.cat([low_level, outputs], dim=1)  # 64 + 192
        outputs = self.conv0(outputs)
        outputs = self.conv1(outputs)
        outputs = self.classification(outputs)  # /4 resolution
        outputs = softmax(outputs, dim=1)
        return outputs