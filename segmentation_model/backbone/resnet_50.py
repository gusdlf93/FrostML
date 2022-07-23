import torch, torchvision
import torch.nn as nn
from torch.nn.functional import softmax
from segmentation_model.utils import *

class ResNet50(nn.Module):
    def __init__(self, pretrained=r'./STCGA_MoCo_ResNet50_k_0199.pth', num_classes=1000, img_size=None):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(num_classes=num_classes)

        self.freeze_initialize()
        try:
            self.load_checkpoint(pretrained)
        except:
            print("weight doesn't exists!")

        self.low_conv = conv3x3(256, 64)
        self.attention_middle = attention(1024, 128, 64)
        self.attention_high = attention(2048, 512, 128)
        self.middle_conv = conv1x1(1024, 64)
        self.upsampling1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True)
        self.concate_conv = conv1x1(256, 192)
        self.upsampling2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners = True)
        self.conv0 = conv3x3(256, 128)
        self.conv1 = conv3x3(128, 64)
        self.classification = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def freeze_initialize(self):
        for name, param in self.model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.model.fc.bias.data.zero_()

    def load_checkpoint(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=False)

    def forward(self, x):
        for module in self.model._modules:
            if module == 'avgpool':
                break
            else:
                x = self.model._modules[module](x)
            if module == 'layer1':
                low_level = x
            if module == 'layer3':
                mid_level = x
            if module == 'layer4':
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