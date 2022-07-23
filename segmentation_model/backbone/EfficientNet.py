from torch.nn.functional import softmax, sigmoid
from timm import create_model
from segmentation_model.utils import *

class EfficientNet(nn.Module):
    # Feature Extractor만 EfficientNet으로 바꾼 버전입니다. 지금은 라이브러리를 사용하고 있는데, 성능 잘 나오면 그때 고민하기
    def __init__(self, num_classes=4, img_size=None, pretrained=True):
        super(EfficientNet, self).__init__()
        
        self.backbone = create_model('tf_efficientnet_b4_ns', pretrained=pretrained)  # 2, 5, 7
        # self.backbone = create_model('tf_efficientnet_b5_ns', pretrained=pretrained)  # 2, 5, 7

        self.low_conv = conv3x3(self.backbone.feature_info[1]['num_chs'], 80)
        self.attention_middle = attention(self.backbone.feature_info[3]['num_chs'], 240, 80)
        self.middle_conv = conv1x1(self.backbone.feature_info[3]['num_chs'], 80)
        self.attention_high = attention(self.backbone.feature_info[4]['num_chs'], 840, 160)
        self.upsampling1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.concate_conv = conv1x1(320, 160)
        self.upsampling2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv0 = conv3x3(240, 120)
        self.conv1 = conv3x3(120, 60)
        self.classification = nn.Conv2d(60, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.backbone.conv_stem(x)
        for i in range(len(self.backbone.blocks)):
            x = self.backbone.blocks[i](x)
            if i == 1: low_level = x
            if i == 4: mid_level = x
            if i == 6: high_level = x

        low_level = self.low_conv(low_level)  # 176x176
        mid_attention = self.attention_middle(mid_level)
        mid_conv = self.middle_conv(mid_level)  # 44
        high_level = self.attention_high(high_level)
        high_level = self.upsampling1(high_level)  # 22
        outputs = torch.cat([mid_attention, mid_conv, high_level], dim=1)
        outputs = self.concate_conv(outputs)
        outputs = self.upsampling2(outputs)  # 44 -> 176
        outputs = torch.cat([low_level, outputs], dim=1)
        outputs = self.conv0(outputs)
        outputs = self.conv1(outputs)
        outputs = self.classification(outputs)
        outputs = softmax(outputs, dim=1)
        return outputs

