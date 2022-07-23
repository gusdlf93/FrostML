import torch
import torch.nn as nn
from torch.nn.functional import softmax
from collections import OrderedDict

#made by jslee

def conv1x1(in_planes, out_planes):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_planes,momentum=0.3),
        nn.ReLU(inplace=True),
    )
    return model


def conv3x3(in_planes, out_planes):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes,momentum=0.3),
        nn.ReLU(inplace=True),
    )
    return model
class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size

        self.f = nn.Conv2d(input_size, input_size//8, 1, stride=1)
        self.f_bn = nn.BatchNorm2d(input_size//8, momentum=0.3)
        self.g = nn.Conv2d(input_size, input_size//8, 1, stride=1)
        self.g_bn = nn.BatchNorm2d(input_size//8)

    def forward(self, x):
        width = x.size(2)
        height = x.size(3)
        channels = x.size(1)

        # apply (1x1) convolution layer and transpose
        f = self.f(x)
        f = self.f_bn(f)
        f = torch.transpose(f.view(-1, self.input_size//8, width * height), 1, 2)

        # apply (1x1) convolution layer without transpose
        g = self.g(x)
        g = self.g_bn(g)
        g = g.view(-1, self.input_size//8, width * height)

        # make attention mask by matrix multiplication
        att = torch.bmm(f, g)

        return torch.nn.functional.softmax(att, 1)

class SelfAttentionPost(nn.Module):
    def __init__(self, input_size):
        super(SelfAttentionPost, self).__init__()
        self.input_size = input_size

        self.gamma = nn.Parameter(torch.zeros(1))
        self.h = nn.Conv2d(input_size, input_size // 8, 1, stride=1)
        self.h_bn = nn.BatchNorm2d(input_size // 8, momentum=0.3)
        self.i = nn.Conv2d(input_size // 8, input_size, 1, stride=1)
        self.i_bn = nn.BatchNorm2d(input_size, momentum=0.3)

    def forward(self, x, att):
        width = x.size(2)
        height = x.size(3)

        # gamma is weight parameter of whole attention mask
        # gamma is zero because we doesn't want to use attention mask at first
        # we want to use attnetion mask continuously increasing
        h = self.gamma * self.h_bn(self.h(x))
        h = h.permute(0, 2, 3, 1).contiguous().view(-1, width * height, self.input_size // 8)
        h = torch.bmm(att, h)
        h = h.view(-1, width, height, self.input_size // 8).permute(0, 3, 1, 2)
        # add original data like residual path
        x = self.i_bn(self.i(h)) + x

        return x
class attention(nn.Module):
    def __init__(self, input_channel, middle_channel, output_channel):
        super(attention, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, middle_channel, kernel_size=3, stride=1, padding=1, bias=False)

        # Initialize attention block and post attention block
        # Number is input channel size
        self.att = SelfAttention(middle_channel)
        self.att_post = SelfAttentionPost(middle_channel)

        self.conv2 = nn.Conv2d(middle_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        output = self.conv1(x)

        # make attention mask first
        att = self.att(output)
        # apply attention mask next
        output = self.att_post(output, att)

        output = self.conv2(output)

        return output
class ReResSegNet(nn.Module):
    def __init__(self, num_classes=None, img_size=None):
        super(ReResSegNet, self).__init__()
        self.encoder = ReResNet18(n_class = num_classes).encoder

        try:
            state_dict = torch.load('./ReConfig4_010_198.pth')

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')[8:]  # remove `module.`
                new_state_dict[name] = v

            self.encoder.load_state_dict(new_state_dict, strict=False)
        except:
            print(f"!!!!!!!!!!model_weight doesn't exists!!!!!!!!!!!! \n"
                  "you need to download model weights from https://drive.google.com/file/d/1xCs1c-N227xba0bnSlQTvyyis9uMX7Cb/view?usp=sharing"
                  "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.low_conv = conv3x3(128, 80)
        self.attention_middle = attention(512, 240, 80)
        self.middle_conv = conv1x1(512, 80)
        self.attention_high = attention(1024, 840, 160)
        self.upsampling1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.concate_conv = conv1x1(320, 160)
        self.upsampling2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv0 = conv3x3(240, 120)
        self.conv1 = conv3x3(120, 60)
        self.classification = nn.Conv2d(60, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.encoder.stem(x)
        x = self.encoder.layer1(x)
        low_level = x
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        mid_level = x
        x = self.encoder.layer4(x)
        high_level = x

        low_level = self.low_conv(low_level)#176x176
        mid_attention = self.attention_middle(mid_level)
        mid_conv = self.middle_conv(mid_level)#44
        high_level = self.attention_high(high_level)
        high_level = self.upsampling1(high_level)#22
        outputs = torch.cat([mid_attention, mid_conv, high_level], dim=1)
        outputs = self.concate_conv(outputs)
        outputs = self.upsampling2(outputs)#44 -> 176
        outputs = torch.cat([low_level, outputs], dim=1)
        outputs = self.conv0(outputs)
        outputs = self.conv1(outputs)
        outputs = self.classification(outputs)
        outputs = softmax(outputs, dim=1)
        return outputs

class ReResNet18(nn.Module):
    def __init__(self, n_class=1000):
        super().__init__()
        self.encoder = Encoder()
        self.fc = nn.Linear(1024, n_class)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.fc(feature)
        return out



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d( 3, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64))#704->352

        self.layer1 = nn.Sequential(ConvResBlock( 64,  128, 2), ConvResBlock(128, 128, 1))#352->176
        self.layer2 = nn.Sequential(ConvResBlock(128,  256, 2), ConvResBlock(256, 256, 1))#176->88
        self.layer3 = nn.Sequential(ConvResBlock(256,  512, 2), ConvResBlock(512, 512, 1))#88->44
        self.layer4 = nn.Sequential(ConvResBlock(512, 1024, 2), ConvTransBlock(1024))#44->22

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x



class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)
        return out



class ConvTransBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        identity = x
        out = self.dwconv(x)
        out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.gelu(out)
        out = self.pwconv2(out)
        out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        out += identity
        return out
