import torch, torchvision
import torch.nn as nn
from torch.nn.functional import softmax

def conv1x1(in_planes, out_planes):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_planes,momentum=0.3),
        nn.ReLU(),
    )
    return model


def conv3x3(in_planes, out_planes):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes,momentum=0.3),
        nn.ReLU(),
    )
    return model


def atrous_conv(in_planes, out_planes, atrous_rate):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=atrous_rate, dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_planes,momentum=0.3),
        nn.ReLU(),
    )
    return model

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

# Making Attetion mask
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


# Applying Attention mask
class SelfAttentionPost(nn.Module):
    def __init__(self, input_size):
        super(SelfAttentionPost, self).__init__()
        self.input_size = input_size

        self.gamma = nn.Parameter(torch.zeros(1))
        self.h = nn.Conv2d(input_size, input_size//8, 1, stride=1)
        self.h_bn = nn.BatchNorm2d(input_size//8, momentum=0.3)
        self.i = nn.Conv2d(input_size//8, input_size, 1, stride=1)
        self.i_bn = nn.BatchNorm2d(input_size, momentum=0.3)

    def forward(self, x, att):
        width = x.size(2)
        height = x.size(3)

        # gamma is weight parameter of whole attention mask
        # gamma is zero because we doesn't want to use attention mask at first
        # we want to use attnetion mask continuously increasing
        h = self.gamma * self.h_bn(self.h(x))
        h = h.permute(0, 2, 3, 1).contiguous().view(-1, width * height, self.input_size//8)
        h = torch.bmm(att, h)
        h = h.view(-1, width, height, self.input_size//8).permute(0, 3, 1, 2)
        # add original data like residual path
        x = self.i_bn(self.i(h)) + x

        return x

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, freeze=True, img_size=None):
        super(MobileNetV2, self).__init__()
        torchvision.models.mobilenet_v2()
        self.model = torch.hub.load('pytorch/vision:v0.8.0', 'mobilenet_v2', pretrained=True)
        print("load models!!")

        self.low_conv = conv3x3(24, 64)
        self.attention_middle = attention(96, 128, 64)
        self.attention_high = attention(1280, 512, 128)
        self.middle_conv = conv1x1(96, 64)
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
        state = torch.load(checkpoint, map_location='cpu')
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        model_dict = self.model.state_dict()
        weights = {k: v for k, v in state_dict.items() if k in model_dict}

        model_dict.update(weights)
        self.model.load_state_dict(model_dict)

    def forward(self, x):
        for _, module in enumerate(self.model.features):
            if module == 'avgpool':
                break
            else:
                x = module(x)
            if _ == 3:
                low_level = x
            if _ == 13:
                mid_level = x
        high_level = x

        low_level = self.low_conv(low_level)
        mid_attention = self.attention_middle(mid_level) #mid and low is same channel
        mid_conv = self.middle_conv(mid_level)
        high_level = self.attention_high(high_level)#320->128 for pretty channel less than half
        high_level = self.upsampling1(high_level)
        outputs = torch.cat([mid_attention, mid_conv, high_level], dim=1)#256
        outputs = self.concate_conv(outputs)#192
        outputs = self.upsampling2(outputs)
        outputs = torch.cat([low_level, outputs], dim=1)#64 + 192
        outputs = self.conv0(outputs)
        outputs = self.conv1(outputs)
        outputs = self.classification(outputs)#/4 resolution
        outputs = softmax(outputs, dim=1)
        return outputs