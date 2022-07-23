import torch
import torch.nn as nn

def conv1x1(in_planes, out_planes, kernel_size=1, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.3),
        nn.ReLU(),
    )
    return model


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.3),
        nn.ReLU(),
    )
    return model


def atrous_conv(in_planes, out_planes, kernel_size=3, stride = 1, atrous_rate =1):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=atrous_rate, dilation=atrous_rate,
                  bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.3),
        nn.ReLU(),
    )
    return model

class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size

        self.f = nn.Conv2d(input_size, input_size // 8, 1, stride=1)
        self.f_bn = nn.BatchNorm2d(input_size // 8, momentum=0.3)
        self.g = nn.Conv2d(input_size, input_size // 8, 1, stride=1)
        self.g_bn = nn.BatchNorm2d(input_size // 8)

    def forward(self, x):
        width = x.size(2)
        height = x.size(3)
        channels = x.size(1)

        # apply (1x1) convolution layer and transpose
        f = self.f(x)
        f = self.f_bn(f)
        f = torch.transpose(f.view(-1, self.input_size // 8, width * height), 1, 2)

        # apply (1x1) convolution layer without transpose
        g = self.g(x)
        g = self.g_bn(g)
        g = g.view(-1, self.input_size // 8, width * height)

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