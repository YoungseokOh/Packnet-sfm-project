# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/layers.py

from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F
import torch

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, bias=False)

    def forward(self, x):
        return self.conv(x)
        
class ConvBlock_HR(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock_HR, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # ori
        # self.conv = Conv3x3(in_channels, out_channels)
        # revise - zero-padding
        self.pad = nn.ZeroPad2d(1)
        # 1x1 channel fix layer
        # self.conv1x1 = conv1x1(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        # ReLU
        self.nonlin = nn.ReLU(inplace=True)
        # ELU
        # self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        # zero-padding
        out = self.pad(x)
        # 1x1 channel fix layer
        # out = self.conv1x1(x)
        out = self.conv(out)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        # reflectionpad2d
        # if use_refl:
        #     self.pad = nn.ReflectionPad2d(1)
        # else:
        # zero padding
        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def upsample_DNet(x, sf=2):
    """Upsample input tensor by a factor
    """
    return F.interpolate(x, scale_factor=sf, mode="nearest")

def upsample_DNet(x, sf=2):
    """Upsample input tensor by a factor
    """
    return F.interpolate(x, scale_factor=sf, mode="nearest")


class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Original fully connected layer
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False)
        # )

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1)
        )

        self.sigmoid = nn.Sigmoid()
        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        # self.conv1x1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)
        # New code
        y = self.avg_pool(features)
        y = self.fc(y)
        y = self.sigmoid(y)
        # Not support expand_as
        features = features * y

        # Original code
        # b, c, _, _ = features.size()
        # y = self.avg_pool(features).view(b, c)
        
        # y = self.fc(y).view(b, c, 1, 1)
        # y = self.sigmoid(y)
        # # Not support expand_as
        # features = features * y
        # Original code
        # features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.conv_se = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.input_channels = input_channels

    def forward(self, inputs):
        # x = F.avg_pool2d(inputs, kernel_size=inputs.size(1))
        x = self.squeeze(inputs)
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        # x = x.view(-1, self.input_channels, 1, 1)
        return self.conv_se(inputs * x)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # print(y.shape, x.shape)
        y = self.conv(y.reshape(b, 1, c)).reshape(b, c, 1, 1)
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y
        # Original
        # return x * y.expand_as(x)


# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    