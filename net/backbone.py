import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CBLconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CBLconv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.actv = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.actv(x)
        return x

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit, self).__init__()
        self.res = nn.Sequential(
            CBLconv(in_channels, out_channels, 1),
            CBLconv(out_channels, in_channels, 3),
        )

    def forward(self, x):
        x0 = self.res(x)
        return x + x0

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nums):
        super(ResBlock, self).__init__()
        self.conv0 = CBLconv(in_channels, out_channels, 3, 2)
        self.res = nn.Sequential(
            *[ResUnit(out_channels, in_channels) for _ in range(nums)],
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.res(x)
        return x

class DarkNet53(nn.Module):
    def __init__(self, nums=(1, 2, 8, 8, 4)):
        super(DarkNet53, self).__init__()
        self.channels = (64, 128, 256, 512, 1024)
        self.conv0 = CBLconv(3, 32, 3, 1)
        self.stages = nn.ModuleList([
            ResBlock(32, self.channels[0], nums[0]),
            ResBlock(self.channels[0], self.channels[1], nums[1]),
            ResBlock(self.channels[1], self.channels[2], nums[2]),
            ResBlock(self.channels[2], self.channels[3], nums[3]),
            ResBlock(self.channels[3], self.channels[4], nums[4]),
        ])

    def forward(self, x):
        x = self.conv0(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out0 = self.stages[2](x)
        out1 = self.stages[3](out0)
        out2 = self.stages[4](out1)

        return out0, out1, out2
