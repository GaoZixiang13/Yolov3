from net.backbone import DarkNet53, CBLconv

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Spp(nn.Module):
    def __init__(self, maxpool_sizes=(5, 9, 13)):
        super(Spp, self).__init__()
        self.maxpool_sizes = maxpool_sizes
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(pool_size, stride=1, padding=pool_size//2) for pool_size in self.maxpool_sizes]
        )

    def forward(self, x):
        x0, x1, x2 = [pool(x) for pool in self.pools]
        out = torch.cat([x, x0, x1, x2], dim=1)
        return out

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv0 = CBLconv(in_channels, out_channels, 1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv0(x)
        x = self.up(x)
        return x

class yolov3_head(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(yolov3_head, self).__init__()
        self.conv = CBLconv(in_channels, out_channels, 3)
        self.head = nn.Conv2d(out_channels, 3*(5+num_classes), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.head(self.conv(x))

class yolov3(nn.Module):
    def __init__(self, num_classes):
        super(yolov3, self).__init__()
        self.num_classes = num_classes
        self.backbone = DarkNet53()
        self.conv1 = nn.Sequential(
            CBLconv(1024, 512, 1),
            CBLconv(512, 1024, 3),
            CBLconv(1024, 512, 1),
        )
        self.spp = Spp()
        self.conv2 = nn.Sequential(
            CBLconv(2048, 512, 1),
            CBLconv(512, 1024, 3),
            CBLconv(1024, 512, 1),
        )
        self.conv2_ToOut2 = yolov3_head(512, 1024, self.num_classes)
        self.conv3_up = UpSample(512, 256)
        self.conv4_for5 = nn.Sequential(
            CBLconv(768, 256, 1),
            CBLconv(256, 512, 3),
            CBLconv(512, 256, 1),
            CBLconv(256, 512, 3),
            CBLconv(512, 256, 1),
        )
        self.conv5_forOut1 = yolov3_head(256, 256, self.num_classes)
        self.conv5_up = UpSample(256, 128)
        self.conv6_for5 = nn.Sequential(
            CBLconv(384, 128, 1),
            CBLconv(128, 256, 3),
            CBLconv(256, 128, 1),
            CBLconv(128, 256, 3),
            CBLconv(256, 128, 1),
        )
        self.conv7_forOut0 = yolov3_head(128, 256, self.num_classes)

    def forward(self, x):
        out0, out1, out2 = self.backbone(x)
        out2 = self.conv1(out2)
        out2 = self.spp(out2)
        out2 = self.conv2(out2)
        P2 = self.conv2_ToOut2(out2)
        out2_up = self.conv3_up(out2)
        out1 = torch.cat([out1, out2_up], dim=1)
        out1 = self.conv4_for5(out1)
        P1 = self.conv5_forOut1(out1)
        out1_up = self.conv5_up(out1)
        out0 = torch.cat([out0, out1_up], dim=1)
        out0 = self.conv6_for5(out0)
        P0 = self.conv7_forOut0(out0)

        return P2, P1, P0  # --> 32, 16, 8

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

