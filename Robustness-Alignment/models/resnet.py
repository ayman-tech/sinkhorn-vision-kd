"""CIFAR-style ResNet-{20,56,110}. Depth = 6n+2.

Standard He et al. CIFAR variant: 3x3 stem with stride 1 (no maxpool),
three groups at 32x32 / 16x16 / 8x8 with widths 16/32/64.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.shortcut=nn.Sequential()
        if stride!=1 or in_planes!=planes:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out=out+self.shortcut(x)
        return F.relu(out)


class CIFARResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes=16
        self.conv1=nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.layer1=self._make_layer(16, num_blocks, stride=1)
        self.layer2=self._make_layer(32, num_blocks, stride=2)
        self.layer3=self._make_layer(64, num_blocks, stride=2)
        self.linear=nn.Linear(64, num_classes)
        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides=[stride] + [1]*(num_blocks-1)
        blocks=[]
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes=planes
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features: bool = False):
        out=F.relu(self.bn1(self.conv1(x)))
        f1=self.layer1(out)
        f2=self.layer2(f1)
        f3=self.layer3(f2)
        out=F.avg_pool2d(f3, f3.size(2))
        out=out.view(out.size(0), -1)
        z=self.linear(out)
        if return_features:
            return z, [f1, f2, f3]
        return z


def resnet20(num_classes=10):  return CIFARResNet(num_blocks=3,  num_classes=num_classes)
def resnet56(num_classes=10):  return CIFARResNet(num_blocks=9,  num_classes=num_classes)
def resnet110(num_classes=10): return CIFARResNet(num_blocks=18, num_classes=num_classes)
