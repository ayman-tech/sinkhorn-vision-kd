"""MobileNetV2 trimmed for 32x32 CIFAR inputs.

Diffs vs the ImageNet variant:
  - stem stride 1 (not 2): 32x32 is already small, can't afford another /2
  - the (6, 24) stage uses stride 1 and (6, 160) uses stride 1 — saves spatial
    resolution we'd otherwise lose to ImageNet-style aggressive downsampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.stride=stride
        self.use_res = (stride==1 and in_ch==out_ch)

        hid = in_ch*expand_ratio
        layers=[]
        if expand_ratio!=1:
            # expand 1x1
            layers += [nn.Conv2d(in_ch, hid, 1, bias=False),
                       nn.BatchNorm2d(hid), nn.ReLU6(inplace=True)]
        # depthwise 3x3
        layers += [nn.Conv2d(hid, hid, 3, stride=stride, padding=1, groups=hid, bias=False),
                   nn.BatchNorm2d(hid), nn.ReLU6(inplace=True)]
        # linear bottleneck — no activation here on purpose.
        layers += [nn.Conv2d(hid, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]
        self.conv=nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2CIFAR(nn.Module):
    # (expansion, out_ch, n_blocks, stride). Tweaked from imagenet config.
    CFG=[(1, 16,  1, 1),
         (6, 24,  2, 1),   # imagenet uses stride 2 here
         (6, 32,  3, 2),
         (6, 64,  4, 2),
         (6, 96,  3, 1),
         (6, 160, 3, 1),   # imagenet uses stride 2 here
         (6, 320, 1, 1)]

    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        in_ch  = max(int(32  *width_mult), 16)
        last_ch= max(int(1280*width_mult), 320)

        self.features=nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, in_ch, 3, stride=1, padding=1, bias=False),
                          nn.BatchNorm2d(in_ch), nn.ReLU6(inplace=True))])

        for t,c,n,s in self.CFG:
            out_ch=max(int(c*width_mult), 16)
            for i in range(n):
                stride = s if i==0 else 1
                self.features.append(InvertedResidual(in_ch, out_ch, stride, t))
                in_ch=out_ch

        self.features.append(nn.Sequential(
            nn.Conv2d(in_ch, last_ch, 1, bias=False),
            nn.BatchNorm2d(last_ch), nn.ReLU6(inplace=True)))
        self.classifier=nn.Linear(last_ch, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features: bool = False):
        feats=[]
        out=x
        for i, layer in enumerate(self.features):
            out=layer(out)
            # capture maps after each spatial-resolution boundary
            if return_features and i in {0, 3, 6, 10}:
                feats.append(out)
        out=F.adaptive_avg_pool2d(out, 1)
        out=out.view(out.size(0), -1)
        z=self.classifier(out)
        if return_features:
            return z, feats
        return z


def mobilenetv2(num_classes=10, width_mult=1.0):
    return MobileNetV2CIFAR(num_classes=num_classes, width_mult=width_mult)
