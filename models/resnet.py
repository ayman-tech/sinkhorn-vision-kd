"""
CIFAR-specific ResNet architectures (ResNet-20, ResNet-56, ResNet-110).

These follow the original He et al. (2015) design for CIFAR:
  - Initial 3x3 conv (no max pool, unlike ImageNet variants)
  - Three groups of residual blocks at spatial resolutions 32x32, 16x16, 8x8
  - Global average pooling -> linear classifier

Each model returns both logits and intermediate feature maps (after each group)
so they can be used for feature-based distillation if needed.

Reference: "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class BasicBlock(nn.Module):
    """Standard residual block with two 3x3 convolutions and a skip connection.

    Architecture:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU

    When the spatial dimensions or channel count change (stride > 1 or
    in_planes != planes), a 1x1 conv shortcut is used to match dimensions.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    """ResNet for CIFAR-10/100 (32x32 input images).

    Architecture overview:
        Conv3x3(3, 16) -> BN -> ReLU
        -> Group1: n blocks at 16 channels, stride 1  (32x32)
        -> Group2: n blocks at 32 channels, stride 2  (16x16)
        -> Group3: n blocks at 64 channels, stride 2  (8x8)
        -> AvgPool(8x8) -> Linear(64, num_classes)

    The total depth is 6n + 2:
        ResNet-20:  n=3   (6*3  + 2 = 20)
        ResNet-56:  n=9   (6*9  + 2 = 56)
        ResNet-110: n=18  (6*18 + 2 = 110)

    Args:
        num_blocks: Number of BasicBlocks per group.
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100).
    """

    def __init__(self, num_blocks: int, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, num_blocks, stride=1)
        self.layer2 = self._make_layer(32, num_blocks, stride=2)
        self.layer3 = self._make_layer(64, num_blocks, stride=2)

        self.linear = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming initialization for conv layers, constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]] | torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).
            return_features: If True, also return intermediate feature maps
                from each residual group (useful for feature-based distillation).

        Returns:
            If return_features is False: logits of shape (B, num_classes).
            If return_features is True: (logits, [feat1, feat2, feat3]) where
                feat1 has shape (B, 16, 32, 32),
                feat2 has shape (B, 32, 16, 16),
                feat3 has shape (B, 64, 8, 8).
        """
        out = F.relu(self.bn1(self.conv1(x)))

        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        out = F.avg_pool2d(f3, f3.size(2))
        out = out.view(out.size(0), -1)
        logits = self.linear(out)

        if return_features:
            return logits, [f1, f2, f3]
        return logits


def resnet20(num_classes: int = 10) -> CIFARResNet:
    """ResNet-20 for CIFAR (0.27M params)."""
    return CIFARResNet(num_blocks=3, num_classes=num_classes)


def resnet56(num_classes: int = 10) -> CIFARResNet:
    """ResNet-56 for CIFAR (0.85M params)."""
    return CIFARResNet(num_blocks=9, num_classes=num_classes)


def resnet110(num_classes: int = 10) -> CIFARResNet:
    """ResNet-110 for CIFAR (1.73M params)."""
    return CIFARResNet(num_blocks=18, num_classes=num_classes)
