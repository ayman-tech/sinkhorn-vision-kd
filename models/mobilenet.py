"""
MobileNetV2 adapted for CIFAR-10/100 (32x32 images).

The standard MobileNetV2 is designed for ImageNet (224x224). For CIFAR we make
the following adjustments:
  - Initial conv uses stride 1 instead of stride 2 (images are already small)
  - Remove the first stride-2 bottleneck (would shrink 32x32 too aggressively)
  - Reduce the width multiplier to keep the model lightweight

This gives a compact student model (~0.8M params with width_mult=0.5) that is
architecturally different from ResNet, testing whether OT-KD transfers across
architectures.

Reference: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
           (Sandler et al., CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block.

    Architecture:
        x -> Conv1x1 (expand) -> BN -> ReLU6
          -> DWConv3x3 (stride) -> BN -> ReLU6
          -> Conv1x1 (project) -> BN -> (+x if residual)

    The "inverted" part: expansion happens first (thin -> wide -> thin),
    opposite to standard residual blocks (wide -> thin -> wide).
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, expand_ratio: int
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio
        layers = []

        # Expansion phase (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, 3,
                stride=stride, padding=1, groups=hidden_dim, bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # Projection (linear bottleneck — no activation)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2CIFAR(nn.Module):
    """MobileNetV2 adapted for CIFAR (32x32 inputs).

    Args:
        num_classes: Number of output classes.
        width_mult: Width multiplier (default 1.0). Use 0.5 for a smaller student.
    """

    # (expansion, out_channels, num_blocks, stride)
    # Adjusted for CIFAR: fewer stride-2 layers than ImageNet version
    CIFAR_CONFIG = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),   # stride 1 instead of 2 for CIFAR
        (6, 32, 3, 2),   # 32x32 -> 16x16
        (6, 64, 4, 2),   # 16x16 -> 8x8
        (6, 96, 3, 1),
        (6, 160, 3, 1),  # stride 1 instead of 2 for CIFAR
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        input_channel = max(int(32 * width_mult), 16)
        last_channel = max(int(1280 * width_mult), 320)

        # Initial convolution — stride 1 for CIFAR (not stride 2 like ImageNet)
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True),
            )
        ])

        # Build inverted residual blocks
        for t, c, n, s in self.CIFAR_CONFIG:
            output_channel = max(int(c * width_mult), 16)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel

        # Final expansion layer
        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, last_channel, 1, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True),
            )
        )

        self.classifier = nn.Linear(last_channel, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
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

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]] | torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).
            return_features: If True, return intermediate features at stride
                boundaries (useful for feature-level distillation).

        Returns:
            logits or (logits, feature_list).
        """
        features = []
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            # Capture features after spatial downsampling stages
            if return_features and i in {0, 3, 6, 10}:
                features.append(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        if return_features:
            return logits, features
        return logits


def mobilenetv2(num_classes: int = 10, width_mult: float = 1.0) -> MobileNetV2CIFAR:
    """Create a MobileNetV2 for CIFAR.

    With width_mult=1.0: ~2.3M params
    With width_mult=0.5: ~0.7M params
    """
    return MobileNetV2CIFAR(num_classes=num_classes, width_mult=width_mult)
