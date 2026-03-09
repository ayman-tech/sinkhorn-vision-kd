"""
Evaluation metrics: accuracy, parameter count, and FLOPs estimation.
"""

import torch
import torch.nn as nn
from typing import Tuple


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)):
    """Compute top-k accuracy for the given predictions and ground truth.

    Args:
        output: Model logits of shape (B, num_classes).
        target: Ground truth labels of shape (B,).
        topk: Tuple of k values to compute accuracy for.

    Returns:
        List of accuracy values (as percentages) for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Count all parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> int:
    """Estimate FLOPs for a forward pass using a simple hook-based counter.

    This counts multiply-accumulate operations (MACs) for Conv2d and Linear
    layers, then reports FLOPs = 2 * MACs (one multiply + one add per MAC).

    This is an approximation — it ignores BN, activation, and pooling costs
    (which are negligible compared to conv/linear).

    Args:
        model: The model to profile.
        input_size: Input tensor shape (default: single CIFAR image).

    Returns:
        Estimated FLOPs (int).
    """
    total_flops = [0]
    hooks = []

    def conv_hook(module, input, output):
        # FLOPs = 2 * out_h * out_w * out_c * (k_h * k_w * in_c / groups)
        batch, out_c, out_h, out_w = output.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total_flops[0] += 2 * out_h * out_w * out_c * kernel_ops

    def linear_hook(module, input, output):
        # FLOPs = 2 * in_features * out_features
        total_flops[0] += 2 * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy = torch.randn(input_size, device=device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return total_flops[0]


class AverageMeter:
    """Tracks running mean and current value of a metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
