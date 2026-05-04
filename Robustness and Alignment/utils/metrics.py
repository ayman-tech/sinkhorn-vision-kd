"""accuracy / param count / cheap FLOPs estimate / running-average meter."""
import torch
import torch.nn as nn
from typing import Tuple


def accuracy(output, target, topk: Tuple[int, ...] = (1,)):
    """Top-k accuracy in percent. Returns a list (one entry per k)."""
    with torch.no_grad():
        maxk=max(topk)
        bs=target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred=pred.t()
        correct=pred.eq(target.view(1, -1).expand_as(pred))
        out=[]
        for k in topk:
            ck=correct[:k].reshape(-1).float().sum(0, keepdim=True)
            out.append(ck.mul_(100.0/bs).item())
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model, input_size: Tuple[int, ...] = (1, 3, 32, 32)):
    """Hook-based FLOP estimate: 2*MACs over Conv2d + Linear.

    Ignores BN / activations / pooling — those are <1% of total for these nets.
    """
    total=[0]
    hooks=[]

    def conv_hook(mod, inp, out):
        _, oc, oh, ow = out.shape
        kops = mod.kernel_size[0]*mod.kernel_size[1]*(mod.in_channels // mod.groups)
        total[0] += 2*oh*ow*oc*kops

    def lin_hook(mod, inp, out):
        total[0] += 2*mod.in_features*mod.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))

    dev=next(model.parameters()).device
    dummy=torch.randn(input_size, device=dev)
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return total[0]


class AverageMeter:
    def __init__(self, name=""):
        self.name=name
        self.reset()

    def reset(self):
        self.val=0.0; self.avg=0.0; self.sum=0.0; self.count=0

    def update(self, val, n=1):
        self.val=val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
