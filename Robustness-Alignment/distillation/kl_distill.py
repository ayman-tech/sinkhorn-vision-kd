"""Vanilla Hinton KD baseline.

L = alpha * T^2 * KL(p_T || p_S)  +  (1-alpha) * CE(z_S, y)
the T^2 factor cancels the temperature-induced shrink in soft-target gradients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class KLDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.9):
        super().__init__()
        self.T=temperature
        self.alpha=alpha
        self.ce=nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels) -> Dict[str, torch.Tensor]:
        T=self.T
        # kl_div expects log-probs as input and probs as target.
        log_qS=F.log_softmax(student_logits/T, dim=1)
        qT=F.softmax(teacher_logits/T, dim=1)
        kd=F.kl_div(log_qS, qT, reduction="batchmean")*(T*T)
        ce=self.ce(student_logits, labels)
        total=self.alpha*kd + (1-self.alpha)*ce
        return {"loss":total, "kd_loss":kd.detach(), "ce_loss":ce.detach()}
