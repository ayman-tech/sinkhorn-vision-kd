"""
Baseline Knowledge Distillation using KL Divergence.

Standard KD loss from Hinton et al. (2015):

    L_KD = alpha * L_CE(z_S, y) + (1 - alpha) * T^2 * KL(p_T || p_S)

where:
    - z_S: student logits
    - y: ground truth labels
    - p_T = softmax(z_T / T): softened teacher distribution
    - p_S = softmax(z_S / T): softened student distribution
    - T: temperature (higher T -> softer distributions -> more knowledge transfer)
    - alpha: balance between task loss and distillation loss

The T^2 factor compensates for the gradient magnitude change when using
temperature scaling, ensuring consistent learning dynamics across different T.

Reference: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class KLDistillationLoss(nn.Module):
    """KL-divergence based knowledge distillation loss.

    This is the standard baseline that our Sinkhorn OT-KD methods aim to beat.

    Args:
        temperature: Softmax temperature for knowledge distillation.
            Higher values produce softer probability distributions, revealing
            more of the teacher's "dark knowledge" about inter-class similarities.
            Typical values: 3-20. Default: 4.0.
        alpha: Weight for the KD loss vs cross-entropy loss.
            L_total = alpha * L_KD + (1 - alpha) * L_CE
            alpha=1.0 means pure distillation; alpha=0.0 means pure CE.
            Default: 0.9 (mostly distillation, some task-specific signal).
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.9):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the KL-KD loss.

        Args:
            student_logits: Raw logits from student model, shape (B, C).
            teacher_logits: Raw logits from teacher model, shape (B, C).
            labels: Ground truth class indices, shape (B,).

        Returns:
            Dict with keys:
                "loss": Total combined loss (scalar).
                "kd_loss": KL distillation component (scalar).
                "ce_loss": Cross-entropy component (scalar).
        """
        T = self.temperature

        # Soft probability distributions at temperature T
        # KL(p_T || p_S) = sum p_T * log(p_T / p_S)
        # PyTorch's kl_div expects log-probabilities as input, targets as probabilities
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)

        # KL divergence: sum over classes, mean over batch
        # Multiply by T^2 to compensate for gradient scaling
        kd_loss = F.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean"
        ) * (T * T)

        # Standard cross-entropy on hard labels
        ce_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return {
            "loss": total_loss,
            "kd_loss": kd_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }
