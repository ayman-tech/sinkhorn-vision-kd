"""
Adaptive Sinkhorn OT Knowledge Distillation with LEARNABLE Cost Matrix.

============================================================================
THIS IS THE MAIN NOVEL CONTRIBUTION OF THE PROJECT.
============================================================================

Key idea: Instead of using a fixed cost matrix C, we LEARN C jointly with the
student network via bilevel optimization:

    Outer problem: min_C  L_val(theta*(C), C)     [update C on validation data]
    Inner problem: min_theta  L_train(theta, C)    [update student on training data]

The learned cost matrix C captures the semantic geometry of the label space:
C[i][j] encodes "how costly" it is to confuse class i with class j. After
training on CIFAR-100, we expect C to reveal meaningful structure:
    - Low cost between semantically similar classes (e.g., "cat" <-> "tiger")
    - High cost between dissimilar classes (e.g., "cat" <-> "bridge")
    - Block-diagonal structure reflecting CIFAR-100 superclasses

Parameterization of C (ensuring valid cost matrix):
    1. Raw parameter A: unconstrained nn.Parameter of shape (K, K)
    2. Symmetry: S = (A + A^T) / 2
    3. Non-negativity: C' = softplus(S)
    4. Zero diagonal: C = C' - diag(diag(C'))
    5. Normalization: C = C / max(C)  [keeps values in [0, 1]]

This parameterization guarantees:
    - C >= 0           (softplus output is always positive)
    - C = C^T          (symmetric by construction)
    - diag(C) = 0      (explicitly zeroed)
    - C in [0, 1]      (normalized)

For bilevel optimization, we use simple alternating updates (not MAML):
    - Every K steps: freeze student theta, take gradient step on C using val batch
    - Otherwise: freeze C, take gradient step on theta using train batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .sinkhorn_distill import log_sinkhorn


class LearnableCostMatrix(nn.Module):
    """Learnable cost matrix for Sinkhorn OT knowledge distillation.

    The cost matrix C[i][j] represents the penalty for transporting probability
    mass from class i to class j. It is parameterized to always satisfy:
        - Symmetry: C = C^T
        - Non-negativity: C >= 0
        - Zero diagonal: C[i][i] = 0  (no cost for correct classification)
        - Bounded: C in [0, 1]

    The raw parameter A is unconstrained; the forward pass applies the
    necessary transformations to produce a valid C.

    Args:
        num_classes: Number of classes (determines C shape: K x K).
        init_scale: Initial scale for the raw parameter. Smaller values
            start C closer to uniform; larger values start with more variance.
    """

    def __init__(self, num_classes: int, init_scale: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        # Raw unconstrained parameter
        # Initialize such that after softplus + normalization, C starts ~uniform
        # softplus(init_scale) ≈ init_scale for init_scale > 1, ≈ ln(1+e^x) for small x
        A = torch.randn(num_classes, num_classes) * 0.1 + init_scale
        self.A = nn.Parameter(A)

    def forward(self) -> torch.Tensor:
        """Produce a valid cost matrix from the raw parameter.

        Returns:
            C: Cost matrix of shape (K, K), symmetric, non-negative,
               zero diagonal, values in [0, 1].
        """
        # Step 1: Symmetrize
        S = (self.A + self.A.T) / 2

        # Step 2: Non-negativity via softplus (smooth approximation to ReLU)
        C = F.softplus(S)

        # Step 3: Zero diagonal (self-transport should be free)
        mask = 1.0 - torch.eye(self.num_classes, device=C.device)
        C = C * mask

        # Step 4: Normalize to [0, 1]
        C = C / (C.max() + 1e-8)

        return C

    def get_cost_matrix(self) -> torch.Tensor:
        """Get the current cost matrix as a detached numpy-ready tensor."""
        with torch.no_grad():
            return self.forward().cpu()


class AdaptiveSinkhornKD(nn.Module):
    """Sinkhorn OT-KD with jointly learned cost matrix.

    This module manages:
        1. The learnable cost matrix C
        2. Computing the Sinkhorn distance W_eps(p_T, p_S; C)
        3. The bilevel optimization schedule (when to update C vs theta)

    Loss function:
        L_total = L_CE(z_S, y) + lambda_ot * W_eps(p_T, p_S; C)

    Bilevel optimization (simple alternating):
        - Every `cost_update_freq` training steps:
            * Freeze student params theta
            * Compute L_total on a VALIDATION batch
            * Take one gradient step on C (cost_lr, with gradient clipping)
        - All other steps:
            * Freeze C
            * Compute L_total on a TRAINING batch
            * Update theta via the main optimizer

    Args:
        num_classes: Number of output classes.
        temperature: Softmax temperature.
        lambda_ot: Weight of the OT loss.
        epsilon: Sinkhorn entropic regularization.
        max_iter: Max Sinkhorn iterations.
        threshold: Sinkhorn convergence threshold.
        cost_lr: Learning rate for cost matrix updates.
        cost_update_freq: Update C every this many training steps.
        cost_grad_clip: Max gradient norm for C updates.
        init_scale: Initialization scale for the cost matrix.
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 4.0,
        lambda_ot: float = 0.5,
        epsilon: float = 0.05,
        max_iter: int = 50,
        threshold: float = 1e-3,
        cost_lr: float = 0.01,
        cost_update_freq: int = 10,
        cost_grad_clip: float = 1.0,
        init_scale: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_ot = lambda_ot
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.threshold = threshold
        self.cost_update_freq = cost_update_freq
        self.cost_grad_clip = cost_grad_clip

        self.ce_loss = nn.CrossEntropyLoss()
        self.cost_matrix = LearnableCostMatrix(num_classes, init_scale)

        # Separate optimizer for the cost matrix (outer loop of bilevel opt)
        self.cost_optimizer = torch.optim.Adam(
            self.cost_matrix.parameters(), lr=cost_lr
        )

        self._step_count = 0

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the adaptive OT-KD loss.

        Args:
            student_logits: Shape (B, K).
            teacher_logits: Shape (B, K).
            labels: Shape (B,).

        Returns:
            Dict with "loss", "ot_loss", "ce_loss", "transport_plan", "cost_matrix".
        """
        T = self.temperature

        # Get current cost matrix (with all constraints applied)
        C = self.cost_matrix()

        # Softened distributions
        p_T = F.softmax(teacher_logits / T, dim=1).clamp(min=1e-8)
        p_S = F.softmax(student_logits / T, dim=1).clamp(min=1e-8)

        log_p_T = p_T.log()
        log_p_S = p_S.log()

        # Sinkhorn distance with current C
        ot_loss, transport_plan = log_sinkhorn(
            log_p_T, log_p_S, C,
            epsilon=self.epsilon, max_iter=self.max_iter, threshold=self.threshold,
        )

        # Standard CE loss
        ce_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        total_loss = ce_loss + self.lambda_ot * ot_loss

        return {
            "loss": total_loss,
            "ot_loss": ot_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "transport_plan": transport_plan.detach(),
            "cost_matrix": C.detach(),
        }

    def should_update_cost(self) -> bool:
        """Check if this step should update the cost matrix C (outer loop).

        Returns True every `cost_update_freq` steps.
        """
        return self._step_count % self.cost_update_freq == 0

    def step_cost_matrix(
        self,
        student: nn.Module,
        teacher: nn.Module,
        val_images: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one bilevel outer-loop step: update C on validation data.

        This is called every `cost_update_freq` steps. It:
            1. Freezes the student parameters
            2. Computes the loss on a validation batch
            3. Backpropagates through the Sinkhorn solver to get dL/dC
            4. Updates C via the cost optimizer (with gradient clipping)

        The key insight: since the Sinkhorn iterations are differentiable,
        gradients flow from the loss through the transport plan back to C.
        This tells C which class confusions to penalize more.

        Args:
            student: Student model (will be set to eval mode temporarily).
            teacher: Teacher model (always in eval mode).
            val_images: Validation batch images, shape (B, 3, 32, 32).
            val_labels: Validation batch labels, shape (B,).

        Returns:
            Dict with cost update metrics: "cost_loss", "cost_grad_norm".
        """
        # Teacher always in eval mode
        teacher.eval()

        # Get teacher and student logits (no gradient on teacher)
        with torch.no_grad():
            teacher_logits = teacher(val_images)

        student_logits = student(val_images)

        # Compute loss (gradients flow to C through Sinkhorn)
        result = self.forward(student_logits, teacher_logits, val_labels)
        loss = result["loss"]

        # Update cost matrix
        self.cost_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        cost_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.cost_matrix.parameters(), self.cost_grad_clip
        )

        self.cost_optimizer.step()

        return {
            "cost_loss": loss.item(),
            "cost_grad_norm": cost_grad_norm.item() if isinstance(cost_grad_norm, torch.Tensor) else cost_grad_norm,
        }

    def increment_step(self):
        """Increment the internal step counter (call after each training step)."""
        self._step_count += 1

    def get_cost_matrix_numpy(self):
        """Get current cost matrix as a numpy array for visualization."""
        return self.cost_matrix.get_cost_matrix().numpy()
