"""
Sinkhorn Optimal Transport Knowledge Distillation with FIXED cost matrix.

Instead of KL divergence, we measure the distance between teacher and student
distributions using the entropy-regularized Wasserstein distance (Sinkhorn distance):

    W_eps(p_T, p_S) = min_{pi in Pi(p_T, p_S)} <C, pi> + eps * KL(pi || p_T x p_S)

where:
    - p_T = softmax(z_T / T): teacher distribution over classes
    - p_S = softmax(z_S / T): student distribution over classes
    - C: cost matrix (num_classes x num_classes), encoding class distances
    - pi: transport plan (joint distribution with marginals p_T, p_S)
    - eps: entropic regularization (smaller -> closer to exact OT, but less stable)

The Sinkhorn algorithm solves this via iterative Bregman projections.
We implement it in log-domain for numerical stability.

Why OT is better than KL for KD:
    KL divergence treats all mismatches equally — confusing "cat" with "dog"
    costs the same as confusing "cat" with "truck". OT with a meaningful cost
    matrix penalizes semantically wrong confusions more heavily, giving the
    student richer gradient signal about the structure of the label space.

Reference: Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of
           Optimal Transport"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def log_sinkhorn(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    C: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 50,
    threshold: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the Sinkhorn distance in log-domain for numerical stability.

    Solves the entropy-regularized OT problem:
        W_eps = min_{pi} <C, pi> + eps * KL(pi || a x b)
        subject to: pi @ 1 = a,  pi^T @ 1 = b

    The dual variables f, g are updated via the Sinkhorn iterations:
        f_i = -eps * log( sum_j exp((g_j - C_{ij}) / eps) * b_j )
        g_j = -eps * log( sum_i exp((f_i - C_{ij}) / eps) * a_i )

    In log-domain, these become numerically stable logsumexp operations.

    Args:
        log_a: Log of source distribution, shape (B, K) where K = num_classes.
        log_b: Log of target distribution, shape (B, K).
        C: Cost matrix, shape (K, K). Must be non-negative.
        epsilon: Entropic regularization. Smaller = sharper transport plan
            but harder to optimize. Typical range: 0.01-0.1.
        max_iter: Maximum Sinkhorn iterations.
        threshold: Convergence threshold on marginal violation.

    Returns:
        (sinkhorn_cost, transport_plan):
            sinkhorn_cost: Scalar, the Sinkhorn distance averaged over batch.
            transport_plan: Shape (B, K, K), the optimal coupling.
    """
    B, K = log_a.shape

    # Cost matrix: (K, K) -> (1, K, K) for broadcasting with batch
    M = C.unsqueeze(0) / epsilon  # (1, K, K), scaled cost

    # Initialize dual variables to zero
    f = torch.zeros(B, K, device=log_a.device, dtype=log_a.dtype)  # (B, K)
    g = torch.zeros(B, K, device=log_b.device, dtype=log_b.dtype)  # (B, K)

    for iteration in range(max_iter):
        f_prev = f.clone()

        # Update f: f_i = -eps * logsumexp_j( (g_j - C_{ij}) / eps + log_b_j )
        #         = -eps * logsumexp_j( g_j/eps - C_{ij}/eps + log_b_j )
        # In matrix form with broadcasting:
        #   inner = g.unsqueeze(1) / eps - M + log_b.unsqueeze(1)  # (B, 1, K) - (1, K, K) + (B, 1, K)
        # But we keep f,g unscaled and absorb eps into M:
        # f_i = - logsumexp_j( g_j - M_{ij} + log_b_j )  [all divided by eps already in M]

        # f update: for each i, sum over j
        log_kernel_f = g.unsqueeze(1) - M + log_b.unsqueeze(1)  # (B, K, K)
        f = -torch.logsumexp(log_kernel_f, dim=2)  # (B, K)

        # g update: for each j, sum over i
        log_kernel_g = f.unsqueeze(2) - M + log_a.unsqueeze(2)  # (B, K, K)
        g = -torch.logsumexp(log_kernel_g, dim=1)  # (B, K)

        # Check convergence: ||f - f_prev||_inf
        err = (f - f_prev).abs().max().item()
        if err < threshold:
            break

    # Compute transport plan: pi_{ij} = exp(f_i/eps + g_j/eps - C_{ij}/eps + log_a_i + log_b_j)
    # With our scaling: pi_{ij} = exp(f_i + g_j - M_{ij} + log_a_i + log_b_j)
    log_pi = (
        f.unsqueeze(2) + g.unsqueeze(1) - M + log_a.unsqueeze(2) + log_b.unsqueeze(1)
    )
    pi = log_pi.exp()  # (B, K, K)

    # Sinkhorn cost: <C, pi> = sum_{ij} C_{ij} * pi_{ij}, averaged over batch
    # Note: we use the original C (not M) for the actual cost
    cost = (C.unsqueeze(0) * pi).sum(dim=(1, 2)).mean()

    return cost, pi


def build_cost_matrix(
    num_classes: int,
    cost_type: str = "uniform",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build a fixed cost matrix for the Sinkhorn distance.

    Args:
        num_classes: Number of classes (K).
        cost_type: Type of cost matrix:
            - "uniform": C_{ij} = 1 - delta_{ij}. All misclassifications
              cost equally. Equivalent to 0-1 loss structure.
            - "label_distance": C_{ij} = |i - j| / (K - 1). Assumes classes
              have an ordinal relationship (useful when class indices encode
              some semantic ordering).
            - "random": Random symmetric matrix with zero diagonal, values in [0,1].
              Used as a baseline to show that LEARNED C > random C.
        device: Torch device.

    Returns:
        Cost matrix C of shape (K, K), symmetric, non-negative, zero diagonal.
    """
    if cost_type == "uniform":
        C = torch.ones(num_classes, num_classes, device=device)
        C.fill_diagonal_(0.0)

    elif cost_type == "label_distance":
        idx = torch.arange(num_classes, dtype=torch.float32, device=device)
        C = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        C = C / (num_classes - 1)  # Normalize to [0, 1]

    elif cost_type == "random":
        # Random symmetric matrix
        A = torch.rand(num_classes, num_classes, device=device)
        C = (A + A.T) / 2
        C.fill_diagonal_(0.0)

    else:
        raise ValueError(f"Unknown cost_type: {cost_type}. "
                         f"Choose from: uniform, label_distance, random.")

    return C


class SinkhornDistillationLoss(nn.Module):
    """Knowledge distillation loss using Sinkhorn optimal transport with a FIXED cost matrix.

    L_total = L_CE(z_S, y) + lambda_ot * W_eps(p_T, p_S; C)

    This serves as an intermediate baseline between:
      - KL-KD (no cost structure at all)
      - Adaptive Sinkhorn KD (learned cost structure)

    Args:
        num_classes: Number of output classes.
        temperature: Softmax temperature for softening distributions.
        lambda_ot: Weight of the OT loss relative to cross-entropy.
        epsilon: Sinkhorn entropic regularization.
        max_iter: Maximum Sinkhorn iterations.
        threshold: Convergence threshold.
        cost_type: Fixed cost matrix type ("uniform", "label_distance", "random").
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 4.0,
        lambda_ot: float = 0.5,
        epsilon: float = 0.05,
        max_iter: int = 50,
        threshold: float = 1e-3,
        cost_type: str = "uniform",
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_ot = lambda_ot
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.threshold = threshold
        self.ce_loss = nn.CrossEntropyLoss()

        # Fixed cost matrix (not a parameter — no gradients)
        C = build_cost_matrix(num_classes, cost_type)
        self.register_buffer("C", C)

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute OT-KD loss with fixed cost matrix.

        Args:
            student_logits: Shape (B, K).
            teacher_logits: Shape (B, K).
            labels: Shape (B,).

        Returns:
            Dict with "loss", "ot_loss", "ce_loss", and "transport_plan".
        """
        T = self.temperature

        # Softened distributions (add small epsilon for numerical safety in log)
        p_T = F.softmax(teacher_logits / T, dim=1).clamp(min=1e-8)
        p_S = F.softmax(student_logits / T, dim=1).clamp(min=1e-8)

        log_p_T = p_T.log()
        log_p_S = p_S.log()

        # Sinkhorn distance
        ot_loss, transport_plan = log_sinkhorn(
            log_p_T, log_p_S, self.C,
            epsilon=self.epsilon, max_iter=self.max_iter, threshold=self.threshold,
        )

        # Cross-entropy on hard labels
        ce_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        total_loss = ce_loss + self.lambda_ot * ot_loss

        return {
            "loss": total_loss,
            "ot_loss": ot_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "transport_plan": transport_plan.detach(),
        }
