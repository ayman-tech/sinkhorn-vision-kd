"""Fixed-cost Sinkhorn OT distillation.

W_eps(p_T, p_S; C) = min_pi <C, pi> + eps*KL(pi || p_T x p_S)
solved with log-domain Sinkhorn iterations for numerical stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def log_sinkhorn(log_a, log_b, C, epsilon=0.05, max_iter=50, threshold=1e-3):
    """Batched Sinkhorn in log-domain.

    log_a, log_b: (B, K) log-marginals, must sum (in exp) to 1 along K.
    C:            (K, K) non-negative cost matrix (the K dim is shared across batch).
    Returns scalar mean cost <C, pi> and the (B, K, K) transport plan pi.

    The trick: we keep the dual potentials f, g unscaled and absorb 1/eps into
    M = C/eps so the iteration matches the standard "log-sum-exp" form without
    further bookkeeping.
    """
    B, K = log_a.shape
    M = C.unsqueeze(0)/epsilon

    f=torch.zeros(B, K, device=log_a.device, dtype=log_a.dtype)
    g=torch.zeros(B, K, device=log_b.device, dtype=log_b.dtype)

    for _ in range(max_iter):
        f_old=f.clone()
        # f_i  =  -lse_j ( g_j - M_ij + log_b_j )
        kf=g.unsqueeze(1) - M + log_b.unsqueeze(1)   # (B, K, K)
        f=-torch.logsumexp(kf, dim=2)
        # g_j  =  -lse_i ( f_i - M_ij + log_a_i )
        kg=f.unsqueeze(2) - M + log_a.unsqueeze(2)   # (B, K, K)
        g=-torch.logsumexp(kg, dim=1)
        if (f-f_old).abs().max().item()<threshold:
            break

    log_pi = f.unsqueeze(2) + g.unsqueeze(1) - M + log_a.unsqueeze(2) + log_b.unsqueeze(1)
    pi=log_pi.exp()
    cost=(C.unsqueeze(0)*pi).sum(dim=(1,2)).mean()
    return cost, pi


def build_cost_matrix(num_classes, cost_type="uniform", device=torch.device("cpu")):
    """Returns a (K, K) cost matrix. Always symmetric, non-negative, zero diagonal."""
    K=num_classes
    if cost_type=="uniform":
        C=torch.ones(K, K, device=device)
        C.fill_diagonal_(0.0)
    elif cost_type=="label_distance":
        idx=torch.arange(K, dtype=torch.float32, device=device)
        C=(idx.unsqueeze(0)-idx.unsqueeze(1)).abs()/(K-1)
    elif cost_type=="random":
        A=torch.rand(K, K, device=device)
        C=(A+A.T)/2
        C.fill_diagonal_(0.0)
    else:
        raise ValueError(f"unknown cost_type {cost_type!r}; want uniform|label_distance|random")
    return C


class SinkhornDistillationLoss(nn.Module):
    """Sinkhorn OT-KD with a fixed cost matrix.

    L = CE(z_S, y) + lambda_ot * W_eps(p_T, p_S; C)
    """
    def __init__(self, num_classes, temperature=4.0, lambda_ot=0.5, epsilon=0.05,
                 max_iter=50, threshold=1e-3, cost_type="uniform"):
        super().__init__()
        self.T=temperature
        self.lam=lambda_ot
        self.eps=epsilon
        self.max_iter=max_iter
        self.threshold=threshold
        self.ce=nn.CrossEntropyLoss()
        # buffer (not parameter) — moves with .to(device) but no gradients.
        self.register_buffer("C", build_cost_matrix(num_classes, cost_type))

    def forward(self, student_logits, teacher_logits, labels) -> Dict[str, torch.Tensor]:
        T=self.T
        pT=F.softmax(teacher_logits/T, dim=1).clamp(min=1e-8)
        pS=F.softmax(student_logits/T, dim=1).clamp(min=1e-8)
        ot, pi = log_sinkhorn(pT.log(), pS.log(), self.C,
                              epsilon=self.eps, max_iter=self.max_iter,
                              threshold=self.threshold)
        ce=self.ce(student_logits, labels)
        return {"loss": ce + self.lam*ot,
                "ot_loss": ot.detach(),
                "ce_loss": ce.detach(),
                "transport_plan": pi.detach()}
