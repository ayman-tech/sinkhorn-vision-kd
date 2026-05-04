"""Adaptive (learnable-cost) Sinkhorn OT distillation — the main contribution.

Bilevel:
    outer: min_C  L_val(theta*(C), C)     [update C on a held-out val batch]
    inner: min_theta  L_train(theta, C)    [normal SGD on student]

The cost matrix C is parameterized so it stays valid under any raw A:
  S = (A + A^T)/2          symmetric
  C' = softplus(S)         non-negative (smooth ReLU)
  C  = C' - diag(C')       zero diagonal
  C  = C / max(C)          normalize to [0, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .sinkhorn_distill import log_sinkhorn


class LearnableCostMatrix(nn.Module):
    def __init__(self, num_classes: int, init_scale: float = 0.5):
        super().__init__()
        self.K=num_classes
        # small noise around init_scale so softplus(S) starts roughly uniform.
        A=torch.randn(num_classes, num_classes)*0.1 + init_scale
        self.A=nn.Parameter(A)

    def forward(self):
        S=(self.A + self.A.T)/2
        C=F.softplus(S)
        mask=1.0 - torch.eye(self.K, device=C.device)
        C=C*mask
        C=C/(C.max()+1e-8)
        return C

    def get_cost_matrix(self):
        with torch.no_grad():
            return self.forward().cpu()


class AdaptiveSinkhornKD(nn.Module):
    """Sinkhorn OT-KD with a jointly learned cost matrix.

    L = CE(z_S, y) + lambda_ot * W_eps(p_T, p_S; C(A))
    Trainer is expected to call:
      - forward(...) every step (returns inner-loop loss),
      - should_update_cost() to decide if outer step is due,
      - step_cost_matrix(...) to do the outer update on a val batch,
      - increment_step() at the end of every iteration.
    """
    def __init__(self, num_classes, temperature=4.0, lambda_ot=0.5, epsilon=0.05,
                 max_iter=50, threshold=1e-3, cost_lr=0.01, cost_update_freq=10,
                 cost_grad_clip=1.0, init_scale=0.5):
        super().__init__()
        self.T=temperature
        self.lam=lambda_ot
        self.eps=epsilon
        self.max_iter=max_iter
        self.threshold=threshold
        self.cost_update_freq=cost_update_freq
        self.cost_grad_clip=cost_grad_clip

        self.ce=nn.CrossEntropyLoss()
        self.cost_matrix=LearnableCostMatrix(num_classes, init_scale)
        # separate optimizer for the outer loop — Adam handles the C grads
        # better than SGD here (they're small and noisy).
        self.cost_optimizer=torch.optim.Adam(self.cost_matrix.parameters(), lr=cost_lr)
        self._step=0

    def forward(self, student_logits, teacher_logits, labels) -> Dict[str, torch.Tensor]:
        T=self.T
        C=self.cost_matrix()
        pT=F.softmax(teacher_logits/T, dim=1).clamp(min=1e-8)
        pS=F.softmax(student_logits/T, dim=1).clamp(min=1e-8)
        ot, pi = log_sinkhorn(pT.log(), pS.log(), C,
                              epsilon=self.eps, max_iter=self.max_iter,
                              threshold=self.threshold)
        ce=self.ce(student_logits, labels)
        return {"loss": ce + self.lam*ot,
                "ot_loss": ot.detach(),
                "ce_loss": ce.detach(),
                "transport_plan": pi.detach(),
                "cost_matrix": C.detach()}

    def should_update_cost(self) -> bool:
        return self._step % self.cost_update_freq == 0

    def step_cost_matrix(self, student, teacher, val_images, val_labels):
        """One outer-loop step: take a gradient on C using a validation batch.

        We don't freeze student grads explicitly — gradients on theta are just
        thrown away after the C-optimizer step (theta has its own optimizer
        that the trainer calls separately).
        """
        teacher.eval()
        with torch.no_grad():
            tz=teacher(val_images)
        sz=student(val_images)
        r=self(sz, tz, val_labels)

        self.cost_optimizer.zero_grad()
        r["loss"].backward()
        gnorm=torch.nn.utils.clip_grad_norm_(self.cost_matrix.parameters(),
                                             self.cost_grad_clip)
        self.cost_optimizer.step()
        return {"cost_loss": r["loss"].item(),
                "cost_grad_norm": gnorm.item() if isinstance(gnorm, torch.Tensor) else gnorm}

    def increment_step(self):
        self._step += 1

    def get_cost_matrix_numpy(self):
        return self.cost_matrix.get_cost_matrix().numpy()
