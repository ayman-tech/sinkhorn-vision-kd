from .kl_distill import KLDistillationLoss
from .sinkhorn_distill import SinkhornDistillationLoss
from .adaptive_sinkhorn import AdaptiveSinkhornKD, LearnableCostMatrix

__all__ = [
    "KLDistillationLoss",
    "SinkhornDistillationLoss",
    "AdaptiveSinkhornKD",
    "LearnableCostMatrix",
]
