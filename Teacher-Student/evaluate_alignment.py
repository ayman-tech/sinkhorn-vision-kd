"""
Teacher-student alignment metrics.

For each distilled student (KL-KD, Fixed-OT-KD, Adaptive-OT-KD, plus the no-KD
baseline as a reference), measures how closely the student's outputs track the
teacher on the test set:

  - KL(p_T || p_S)          divergence of softened distributions (lower = closer)
  - Sinkhorn W_eps(p_T, p_S) entropy-regularized Wasserstein on class-prob simplex
  - Cosine similarity        of raw logit vectors (per sample, then averaged)
  - Top-1 agreement          fraction of samples where argmax matches the teacher
  - Top-5 agreement          fraction where teacher's top-1 is in student's top-5

KL and Sinkhorn use the same temperature T as training (default 4.0). Sinkhorn
uses a uniform cost (C_{ij} = 1 - delta_{ij}) so the metric is comparable
across all methods regardless of which cost they trained with.

Usage:
    python evaluate_alignment.py --dataset cifar10
    python evaluate_alignment.py --dataset cifar100 --temperature 4.0
"""

import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet20, resnet56, resnet110, mobilenetv2
from utils.data_loader import get_cifar_loaders
from utils.metrics import AverageMeter
from distillation.sinkhorn_distill import log_sinkhorn, build_cost_matrix


MODEL_FACTORY = {
    "resnet20": resnet20,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "mobilenetv2": mobilenetv2,
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(arch: str, num_classes: int, ckpt_path: str, device):
    if arch not in MODEL_FACTORY:
        raise ValueError(f"Unknown arch: {arch}")
    model = MODEL_FACTORY[arch](num_classes=num_classes).to(device)
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def alignment_metrics(student: nn.Module, teacher: nn.Module, loader, device,
                      temperature: float, cost_matrix: torch.Tensor,
                      epsilon: float, max_iter: int) -> Dict[str, float]:
    """Compute alignment metrics on the full loader."""
    kl_meter = AverageMeter()
    w_meter = AverageMeter()
    cos_meter = AverageMeter()
    top1_agree = AverageMeter()
    top5_agree = AverageMeter()

    T = temperature
    teacher.eval()
    student.eval()

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        z_T = teacher(images)
        z_S = student(images)

        # Softened distributions
        p_T = F.softmax(z_T / T, dim=1).clamp(min=1e-8)
        p_S = F.softmax(z_S / T, dim=1).clamp(min=1e-8)
        log_p_T = p_T.log()
        log_p_S = p_S.log()

        # KL(p_T || p_S), per-sample then averaged. Scale by T^2 to match the
        # standard KD-loss convention so numbers are comparable to training.
        kl_per_sample = (p_T * (log_p_T - log_p_S)).sum(dim=1)
        kl = kl_per_sample.mean() * (T * T)
        kl_meter.update(kl.item(), images.size(0))

        # Sinkhorn distance on softened distributions with a uniform cost.
        w_cost, _ = log_sinkhorn(
            log_p_T, log_p_S, cost_matrix,
            epsilon=epsilon, max_iter=max_iter, threshold=1e-3,
        )
        w_meter.update(w_cost.item(), images.size(0))

        # Cosine similarity of raw logits (per sample).
        cos = F.cosine_similarity(z_T, z_S, dim=1).mean()
        cos_meter.update(cos.item(), images.size(0))

        # Agreement: argmax_S matches argmax_T.
        pred_T = z_T.argmax(dim=1)
        pred_S_top1 = z_S.argmax(dim=1)
        top1_agree.update(
            (pred_S_top1 == pred_T).float().mean().item() * 100.0, images.size(0)
        )

        # Top-5 student contains teacher top-1.
        k = min(5, z_S.size(1))
        top5_S = z_S.topk(k, dim=1).indices
        in_top5 = (top5_S == pred_T.unsqueeze(1)).any(dim=1)
        top5_agree.update(in_top5.float().mean().item() * 100.0, images.size(0))

    return {
        "kl": kl_meter.avg,
        "wasserstein": w_meter.avg,
        "cosine": cos_meter.avg,
        "top1_agree": top1_agree.avg,
        "top5_agree": top5_agree.avg,
    }


def collect_students(args, device, num_classes):
    ckpt_dir = args.checkpoint_dir
    specs = [
        ("Student (no KD)", args.student,
         os.path.join(ckpt_dir, f"{args.student}_no_kd_best.pth")),
        ("KL-KD", args.student,
         os.path.join(ckpt_dir, "kl_kd_best.pth")),
        ("Fixed-OT-KD", args.student,
         os.path.join(ckpt_dir, "sinkhorn_kd_best.pth")),
        ("Adaptive-OT-KD", args.student,
         os.path.join(ckpt_dir, "adaptive_sinkhorn_kd_best.pth")),
    ]
    students = {}
    for name, arch, path in specs:
        m = load_model(arch, num_classes, path, device)
        if m is None:
            print(f"[skip] {name}: checkpoint not found at {path}")
        else:
            students[name] = m
    return students


def print_table(results: Dict[str, Dict[str, float]]):
    print("\n" + "=" * 92)
    print("TEACHER-STUDENT ALIGNMENT (test set)")
    print("=" * 92)
    header = (f"{'Method':<22} | {'KL(T||S)':>10} | {'W_eps':>10} | "
              f"{'cos(z_T,z_S)':>13} | {'top-1 agree':>12} | {'top-5 agree':>12}")
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<22} | {m['kl']:>10.4f} | {m['wasserstein']:>10.4f} | "
              f"{m['cosine']:>13.4f} | {m['top1_agree']:>11.2f}% | "
              f"{m['top5_agree']:>11.2f}%")
    print("=" * 92)
    print("Lower KL / W_eps and higher cosine / agreement = closer to teacher.")


def parse_args():
    p = argparse.ArgumentParser(description="Teacher-student alignment metrics")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Defaults to ./checkpoints/<dataset>")
    p.add_argument("--teacher", type=str, default="resnet110")
    p.add_argument("--student", type=str, default="resnet20")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--sinkhorn_iter", type=int, default=50)
    p.add_argument("--save_csv", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"./checkpoints/{args.dataset}"

    device = get_device()
    num_classes = 10 if args.dataset == "cifar10" else 100
    print(f"Device: {device} | dataset: {args.dataset} | ckpt_dir: {args.checkpoint_dir}")

    teacher_path = os.path.join(
        args.checkpoint_dir, f"{args.dataset}_{args.teacher}_teacher.pth"
    )
    teacher = load_model(args.teacher, num_classes, teacher_path, device)
    if teacher is None:
        raise FileNotFoundError(f"Teacher checkpoint missing: {teacher_path}")

    students = collect_students(args, device, num_classes)
    if not students:
        print("No student checkpoints found.")
        return

    _, _, test_loader = get_cifar_loaders(
        dataset=args.dataset, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cost_matrix = build_cost_matrix(num_classes, "uniform", device=device)

    results: Dict[str, Dict[str, float]] = {}
    for name, model in students.items():
        print(f"\n[evaluating] {name}")
        results[name] = alignment_metrics(
            model, teacher, test_loader, device,
            temperature=args.temperature,
            cost_matrix=cost_matrix,
            epsilon=args.epsilon, max_iter=args.sinkhorn_iter,
        )

    print_table(results)

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "kl", "wasserstein", "cosine",
                        "top1_agree", "top5_agree"])
            for name, m in results.items():
                w.writerow([name, m["kl"], m["wasserstein"], m["cosine"],
                            m["top1_agree"], m["top5_agree"]])
        print(f"\nSaved CSV to {args.save_csv}")


if __name__ == "__main__":
    main()
