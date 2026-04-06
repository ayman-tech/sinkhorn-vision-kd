"""
Evaluation and comparison script.

Loads all saved checkpoints, prints a comparison table, generates visualizations,
and optionally computes statistical significance across multiple seeds.

Usage:
    python evaluate.py --dataset cifar100 --checkpoint_dir ./checkpoints/cifar100
    python evaluate.py --dataset cifar100 --run_seeds --num_seeds 3
"""

import argparse
import os
import glob
import re

import numpy as np
import torch
import torch.nn as nn

from models import resnet20, resnet56, resnet110, mobilenetv2
from utils.data_loader import get_cifar_loaders, get_class_names
from utils.metrics import accuracy, count_parameters, estimate_flops, AverageMeter
from utils.visualization import (
    plot_cost_matrix,
    plot_training_curves,
    plot_compression_tradeoff,
    plot_cost_matrix_evolution,
)


MODEL_FACTORY = {
    "resnet20": resnet20,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "mobilenetv2": mobilenetv2,
}


def get_device():
    """Select best available device: CUDA > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(arch: str, num_classes: int, ckpt_path: str, device: torch.device):
    """Load a model from a checkpoint file."""
    if arch == "mobilenetv2":
        model = MODEL_FACTORY[arch](num_classes=num_classes)
    else:
        model = MODEL_FACTORY[arch](num_classes=num_classes)
    model = model.to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt
    return model, None


def evaluate_checkpoint(model: nn.Module, test_loader, device):
    """Evaluate model on test set, return (top1_acc, top5_acc)."""
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
    return top1.avg, top5.avg


def print_comparison_table(results: list):
    """Print a formatted comparison table of all methods."""
    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)
    header = f"{'Method':<25} | {'Top-1 Acc':>10} | {'Top-5 Acc':>10} | {'Params':>10} | {'FLOPs':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        acc1_str = f"{r['top1_acc']:.2f}%" if r['top1_acc'] is not None else "N/A"
        acc5_str = f"{r['top5_acc']:.2f}%" if r.get('top5_acc') is not None else "N/A"
        params_str = f"{r['params']/1e6:.2f}M"
        flops_str = f"{r['flops']/1e6:.1f}M" if r['flops'] else "N/A"
        print(f"{r['method']:<25} | {acc1_str:>10} | {acc5_str:>10} | {params_str:>10} | {flops_str:>12}")
    print("=" * 90)


def collect_results(args):
    """Collect results from all saved checkpoints."""
    device = get_device()
    num_classes = 10 if args.dataset == "cifar10" else 100
    ckpt_dir = args.checkpoint_dir

    _, _, test_loader = get_cifar_loaders(
        dataset=args.dataset, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    results = []

    def _get_accs(ckpt, model):
        """Extract top-1 and top-5 from checkpoint, or evaluate if missing."""
        acc1 = ckpt.get("best_acc") if ckpt else None
        acc5 = ckpt.get("best_acc5") if ckpt else None
        # If top-5 not saved in checkpoint, evaluate the model
        if acc1 is not None and acc5 is None and ckpt:
            acc1, acc5 = evaluate_checkpoint(model, test_loader, device)
        return acc1, acc5

    # ── Teacher ───────────────────────────────────────────────────────────
    teacher_path = os.path.join(ckpt_dir, f"{args.dataset}_{args.teacher}_teacher.pth")
    teacher, teacher_ckpt = load_model(args.teacher, num_classes, teacher_path, device)
    t_acc1, t_acc5 = _get_accs(teacher_ckpt, teacher)
    results.append({
        "method": f"Teacher ({args.teacher})",
        "top1_acc": t_acc1, "top5_acc": t_acc5,
        "params": count_parameters(teacher),
        "flops": estimate_flops(teacher),
        "params_M": count_parameters(teacher) / 1e6,
    })

    # ── Student baseline (no KD) ─────────────────────────────────────────
    baseline_path = os.path.join(ckpt_dir, f"{args.student}_no_kd_best.pth")
    student_base, base_ckpt = load_model(args.student, num_classes, baseline_path, device)
    b_acc1, b_acc5 = _get_accs(base_ckpt, student_base)
    results.append({
        "method": "Student (no KD)",
        "top1_acc": b_acc1, "top5_acc": b_acc5,
        "params": count_parameters(student_base),
        "flops": estimate_flops(student_base),
        "params_M": count_parameters(student_base) / 1e6,
    })

    # ── KL-KD ─────────────────────────────────────────────────────────────
    kl_path = os.path.join(ckpt_dir, "kl_kd_best.pth")
    student_kl, kl_ckpt = load_model(args.student, num_classes, kl_path, device)
    kl_acc1, kl_acc5 = _get_accs(kl_ckpt, student_kl)
    results.append({
        "method": "KL-KD",
        "top1_acc": kl_acc1, "top5_acc": kl_acc5,
        "params": count_parameters(student_kl),
        "flops": estimate_flops(student_kl),
        "params_M": count_parameters(student_kl) / 1e6,
    })

    # ── Fixed Sinkhorn KD ────────────────────────────────────────────────
    sink_path = os.path.join(ckpt_dir, "sinkhorn_kd_best.pth")
    student_sink, sink_ckpt = load_model(args.student, num_classes, sink_path, device)
    s_acc1, s_acc5 = _get_accs(sink_ckpt, student_sink)
    results.append({
        "method": "Fixed-OT-KD",
        "top1_acc": s_acc1, "top5_acc": s_acc5,
        "params": count_parameters(student_sink),
        "flops": estimate_flops(student_sink),
        "params_M": count_parameters(student_sink) / 1e6,
    })

    # ── Adaptive Sinkhorn KD ─────────────────────────────────────────────
    adaptive_path = os.path.join(ckpt_dir, "adaptive_sinkhorn_kd_best.pth")
    student_adapt, adapt_ckpt = load_model(args.student, num_classes, adaptive_path, device)
    a_acc1, a_acc5 = _get_accs(adapt_ckpt, student_adapt)
    results.append({
        "method": "Adaptive-OT-KD (Ours)",
        "top1_acc": a_acc1, "top5_acc": a_acc5,
        "params": count_parameters(student_adapt),
        "flops": estimate_flops(student_adapt),
        "params_M": count_parameters(student_adapt) / 1e6,
    })

    return results, adapt_ckpt


def generate_visualizations(args, results, adapt_ckpt):
    """Generate all comparison plots."""
    ckpt_dir = args.checkpoint_dir
    num_classes = 10 if args.dataset == "cifar10" else 100
    class_names = get_class_names(args.dataset)
    fig_dir = os.path.join(ckpt_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── 1. Cost matrix heatmap ────────────────────────────────────────────
    if adapt_ckpt and "cost_matrix" in adapt_ckpt:
        C = adapt_ckpt["cost_matrix"]
        if isinstance(C, torch.Tensor):
            C = C.numpy()
        plot_cost_matrix(
            C, class_names,
            save_path=os.path.join(fig_dir, "learned_cost_matrix.png"),
            title=f"Learned Cost Matrix ({args.dataset.upper()})",
        )

    # ── 2. Training curves ────────────────────────────────────────────────
    curves = {}
    for method_key, label in [("kl_kd", "KL-KD"), ("sinkhorn_kd", "Fixed-OT-KD"),
                               ("adaptive_sinkhorn_kd", "Adaptive-OT-KD")]:
        results_path = os.path.join(ckpt_dir, f"{method_key}_results.pth")
        if os.path.exists(results_path):
            data = torch.load(results_path, map_location="cpu", weights_only=False)
            if "history" in data:
                curves[label] = data["history"]

    if curves:
        plot_training_curves(
            curves, save_path=os.path.join(fig_dir, "training_curves.png"),
        )

    # ── 3. Compression trade-off ──────────────────────────────────────────
    plot_data = [
        {"method": r["method"], "params_M": r["params_M"], "top1_acc": r["top1_acc"]}
        for r in results if r["top1_acc"] is not None
    ]
    if plot_data:
        plot_compression_tradeoff(
            plot_data, save_path=os.path.join(fig_dir, "compression_tradeoff.png"),
        )

    # ── 4. Cost matrix evolution (sorted by epoch number) ─────────────────
    epoch_ckpts = glob.glob(os.path.join(ckpt_dir, "adaptive_sinkhorn_kd_epoch*.pth"))
    # Sort by epoch number (numeric), not by filename string
    def _extract_epoch(path):
        m = re.search(r"epoch(\d+)\.pth$", path)
        return int(m.group(1)) if m else 0
    epoch_ckpts.sort(key=_extract_epoch)

    if epoch_ckpts:
        cost_matrices = []
        epochs = []
        for path in epoch_ckpts:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "cost_matrix" in ckpt:
                C = ckpt["cost_matrix"]
                if isinstance(C, torch.Tensor):
                    C = C.numpy()
                cost_matrices.append(C)
                epochs.append(ckpt["epoch"] + 1)

        if cost_matrices:
            # Show at most 5 snapshots
            if len(cost_matrices) > 5:
                indices = np.linspace(0, len(cost_matrices) - 1, 5, dtype=int)
                cost_matrices = [cost_matrices[i] for i in indices]
                epochs = [epochs[i] for i in indices]

            plot_cost_matrix_evolution(
                cost_matrices, epochs, class_names,
                save_path=os.path.join(fig_dir, "cost_evolution.png"),
            )

    print(f"\nAll figures saved to {fig_dir}/")


def run_multi_seed(args):
    """Run all methods across multiple seeds and report mean +/- std."""
    from train import train_distillation, train_student_baseline, set_seed

    seeds = list(range(args.seed, args.seed + args.num_seeds))
    all_results = {method: [] for method in ["student_baseline", "kl_kd", "sinkhorn_kd", "adaptive_sinkhorn_kd"]}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        args.seed = seed

        # Student baseline
        args.mode = "student_baseline"
        base_acc = train_student_baseline(args)
        all_results["student_baseline"].append(base_acc)

        # KL-KD
        args.method = "kl_kd"
        _, kl_acc = train_distillation(args)
        all_results["kl_kd"].append(kl_acc)

        # Fixed Sinkhorn
        args.method = "sinkhorn_kd"
        _, sink_acc = train_distillation(args)
        all_results["sinkhorn_kd"].append(sink_acc)

        # Adaptive Sinkhorn
        args.method = "adaptive_sinkhorn_kd"
        _, adapt_acc = train_distillation(args)
        all_results["adaptive_sinkhorn_kd"].append(adapt_acc)

    # Report
    print("\n" + "=" * 60)
    print(f"MULTI-SEED RESULTS ({args.num_seeds} seeds)")
    print("=" * 60)
    print(f"{'Method':<25} | {'Mean Acc':>10} | {'Std':>8}")
    print("-" * 50)
    for method, accs in all_results.items():
        mean = np.mean(accs)
        std = np.std(accs)
        print(f"{method:<25} | {mean:>9.2f}% | {std:>7.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare KD methods")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/cifar100")
    parser.add_argument("--teacher", type=str, default="resnet110")
    parser.add_argument("--student", type=str, default="resnet20")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_seeds", action="store_true",
                        help="Run all methods across multiple seeds.")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    if args.run_seeds:
        run_multi_seed(args)
    else:
        results, adapt_ckpt = collect_results(args)
        print_comparison_table(results)
        generate_visualizations(args, results, adapt_ckpt)


if __name__ == "__main__":
    main()
