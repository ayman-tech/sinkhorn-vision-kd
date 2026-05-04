"""
Evaluation and comparison script.

Loads all saved checkpoints, prints a comparison table, generates visualizations,
and optionally runs:
  --alignment   Teacher-student alignment metrics (KL, Sinkhorn W_eps, cosine, agreement)
  --robustness  CIFAR-10-C / CIFAR-100-C corruption sweep (downloads ~2.9 GB on first run)
  --no_entropy  Skip predictive entropy measurement (runs by default)

Usage:
    python evaluate.py --dataset cifar10 --checkpoint_dir ./checkpoints/cifar10
    python evaluate.py --dataset cifar10 --alignment
    python evaluate.py --dataset cifar10 --robustness --corruptions gaussian_noise gaussian_blur
    python evaluate.py --dataset cifar100 --alignment --robustness --alignment_csv align.csv
    python evaluate.py --dataset cifar10 --run_seeds --num_seeds 3
"""

import argparse
import csv
import math
import os
import glob
import re
import tarfile
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from models import resnet20, resnet56, resnet110, mobilenetv2
from utils.data_loader import (
    get_cifar_loaders, get_class_names,
    CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD,
)
from utils.metrics import (
    accuracy, count_parameters, estimate_flops, AverageMeter,
    collect_predictions, compute_ece, compute_nll, compute_brier_score,
)
from utils.visualization import (
    plot_cost_matrix,
    plot_training_curves,
    plot_compression_tradeoff,
    plot_cost_matrix_evolution,
    plot_reliability_diagram,
    plot_class_clustering,
    plot_nearest_neighbor_costs,
)
from distillation.sinkhorn_distill import log_sinkhorn, build_cost_matrix


MODEL_FACTORY = {
    "resnet20": resnet20,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "mobilenetv2": mobilenetv2,
}


# ── CIFAR-C constants ──────────────────────────────────────────────────────────

ALL_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur",
]
DEFAULT_CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "spatter"]

# Zenodo hosts the official Hendrycks & Dietterich (2019) CIFAR-C tarballs (~2.9 GB each).
ZENODO = {
    "cifar10":  ("https://zenodo.org/records/2535967/files/CIFAR-10-C.tar",
                 "CIFAR-10-C.tar",  "CIFAR-10-C"),
    "cifar100": ("https://zenodo.org/records/3555552/files/CIFAR-100-C.tar",
                 "CIFAR-100-C.tar", "CIFAR-100-C"),
}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(arch: str, num_classes: int, ckpt_path: str, device: torch.device):
    """Load a model from a checkpoint file. Returns (model, ckpt_dict or None)."""
    model = MODEL_FACTORY[arch](num_classes=num_classes).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt
    return model, None


def evaluate_checkpoint(model: nn.Module, test_loader, device):
    """Evaluate model on test set. Returns (top1_acc, top5_acc)."""
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


# ── Standard evaluation (accuracy + calibration) ──────────────────────────────

def print_comparison_table(results: list):
    """Print a formatted comparison table of all methods including calibration."""
    print("\n" + "=" * 115)
    print("RESULTS COMPARISON")
    print("=" * 115)
    header = (
        f"{'Method':<25} | {'Top-1':>7} | {'Top-5':>7} | "
        f"{'ECE':>7} | {'NLL':>7} | {'Brier':>7} | "
        f"{'Params':>8} | {'FLOPs':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        acc1_str   = f"{r['top1_acc']:.2f}%"  if r.get("top1_acc")  is not None else "N/A"
        acc5_str   = f"{r['top5_acc']:.2f}%"  if r.get("top5_acc")  is not None else "N/A"
        ece_str    = f"{r['ece']:.4f}"          if r.get("ece")       is not None else "N/A"
        nll_str    = f"{r['nll']:.4f}"          if r.get("nll")       is not None else "N/A"
        brier_str  = f"{r['brier']:.4f}"        if r.get("brier")     is not None else "N/A"
        params_str = f"{r['params']/1e6:.2f}M"
        flops_str  = f"{r['flops']/1e6:.1f}M"  if r.get("flops")     else "N/A"
        print(
            f"{r['method']:<25} | {acc1_str:>7} | {acc5_str:>7} | "
            f"{ece_str:>7} | {nll_str:>7} | {brier_str:>7} | "
            f"{params_str:>8} | {flops_str:>10}"
        )
    print("=" * 115)


def collect_results(args):
    """Collect accuracy + calibration metrics from all saved checkpoints.

    Returns:
        results:     List of per-method metric dicts.
        adapt_ckpt:  Adaptive-OT-KD checkpoint dict (contains cost_matrix).
        predictions: {label: {"probs": ndarray, "labels": ndarray}}
        models:      {label: nn.Module}   reused by alignment / robustness
        test_loader: The CIFAR test DataLoader (reused by alignment)
    """
    device = get_device()
    num_classes = 10 if args.dataset == "cifar10" else 100
    ckpt_dir = args.checkpoint_dir

    _, _, test_loader = get_cifar_loaders(
        dataset=args.dataset, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    results = []
    predictions = {}
    models = {}

    def _eval(model, ckpt, label):
        if ckpt is None:
            return None, None, None, None, None
        probs, labels = collect_predictions(model, test_loader, device)
        predictions[label] = {"probs": probs.numpy(), "labels": labels.numpy()}
        preds = probs.argmax(dim=1)
        acc1  = preds.eq(labels).float().mean().item() * 100
        acc5  = probs.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100
        return (acc1, acc5,
                compute_ece(probs, labels),
                compute_nll(probs, labels),
                compute_brier_score(probs, labels, num_classes))

    checkpoints = [
        (f"Teacher ({args.teacher})", args.teacher,
         os.path.join(ckpt_dir, f"{args.dataset}_{args.teacher}_teacher.pth")),
        ("Student (no KD)", args.student,
         os.path.join(ckpt_dir, f"{args.student}_no_kd_best.pth")),
        ("KL-KD", args.student,
         os.path.join(ckpt_dir, "kl_kd_best.pth")),
        ("Fixed-OT-KD", args.student,
         os.path.join(ckpt_dir, "sinkhorn_kd_best.pth")),
        ("Adaptive-OT-KD (Ours)", args.student,
         os.path.join(ckpt_dir, "adaptive_sinkhorn_kd_best.pth")),
    ]

    adapt_ckpt = None
    for label, arch, path in checkpoints:
        model, ckpt = load_model(arch, num_classes, path, device)
        acc1, acc5, ece, nll, brier = _eval(model, ckpt, label)
        results.append({
            "method": label,
            "top1_acc": acc1, "top5_acc": acc5,
            "ece": ece, "nll": nll, "brier": brier,
            "params": count_parameters(model),
            "flops": estimate_flops(model),
            "params_M": count_parameters(model) / 1e6,
        })
        if ckpt is not None:
            models[label] = model
            if label == "Adaptive-OT-KD (Ours)":
                adapt_ckpt = ckpt

    return results, adapt_ckpt, predictions, models, test_loader


# ── Visualizations ─────────────────────────────────────────────────────────────

def generate_visualizations(args, results, adapt_ckpt, predictions,
                             alignment_res=None, robustness_res=None):
    """Generate all comparison plots and save to <output_dir>/figures/."""
    ckpt_dir = args.checkpoint_dir
    class_names = get_class_names(args.dataset)
    root = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(root, "figures", args.dataset)
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Learned cost matrix heatmap
    C = None
    if adapt_ckpt and "cost_matrix" in adapt_ckpt:
        C = adapt_ckpt["cost_matrix"]
        if isinstance(C, torch.Tensor):
            C = C.numpy()
        plot_cost_matrix(
            C, class_names,
            save_path=os.path.join(fig_dir, "learned_cost_matrix.png"),
            title=f"Learned Cost Matrix ({args.dataset.upper()})",
        )

    # 2. Training curves
    curves = {}
    for method_key, label in [("kl_kd", "KL-KD"), ("sinkhorn_kd", "Fixed-OT-KD"),
                               ("adaptive_sinkhorn_kd", "Adaptive-OT-KD")]:
        results_path = os.path.join(ckpt_dir, f"{method_key}_results.pth")
        if os.path.exists(results_path):
            data = torch.load(results_path, map_location="cpu", weights_only=False)
            if "history" in data:
                curves[label] = data["history"]
    if curves:
        plot_training_curves(curves, save_path=os.path.join(fig_dir, "training_curves.png"))

    # 3. Compression trade-off
    plot_data = [
        {"method": r["method"], "params_M": r["params_M"], "top1_acc": r["top1_acc"]}
        for r in results if r.get("top1_acc") is not None
    ]
    if plot_data:
        plot_compression_tradeoff(
            plot_data, save_path=os.path.join(fig_dir, "compression_tradeoff.png"),
        )

    # 4. Cost matrix evolution across training
    epoch_ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "adaptive_sinkhorn_kd_epoch*.pth")),
        key=lambda p: int(m.group(1)) if (m := re.search(r"epoch(\d+)\.pth$", p)) else 0,
    )
    if epoch_ckpts:
        cost_matrices, epochs = [], []
        for path in epoch_ckpts:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "cost_matrix" in ckpt:
                cm = ckpt["cost_matrix"]
                if isinstance(cm, torch.Tensor):
                    cm = cm.numpy()
                cost_matrices.append(cm)
                epochs.append(ckpt["epoch"] + 1)
        if cost_matrices:
            if len(cost_matrices) > 5:
                idx = np.linspace(0, len(cost_matrices) - 1, 5, dtype=int)
                cost_matrices = [cost_matrices[i] for i in idx]
                epochs = [epochs[i] for i in idx]
            plot_cost_matrix_evolution(
                cost_matrices, epochs, class_names,
                save_path=os.path.join(fig_dir, "cost_evolution.png"),
            )

    # 5. Reliability diagrams (calibration)
    if predictions:
        plot_reliability_diagram(
            predictions, save_path=os.path.join(fig_dir, "reliability_diagrams.png"),
        )

    # 6. t-SNE class clustering from cost matrix
    if C is not None:
        plot_class_clustering(
            C, class_names,
            save_path=os.path.join(fig_dir, "class_clustering.png"),
            title=f"Class Geometry from Learned Cost Matrix ({args.dataset.upper()}, t-SNE)",
        )

    # 7. Nearest-neighbor costs
    if C is not None:
        plot_nearest_neighbor_costs(
            C, class_names, k=5,
            save_path=os.path.join(fig_dir, "nearest_neighbor_costs.png"),
            txt_path=os.path.join(fig_dir, "nearest_neighbor_costs.txt"),
        )

    # 8. Alignment bar charts (if --alignment was run)
    if alignment_res:
        _plot_alignment_bars(
            alignment_res,
            save_path=os.path.join(fig_dir, "alignment_bars.png"),
        )

    # 9. Robustness curves (if --robustness was run)
    if robustness_res:
        _plot_robustness_curves(
            robustness_res, args.dataset,
            save_path=os.path.join(fig_dir, "robustness_curves.png"),
        )

    print(f"\nAll figures saved to {fig_dir}")


# ── Alignment ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _alignment_metrics(student, teacher, loader, device, T, C, eps, max_iter) -> dict:
    """Compute KL(T||S), Sinkhorn W_eps, cosine(z_T, z_S), top-1/top-5 agreement."""
    kl_m = AverageMeter(); w_m = AverageMeter(); cos_m = AverageMeter()
    a1 = AverageMeter(); a5 = AverageMeter()
    teacher.eval(); student.eval()

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        zT = teacher(x); zS = student(x)

        pT = F.softmax(zT / T, dim=1).clamp(min=1e-8)
        pS = F.softmax(zS / T, dim=1).clamp(min=1e-8)

        # KL scaled by T^2 to match Hinton training loss magnitude
        kl_m.update((pT * (pT.log() - pS.log())).sum(dim=1).mean().item() * T * T, x.size(0))

        w_cost, _ = log_sinkhorn(pT.log(), pS.log(), C,
                                 epsilon=eps, max_iter=max_iter, threshold=1e-3)
        w_m.update(w_cost.item(), x.size(0))

        cos_m.update(F.cosine_similarity(zT, zS, dim=1).mean().item(), x.size(0))

        teacher_pred = zT.argmax(dim=1)
        a1.update((zS.argmax(dim=1) == teacher_pred).float().mean().item() * 100.0, x.size(0))
        k = min(5, zS.size(1))
        in_top5 = (zS.topk(k, dim=1).indices == teacher_pred.unsqueeze(1)).any(dim=1)
        a5.update(in_top5.float().mean().item() * 100.0, x.size(0))

    return {"kl": kl_m.avg, "wasserstein": w_m.avg, "cosine": cos_m.avg,
            "top1_agree": a1.avg, "top5_agree": a5.avg}


def run_alignment(args, models: dict, test_loader) -> dict:
    """Compute teacher-student alignment metrics for all loaded students."""
    device = get_device()
    num_classes = 10 if args.dataset == "cifar10" else 100
    teacher_label = f"Teacher ({args.teacher})"

    teacher = models.get(teacher_label)
    if teacher is None:
        print("[alignment] teacher checkpoint not found, skipping.")
        return {}

    C = build_cost_matrix(num_classes, "uniform", device=device)
    student_labels = ["Student (no KD)", "KL-KD", "Fixed-OT-KD", "Adaptive-OT-KD (Ours)"]

    res = {}
    for name in student_labels:
        m = models.get(name)
        if m is None:
            print(f"[alignment] skip {name}: no checkpoint")
            continue
        print(f"[alignment] {name}")
        res[name] = _alignment_metrics(
            m, teacher, test_loader, device,
            T=args.temperature, C=C,
            eps=args.epsilon, max_iter=args.sinkhorn_iter,
        )

    _print_alignment_table(res)

    if args.alignment_csv and res:
        with open(args.alignment_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "kl", "wasserstein", "cosine", "top1_agree", "top5_agree"])
            for name, m in res.items():
                w.writerow([name, m["kl"], m["wasserstein"],
                            m["cosine"], m["top1_agree"], m["top5_agree"]])
        print(f"alignment csv → {args.alignment_csv}")

    return res


def _print_alignment_table(res: dict):
    print("\n" + "=" * 95)
    print("TEACHER-STUDENT ALIGNMENT (test set)")
    print("=" * 95)
    head = (f"{'Method':<25} | {'KL(T||S)':>10} | {'W_eps':>10} | "
            f"{'cos(zT,zS)':>12} | {'top-1 agree':>12} | {'top-5 agree':>12}")
    print(head)
    print("-" * len(head))
    for name, m in res.items():
        print(f"{name:<25} | {m['kl']:>10.4f} | {m['wasserstein']:>10.4f} | "
              f"{m['cosine']:>12.4f} | {m['top1_agree']:>11.2f}% | {m['top5_agree']:>11.2f}%")
    print("=" * 95)
    print("lower KL / W_eps = closer to teacher;  higher cosine / agree = closer.")


def _plot_alignment_bars(res: dict, save_path: str):
    """Four-panel bar chart, one panel per alignment metric."""
    metrics = [
        ("kl",          "KL(T‖S)",         "lower is better"),
        ("wasserstein", "Sinkhorn W_ε",    "lower is better"),
        ("cosine",      "cos(z_T, z_S)",   "higher is better"),
        ("top1_agree",  "top-1 agree (%)", "higher is better"),
    ]
    methods = list(res.keys())
    short = [m.replace(" (Ours)", "") for m in methods]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, (col, title, note) in zip(axes, metrics):
        vals = [res[m][col] for m in methods]
        bars = ax.bar(range(len(methods)), vals)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{title}\n({note})", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Alignment bar chart saved to {save_path}")


# ── Robustness (CIFAR-C) ───────────────────────────────────────────────────────

def _download_file(url: str, dst: str):
    print(f"  downloading {url}\n          to {dst}")
    last = [-1]
    def _hook(blocks, bs, total):
        if total <= 0:
            return
        pct = int(100 * blocks * bs / total)
        if pct != last[0] and pct % 5 == 0:
            print(f"    {pct:3d}%  ({blocks*bs/1e6:6.1f} / {total/1e6:6.1f} MB)")
            last[0] = pct
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    os.replace(tmp, dst)


def _extract_npy_files(tar_path: str, target_dir: str, names: List[str]):
    want = {f"{n}.npy" for n in names}
    got: set = set()
    print(f"  extracting {len(want)} files from {os.path.basename(tar_path)}")
    with tarfile.open(tar_path, "r") as tf:
        for mem in tf:
            if not mem.isfile():
                continue
            base = os.path.basename(mem.name)
            if base in want:
                src = tf.extractfile(mem)
                if src is None:
                    continue
                with open(os.path.join(target_dir, base), "wb") as out:
                    out.write(src.read())
                print(f"    + {base}")
                got.add(base)
                if got == want:
                    break
    miss = want - got
    if miss:
        raise RuntimeError(f"tar missing: {miss}. Delete {tar_path} and retry.")


def ensure_cifar_c(dataset: str, corruptions: List[str], data_dir: str, keep_tar: bool = False) -> str:
    """Download and extract CIFAR-C .npy files if not already on disk."""
    url, tar_name, dirname = ZENODO[dataset]
    target = os.path.join(data_dir, dirname)
    os.makedirs(target, exist_ok=True)

    for c in corruptions:
        if c not in ALL_CORRUPTIONS:
            raise ValueError(f"unknown corruption {c!r}; choose from: {', '.join(ALL_CORRUPTIONS)}")

    needed = list(corruptions) + ["labels"]
    missing = [n for n in needed if not os.path.exists(os.path.join(target, f"{n}.npy"))]
    if not missing:
        return target

    tar = os.path.join(data_dir, tar_name)
    if not os.path.exists(tar):
        print(f"\nfetching {dirname} from Zenodo (~2.9 GB, one-time download)")
        _download_file(url, tar)
    _extract_npy_files(tar, target, missing)
    if not keep_tar:
        os.remove(tar)
        print(f"  removed {tar} (pass --keep_tar to retain)")
    return target


class _CIFARC(Dataset):
    """One severity slice of a CIFAR-C .npy corruption file (10 000 images)."""
    _SLICE = 10_000

    def __init__(self, root: str, corruption: str, severity: int, mean, std):
        imgs = np.load(os.path.join(root, f"{corruption}.npy"))
        labs = np.load(os.path.join(root, "labels.npy"))
        if imgs.shape[0] != 5 * self._SLICE:
            raise RuntimeError(
                f"{corruption}.npy has shape {imgs.shape}, expected (50000, 32, 32, 3)"
            )
        lo = (severity - 1) * self._SLICE
        self.imgs = imgs[lo : lo + self._SLICE]
        self.labs = labs[lo : lo + self._SLICE]
        self.norm = T.Normalize(mean, std)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = torch.from_numpy(self.imgs[i]).permute(2, 0, 1).contiguous().float() / 255.0
        return self.norm(x), int(self.labs[i])


class _CleanCIFAR(Dataset):
    """Standard CIFAR test set wrapped in the same transform used during training."""
    def __init__(self, dataset: str, data_dir: str, mean, std):
        cls = (torchvision.datasets.CIFAR10 if dataset == "cifar10"
               else torchvision.datasets.CIFAR100)
        self.base = cls(root=data_dir, train=False, download=True, transform=None)
        self.tx = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return self.tx(x), y


@torch.no_grad()
def _eval_loader(model: nn.Module, loader, device) -> float:
    meter = AverageMeter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        meter.update(accuracy(model(x), y, topk=(1,))[0], x.size(0))
    return meter.avg


def run_robustness(args, models: dict) -> dict:
    """Sweep all checkpoints across CIFAR-C corruptions at every severity level."""
    if not models:
        print("[robustness] no models loaded, skipping.")
        return {}

    device = get_device()
    pin = torch.cuda.is_available()
    mean, std = ((CIFAR10_MEAN, CIFAR10_STD) if args.dataset == "cifar10"
                 else (CIFAR100_MEAN, CIFAR100_STD))

    print(f"\nchecking CIFAR-C data ({', '.join(args.corruptions)})")
    cifar_c_dir = ensure_cifar_c(args.dataset, args.corruptions, args.data_dir,
                                 keep_tar=args.keep_tar)
    print(f"  ok — {cifar_c_dir}")

    res = {name: {"clean": None, **{c: {} for c in args.corruptions}}
           for name in models}

    # Clean baseline
    clean_loader = DataLoader(
        _CleanCIFAR(args.dataset, args.data_dir, mean, std),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )
    print("\n[clean]")
    for name, model in models.items():
        a = _eval_loader(model, clean_loader, device)
        res[name]["clean"] = a
        print(f"  {name:<25} {a:.2f}%")

    # Corrupted sweep
    for corr in args.corruptions:
        for sev in args.severities:
            loader = DataLoader(
                _CIFARC(cifar_c_dir, corr, sev, mean, std),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=pin,
            )
            print(f"\n[{corr} | severity {sev}]")
            for name, model in models.items():
                a = _eval_loader(model, loader, device)
                res[name][corr][sev] = a
                print(f"  {name:<25} {a:.2f}%")

    _print_robustness_table(res, args.severities, args.corruptions)

    if args.robustness_csv and res:
        with open(args.robustness_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "corruption", "severity", "top1_acc"])
            for name, d in res.items():
                if d.get("clean") is not None:
                    w.writerow([name, "clean", 0, d["clean"]])
                for corr in args.corruptions:
                    for sev, a in d[corr].items():
                        w.writerow([name, corr, sev, a])
        print(f"robustness csv → {args.robustness_csv}")

    return res


def _print_robustness_table(res: dict, sevs: list, corruptions: list):
    methods = list(res.keys())
    print("\n" + "=" * 110)
    print("ROBUSTNESS — CIFAR-C top-1 (%)")
    print("=" * 110)
    for corr in ["clean"] + corruptions:
        cols = ["clean"] if corr == "clean" else [f"sev{s}" for s in sevs]
        head = f"{'Method':<25} | " + " | ".join(f"{h:>7}" for h in cols)
        if corr != "clean":
            head += f" | {'mean':>7} | {'drop':>7}"
        print(f"\n[{corr}]")
        print(head)
        print("-" * len(head))
        for name in methods:
            row = f"{name:<25}"
            if corr == "clean":
                v = res[name].get("clean")
                row += f" | {v:>6.2f}%" if v is not None else f" | {'N/A':>7}"
            else:
                accs = [res[name][corr].get(s) for s in sevs]
                cells = [f"{v:>6.2f}%" if v is not None else f"{'N/A':>7}" for v in accs]
                ok = [float(a) for a in accs if a is not None]
                mu = sum(ok) / len(ok) if ok else None
                cl = res[name].get("clean")
                drop = (cl - mu) if (cl is not None and mu is not None) else None
                row += " | " + " | ".join(cells)
                row += f" | {mu:>6.2f}%" if mu is not None else f" | {'N/A':>7}"
                row += f" | {drop:>6.2f}" if drop is not None else f" | {'N/A':>7}"
            print(row)
    print("=" * 110)


def _plot_robustness_curves(res: dict, dataset: str, save_path: str):
    """Line plot: accuracy vs severity, one panel per corruption."""
    corruptions = [c for c in list(next(iter(res.values())).keys()) if c != "clean"]
    if not corruptions:
        return
    methods = list(res.keys())
    fig, axes = plt.subplots(1, len(corruptions),
                             figsize=(5 * len(corruptions), 4), sharey=True)
    if len(corruptions) == 1:
        axes = [axes]
    for ax, corr in zip(axes, corruptions):
        for name in methods:
            d = res[name][corr]
            sevs = sorted(d.keys())
            ax.plot(sevs, [d[s] for s in sevs], marker="o",
                    label=name.replace(" (Ours)", ""))
        # Reference line at the lowest clean accuracy
        clean_vals = [res[n]["clean"] for n in methods if res[n]["clean"] is not None]
        if clean_vals:
            ax.axhline(min(clean_vals), color="gray", linestyle="--",
                       alpha=0.4, linewidth=0.8)
        ax.set_title(f"{dataset} — {corr.replace('_', ' ')}", fontsize=9)
        ax.set_xlabel("severity")
        if ax is axes[0]:
            ax.set_ylabel("top-1 acc (%)")
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Robustness curves saved to {save_path}")


# ── Entropy ────────────────────────────────────────────────────────────────────

def run_entropy(args, models: dict, test_loader) -> dict:
    """Compute predictive entropy for each loaded model."""
    device = get_device()
    num_classes = 10 if args.dataset == "cifar10" else 100

    print("\n" + "=" * 57)
    print("OUTPUT ENTROPY (test set)")
    print("=" * 57)
    print(f"{'Method':<25} | {'Mean Entropy':>14} | {'Max Entropy':>12}")
    print("-" * 57)

    res = {}
    for name, model in models.items():
        probs, _ = collect_predictions(model, test_loader, device)
        entropy = torch.distributions.Categorical(probs=probs).entropy()
        mean_h = entropy.mean().item()
        max_h  = entropy.max().item()
        res[name] = {"mean": mean_h, "max": max_h}
        print(f"{name:<25} | {mean_h:>14.4f} | {max_h:>12.4f}")

    print(f"\n(Max possible entropy for {num_classes} classes"
          f" = ln({num_classes}) = {math.log(num_classes):.4f})")
    print("=" * 57)

    if getattr(args, "entropy_csv", None) and res:
        with open(args.entropy_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "mean_entropy", "max_entropy"])
            for name, m in res.items():
                w.writerow([name, m["mean"], m["max"]])
        print(f"entropy csv → {args.entropy_csv}")

    return res


# ── Multi-seed ─────────────────────────────────────────────────────────────────

def run_multi_seed(args):
    """Train all methods across multiple seeds and report mean ± std accuracy."""
    from train import train_distillation, train_student_baseline, set_seed

    _default_seeds = [42, 123, 456, 789, 1234]
    seeds = _default_seeds[:args.num_seeds]
    all_results = {m: [] for m in ["student_baseline", "kl_kd", "sinkhorn_kd", "adaptive_sinkhorn_kd"]}

    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        args.seed = seed
        args.mode = "student_baseline"
        all_results["student_baseline"].append(train_student_baseline(args))
        for method in ["kl_kd", "sinkhorn_kd", "adaptive_sinkhorn_kd"]:
            args.method = method
            _, acc = train_distillation(args)
            all_results[method].append(acc)

    print("\n" + "=" * 60)
    print(f"MULTI-SEED RESULTS ({args.num_seeds} seeds)")
    print("=" * 60)
    print(f"{'Method':<25} | {'Mean Acc':>10} | {'Std':>8}")
    print("-" * 50)
    for method, accs in all_results.items():
        print(f"{method:<25} | {np.mean(accs):>9.2f}% | {np.std(accs):>7.2f}%")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare KD methods")

    # Shared
    parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/cifar100")
    parser.add_argument("--teacher", default="resnet110")
    parser.add_argument("--student", default="resnet20")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # Modes
    parser.add_argument("--run_seeds", action="store_true",
                        help="Train all methods across N seeds and report mean ± std.")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alignment", action="store_true",
                        help="Run teacher-student alignment metrics.")
    parser.add_argument("--robustness", action="store_true",
                        help="Run CIFAR-C robustness sweep (~2.9 GB download on first run).")
    parser.add_argument("--no_entropy", action="store_true",
                        help="Skip predictive entropy measurement.")

    # Alignment options
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Softmax temperature for alignment KL / Sinkhorn.")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Sinkhorn entropic regularization for alignment W_eps.")
    parser.add_argument("--sinkhorn_iter", type=int, default=50,
                        help="Max Sinkhorn iterations for alignment.")
    parser.add_argument("--alignment_csv", default=None,
                        help="Save alignment results to this CSV path.")

    # Robustness options
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Severity levels to evaluate (1–5).")
    parser.add_argument("--corruptions", type=str, nargs="+", default=DEFAULT_CORRUPTIONS,
                        help=f"Corruption types. Any of: {', '.join(ALL_CORRUPTIONS)}")
    parser.add_argument("--keep_tar", action="store_true",
                        help="Keep the downloaded CIFAR-C tar after extraction (~2.9 GB).")
    parser.add_argument("--robustness_csv", default=None,
                        help="Save robustness results to this CSV path.")
    parser.add_argument("--entropy_csv", default=None,
                        help="Save entropy results to this CSV path.")

    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"Device: {get_device()}")

    if args.run_seeds:
        run_multi_seed(args)
        return

    # Base evaluation always runs
    results, adapt_ckpt, predictions, models, test_loader = collect_results(args)
    print_comparison_table(results)

    alignment_res  = run_alignment(args, models, test_loader) if args.alignment else None
    robustness_res = run_robustness(args, models) if args.robustness else None
    if not args.no_entropy:
        run_entropy(args, models, test_loader)

    generate_visualizations(args, results, adapt_ckpt, predictions,
                            alignment_res=alignment_res,
                            robustness_res=robustness_res)


if __name__ == "__main__":
    main()
