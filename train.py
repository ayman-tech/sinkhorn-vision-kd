"""
Main training script for Sinkhorn OT Knowledge Distillation.

Supports three distillation methods:
    1. kl_kd:               Standard KL-divergence KD (Hinton et al., 2015)
    2. sinkhorn_kd:         Sinkhorn OT-KD with FIXED cost matrix
    3. adaptive_sinkhorn_kd: Sinkhorn OT-KD with LEARNABLE cost matrix [OURS]

Usage examples:
    # Pretrain teacher
    python train.py --mode pretrain_teacher --teacher resnet110 --dataset cifar100

    # Baseline: KL-KD
    python train.py --method kl_kd --teacher resnet110 --student resnet20 --dataset cifar100

    # Fixed OT-KD
    python train.py --method sinkhorn_kd --teacher resnet110 --student resnet20 --dataset cifar100

    # Our method: Adaptive OT-KD
    python train.py --method adaptive_sinkhorn_kd --teacher resnet110 --student resnet20 --dataset cifar100
"""

import argparse
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from models import resnet20, resnet56, resnet110, mobilenetv2
from distillation import KLDistillationLoss, SinkhornDistillationLoss, AdaptiveSinkhornKD
from utils.data_loader import get_cifar_loaders, get_class_names
from utils.metrics import accuracy, count_parameters, estimate_flops, AverageMeter
from utils.visualization import plot_cost_matrix, plot_training_curves


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

MODEL_FACTORY = {
    "resnet20": resnet20,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "mobilenetv2": mobilenetv2,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch not in MODEL_FACTORY:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(MODEL_FACTORY.keys())}")
    if arch == "mobilenetv2":
        return MODEL_FACTORY[arch](num_classes=num_classes, width_mult=1.0)
    return MODEL_FACTORY[arch](num_classes=num_classes)


def build_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ──────────────────────────────────────────────────────────────────────────────
# Teacher Pretraining
# ──────────────────────────────────────────────────────────────────────────────

def pretrain_teacher(args):
    """Pretrain teacher model from scratch on CIFAR."""
    device = get_device()
    set_seed(args.seed)

    num_classes = 10 if args.dataset == "cifar10" else 100
    train_loader, _, test_loader = get_cifar_loaders(
        dataset=args.dataset, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = build_model(args.teacher, num_classes).to(device)
    optimizer = build_optimizer(model, lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, args.pretrain_epochs, warmup_epochs=5)
    criterion = nn.CrossEntropyLoss()

    print(f"Pretraining {args.teacher} on {args.dataset}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'LR':>8}")
    print("-" * 60)

    best_acc = 0.0
    for epoch in range(args.pretrain_epochs):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(logits, labels)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))

        scheduler.step()

        # Evaluate
        test_acc = evaluate_model(model, test_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch+1:>6} | {losses.avg:>10.4f} | {top1.avg:>8.2f}% | {test_acc:>7.2f}% | {lr_now:>8.5f}")

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_dir = args.checkpoint_dir or f"./checkpoints/{args.dataset}"
            save_checkpoint(
                {"arch": args.teacher, "state_dict": model.state_dict(),
                 "num_classes": num_classes, "best_acc": best_acc, "epoch": epoch},
                os.path.join(ckpt_dir, f"{args.dataset}_{args.teacher}_teacher.pth"),
            )

    print(f"\nTeacher pretraining complete. Best accuracy: {best_acc:.2f}%")


def evaluate_model(model: nn.Module, loader, device: torch.device) -> float:
    """Evaluate a model on a data loader, return top-1 accuracy."""
    model.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            acc = accuracy(logits, labels)[0]
            top1.update(acc, images.size(0))
    return top1.avg


# ──────────────────────────────────────────────────────────────────────────────
# Distillation Training
# ──────────────────────────────────────────────────────────────────────────────

def train_distillation(args):
    """Train student with knowledge distillation from teacher."""
    device = get_device()
    set_seed(args.seed)

    num_classes = 10 if args.dataset == "cifar10" else 100
    ckpt_dir = args.checkpoint_dir or f"./checkpoints/{args.dataset}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    need_val = args.method == "adaptive_sinkhorn_kd"
    val_fraction = args.val_fraction if need_val else 0.0

    train_loader, val_loader, test_loader = get_cifar_loaders(
        dataset=args.dataset, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        val_fraction=val_fraction, seed=args.seed,
    )
    val_iter = iter(val_loader) if val_loader is not None else None

    # ── Models ────────────────────────────────────────────────────────────
    teacher = build_model(args.teacher, num_classes).to(device)
    teacher_ckpt_path = args.teacher_path or os.path.join(
        ckpt_dir, f"{args.dataset}_{args.teacher}_teacher.pth"
    )
    if os.path.exists(teacher_ckpt_path):
        ckpt = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["state_dict"])
        print(f"Loaded teacher from {teacher_ckpt_path} (acc: {ckpt.get('best_acc', '?')}%)")
    else:
        print(f"WARNING: Teacher checkpoint not found at {teacher_ckpt_path}")
        print("Training with randomly initialized teacher (results won't be meaningful)")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = build_model(args.student, num_classes).to(device)

    print(f"\nMethod: {args.method}")
    print(f"Teacher: {args.teacher} ({count_parameters(teacher):,} params)")
    print(f"Student: {args.student} ({count_parameters(student):,} params)")
    print(f"FLOPs (student): {estimate_flops(student):,}")

    # ── Distillation criterion ────────────────────────────────────────────
    if args.method == "kl_kd":
        criterion = KLDistillationLoss(
            temperature=args.temperature, alpha=args.alpha,
        )
    elif args.method == "sinkhorn_kd":
        criterion = SinkhornDistillationLoss(
            num_classes=num_classes, temperature=args.temperature,
            lambda_ot=args.lambda_ot, epsilon=args.epsilon,
            max_iter=args.sinkhorn_max_iter, threshold=args.sinkhorn_threshold,
            cost_type=args.cost_type,
        ).to(device)
    elif args.method == "adaptive_sinkhorn_kd":
        criterion = AdaptiveSinkhornKD(
            num_classes=num_classes, temperature=args.temperature,
            lambda_ot=args.lambda_ot, epsilon=args.epsilon,
            max_iter=args.sinkhorn_max_iter, threshold=args.sinkhorn_threshold,
            cost_lr=args.cost_lr, cost_update_freq=args.cost_update_freq,
            cost_grad_clip=args.cost_grad_clip,
        ).to(device)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = build_optimizer(student, args.lr, args.momentum, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    # ── Training loop ─────────────────────────────────────────────────────
    history = {
        "train_acc": [], "val_acc": [], "train_loss": [],
        "ot_loss": [], "kd_loss": [], "ce_loss": [],
    }
    best_acc = 0.0

    header = f"{'Epoch':>6} | {'Loss':>8} | {'OT/KD':>8} | {'CE':>8} | {'Train':>7} | {'Test':>7} | {'LR':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(args.epochs):
        student.train()
        losses = AverageMeter()
        ot_losses = AverageMeter()
        ce_losses = AverageMeter()
        top1 = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # ── Bilevel outer loop: update cost matrix on val data ─────
            if args.method == "adaptive_sinkhorn_kd" and criterion.should_update_cost():
                # Get a validation batch (cycle through val_loader)
                try:
                    val_images, val_labels = next(val_iter)
                except (StopIteration, TypeError):
                    val_iter = iter(val_loader)
                    val_images, val_labels = next(val_iter)
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                # Update C (outer loop) — student gradients DON'T update theta here
                cost_info = criterion.step_cost_matrix(
                    student, teacher, val_images, val_labels,
                )

            # ── Inner loop: update student on training data ───────────
            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            result = criterion(student_logits, teacher_logits, labels)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.method == "adaptive_sinkhorn_kd":
                criterion.increment_step()

            # ── Logging ───────────────────────────────────────────────
            acc = accuracy(student_logits, labels)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))

            ot_val = result.get("ot_loss", result.get("kd_loss", torch.tensor(0.0)))
            if isinstance(ot_val, torch.Tensor):
                ot_val = ot_val.item()
            ot_losses.update(ot_val, images.size(0))
            ce_losses.update(result["ce_loss"].item(), images.size(0))

            pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.1f}%")

        scheduler.step()

        # ── Evaluation ────────────────────────────────────────────────────
        test_acc = evaluate_model(student, test_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"{epoch+1:>6} | {losses.avg:>8.4f} | {ot_losses.avg:>8.4f} | "
              f"{ce_losses.avg:>8.4f} | {top1.avg:>6.2f}% | {test_acc:>6.2f}% | {lr_now:>8.5f}")

        history["train_acc"].append(top1.avg)
        history["val_acc"].append(test_acc)
        history["train_loss"].append(losses.avg)
        history["ot_loss"].append(ot_losses.avg)
        history["ce_loss"].append(ce_losses.avg)

        # ── Checkpointing ─────────────────────────────────────────────────
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                "method": args.method, "epoch": epoch,
                "student_arch": args.student, "teacher_arch": args.teacher,
                "state_dict": student.state_dict(),
                "best_acc": best_acc, "history": history,
            }
            if args.method == "adaptive_sinkhorn_kd":
                state["cost_matrix"] = criterion.get_cost_matrix_numpy()
            save_checkpoint(state, os.path.join(ckpt_dir, f"{args.method}_best.pth"))

        if (epoch + 1) % args.save_freq == 0:
            state = {
                "method": args.method, "epoch": epoch,
                "student_arch": args.student, "state_dict": student.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_acc": best_acc, "history": history,
            }
            if args.method == "adaptive_sinkhorn_kd":
                state["cost_matrix"] = criterion.get_cost_matrix_numpy()
            save_checkpoint(state, os.path.join(ckpt_dir, f"{args.method}_epoch{epoch+1}.pth"))

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")

    # ── Save final results + visualizations ───────────────────────────────
    save_checkpoint(
        {"history": history, "best_acc": best_acc, "args": vars(args)},
        os.path.join(ckpt_dir, f"{args.method}_results.pth"),
    )

    if args.method == "adaptive_sinkhorn_kd":
        C = criterion.get_cost_matrix_numpy()
        class_names = get_class_names(args.dataset)
        plot_cost_matrix(
            C, class_names,
            save_path=os.path.join(ckpt_dir, "learned_cost_matrix.png"),
            title=f"Learned Cost Matrix ({args.dataset.upper()})",
        )

    return history, best_acc


# ──────────────────────────────────────────────────────────────────────────────
# Train student WITHOUT distillation (baseline)
# ──────────────────────────────────────────────────────────────────────────────

def train_student_baseline(args):
    """Train student from scratch without any distillation (lower bound)."""
    device = get_device()
    set_seed(args.seed)

    num_classes = 10 if args.dataset == "cifar10" else 100
    ckpt_dir = args.checkpoint_dir or f"./checkpoints/{args.dataset}"

    train_loader, _, test_loader = get_cifar_loaders(
        dataset=args.dataset, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    student = build_model(args.student, num_classes).to(device)
    optimizer = build_optimizer(student, args.lr, args.momentum, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {args.student} from scratch (no distillation)")
    print(f"Parameters: {count_parameters(student):,}")

    best_acc = 0.0
    for epoch in range(args.epochs):
        student.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = student(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(logits, labels)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))

        scheduler.step()
        test_acc = evaluate_model(student, test_loader, device)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:>4} | Loss: {losses.avg:.4f} | "
                  f"Train: {top1.avg:.2f}% | Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                {"arch": args.student, "state_dict": student.state_dict(), "best_acc": best_acc},
                os.path.join(ckpt_dir, f"{args.student}_no_kd_best.pth"),
            )

    print(f"Student baseline complete. Best accuracy: {best_acc:.2f}%")
    return best_acc


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sinkhorn OT Knowledge Distillation for Vision Models"
    )

    # Mode
    parser.add_argument("--mode", type=str, default="distill",
                        choices=["pretrain_teacher", "distill", "student_baseline"],
                        help="Training mode.")

    # Method
    parser.add_argument("--method", type=str, default="adaptive_sinkhorn_kd",
                        choices=["kl_kd", "sinkhorn_kd", "adaptive_sinkhorn_kd"],
                        help="Distillation method.")

    # Architecture
    parser.add_argument("--teacher", type=str, default="resnet110",
                        choices=["resnet56", "resnet110"])
    parser.add_argument("--student", type=str, default="resnet20",
                        choices=["resnet20", "resnet56", "mobilenetv2"])

    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Distillation
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="KD loss weight for KL-KD method.")
    parser.add_argument("--lambda_ot", type=float, default=0.5,
                        help="OT loss weight for Sinkhorn methods.")

    # Sinkhorn
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Sinkhorn entropic regularization.")
    parser.add_argument("--sinkhorn_max_iter", type=int, default=50)
    parser.add_argument("--sinkhorn_threshold", type=float, default=1e-3)
    parser.add_argument("--cost_type", type=str, default="uniform",
                        choices=["uniform", "label_distance", "random"],
                        help="Fixed cost matrix type (for sinkhorn_kd).")

    # Adaptive (learnable cost)
    parser.add_argument("--cost_lr", type=float, default=0.01,
                        help="Learning rate for cost matrix C.")
    parser.add_argument("--cost_update_freq", type=int, default=10,
                        help="Update C every K training steps.")
    parser.add_argument("--cost_grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of train data for cost matrix validation.")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--teacher_path", type=str, default=None,
                        help="Path to pretrained teacher checkpoint.")
    parser.add_argument("--save_freq", type=int, default=10)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # Config file (overrides CLI args)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file.")

    return parser.parse_args()


def load_config(args):
    """Override args with values from YAML config file if provided."""
    if args.config is None:
        return args
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Flatten nested config into args namespace
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
        else:
            continue

    for key, val in flat.items():
        if hasattr(args, key):
            setattr(args, key, val)

    return args


def main():
    args = parse_args()
    args = load_config(args)

    print("=" * 60)
    print("Sinkhorn OT Knowledge Distillation for Vision Models")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"Seed: {args.seed}")
    print()

    if args.mode == "pretrain_teacher":
        pretrain_teacher(args)
    elif args.mode == "student_baseline":
        train_student_baseline(args)
    elif args.mode == "distill":
        train_distillation(args)


if __name__ == "__main__":
    main()
