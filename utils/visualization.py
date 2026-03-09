"""
Visualization utilities for the Sinkhorn OT-KD project.

Generates publication-quality figures for:
  1. Learned cost matrix C (heatmap showing class semantic geometry)
  2. Optimal transport plans (showing how probability mass is moved)
  3. Training curves (accuracy/loss comparison across methods)
  4. Compression trade-off (Pareto frontier: model size vs accuracy)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, List, Optional

# Use non-interactive backend for server environments
matplotlib.use("Agg")

# Publication style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})


def plot_cost_matrix(
    C: np.ndarray,
    class_names: List[str],
    save_path: str = "cost_matrix.png",
    title: str = "Learned Cost Matrix C",
    figsize: Optional[tuple] = None,
):
    """Plot the cost matrix as a heatmap with class labels.

    The cost matrix C[i][j] encodes "how wrong" it is to transport probability
    mass from class i to class j. After training, semantically similar classes
    should have LOW cost (dark regions), while dissimilar classes should have
    HIGH cost (bright regions).

    For CIFAR-100 we expect to see block-diagonal structure: animal classes
    cluster together, vehicle classes cluster together, etc.

    Args:
        C: Cost matrix of shape (num_classes, num_classes). Should be symmetric
           with zero diagonal.
        class_names: List of class name strings.
        save_path: Where to save the figure.
        title: Plot title.
        figsize: Figure size. Auto-scaled if None.
    """
    num_classes = len(class_names)
    if figsize is None:
        figsize = (8, 7) if num_classes <= 20 else (18, 16)

    fig, ax = plt.subplots(figsize=figsize)

    # For large matrices (CIFAR-100), skip tick labels to avoid clutter
    show_labels = num_classes <= 30

    sns.heatmap(
        C,
        ax=ax,
        cmap="YlOrRd",
        square=True,
        xticklabels=class_names if show_labels else False,
        yticklabels=class_names if show_labels else False,
        cbar_kws={"label": "Transport cost", "shrink": 0.8},
        linewidths=0.1 if num_classes <= 20 else 0,
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Target class j")
    ax.set_ylabel("Source class i")

    if show_labels:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Cost matrix heatmap saved to {save_path}")


def plot_transport_plan(
    pi: np.ndarray,
    class_names: List[str],
    batch_idx: int = 0,
    save_path: str = "transport_plan.png",
    title: str = "Optimal Transport Plan",
):
    """Visualize the optimal transport plan pi for a single sample.

    The transport plan pi[i][j] shows how much probability mass is moved
    from teacher class i to student class j. A well-distilled student
    should have pi concentrated near the diagonal.

    Args:
        pi: Transport plan matrix of shape (num_classes, num_classes).
        class_names: Class name strings.
        batch_idx: Index of the sample in the batch (for labeling).
        save_path: Where to save the figure.
        title: Plot title.
    """
    num_classes = len(class_names)
    figsize = (8, 7) if num_classes <= 20 else (16, 14)
    show_labels = num_classes <= 30

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pi,
        ax=ax,
        cmap="Blues",
        square=True,
        xticklabels=class_names if show_labels else False,
        yticklabels=class_names if show_labels else False,
        cbar_kws={"label": "Mass transported", "shrink": 0.8},
    )

    ax.set_title(f"{title} (sample {batch_idx})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Student class (target)")
    ax.set_ylabel("Teacher class (source)")

    if show_labels:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Transport plan saved to {save_path}")


def plot_training_curves(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str = "training_curves.png",
):
    """Plot training curves comparing KL-KD, Fixed-OT-KD, and Adaptive-OT-KD.

    Creates a 2x1 figure with accuracy curves (top) and loss curves (bottom).

    Args:
        results: Dictionary mapping method names to their metrics:
            {
                "KL-KD": {
                    "train_acc": [...], "val_acc": [...],
                    "train_loss": [...], "val_loss": [...]
                },
                "Fixed-OT-KD": { ... },
                "Adaptive-OT-KD": { ... },
            }
        save_path: Where to save the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {"KL-KD": "#2196F3", "Fixed-OT-KD": "#FF9800", "Adaptive-OT-KD": "#4CAF50"}
    default_colors = plt.cm.tab10.colors

    # Top: Accuracy curves
    ax = axes[0]
    for i, (method, metrics) in enumerate(results.items()):
        color = colors.get(method, default_colors[i % len(default_colors)])
        epochs = range(1, len(metrics["val_acc"]) + 1)
        ax.plot(epochs, metrics["val_acc"], label=f"{method} (val)", color=color, linewidth=2)
        ax.plot(epochs, metrics["train_acc"], label=f"{method} (train)",
                color=color, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training Progress: Accuracy", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Bottom: Loss curves
    ax = axes[1]
    for i, (method, metrics) in enumerate(results.items()):
        color = colors.get(method, default_colors[i % len(default_colors)])
        epochs = range(1, len(metrics["train_loss"]) + 1)
        ax.plot(epochs, metrics["train_loss"], label=method, color=color, linewidth=2)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.set_title("Training Progress: Loss", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def plot_compression_tradeoff(
    results: List[Dict],
    save_path: str = "compression_tradeoff.png",
):
    """Plot model size vs accuracy (Pareto frontier).

    Shows each method as a point on a scatter plot. The ideal position is
    upper-left (high accuracy, low parameters).

    Args:
        results: List of dicts, each with keys:
            {"method": str, "params_M": float, "top1_acc": float, "marker": str}
        save_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "Teacher": "#9E9E9E",
        "Student (no KD)": "#F44336",
        "KL-KD": "#2196F3",
        "Fixed-OT-KD": "#FF9800",
        "Adaptive-OT-KD": "#4CAF50",
    }
    markers = {
        "Teacher": "D",
        "Student (no KD)": "s",
        "KL-KD": "o",
        "Fixed-OT-KD": "^",
        "Adaptive-OT-KD": "*",
    }

    for r in results:
        method = r["method"]
        color = colors.get(method, "#000000")
        marker = markers.get(method, r.get("marker", "o"))
        ax.scatter(
            r["params_M"], r["top1_acc"],
            color=color, marker=marker,
            s=200 if method == "Adaptive-OT-KD" else 120,
            label=method, zorder=5, edgecolors="black", linewidth=0.5,
        )
        ax.annotate(
            f'{r["top1_acc"]:.1f}%',
            (r["params_M"], r["top1_acc"]),
            textcoords="offset points", xytext=(8, 5), fontsize=10,
        )

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Compression-Accuracy Trade-off", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Compression trade-off plot saved to {save_path}")


def plot_cost_matrix_evolution(
    cost_matrices: List[np.ndarray],
    epochs: List[int],
    class_names: List[str],
    save_path: str = "cost_evolution.png",
):
    """Show how the learned cost matrix C evolves during training.

    Useful for understanding how the model learns class relationships over time.

    Args:
        cost_matrices: List of cost matrices at different epochs.
        epochs: Corresponding epoch numbers.
        class_names: Class name strings.
        save_path: Where to save the figure.
    """
    n = len(cost_matrices)
    num_classes = cost_matrices[0].shape[0]
    show_labels = num_classes <= 20

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    # Use consistent color scale across all subplots
    vmin = min(C.min() for C in cost_matrices)
    vmax = max(C.max() for C in cost_matrices)

    for ax, C, epoch in zip(axes, cost_matrices, epochs):
        sns.heatmap(
            C, ax=ax, cmap="YlOrRd", square=True, vmin=vmin, vmax=vmax,
            xticklabels=class_names if show_labels else False,
            yticklabels=class_names if show_labels else False,
        )
        ax.set_title(f"Epoch {epoch}", fontweight="bold")
        if show_labels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Cost Matrix Evolution During Training", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Cost matrix evolution saved to {save_path}")
