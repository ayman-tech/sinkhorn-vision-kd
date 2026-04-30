"""
Visualization utilities for the Sinkhorn OT-KD project.

Generates publication-quality figures for:
  1. Learned cost matrix C (heatmap showing class semantic geometry)
  2. Optimal transport plans (showing how probability mass is moved)
  3. Training curves (accuracy/loss comparison across methods)
  4. Compression trade-off (Pareto frontier: model size vs accuracy)
  5. Reliability diagrams (calibration analysis)
  6. Class clustering via t-SNE of cost matrix rows
  7. Nearest-neighbor costs table
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, List, Optional, Tuple

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


# ──────────────────────────────────────────────────────────────────────────────
# Calibration: Reliability Diagrams
# ──────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    predictions: Dict[str, Dict[str, np.ndarray]],
    n_bins: int = 15,
    save_path: str = "reliability_diagram.png",
):
    """Multi-panel reliability diagram comparing calibration across methods.

    Each panel shows mean confidence (x) vs actual accuracy (y) per bin,
    with a perfect-calibration diagonal for reference. ECE is annotated on
    each panel.

    Args:
        predictions: {method_label: {"probs": ndarray(N,C), "labels": ndarray(N,)}}
        n_bins: Number of confidence bins.
        save_path: Where to save the figure.
    """
    methods = list(predictions.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    method_colors = {
        "KL-KD": "#2196F3",
        "Fixed-OT-KD": "#FF9800",
        "Adaptive-OT-KD (Ours)": "#4CAF50",
        "Student (no KD)": "#F44336",
        "Teacher": "#9E9E9E",
    }
    default_colors = list(plt.cm.tab10.colors)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for ax_idx, (ax, method) in enumerate(zip(axes, methods)):
        data = predictions[method]
        probs = np.array(data["probs"])
        labels = np.array(data["labels"])

        confidences = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)

        bin_accs, bin_confs, bin_counts = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences >= lo) & (confidences <= hi)
            if mask.sum() == 0:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                bin_counts.append(0)
            else:
                bin_accs.append(correct[mask].mean())
                bin_confs.append(confidences[mask].mean())
                bin_counts.append(int(mask.sum()))

        bin_accs = np.array(bin_accs)
        bin_confs = np.array(bin_confs)
        valid = ~np.isnan(bin_accs)

        color = method_colors.get(method, default_colors[ax_idx % len(default_colors)])

        # Diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, zorder=1)

        # Confidence bars
        ax.bar(
            bin_centers[valid], bin_accs[valid],
            width=1.0 / n_bins, align="center",
            color=color, alpha=0.8, edgecolor="white", linewidth=0.5, zorder=2,
        )

        # Gap shading (over/under-confidence)
        ax.bar(
            bin_centers[valid], bin_centers[valid] - bin_accs[valid],
            bottom=bin_accs[valid],
            width=1.0 / n_bins, align="center",
            color="red", alpha=0.2, edgecolor="none", zorder=2,
        )

        n_total = sum(bin_counts)
        ece = sum(
            (cnt / n_total) * abs(acc - conf)
            for cnt, acc, conf in zip(bin_counts, bin_accs, bin_confs)
            if cnt > 0 and not np.isnan(acc)
        )

        ax.set_title(f"{method}\nECE = {ece:.3f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Accuracy", fontsize=10)
    fig.suptitle("Reliability Diagrams", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagram saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Interpretability: Class Clustering (t-SNE of cost matrix rows)
# ──────────────────────────────────────────────────────────────────────────────

# CIFAR-100 class index (alphabetical, matching data_loader.py) → superclass index (0–19)
_CIFAR100_SUPERCLASS_IDX = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

_CIFAR100_SUPERCLASS_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit & veg.",
    "household devices", "furniture", "insects", "large carnivores",
    "man-made outdoor", "natural outdoor", "large herbivores", "medium mammals",
    "invertebrates", "people", "reptiles", "small mammals", "trees",
    "vehicles 1", "vehicles 2",
]


def plot_class_clustering(
    C: np.ndarray,
    class_names: List[str],
    save_path: str = "class_clustering.png",
    title: str = "Class Geometry from Learned Cost Matrix (t-SNE)",
):
    """t-SNE embedding of cost matrix rows revealing class semantic structure.

    Each class is a point positioned by its cost profile against all others.
    For CIFAR-100, colors encode the 20 superclasses. For CIFAR-10, each
    class gets its own color and label.

    Args:
        C: Cost matrix, shape (num_classes, num_classes).
        class_names: List of class name strings.
        save_path: Where to save the figure.
        title: Plot title.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not found — skipping class clustering. Install with: pip install scikit-learn")
        return

    num_classes = len(class_names)
    is_cifar100 = (num_classes == 100)

    emb = TSNE(
        n_components=2,
        perplexity=min(30, num_classes - 1),
        random_state=42,
        max_iter=2000,
        init="pca",
    ).fit_transform(C)

    fig, ax = plt.subplots(figsize=(10, 8))

    if is_cifar100:
        cmap = plt.cm.get_cmap("tab20", 20)
        for cls_idx, (x, y) in enumerate(emb):
            sc = _CIFAR100_SUPERCLASS_IDX[cls_idx]
            ax.scatter(x, y, color=cmap(sc), s=55, zorder=3,
                       edgecolors="white", linewidth=0.3)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cmap(i), markersize=8, label=name)
            for i, name in enumerate(_CIFAR100_SUPERCLASS_NAMES)
        ]
        ax.legend(handles=handles, loc="best", fontsize=7, ncol=2,
                  framealpha=0.85, title="Superclass", title_fontsize=8)
    else:
        cmap = plt.cm.get_cmap("tab10", num_classes)
        for cls_idx, (x, y) in enumerate(emb):
            ax.scatter(x, y, color=cmap(cls_idx), s=100, zorder=3,
                       edgecolors="white", linewidth=0.5)
            ax.annotate(class_names[cls_idx], (x, y),
                        textcoords="offset points", xytext=(5, 3), fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Class clustering plot saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Interpretability: Nearest-Neighbor Costs Table
# ──────────────────────────────────────────────────────────────────────────────

def plot_nearest_neighbor_costs(
    C: np.ndarray,
    class_names: List[str],
    k: int = 5,
    save_path: str = "nearest_neighbor_costs.png",
    txt_path: Optional[str] = None,
):
    """Table showing the k nearest-neighbor classes by learned cost.

    For CIFAR-100 the figure shows one representative class per superclass
    (20 rows). The full 100-class table is saved to txt_path if given.

    Args:
        C: Cost matrix, shape (num_classes, num_classes).
        class_names: List of class name strings.
        k: Number of nearest neighbors per class.
        save_path: Where to save the figure (PNG).
        txt_path: Optional path for the full plain-text table.
    """
    num_classes = len(class_names)
    is_cifar100 = (num_classes == 100)

    # Build full nearest-neighbor list for every class
    all_rows: List[Tuple[str, List[Tuple[str, float]]]] = []
    for i in range(num_classes):
        costs = C[i].copy()
        costs[i] = np.inf
        nn_idx = np.argsort(costs)[:k]
        all_rows.append((class_names[i], [(class_names[j], float(C[i, j])) for j in nn_idx]))

    # Optionally save full table to text
    if txt_path:
        with open(txt_path, "w") as f:
            header = f"{'Class':<22}" + "".join(f"  NN{n+1:<2} (cost)" for n in range(k))
            f.write(header + "\n" + "-" * len(header) + "\n")
            for cls_name, neighbors in all_rows:
                nn_str = "".join(f"  {n:<18} ({c:.3f})" for n, c in neighbors)
                f.write(f"{cls_name:<22}{nn_str}\n")
        print(f"Full nearest-neighbor table saved to {txt_path}")

    # For the figure: one representative class per superclass (CIFAR-100),
    # or all classes (CIFAR-10)
    if is_cifar100:
        seen_sc: set = set()
        rep_indices: List[int] = []
        for cls_idx, sc in enumerate(_CIFAR100_SUPERCLASS_IDX):
            if sc not in seen_sc:
                seen_sc.add(sc)
                rep_indices.append(cls_idx)
        rep_indices.sort()
    else:
        rep_indices = list(range(num_classes))

    rep_rows = [all_rows[i] for i in rep_indices]
    n_rep = len(rep_rows)

    col_labels = ["Class"] + [f"Neighbor {n + 1}" for n in range(k)]
    table_data = [
        [cls_name] + [f"{n}  ({c:.3f})" for n, c in neighbors]
        for cls_name, neighbors in rep_rows
    ]

    fig_h = max(4.0, 0.38 * n_rep + 1.5)
    fig_w = max(10, 2.8 * k + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")

    for row_i in range(1, n_rep + 1):
        bg = "#F5F5F5" if row_i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[row_i, j].set_facecolor(bg)

    ax.set_title(
        f"Top-{k} Nearest Neighbors by Learned Cost Matrix",
        fontsize=12, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Nearest-neighbor table saved to {save_path}")
