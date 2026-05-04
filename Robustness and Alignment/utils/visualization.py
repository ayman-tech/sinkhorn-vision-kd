"""Plotting helpers: cost-matrix heatmap, transport plan, training curves,
compression trade-off, cost-matrix evolution.

Forces the Agg backend so this works on headless boxes.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

plt.rcParams.update({"font.size":12, "axes.labelsize":14, "axes.titlesize":14,
                     "xtick.labelsize":10, "ytick.labelsize":10,
                     "legend.fontsize":11, "figure.dpi":150})


def plot_cost_matrix(C, class_names, save_path="cost_matrix.png",
                     title="Learned Cost Matrix C", figsize=None):
    K=len(class_names)
    if figsize is None:
        figsize=(8,7) if K<=20 else (18,16)
    fig, ax = plt.subplots(figsize=figsize)
    show=K<=30

    sns.heatmap(C, ax=ax, cmap="YlOrRd", square=True,
                xticklabels=class_names if show else False,
                yticklabels=class_names if show else False,
                cbar_kws={"label":"Transport cost", "shrink":0.8},
                linewidths=0.1 if K<=20 else 0)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Target class j")
    ax.set_ylabel("Source class i")
    if show:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"cost matrix -> {save_path}")


def plot_transport_plan(pi, class_names, batch_idx=0,
                        save_path="transport_plan.png",
                        title="Optimal Transport Plan"):
    K=len(class_names)
    figsize=(8,7) if K<=20 else (16,14)
    show=K<=30
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pi, ax=ax, cmap="Blues", square=True,
                xticklabels=class_names if show else False,
                yticklabels=class_names if show else False,
                cbar_kws={"label":"Mass transported", "shrink":0.8})
    ax.set_title(f"{title} (sample {batch_idx})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Student class (target)")
    ax.set_ylabel("Teacher class (source)")
    if show:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"transport plan -> {save_path}")


def plot_training_curves(results: Dict[str, Dict[str, List[float]]],
                         save_path="training_curves.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    palette={"KL-KD":"#2196F3", "Fixed-OT-KD":"#FF9800", "Adaptive-OT-KD":"#4CAF50"}
    fallback=plt.cm.tab10.colors

    ax=axes[0]
    for i, (m, h) in enumerate(results.items()):
        c=palette.get(m, fallback[i % len(fallback)])
        ep=range(1, len(h["val_acc"])+1)
        ax.plot(ep, h["val_acc"],   label=f"{m} (val)",   color=c, linewidth=2)
        ax.plot(ep, h["train_acc"], label=f"{m} (train)", color=c,
                linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training Progress: Accuracy", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax=axes[1]
    for i, (m, h) in enumerate(results.items()):
        c=palette.get(m, fallback[i % len(fallback)])
        ep=range(1, len(h["train_loss"])+1)
        ax.plot(ep, h["train_loss"], label=m, color=c, linewidth=2)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.set_title("Training Progress: Loss", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"training curves -> {save_path}")


def plot_compression_tradeoff(results: List[Dict], save_path="compression_tradeoff.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    palette={"Teacher":"#9E9E9E", "Student (no KD)":"#F44336",
             "KL-KD":"#2196F3", "Fixed-OT-KD":"#FF9800",
             "Adaptive-OT-KD":"#4CAF50"}
    markers={"Teacher":"D", "Student (no KD)":"s",
             "KL-KD":"o", "Fixed-OT-KD":"^", "Adaptive-OT-KD":"*"}

    for r in results:
        name=r["method"]
        c=palette.get(name, "#000000")
        mk=markers.get(name, r.get("marker", "o"))
        ax.scatter(r["params_M"], r["top1_acc"],
                   color=c, marker=mk,
                   s=200 if name=="Adaptive-OT-KD" else 120,
                   label=name, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(f'{r["top1_acc"]:.1f}%',
                    (r["params_M"], r["top1_acc"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Compression-Accuracy Trade-off", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"compression tradeoff -> {save_path}")


def plot_cost_matrix_evolution(cost_matrices, epochs, class_names,
                               save_path="cost_evolution.png"):
    n=len(cost_matrices)
    K=cost_matrices[0].shape[0]
    show=K<=20

    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n==1:
        axes=[axes]

    # shared colour scale across snapshots so the eye picks up the actual change.
    vmin=min(C.min() for C in cost_matrices)
    vmax=max(C.max() for C in cost_matrices)

    for ax, C, ep in zip(axes, cost_matrices, epochs):
        sns.heatmap(C, ax=ax, cmap="YlOrRd", square=True, vmin=vmin, vmax=vmax,
                    xticklabels=class_names if show else False,
                    yticklabels=class_names if show else False)
        ax.set_title(f"Epoch {ep}", fontweight="bold")
        if show:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Cost Matrix Evolution During Training",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"cost evolution -> {save_path}")
