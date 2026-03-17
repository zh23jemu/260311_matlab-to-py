from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_lines(data: np.ndarray, title: str, path: Path, max_rows: int = 400) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data[:max_rows])
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_label_sequence(labels: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(labels, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Label")
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_training_history(losses: list[float], accuracies: list[float], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses)
    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.2)
    axes[1].plot(accuracies)
    axes[1].set_title(f"{title} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.2)
    _save(fig, path)


def plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_network_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = [
        ("Input\n50", 0.05),
        ("FC\n50", 0.18),
        ("FC\n45", 0.31),
        ("Bottleneck\n40", 0.44),
        ("FC\n45", 0.57),
        ("FC\n50", 0.70),
        ("Output\n50", 0.83),
    ]
    for text, xpos in boxes:
        ax.text(
            xpos,
            0.55,
            text,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f2e8c9", "edgecolor": "#7b6d42"},
            transform=ax.transAxes,
        )
    for start, end in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.05, 0.55), xytext=(start[1] + 0.05, 0.55), xycoords=ax.transAxes,
                    arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_title("Figure 4.10 Network Structure")
    _save(fig, path)
