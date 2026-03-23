from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    """统一保存图像，避免每个绘图函数重复写落盘逻辑。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_lines(data: np.ndarray, title: str, path: Path, max_rows: int = 400) -> None:
    """画矩阵前若干行的折线，用来对应文档里的数据曲线图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data[:max_rows])
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_label_sequence(labels: np.ndarray, title: str, path: Path) -> None:
    """绘制标签序列图，对应文档里的 Figure 4.9。"""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(labels, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Label")
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_training_history(losses: list[float], accuracies: list[float], title: str, path: Path) -> None:
    """同时画 loss 和 accuracy，两张子图放在同一张图上。"""
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
    """画按行归一化的热力图，用来展示各故障的混淆情况。"""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)

    # 在每个格子中写入数值，方便直接阅读具体混淆比例。
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_network_diagram(path: Path) -> None:
    """画一个示意性的 DAE 结构图，对应文档里的网络结构图。"""
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

    # 用箭头把各层串起来，形成简化的网络流程图。
    for start, end in zip(boxes, boxes[1:]):
        ax.annotate(
            "",
            xy=(end[1] - 0.05, 0.55),
            xytext=(start[1] + 0.05, 0.55),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.5},
        )
    ax.set_title("Figure 4.10 Network Structure")
    _save(fig, path)
