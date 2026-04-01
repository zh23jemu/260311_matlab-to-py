from __future__ import annotations

"""实验结果出图函数。

本模块只关心“如何把已有数据画出来”，不负责训练、评估或文件结构管理。
主流程在拿到矩阵、标签、训练历史之后，统一调用这里的函数导出图片。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    """统一保存图像。

    这里集中处理保存细节：
    - 自动创建父目录
    - 调用 `tight_layout()` 压紧版面
    - 统一使用固定 dpi 和边距
    - 保存后立即关闭 figure，避免长时间运行时图对象堆积
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_lines(data: np.ndarray, title: str, path: Path, max_rows: int = 400) -> None:
    """绘制矩阵前若干行的折线图。

    常用于展示：
    - 原始数据的多变量变化趋势
    - 标准化后数据的形态
    - 训练特征或测试特征的整体分布情况

    `max_rows` 用来限制绘制样本数，避免线条过多导致图像不可读。
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data[:max_rows])
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_label_sequence(labels: np.ndarray, title: str, path: Path) -> None:
    """绘制标签序列图。

    这个图主要用于直观展示测试样本在拼接后的标签排列情况，
    对应文档里的标签序列可视化。
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(labels, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Label")
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_training_history(losses: list[float], accuracies: list[float], title: str, path: Path) -> None:
    """绘制训练历史图。

    输出是一张包含两个子图的图片：
    - 左侧是 loss 曲线
    - 右侧是 accuracy 曲线

    这样 DAE 和分类器都可以复用同一套出图逻辑。
    对 DAE 来说，accuracy 会是一条 0 线，占位但不影响图结构统一。
    """
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
    """绘制按行归一化的混淆热力图。

    输入矩阵通常来自 `build_confusion()`，已经完成按行归一化。
    图中每个格子的数值表示：
    “某个真实故障被预测成某个类别的比例”。

    对角线越高，说明该类别识别越准确；非对角线越高，说明该类别越容易与其他类混淆。
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)

    # 在每个格子里写出数值，便于报告阅读时直接查看比例，
    # 不必再额外对照原始矩阵。
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_network_diagram(path: Path) -> None:
    """绘制 DAE 网络结构示意图。

    这不是从真实 PyTorch 图自动导出的结构图，而是为了报告展示而手工画的简化示意图。
    它强调的是层级尺寸变化和整体编码/解码结构，而不是底层计算图细节。
    """
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
    # 先画出每一层对应的盒子，突出维度变化过程。
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

    # 再用箭头按顺序连接各层，形成从输入到输出的结构示意。
    for start, end in zip(boxes, boxes[1:]):
        ax.annotate(
            "",
            xy=(end[1] - 0.05, 0.55),
            xytext=(start[1] + 0.05, 0.55),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.5},
        )
    ax.set_title("Figure 1.10 Network Structure")
    _save(fig, path)
