from __future__ import annotations

"""训练相关逻辑集中入口。

这里把 DAE 训练、分类器训练以及共享训练工具放在一个文件中，
方便按“训练”这个主题集中阅读。
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from te_dae.data_standar import TRAIN_FAULT_IDS


@dataclass
class TrainHistory:
    """保存训练历史。

    - `losses`：每个 epoch 的平均损失
    - `accuracies`：每个 epoch 的准确率

    对 DAE 来说没有分类准确率，因此这里会放 0.0 作为占位，
    这样绘图函数就可以复用统一的输入结构。
    """

    losses: list[float]
    accuracies: list[float]


def set_seed(seed: int) -> None:
    """统一固定随机种子。

    当前项目主要随机性来源有两个：
    - NumPy：给 DAE 训练数据加噪
    - PyTorch：参数初始化、DataLoader 打乱顺序

    这里集中设置后，实验会更容易复现。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(features: np.ndarray, targets: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    """把 NumPy 数组包装成 `DataLoader`。

    这个函数同时服务于两类任务：
    - DAE 重建任务：输入和目标都是特征矩阵
    - 分类任务：输入是特征矩阵，目标是离散标签

    统一封装后，训练循环里的代码会更简洁。
    """
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    if targets is None:
        # 如果没有显式传目标，就默认构造“输入=目标”的重建任务数据集。
        dataset = TensorDataset(feature_tensor, feature_tensor)
    else:
        target_tensor = torch.from_numpy(targets)
        dataset = TensorDataset(feature_tensor, target_tensor)

    # 固定 generator 的种子，是为了让 DataLoader 的 shuffle 顺序也尽量稳定。
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def should_log(epoch: int, epochs: int, log_interval: int) -> bool:
    """判断当前 epoch 是否应该打印日志。

    保证：
    - 第一轮一定打印，便于确认训练确实开始了
    - 最后一轮一定打印，便于查看收尾结果
    - 中间按固定周期打印，防止日志太密
    """
    return epoch == 1 or epoch == epochs or epoch % log_interval == 0


def train_autoencoder(
    model: nn.Module,
    noisy_features: np.ndarray,
    clean_features: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_interval: int = 100,
) -> TrainHistory:
    """训练降噪自编码器。

    这里的训练目标是让模型学会：
    “从带噪输入中恢复出干净信号”。

    训练完成后，模型的编码部分就会变成一个更稳健的特征提取器，
    供后续分类器使用。
    """
    loader = build_loader(noisy_features, clean_features, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.MSELoss()
    model.to(device)
    history = TrainHistory(losses=[], accuracies=[])

    print(
        f"[DAE] start training: samples={len(loader.dataset)}, batch_size={batch_size}, "
        f"epochs={epochs}, lr={learning_rate}, device={device}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for batch_x, batch_y in loader:
            # batch_x 是带噪输入，batch_y 是对应的干净目标。
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)

        epoch_loss = running_loss / max(total, 1)
        history.losses.append(epoch_loss)
        # DAE 不是分类任务，这里没有真正意义上的 accuracy。
        history.accuracies.append(0.0)
        if should_log(epoch, epochs, log_interval):
            print(f"[DAE] epoch {epoch}/{epochs} loss={epoch_loss:.6f}")
    return history


def extract_encoded_features(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    mode: str = "fc3_linear",
) -> np.ndarray:
    """使用训练好的 DAE 提取编码特征。

    这里不会更新参数，只做前向推理。
    输出的就是后续分类器真正吃到的输入特征。
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        encoded = model.extract_features(tensor, mode=mode).cpu().numpy().astype(np.float32)
    return encoded


def encode_features(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    mode: str = "fc3_linear",
) -> np.ndarray:
    """`extract_encoded_features` 的短别名。"""
    return extract_encoded_features(model, features, device=device, mode=mode)


def stack_classifier_features(
    feature_map: dict[int, np.ndarray],
    label_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """把各故障特征按固定顺序拼接成分类器训练输入。

    这个步骤的重点是“顺序固定”。
    因为只要拼接顺序变了，标签映射、混淆矩阵含义和最终每类准确率解释
    都会跟着变。
    """
    features = np.vstack([feature_map[fault_id] for fault_id in TRAIN_FAULT_IDS]).astype(np.float32)
    labels = np.concatenate(
        [np.full(feature_map[fault_id].shape[0], label_map[fault_id] - 1, dtype=np.int64) for fault_id in TRAIN_FAULT_IDS]
    )
    return features, labels


def train_classifier(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_interval: int = 20,
) -> TrainHistory:
    """训练故障分类器。

    输入特征默认已经完成两件事：
    - 经过 DAE 编码
    - 再按 d00 编码统计量做了第二次标准化

    因此这一阶段只专注于学习“编码特征 -> 故障类别”的监督映射。
    """
    loader = build_loader(features, labels.astype(np.int64), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    history = TrainHistory(losses=[], accuracies=[])

    print(
        f"[CLF] start training: samples={len(loader.dataset)}, batch_size={batch_size}, "
        f"epochs={epochs}, lr={learning_rate}, device={device}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            # 分类任务使用交叉熵损失，目标是 0-based 的类别编号。
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_x.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        history.losses.append(epoch_loss)
        history.accuracies.append(epoch_acc)
        if should_log(epoch, epochs, log_interval):
            print(f"[CLF] epoch {epoch}/{epochs} loss={epoch_loss:.6f} acc={epoch_acc:.4f}")
    return history


def predict_classes(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    """对输入特征做分类预测。

    返回结果会从模型内部的 0-based 编号重新映射成 1-based，
    这样更方便和项目里的 F1/F2/F13 这类故障编号口径对齐。
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = model(tensor)
        predictions = logits.argmax(dim=1).cpu().numpy() + 1
    return predictions.astype(np.int64)


def build_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    """构造按行归一化的混淆矩阵。

    按行归一化后，每一行都可以理解为：
    “某个真实类别被预测成各类别的比例分布”。

    这样不同类别样本数即使不完全相同，也仍然能直接比较识别效果。
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


def _save(fig: plt.Figure, path: Path) -> None:
    """统一保存图像。

    这里集中处理三个细节：
    - 自动创建父目录
    - 保存前执行 `tight_layout()`
    - 保存后立即关闭图对象，避免长时间运行时内存堆积
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_lines(data: np.ndarray, title: str, path: Path, max_rows: int = 400) -> None:
    """绘制矩阵前若干行的折线图。

    常用于查看原始数据、标准化结果或者训练特征的大致形态。
    `max_rows` 用来避免一次画太多样本导致图像完全糊成一团。
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

    这张图的意义主要不是“好看”，而是帮助确认测试样本在拼接之后
    的标签排列顺序是否符合预期。
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

    统一输出左右两个子图：
    - 左边看 loss 收敛情况
    - 右边看 accuracy 变化趋势

    即使 DAE 没有真实 accuracy，也保留相同版式，方便报告保持统一。
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

    对角线越亮，说明该类识别越准确；
    非对角线越亮，说明该类越容易和别的故障混淆。
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)

    # 在每个格子上写数值，是为了做报告时不用再另外对照原矩阵。
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_network_diagram(path: Path) -> None:
    """绘制 DAE 网络结构示意图。

    这张图不是从 PyTorch 自动导出的真实计算图，
    而是为了报告展示而手工绘制的“结构示意图”。
    它强调的是层宽变化和编码/解码对称关系。
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
    # 先画出每一层对应的盒子，突出 50 -> 40 -> 50 的结构变化。
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

    # 再用箭头把层级连接起来，让读者一眼看到前向流向。
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


__all__ = [
    "TrainHistory",
    "build_confusion",
    "build_loader",
    "encode_features",
    "extract_encoded_features",
    "predict_classes",
    "plot_heatmap",
    "plot_label_sequence",
    "plot_matrix_lines",
    "plot_network_diagram",
    "plot_training_history",
    "set_seed",
    "should_log",
    "stack_classifier_features",
    "train_autoencoder",
    "train_classifier",
]
