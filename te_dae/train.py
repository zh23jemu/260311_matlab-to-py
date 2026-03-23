from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainHistory:
    """保存每个 epoch 的损失和准确率，供后续画图使用。"""

    losses: list[float]
    accuracies: list[float]


def set_seed(seed: int) -> None:
    """统一固定 NumPy 和 PyTorch 随机种子，保证实验可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_loader(features: np.ndarray, targets: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    """
    将 NumPy 数据包装成 PyTorch DataLoader。

    - DAE 训练时，输入和目标都是连续值，因此 targets 也是特征矩阵
    - 分类器训练时，targets 是类别标签
    """
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    if targets is None:
        dataset = TensorDataset(feature_tensor, feature_tensor)
    else:
        target_tensor = torch.from_numpy(targets)
        dataset = TensorDataset(feature_tensor, target_tensor)

    # 使用固定种子的生成器，保证 shuffle 顺序在同一 seed 下可重复。
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _should_log(epoch: int, epochs: int, log_interval: int) -> bool:
    """决定当前 epoch 是否需要打印日志。"""
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
    """训练 DAE，让带噪输入重建为干净输入。"""
    loader = _build_loader(noisy_features, clean_features, batch_size=batch_size, shuffle=True)
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
        # DAE 是回归重建任务，没有严格意义上的分类准确率，这里留 0 仅用于画图占位。
        history.accuracies.append(0.0)
        if _should_log(epoch, epochs, log_interval):
            print(f"[DAE] epoch {epoch}/{epochs} loss={epoch_loss:.6f}")
    return history


def encode_features(model: nn.Module, features: np.ndarray, device: torch.device, mode: str = "fc3_linear") -> np.ndarray:
    """用训练好的 DAE 提取编码特征。"""
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        encoded = model.extract_features(tensor, mode=mode).cpu().numpy().astype(np.float32)
    return encoded


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
    """训练故障分类器，输入为 DAE 编码后的特征。"""
    loader = _build_loader(features, labels.astype(np.int64), batch_size=batch_size, shuffle=True)
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
        if _should_log(epoch, epochs, log_interval):
            print(f"[CLF] epoch {epoch}/{epochs} loss={epoch_loss:.6f} acc={epoch_acc:.4f}")
    return history


def predict_classes(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    """对测试特征做前向推理，输出 1~17 的故障标签。"""
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = model(tensor)
        predictions = logits.argmax(dim=1).cpu().numpy() + 1
    return predictions.astype(np.int64)


def build_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    """构造按行归一化的混淆矩阵，便于直接画热力图。"""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums
