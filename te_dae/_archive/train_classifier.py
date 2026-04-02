from __future__ import annotations

"""分类器训练、预测与评估辅助函数。

本模块处理 DAE 之后的监督学习阶段，包括：
1. 把不同故障的特征拼接成统一训练矩阵。
2. 训练故障分类器。
3. 生成测试预测结果。
4. 构建混淆矩阵，供热力图绘制使用。
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from te_dae.constants import TRAIN_FAULT_IDS
from te_dae.training_common import TrainHistory, build_loader, should_log


def stack_classifier_features(
    feature_map: dict[int, np.ndarray],
    label_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """把各故障特征按固定顺序拼接成分类器训练输入。

    返回：
    - `features`：纵向拼接后的训练特征矩阵
    - `labels`：与之对齐的 0-based 标签数组

    这里使用 0-based 标签，是为了直接适配 PyTorch 的 `CrossEntropyLoss`。
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

    输入特征已经是：
    - 经过 DAE 编码
    - 再按 d00 编码统计量完成第二次标准化

    因此这个阶段的任务很明确，就是学习“编码特征 -> 故障类别”的映射关系。
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
            # 分类阶段的目标值是离散标签，因此损失函数使用交叉熵。
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

    返回值是 1~17 的标签编号，而不是 0~16。
    这样更方便和项目里的 `classifier_labels` 以及最终 `F1`、`F2` 等故障编号口径对齐。
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

    这里做的是“每一行除以该真实类别的样本总数”，因此矩阵中的值可以解释为：
    某个真实故障被预测成各类别的比例分布。

    这种形式特别适合画热力图，因为不同类别的样本总数即使不完全相同，
    也能直接比较分类正确率和混淆方向。
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums
