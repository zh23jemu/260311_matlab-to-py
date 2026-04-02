from __future__ import annotations

"""DAE 训练与特征提取。

本模块只负责两件事：
1. 训练降噪自编码器。
2. 用训练好的 DAE 提取编码特征。

它不负责分类器训练，也不负责最终评估与导出。
"""

import numpy as np
import torch
from torch import nn

from te_dae.training_common import TrainHistory, build_loader, should_log


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

    输入是加噪后的特征矩阵，目标是原始干净特征矩阵。
    训练完成后，模型应当学会在噪声干扰下恢复主要结构信息，
    从而为后续分类阶段提供更稳定的编码特征。
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
        # DAE 是重建任务，不定义分类准确率，这里统一写 0.0 作为占位。
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

    这里不会做梯度计算，也不会更新模型参数，只是把输入前向送入编码器，
    然后取出指定模式下的中间表示，作为后续分类器的输入。
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        encoded = model.extract_features(tensor, mode=mode).cpu().numpy().astype(np.float32)
    return encoded
