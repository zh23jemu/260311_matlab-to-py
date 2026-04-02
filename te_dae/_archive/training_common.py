from __future__ import annotations

"""训练公共工具。

这里放的是 DAE 和分类器训练都会复用的基础能力，例如：
- 训练历史记录结构
- 随机种子设置
- DataLoader 构建
- 日志打印时机判断

单独拆成一个文件，是为了避免这些“训练基础设施”分别复制到两个训练模块里。
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainHistory:
    """保存训练历史。

    - `losses`：每个 epoch 的平均损失
    - `accuracies`：每个 epoch 的准确率

    对 DAE 来说没有严格意义上的分类准确率，因此会放占位值 0.0，
    主要是为了让出图函数保持统一输入结构。
    """

    losses: list[float]
    accuracies: list[float]


def set_seed(seed: int) -> None:
    """统一固定随机种子，尽量保证实验可复现。

    当前同时固定：
    - NumPy 随机数
    - PyTorch CPU 随机数

    对当前项目的 CPU 训练和数据加噪流程，这已经足够覆盖主要随机来源。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(features: np.ndarray, targets: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    """把 NumPy 数组包装成 PyTorch 的 `DataLoader`。

    使用约定：
    - 当 `targets is None` 时，表示自监督重建任务，输入和目标相同。
    - 当 `targets` 不为空时，表示监督分类任务，目标是类别标签。

    这里统一使用固定种子的 `generator`，这样在相同 seed 下，shuffle 顺序也能保持稳定。
    """
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    if targets is None:
        dataset = TensorDataset(feature_tensor, feature_tensor)
    else:
        target_tensor = torch.from_numpy(targets)
        dataset = TensorDataset(feature_tensor, target_tensor)

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def should_log(epoch: int, epochs: int, log_interval: int) -> bool:
    """判断当前 epoch 是否应该打印日志。

    规则是：
    - 第 1 个 epoch 一定打印
    - 最后 1 个 epoch 一定打印
    - 其余 epoch 按 `log_interval` 周期打印

    这样既能看到训练是否正常启动，也能避免日志刷得过密。
    """
    return epoch == 1 or epoch == epochs or epoch % log_interval == 0
