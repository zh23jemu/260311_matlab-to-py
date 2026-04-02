from __future__ import annotations

"""TE 数据读取与预处理。

这个文件现在直接承载数据相关的核心逻辑，尽量保持成一个
“看文件名就知道用途”的集中入口，避免数据规则再散落到多个小文件里。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# 参与分类器训练与评估的故障编号顺序。
# 这里故意不是 1~20 全部故障都上，因为当前项目要对齐原 MATLAB 流程里
# 真正参与分类器训练的那 17 个故障类。
# 这个列表会同时影响：
# 1. 标签映射顺序
# 2. 训练特征拼接顺序
# 3. 最终热力图坐标顺序
TRAIN_FAULT_IDS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]

# DAE 训练阶段使用 d00~d20 的全部工况。
# 和分类器不同，DAE 这里是做“重建式特征学习”，因此会把正常类和故障类一起喂进去，
# 让编码器先学到整个数据空间的共性结构。
ALL_IDS = list(range(21))

# 原始数据是 52 维，按文档要求删除第 46 列和第 50 列。
# Python 使用 0-based 下标，所以这里对应 45 和 49。
DROP_COLS = [45, 49]


def test_slice_count(fault_id: int) -> int:
    """返回某个故障在测试阶段采用的样本数。

    这里不是简单地把所有测试样本全部拿来评估，而是沿用 MATLAB 代码里的口径：
    - F6 只统计 247 个样本
    - 其他参与评估的故障统一统计 2000 个样本

    这样做的目的不是数学上“更优”，而是为了让 Python 结果和原始流程
    能够直接横向对比。
    """
    return 247 if fault_id == 6 else 2000


@dataclass
class DatasetBundle:
    """封装数据处理阶段的中间结果。

    这个 dataclass 的作用是把实验流程里跨阶段要反复使用的数据集中放在一起，
    避免主流程里漂浮着很多散乱变量。

    这些字段大致分成三层：
    - 原始/第一次标准化后的输入数据
    - DAE 训练输入
    - 第二次标准化后的分类器输入
    """

    train_raw: dict[int, np.ndarray]
    test_raw: dict[int, np.ndarray]
    train_standardized: dict[int, np.ndarray]
    test_standardized: dict[int, np.ndarray]
    dae_train_data: np.ndarray
    classifier_train_features: dict[int, np.ndarray]
    classifier_test_features: dict[int, np.ndarray]
    classifier_labels: dict[int, int]
    base_mean: np.ndarray
    base_std: np.ndarray
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None


def _fault_key(prefix: str, fault_id: int, suffix: str) -> str:
    """生成 `.mat` 文件中的变量名。

    MATLAB 数据文件中的变量名是固定模式，例如：
    - `d00_6`
    - `d01_5`
    - `d20_7`

    这里统一封装后，外层流程就不需要到处手动拼字符串了。
    """
    return f"{prefix}{fault_id:02d}_{suffix}"


def _select_features(array: np.ndarray) -> np.ndarray:
    """删除原始 52 维信号中不参与建模的两列。

    删除后特征维度会从 52 维变成 50 维，同时统一转成 `float32`，
    这样后续送入 PyTorch 时不会重复做类型转换。
    """
    keep = [idx for idx in range(array.shape[1]) if idx not in DROP_COLS]
    return array[:, keep].astype(np.float32)


def load_te_dataset(mat_path: Path) -> DatasetBundle:
    """读取 TE 数据集并完成基础预处理。

    这里会一次性完成进入模型训练前最核心的准备工作：
    1. 读取训练片段 `_6`
    2. 读取测试片段 `_5` 和 `_7`，并按行拼接
    3. 删除固定两列，使输入维度变成 50
    4. 用 d00 的统计量对所有 train/test 数据做第一次 z-score
    5. 拼出 DAE 训练总矩阵
    6. 建立故障编号到连续标签的映射

    注意这里的“标准化统计量”始终来自 d00，而不是每个故障各算各的，
    这是整个项目里非常重要的对齐规则。
    """
    data = loadmat(mat_path)

    # 训练集固定读取 `_6` 片段，覆盖 d00~d20 的全部工况。
    train_raw = {
        fault_id: _select_features(data[_fault_key("d", fault_id, "6")]) for fault_id in ALL_IDS
    }

    # 测试集沿用 MATLAB 习惯，由 `_5` 和 `_7` 两段首尾拼接而成。
    # 之所以不直接只取一个片段，是为了保持与原流程相同的测试样本组织方式。
    test_raw = {}
    for fault_id in ALL_IDS:
        part_5 = _select_features(data[_fault_key("d", fault_id, "5")])
        part_7 = _select_features(data[_fault_key("d", fault_id, "7")])
        test_raw[fault_id] = np.vstack([part_5, part_7]).astype(np.float32)

    # 第一次标准化严格以 d00 为基准。
    # 这意味着“正常工况”被视为整个特征空间的参考中心。
    base_mean = train_raw[0].mean(axis=0)
    base_std = train_raw[0].std(axis=0, ddof=1)
    base_std[base_std == 0] = 1.0

    # 训练集和测试集都使用同一套 d00 统计量，避免信息泄漏，也保证口径一致。
    train_standardized = {
        fault_id: ((values - base_mean) / base_std).astype(np.float32)
        for fault_id, values in train_raw.items()
    }
    test_standardized = {
        fault_id: ((values - base_mean) / base_std).astype(np.float32)
        for fault_id, values in test_raw.items()
    }

    # DAE 训练数据是把所有工况的标准化训练样本纵向拼起来，
    # 让自编码器先学习“如何重建这个整体数据分布”。
    dae_train_data = np.vstack([train_standardized[fault_id] for fault_id in ALL_IDS]).astype(np.float32)

    # 分类标签映射成连续整数 1~17，便于后续统一做训练和评估。
    classifier_labels = {fault_id: index + 1 for index, fault_id in enumerate(TRAIN_FAULT_IDS)}

    return DatasetBundle(
        train_raw=train_raw,
        test_raw=test_raw,
        train_standardized=train_standardized,
        test_standardized=test_standardized,
        dae_train_data=dae_train_data,
        classifier_train_features={},
        classifier_test_features={},
        classifier_labels=classifier_labels,
        base_mean=base_mean.astype(np.float32),
        base_std=base_std.astype(np.float32),
    )


def add_noise(data: np.ndarray, wuc: float, seed: int) -> np.ndarray:
    """给 DAE 输入添加高斯噪声。

    DAE 训练不是“输入什么就重建什么”的普通自编码器，而是：
    - 输入：带噪数据
    - 目标：干净数据

    因此这里会按照 `sqrt(wuc)` 作为标准差生成高斯噪声，
    构造出“带噪输入 -> 干净输出”的训练对。
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(wuc), size=data.shape)
    return (data + noise).astype(np.float32)


__all__ = [
    "ALL_IDS",
    "DROP_COLS",
    "TRAIN_FAULT_IDS",
    "DatasetBundle",
    "add_noise",
    "load_te_dataset",
    "test_slice_count",
]
