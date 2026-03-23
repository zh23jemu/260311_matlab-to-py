from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# 文档中实际参与分类训练的故障编号。
TRAIN_FAULT_IDS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
# DAE 训练会使用 d00 到 d20 全部过程数据。
ALL_IDS = list(range(21))
# 去掉原始 52 维中的第 46 和第 50 列，对应 Python 的 0-based 索引 45 和 49。
DROP_COLS = [45, 49]


@dataclass
class DatasetBundle:
    """统一封装数据处理各阶段的中间结果。"""

    # 原始训练/测试数据，已经完成列删除，但尚未标准化。
    train_raw: dict[int, np.ndarray]
    test_raw: dict[int, np.ndarray]
    # 以 d00 统计量做 z-score 之后的训练/测试数据。
    train_standardized: dict[int, np.ndarray]
    test_standardized: dict[int, np.ndarray]
    # DAE 训练输入，按 d00~d20 拼接得到。
    dae_train_data: np.ndarray
    # 分类器输入特征，会在 pipeline 中根据 DAE 编码结果填充。
    classifier_train_features: dict[int, np.ndarray]
    classifier_test_features: dict[int, np.ndarray]
    # 将故障编号映射到分类标签 1~17。
    classifier_labels: dict[int, int]
    # d00 原始统计量，用于第一次标准化。
    base_mean: np.ndarray
    base_std: np.ndarray
    # d00 编码特征统计量，用于第二次标准化。
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None


def _fault_key(prefix: str, fault_id: int, suffix: str) -> str:
    """生成 MATLAB `.mat` 文件里的变量名，例如 d01_6、d10_5。"""
    return f"{prefix}{fault_id:02d}_{suffix}"


def _select_features(array: np.ndarray) -> np.ndarray:
    """删除文档中明确要求剔除的两列，使 52 维变成 50 维。"""
    keep = [idx for idx in range(array.shape[1]) if idx not in DROP_COLS]
    return array[:, keep].astype(np.float32)


def load_te_dataset(mat_path: Path) -> DatasetBundle:
    """
    读取 TE 数据集，并完成第一阶段预处理。

    约定：
    - 训练集使用 `_6`
    - 测试集使用 `_5` 和 `_7` 纵向拼接
    - 全部标准化统计量都来自 d00
    """
    data = loadmat(mat_path)

    # 读取 d00_6 到 d20_6，作为训练部分。
    train_raw = {
        fault_id: _select_features(data[_fault_key("d", fault_id, "6")]) for fault_id in ALL_IDS
    }

    # 测试集按文档要求由 `_5` 和 `_7` 两段拼接得到。
    test_raw = {}
    for fault_id in ALL_IDS:
        part_5 = _select_features(data[_fault_key("d", fault_id, "5")])
        part_7 = _select_features(data[_fault_key("d", fault_id, "7")])
        test_raw[fault_id] = np.vstack([part_5, part_7]).astype(np.float32)

    # 文档和 MATLAB 程序都使用 d00 的均值/标准差做第一次 z-score 标准化。
    base_mean = train_raw[0].mean(axis=0)
    base_std = train_raw[0].std(axis=0, ddof=1)
    base_std[base_std == 0] = 1.0

    train_standardized = {
        fault_id: ((values - base_mean) / base_std).astype(np.float32)
        for fault_id, values in train_raw.items()
    }
    test_standardized = {
        fault_id: ((values - base_mean) / base_std).astype(np.float32)
        for fault_id, values in test_raw.items()
    }

    # DAE 会对全部工况做编码，因此这里拼接 d00~d20。
    dae_train_data = np.vstack([train_standardized[fault_id] for fault_id in ALL_IDS]).astype(np.float32)
    # 分类标签按文档里的故障顺序映射成 1~17。
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
    """
    给 DAE 训练输入添加高斯噪声。

    MATLAB 文档里的写法是 `sqrt(wuc) * randn(...)`，这里做等价实现。
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(wuc), size=data.shape)
    return (data + noise).astype(np.float32)
