from __future__ import annotations

"""TE 数据读取与第一阶段预处理。

这个模块专门处理“进入模型训练之前”的原始数据准备工作，包括：
1. 从 MATLAB 的 `.mat` 文件中读取各工况数据。
2. 删除文档明确要求剔除的两列。
3. 按 d00 的统计量对训练集和测试集做第一次 z-score 标准化。
4. 生成 DAE 训练所需的带噪输入。

注意：
- 本模块不负责模型训练。
- 本模块也不负责第二次特征标准化，那一步发生在 DAE 编码之后。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from te_dae.constants import ALL_IDS, DROP_COLS, TRAIN_FAULT_IDS


@dataclass
class DatasetBundle:
    """统一封装数据处理阶段会反复用到的中间结果。

    之所以用一个 dataclass 集中承载，而不是在主流程里散落多个局部变量，
    是因为这个项目的数据会经历多个阶段：
    - 原始数据
    - 第一次标准化后的数据
    - DAE 训练输入
    - DAE 编码后的分类特征
    - 第二次标准化后的分类输入

    把这些结果统一放在一个对象里，更方便主流程在不同阶段之间传递与补充。
    """

    # 删除指定列后的原始训练数据，键为 fault_id。
    train_raw: dict[int, np.ndarray]
    # 删除指定列后的原始测试数据，测试集由 `_5` 和 `_7` 两段拼接而成。
    test_raw: dict[int, np.ndarray]
    # 以 d00 统计量完成第一次 z-score 后的训练数据。
    train_standardized: dict[int, np.ndarray]
    # 以 d00 统计量完成第一次 z-score 后的测试数据。
    test_standardized: dict[int, np.ndarray]
    # 将 d00~d20 的标准化训练数据纵向拼接后得到的 DAE 输入。
    dae_train_data: np.ndarray
    # 第二次标准化后的分类训练特征，会在主流程中填充。
    classifier_train_features: dict[int, np.ndarray]
    # 第二次标准化后的分类测试特征，会在主流程中填充。
    classifier_test_features: dict[int, np.ndarray]
    # 故障编号到分类标签编号的映射，例如 F1 -> 1, F2 -> 2。
    classifier_labels: dict[int, int]
    # 第一次标准化使用的 d00 均值。
    base_mean: np.ndarray
    # 第一次标准化使用的 d00 标准差。
    base_std: np.ndarray
    # 第二次标准化使用的 d00 编码特征均值。
    feature_mean: np.ndarray | None = None
    # 第二次标准化使用的 d00 编码特征标准差。
    feature_std: np.ndarray | None = None


def _fault_key(prefix: str, fault_id: int, suffix: str) -> str:
    """生成 `.mat` 文件中的变量名。

    例如：
    - `fault_id=1, suffix="6"` -> `d01_6`
    - `fault_id=10, suffix="5"` -> `d10_5`

    MATLAB 数据文件中各工况变量名遵循固定模式，所以这里统一生成，避免主逻辑里
    到处拼字符串。
    """
    return f"{prefix}{fault_id:02d}_{suffix}"


def _select_features(array: np.ndarray) -> np.ndarray:
    """删除原始 52 维信号中不参与建模的两列。

    文档要求剔除原始第 46 列和第 50 列，因此这里会把数据从 52 维变成 50 维。
    同时统一转换成 `float32`，便于后续交给 PyTorch 使用。
    """
    keep = [idx for idx in range(array.shape[1]) if idx not in DROP_COLS]
    return array[:, keep].astype(np.float32)


def load_te_dataset(mat_path: Path) -> DatasetBundle:
    """读取 TE 数据集并完成进入模型前的基础预处理。

    该函数会依次完成以下工作：
    1. 从 `.mat` 文件读取 d00~d20 的训练部分 `_6`。
    2. 从 `.mat` 文件读取测试部分 `_5` 与 `_7`，并按行拼接。
    3. 删除固定两列，使输入维度从 52 变为 50。
    4. 用 d00 的均值和标准差对全部训练/测试数据做第一次 z-score 标准化。
    5. 拼出 DAE 所需的总训练矩阵。
    6. 建立故障编号到分类标签编号的映射。

    返回结果不会直接进入分类器训练，因为后面还要经过 DAE 编码和第二次标准化。
    """
    data = loadmat(mat_path)

    # 训练集使用 `_6` 片段，覆盖 d00~d20 全部工况。
    train_raw = {
        fault_id: _select_features(data[_fault_key("d", fault_id, "6")]) for fault_id in ALL_IDS
    }

    # 测试集按 MATLAB 原流程由 `_5` 和 `_7` 两段拼接得到。
    test_raw = {}
    for fault_id in ALL_IDS:
        part_5 = _select_features(data[_fault_key("d", fault_id, "5")])
        part_7 = _select_features(data[_fault_key("d", fault_id, "7")])
        test_raw[fault_id] = np.vstack([part_5, part_7]).astype(np.float32)

    # 第一次标准化严格使用 d00 的统计量，而不是各故障各算各的。
    # 这是为了保持与原始 MATLAB 工作流一致。
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

    # DAE 训练不是只看故障类，而是把 d00~d20 的训练数据都拿来做去噪重建学习。
    dae_train_data = np.vstack([train_standardized[fault_id] for fault_id in ALL_IDS]).astype(np.float32)

    # 将实际参与分类的故障编号映射为连续标签 1~17。
    # 注意：分类器训练内部最终还是会转成 0~16，以适配 CrossEntropyLoss。
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
    """给 DAE 输入添加高斯噪声，构造“带噪输入 -> 干净输出”的训练对。

    参数说明：
    - `data`：干净的标准化输入
    - `wuc`：噪声强度参数，实际标准差为 `sqrt(wuc)`
    - `seed`：随机种子，保证不同实验可复现

    这里采用的写法与 MATLAB 文档中的 `sqrt(wuc) * randn(...)` 等价。
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(wuc), size=data.shape)
    return (data + noise).astype(np.float32)
