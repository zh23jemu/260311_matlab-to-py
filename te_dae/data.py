from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


TRAIN_FAULT_IDS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
ALL_IDS = list(range(21))
DROP_COLS = [45, 49]


@dataclass
class DatasetBundle:
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
    return f"{prefix}{fault_id:02d}_{suffix}"


def _select_features(array: np.ndarray) -> np.ndarray:
    keep = [idx for idx in range(array.shape[1]) if idx not in DROP_COLS]
    return array[:, keep].astype(np.float32)


def load_te_dataset(mat_path: Path) -> DatasetBundle:
    data = loadmat(mat_path)
    train_raw = {
        fault_id: _select_features(data[_fault_key("d", fault_id, "6")]) for fault_id in ALL_IDS
    }
    test_raw = {}
    for fault_id in ALL_IDS:
        part_5 = _select_features(data[_fault_key("d", fault_id, "5")])
        part_7 = _select_features(data[_fault_key("d", fault_id, "7")])
        test_raw[fault_id] = np.vstack([part_5, part_7]).astype(np.float32)

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

    dae_train_data = np.vstack([train_standardized[fault_id] for fault_id in ALL_IDS]).astype(np.float32)
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
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(wuc), size=data.shape)
    return (data + noise).astype(np.float32)
