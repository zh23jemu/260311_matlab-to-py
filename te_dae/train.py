from __future__ import annotations

"""旧训练模块名的兼容包装层。

项目重组后，训练逻辑已经拆分为：
- `te_dae.train_dae`
- `te_dae.train_classifier`
- `te_dae.training_common`

这个文件现在只负责把旧接口重新导出出去，保证旧导入路径仍然有效。
"""

from te_dae.train_classifier import build_confusion, predict_classes, train_classifier
from te_dae.train_dae import extract_encoded_features as encode_features
from te_dae.train_dae import train_autoencoder
from te_dae.training_common import TrainHistory, set_seed

__all__ = [
    "TrainHistory",
    "build_confusion",
    "encode_features",
    "predict_classes",
    "set_seed",
    "train_autoencoder",
    "train_classifier",
]
