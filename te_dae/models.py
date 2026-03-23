from __future__ import annotations

import torch
from torch import nn


class DenoisingAutoencoder(nn.Module):
    """与文档一致的 6 层全连接 DAE。"""

    def __init__(self, input_dim: int = 50) -> None:
        super().__init__()
        # 编码部分：50 -> 50 -> 45 -> 40
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 45)
        self.fc3 = nn.Linear(45, 40)
        # 解码部分：40 -> 45 -> 50 -> 50
        self.fc4 = nn.Linear(40, 45)
        self.fc5 = nn.Linear(45, 50)
        self.fc6 = nn.Linear(50, input_dim)
        self.act = nn.LeakyReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """返回经过第三个激活函数后的 bottleneck 特征。"""
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return x

    def encode_fc3_linear(self, x: torch.Tensor) -> torch.Tensor:
        """返回 `fc3` 线性层输出，不经过第三次激活。"""
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

    def extract_features(self, x: torch.Tensor, mode: str = "fc3_linear") -> torch.Tensor:
        """
        统一封装两种特征提取方式，便于实验比较。

        - `fc3_linear`: 更贴近“按层号取 fc3”的解释
        - `bottleneck_relu`: 更贴近“取编码瓶颈输出”的解释
        """
        if mode == "fc3_linear":
            return self.encode_fc3_linear(x)
        if mode == "bottleneck_relu":
            return self.encode(x)
        raise ValueError(f"Unsupported feature extraction mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整前向过程：先编码，再解码重建输入。"""
        x = self.encode(x)
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        x = self.fc6(x)
        return x


class ClassifierNet(nn.Module):
    """文档中用于故障分类的三层全连接网络。"""

    def __init__(self, input_dim: int = 40, num_classes: int = 17) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.Tanh(),
            nn.Linear(400, 250),
            nn.Tanh(),
            nn.Linear(250, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
