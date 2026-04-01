from __future__ import annotations

import torch
from torch import nn


class DenoisingAutoencoder(nn.Module):
    """降噪自编码器。

    该模型与文档中的结构保持一致，整体是一个对称的全连接自编码器：
    `50 -> 50 -> 45 -> 40 -> 45 -> 50 -> 50`

    其中：
    - 前三层负责把 50 维输入压缩到 40 维瓶颈特征
    - 后三层负责把 40 维瓶颈特征重建回原始输入空间
    - 隐藏层统一使用 `LeakyReLU`
    """

    def __init__(self, input_dim: int = 50) -> None:
        super().__init__()
        # 编码部分：逐层压缩输入，最终得到 40 维瓶颈表示。
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 45)
        self.fc3 = nn.Linear(45, 40)
        # 解码部分：把瓶颈表示逐层还原回输入空间，用于重建干净信号。
        self.fc4 = nn.Linear(40, 45)
        self.fc5 = nn.Linear(45, 50)
        self.fc6 = nn.Linear(50, input_dim)
        self.act = nn.LeakyReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """返回瓶颈层经过激活后的编码结果。

        这个输出对应项目中默认使用的 `bottleneck_relu` 特征模式。
        """
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return x

    def encode_fc3_linear(self, x: torch.Tensor) -> torch.Tensor:
        """返回 `fc3` 的线性输出，不经过第三次激活。

        这个输出对应项目里保留的 `fc3_linear` 特征模式，
        便于和 MATLAB 文档中“取 fc3 层输出”的解释做对照。
        """
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

    def extract_features(self, x: torch.Tensor, mode: str = "fc3_linear") -> torch.Tensor:
        """
        统一封装两种特征提取方式。

        支持的模式：
        - `fc3_linear`：取第三个全连接层的线性输出
        - `bottleneck_relu`：取第三个激活函数之后的瓶颈输出

        这样主流程不需要关心具体调用哪个编码函数，只需要传入模式名即可。
        """
        if mode == "fc3_linear":
            return self.encode_fc3_linear(x)
        if mode == "bottleneck_relu":
            return self.encode(x)
        raise ValueError(f"不支持的特征提取模式: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整前向传播。

        训练 DAE 时，输入是带噪数据，输出是重建结果。
        损失函数会拿这个重建结果去和“干净输入”做比较。
        """
        x = self.encode(x)
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        x = self.fc6(x)
        return x


class ClassifierNet(nn.Module):
    """故障分类网络。

    分类器接收的是 DAE 编码后的 40 维特征，而不是原始 50 维输入。
    网络结构为：
    `40 -> 400 -> 250 -> 17`

    最后一层直接输出 17 个类别的 logits，交给 `CrossEntropyLoss` 处理。
    """

    def __init__(self, input_dim: int = 40, num_classes: int = 17) -> None:
        super().__init__()
        # 使用简单的三层全连接结构，保持与文档设定一致。
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.Tanh(),
            nn.Linear(400, 250),
            nn.Tanh(),
            nn.Linear(250, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回各故障类别的原始 logits。"""
        return self.net(x)
