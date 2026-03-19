from __future__ import annotations

import torch
from torch import nn


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 50) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 45)
        self.fc3 = nn.Linear(45, 40)
        self.fc4 = nn.Linear(40, 45)
        self.fc5 = nn.Linear(45, 50)
        self.fc6 = nn.Linear(50, input_dim)
        self.act = nn.LeakyReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return x

    def encode_fc3_linear(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

    def extract_features(self, x: torch.Tensor, mode: str = "fc3_linear") -> torch.Tensor:
        if mode == "fc3_linear":
            return self.encode_fc3_linear(x)
        if mode == "bottleneck_relu":
            return self.encode(x)
        raise ValueError(f"Unsupported feature extraction mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        x = self.fc6(x)
        return x


class ClassifierNet(nn.Module):
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
