from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainHistory:
    losses: list[float]
    accuracies: list[float]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_loader(features: np.ndarray, targets: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    if targets is None:
        dataset = TensorDataset(feature_tensor, feature_tensor)
    else:
        target_tensor = torch.from_numpy(targets)
        dataset = TensorDataset(feature_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_autoencoder(
    model: nn.Module,
    noisy_features: np.ndarray,
    clean_features: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> TrainHistory:
    dataset = TensorDataset(
        torch.from_numpy(noisy_features.astype(np.float32)),
        torch.from_numpy(clean_features.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.to(device)
    history = TrainHistory(losses=[], accuracies=[])

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)
        history.losses.append(running_loss / max(total, 1))
        history.accuracies.append(0.0)
    return history


def encode_features(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        encoded = model.encode(tensor).cpu().numpy().astype(np.float32)
    return encoded


def train_classifier(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> TrainHistory:
    loader = _build_loader(features, labels.astype(np.int64), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    history = TrainHistory(losses=[], accuracies=[])

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_x.size(0)
        history.losses.append(running_loss / max(total, 1))
        history.accuracies.append(correct / max(total, 1))
    return history


def predict_classes(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = model(tensor)
        predictions = logits.argmax(dim=1).cpu().numpy() + 1
    return predictions.astype(np.int64)


def build_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums
