from __future__ import annotations

"""简版单文件 DAE 程序。

这个版本专门面向“目录尽量简单、直接运行”的使用场景：
- 当前目录直接放 `data567.mat`
- 直接运行 `main.py` 或 `DAE_main.py`
- 自动在当前目录下生成 `outputs/`
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TRAIN_FAULT_IDS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
ALL_IDS = list(range(21))
DROP_COLS = [45, 49]


def test_slice_count(fault_id: int) -> int:
    """返回测试阶段采用的样本数。"""
    return 247 if fault_id == 6 else 2000


@dataclass
class DatasetBundle:
    """封装数据处理中间结果。"""

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


@dataclass
class TrainHistory:
    """保存训练历史。"""

    losses: list[float]
    accuracies: list[float]


class DenoisingAutoencoder(nn.Module):
    """降噪自编码器。"""

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
        raise ValueError(f"不支持的特征提取模式: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        x = self.fc6(x)
        return x


class ClassifierNet(nn.Module):
    """故障分类网络。"""

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


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def should_log(epoch: int, epochs: int, log_interval: int) -> bool:
    """判断是否打印当前 epoch 日志。"""
    return epoch == 1 or epoch == epochs or epoch % log_interval == 0


def build_loader(features: np.ndarray, targets: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    """构建 DataLoader。"""
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    if targets is None:
        dataset = TensorDataset(feature_tensor, feature_tensor)
    else:
        dataset = TensorDataset(feature_tensor, torch.from_numpy(targets))
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _fault_key(prefix: str, fault_id: int, suffix: str) -> str:
    """生成 mat 变量名。"""
    return f"{prefix}{fault_id:02d}_{suffix}"


def _select_features(array: np.ndarray) -> np.ndarray:
    """删除不参与建模的两列。"""
    keep = [idx for idx in range(array.shape[1]) if idx not in DROP_COLS]
    return array[:, keep].astype(np.float32)


def load_te_dataset(mat_path: Path) -> DatasetBundle:
    """读取 TE 数据集并完成基础预处理。"""
    data = loadmat(mat_path)
    train_raw = {
        fault_id: _select_features(data[_fault_key("d", fault_id, "6")]) for fault_id in ALL_IDS
    }

    test_raw: dict[int, np.ndarray] = {}
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
    """给 DAE 输入添加高斯噪声。"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(wuc), size=data.shape)
    return (data + noise).astype(np.float32)


def train_autoencoder(
    model: nn.Module,
    noisy_features: np.ndarray,
    clean_features: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_interval: int = 100,
) -> TrainHistory:
    """训练 DAE。"""
    loader = build_loader(noisy_features, clean_features, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.MSELoss()
    model.to(device)
    history = TrainHistory(losses=[], accuracies=[])

    print(
        f"[DAE] start training: samples={len(loader.dataset)}, batch_size={batch_size}, "
        f"epochs={epochs}, lr={learning_rate}, device={device}"
    )
    for epoch in range(1, epochs + 1):
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

        epoch_loss = running_loss / max(total, 1)
        history.losses.append(epoch_loss)
        history.accuracies.append(0.0)
        if should_log(epoch, epochs, log_interval):
            print(f"[DAE] epoch {epoch}/{epochs} loss={epoch_loss:.6f}")
    return history


def extract_encoded_features(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    mode: str = "fc3_linear",
) -> np.ndarray:
    """提取编码特征。"""
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        encoded = model.extract_features(tensor, mode=mode).cpu().numpy().astype(np.float32)
    return encoded


def stack_classifier_features(
    feature_map: dict[int, np.ndarray],
    label_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """拼接分类器训练输入。"""
    features = np.vstack([feature_map[fault_id] for fault_id in TRAIN_FAULT_IDS]).astype(np.float32)
    labels = np.concatenate(
        [np.full(feature_map[fault_id].shape[0], label_map[fault_id] - 1, dtype=np.int64) for fault_id in TRAIN_FAULT_IDS]
    )
    return features, labels


def train_classifier(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_interval: int = 20,
) -> TrainHistory:
    """训练分类器。"""
    loader = build_loader(features, labels.astype(np.int64), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    history = TrainHistory(losses=[], accuracies=[])

    print(
        f"[CLF] start training: samples={len(loader.dataset)}, batch_size={batch_size}, "
        f"epochs={epochs}, lr={learning_rate}, device={device}"
    )
    for epoch in range(1, epochs + 1):
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

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        history.losses.append(epoch_loss)
        history.accuracies.append(epoch_acc)
        if should_log(epoch, epochs, log_interval):
            print(f"[CLF] epoch {epoch}/{epochs} loss={epoch_loss:.6f} acc={epoch_acc:.4f}")
    return history


def predict_classes(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    """预测分类标签。"""
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = model(tensor)
        predictions = logits.argmax(dim=1).cpu().numpy() + 1
    return predictions.astype(np.int64)


def build_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    """构造按行归一化的混淆矩阵。"""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


def _apply_chinese_font() -> None:
    """尽量使用常见中文字体，避免中文乱码。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, path: Path) -> None:
    """统一保存图像。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_lines(data: np.ndarray, title: str, path: Path, max_rows: int = 400) -> None:
    """绘制矩阵折线图。"""
    _apply_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data[:max_rows])
    ax.set_title(title)
    ax.set_xlabel("样本序号")
    ax.set_ylabel("数值")
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_label_sequence(labels: np.ndarray, title: str, path: Path) -> None:
    """绘制标签序列图。"""
    _apply_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(labels, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("样本序号")
    ax.set_ylabel("标签")
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_training_history(losses: list[float], accuracies: list[float], title: str, path: Path) -> None:
    """绘制分类器训练过程图。"""
    _apply_chinese_font()
    epochs = list(range(1, len(losses) + 1))
    accuracy_percent = np.asarray(accuracies, dtype=np.float64) * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(epochs, accuracy_percent, color="#1f77b4", linewidth=2.0)
    axes[0].set_title("训练过程")
    axes[0].set_ylabel("准确率(%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, losses, color="#ff7f0e", linewidth=2.0)
    axes[1].set_xlabel("迭代次数")
    axes[1].set_ylabel("损失")
    axes[1].grid(alpha=0.25)
    _save(fig, path)


def plot_dae_training_history(losses: list[float], title: str, path: Path) -> None:
    """绘制 DAE 训练图。"""
    _apply_chinese_font()
    epochs = list(range(1, len(losses) + 1))
    loss_array = np.asarray(losses, dtype=np.float64)
    initial_loss = max(loss_array[0], 1e-12)
    convergence = (1.0 - loss_array / initial_loss) * 100.0
    convergence = np.clip(convergence, 0.0, 100.0)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(epochs, convergence, color="#1f77b4", linewidth=2.0)
    axes[0].set_title(f"训练进度（{timestamp}）")
    axes[0].set_ylabel("收敛率(%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, loss_array, color="#ff7f0e", linewidth=2.0)
    axes[1].set_xlabel("迭代次数")
    axes[1].set_ylabel("均方误差")
    axes[1].grid(alpha=0.25)
    _save(fig, path)


def plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    """绘制热值图。"""
    _apply_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_network_diagram(path: Path) -> None:
    """绘制网络结构示意图。"""
    _apply_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = [
        ("输入\n50", 0.05),
        ("全连接\n50", 0.18),
        ("全连接\n45", 0.31),
        ("瓶颈层\n40", 0.44),
        ("全连接\n45", 0.57),
        ("全连接\n50", 0.70),
        ("输出\n50", 0.83),
    ]
    for text, xpos in boxes:
        ax.text(
            xpos,
            0.55,
            text,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f2e8c9", "edgecolor": "#7b6d42"},
            transform=ax.transAxes,
        )
    for start, end in zip(boxes, boxes[1:]):
        ax.annotate(
            "",
            xy=(end[1] - 0.05, 0.55),
            xytext=(start[1] + 0.05, 0.55),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.5},
        )
    ax.set_title("网络结构图")
    _save(fig, path)


def _output_dirs(root: Path, output_dir: str) -> dict[str, Path]:
    """创建输出目录。"""
    outputs = root / output_dir
    paths = {
        "root": outputs,
        "figures": outputs / "figures",
        "metrics": outputs / "metrics",
        "models": outputs / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _save_json(path: Path, payload: dict) -> None:
    """保存 JSON。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """解析参数。"""
    parser = argparse.ArgumentParser(description="TE DAE 简版程序")
    parser.add_argument("--dae-epochs", type=int, default=2500)
    parser.add_argument("--clf-epochs", type=int, default=300)
    parser.add_argument("--dae-lr", type=float, default=0.0001)
    parser.add_argument("--clf-lr", type=float, default=0.0001)
    parser.add_argument("--wuc", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--dae-feature-mode", default="bottleneck_relu", choices=["fc3_linear", "bottleneck_relu"])
    parser.add_argument("--data-path", default=None)
    return parser.parse_args()


def _resolve_device(choice: str) -> torch.device:
    """解析设备。"""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_mat_path(root: Path, explicit_path: str | None = None) -> Path:
    """解析数据文件路径。"""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    candidates.extend([root / "data567.mat", root / "CNN" / "data567.mat"])
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"未找到 data567.mat。\n已检查位置：\n{searched}")


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """执行一次完整实验。"""
    root = Path(__file__).resolve().parent
    paths = _output_dirs(root, args.output_dir)
    set_seed(args.seed)

    mat_path = _resolve_mat_path(root, args.data_path)
    device = _resolve_device(args.device)

    print(f"[PIPELINE] root={root}")
    print(f"[PIPELINE] output_dir={paths['root']}")
    print(f"[PIPELINE] seed={args.seed} wuc={args.wuc} device={device}")
    print(f"[PIPELINE] dae_lr={args.dae_lr} clf_lr={args.clf_lr} clf_epochs={args.clf_epochs}")
    print(f"[PIPELINE] dae_feature_mode={args.dae_feature_mode}")
    print(f"[PIPELINE] loading dataset from {mat_path}")

    bundle = load_te_dataset(mat_path)
    print(
        f"[PIPELINE] dataset loaded: dae_train_shape={bundle.dae_train_data.shape}, "
        f"train_faults={len(bundle.train_standardized)}, test_faults={len(bundle.test_standardized)}"
    )

    print("[PIPELINE] adding Gaussian noise for DAE training")
    dae_train_noisy = add_noise(bundle.dae_train_data, wuc=args.wuc, seed=args.seed)
    dae_model = DenoisingAutoencoder(input_dim=bundle.dae_train_data.shape[1])
    dae_history = train_autoencoder(
        dae_model,
        noisy_features=dae_train_noisy,
        clean_features=bundle.dae_train_data,
        epochs=args.dae_epochs,
        batch_size=32,
        learning_rate=args.dae_lr,
        device=device,
        log_interval=max(1, args.dae_epochs // 10),
    )

    print("[PIPELINE] encoding train and test features with DAE")
    encoded_train = {
        fault_id: extract_encoded_features(dae_model, values, device=device, mode=args.dae_feature_mode)
        for fault_id, values in bundle.train_standardized.items()
    }
    encoded_test = {
        fault_id: extract_encoded_features(dae_model, values, device=device, mode=args.dae_feature_mode)
        for fault_id, values in bundle.test_standardized.items()
    }

    d00_features = encoded_train[0]
    feature_mean = d00_features.mean(axis=0)
    feature_std = d00_features.std(axis=0, ddof=1)
    feature_std[feature_std == 0] = 1.0
    bundle.feature_mean = feature_mean.astype(np.float32)
    bundle.feature_std = feature_std.astype(np.float32)

    print("[PIPELINE] standardizing encoded features using d00 statistics")
    bundle.classifier_train_features = {
        fault_id: ((encoded_train[fault_id] - feature_mean) / feature_std).astype(np.float32)
        for fault_id in TRAIN_FAULT_IDS
    }
    bundle.classifier_test_features = {
        fault_id: ((encoded_test[fault_id] - feature_mean) / feature_std).astype(np.float32)
        for fault_id in TRAIN_FAULT_IDS
    }

    clf_train_x, clf_train_y = stack_classifier_features(bundle.classifier_train_features, bundle.classifier_labels)
    print(f"[PIPELINE] classifier training tensor shape={clf_train_x.shape}, labels={clf_train_y.shape}")
    classifier = ClassifierNet(input_dim=clf_train_x.shape[1], num_classes=len(TRAIN_FAULT_IDS))
    clf_history = train_classifier(
        classifier,
        features=clf_train_x,
        labels=clf_train_y,
        epochs=args.clf_epochs,
        batch_size=300,
        learning_rate=args.clf_lr,
        device=device,
        log_interval=max(1, args.clf_epochs // 10),
    )

    print("[PIPELINE] evaluating classifier on test faults")
    per_fault_accuracy: dict[str, float] = {}
    y_true_all = []
    y_pred_all = []
    for fault_id in TRAIN_FAULT_IDS:
        count = min(test_slice_count(fault_id), bundle.classifier_test_features[fault_id].shape[0])
        features = bundle.classifier_test_features[fault_id][:count]
        preds = predict_classes(classifier, features, device=device)
        label = bundle.classifier_labels[fault_id]
        truth = np.full(count, label, dtype=np.int64)
        y_true_all.append(truth)
        y_pred_all.append(preds)
        fault_acc = float((preds == label).mean())
        per_fault_accuracy[f"F{fault_id}"] = fault_acc
        print(f"[EVAL] F{fault_id} accuracy={fault_acc:.4f} samples={count}")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    mean_accuracy = float(np.mean(list(per_fault_accuracy.values())))
    heat = build_confusion(y_true, y_pred, labels=list(range(1, len(TRAIN_FAULT_IDS) + 1)))
    fault_labels = [f"F{fault_id}" for fault_id in TRAIN_FAULT_IDS]

    print("[PIPELINE] saving models and metrics")
    torch.save(
        {
            "state_dict": dae_model.state_dict(),
            "base_mean": bundle.base_mean,
            "base_std": bundle.base_std,
            "feature_mean": bundle.feature_mean,
            "feature_std": bundle.feature_std,
            "feature_mode": args.dae_feature_mode,
            "seed": args.seed,
            "wuc": args.wuc,
            "dae_lr": args.dae_lr,
            "clf_lr": args.clf_lr,
            "clf_epochs": args.clf_epochs,
        },
        paths["models"] / "dae.pt",
    )
    torch.save(
        {"state_dict": classifier.state_dict(), "seed": args.seed, "clf_lr": args.clf_lr, "clf_epochs": args.clf_epochs},
        paths["models"] / "classifier.pt",
    )

    _save_json(
        paths["metrics"] / "metrics.json",
        {
            "device": str(device),
            "seed": args.seed,
            "wuc": args.wuc,
            "dae_epochs": args.dae_epochs,
            "clf_epochs": args.clf_epochs,
            "dae_lr": args.dae_lr,
            "clf_lr": args.clf_lr,
            "dae_feature_mode": args.dae_feature_mode,
            "mean_accuracy": mean_accuracy,
            "per_fault_accuracy": per_fault_accuracy,
        },
    )

    print("[PIPELINE] rendering figures")
    plot_matrix_lines(bundle.train_raw[0], "图 4.2 未标准化 d00", paths["figures"] / "figure_4_2.png")
    plot_matrix_lines(bundle.train_standardized[0], "图 4.3 标准化 d00", paths["figures"] / "figure_4_3.png")
    plot_matrix_lines(bundle.train_standardized[0], "图 4.4 d00 标准化数据", paths["figures"] / "figure_4_4.png")
    plot_matrix_lines(bundle.train_standardized[1], "图 4.5 d01 标准化数据", paths["figures"] / "figure_4_5.png")
    plot_matrix_lines(bundle.train_standardized[2], "图 4.6 d02 标准化数据", paths["figures"] / "figure_4_6.png")
    plot_matrix_lines(clf_train_x, "图 4.7 训练特征标准化", paths["figures"] / "figure_4_7.png")
    plot_matrix_lines(
        np.vstack(
            [bundle.classifier_test_features[fault_id][: min(200, bundle.classifier_test_features[fault_id].shape[0])] for fault_id in TRAIN_FAULT_IDS]
        ),
        "图 4.8 测试特征标准化",
        paths["figures"] / "figure_4_8.png",
        max_rows=200,
    )
    plot_label_sequence(y_true, "图 4.9 数据标签", paths["figures"] / "figure_4_9.png")
    plot_network_diagram(paths["figures"] / "figure_4_10.png")
    plot_training_history(clf_history.losses, clf_history.accuracies, "图 4.11 训练过程", paths["figures"] / "figure_4_11.png")
    plot_heatmap(heat, fault_labels, "图 4.12 热值图", paths["figures"] / "figure_4_12.png")
    plot_dae_training_history(dae_history.losses, "DAE 训练过程", paths["figures"] / "dae_training.png")

    print(f"[RESULT] mean_accuracy={mean_accuracy:.4f}")
    print(f"[RESULT] saved outputs to {paths['root']}")
    return {
        "mean_accuracy": mean_accuracy,
        "per_fault_accuracy": per_fault_accuracy,
        "output_dir": str(paths["root"]),
    }


def main() -> None:
    """命令行入口。"""
    args = _parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
