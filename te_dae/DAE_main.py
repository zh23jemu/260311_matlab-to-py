from __future__ import annotations

"""单次 TE DAE 实验的总调度模块。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from te_dae.dae_result_predict import (
    build_confusion,
    extract_encoded_features,
    plot_heatmap,
    plot_label_sequence,
    plot_matrix_lines,
    plot_network_diagram,
    plot_training_history,
    predict_classes,
    set_seed,
    stack_classifier_features,
    train_autoencoder,
    train_classifier,
)
from te_dae.daelayer import ClassifierNet, DenoisingAutoencoder
from te_dae.data_standar import TRAIN_FAULT_IDS, add_noise, load_te_dataset, test_slice_count


def _output_dirs(root: Path, output_dir: str) -> dict[str, Path]:
    """创建并返回本次实验使用的输出目录结构。

    输出目录固定拆成：
    - `figures`：图片
    - `metrics`：指标 JSON
    - `models`：模型参数和相关统计量

    这样后续不论是本地查看、批量实验归档还是同步结果文件，
    目录结构都会比较稳定。
    """
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
    """以统一格式保存 JSON 文件。

    统一使用 UTF-8 和缩进格式，主要是为了让结果文件既适合程序读取，
    也适合人直接打开查看。
    """
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这里把关键实验参数都暴露出来，是为了同时兼顾两种场景：
    - 本地手动调试
    - 通过脚本或 Slurm 做批量实验
    """
    parser = argparse.ArgumentParser(description="TE DAE Python pipeline")
    parser.add_argument("--dae-epochs", type=int, default=2500)
    parser.add_argument("--clf-epochs", type=int, default=300)
    parser.add_argument("--dae-lr", type=float, default=0.0001)
    parser.add_argument("--clf-lr", type=float, default=0.0001)
    parser.add_argument("--wuc", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--dae-feature-mode", default="bottleneck_relu", choices=["fc3_linear", "bottleneck_relu"])
    return parser.parse_args()


def _resolve_device(choice: str) -> torch.device:
    """根据命令行选择解析训练设备。

    即使用户显式传了 `cuda`，这里也会在 CUDA 不可用时自动回退到 CPU，
    避免程序因为环境差异直接报错退出。
    """
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """执行一次完整实验，并返回摘要结果。

    这是整个项目真正的主流程函数。
    它负责把“数据准备 -> DAE 训练 -> 特征提取 -> 分类训练 -> 评估 -> 导出”
    串成一次完整实验。
    """
    root = Path(__file__).resolve().parent.parent
    paths = _output_dirs(root, args.output_dir)
    set_seed(args.seed)

    # 数据文件路径固定放在项目根目录下，保持和当前仓库结构一致。
    mat_path = root / "CNN" / "data567.mat"
    device = _resolve_device(args.device)

    print(f"[PIPELINE] root={root}")
    print(f"[PIPELINE] output_dir={paths['root']}")
    print(f"[PIPELINE] seed={args.seed} wuc={args.wuc} device={device}")
    print(f"[PIPELINE] dae_lr={args.dae_lr} clf_lr={args.clf_lr} clf_epochs={args.clf_epochs}")
    print(f"[PIPELINE] dae_feature_mode={args.dae_feature_mode}")
    print(f"[PIPELINE] loading dataset from {mat_path}")

    # 第一步：读取原始 TE 数据，并完成删列和第一次标准化。
    bundle = load_te_dataset(mat_path)
    print(
        f"[PIPELINE] dataset loaded: dae_train_shape={bundle.dae_train_data.shape}, "
        f"train_faults={len(bundle.train_standardized)}, test_faults={len(bundle.test_standardized)}"
    )

    # 第二步：给 DAE 训练输入加噪，让模型学习“去噪重建”。
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

    # 第三步：使用训练好的 DAE 对 train/test 数据提取编码特征。
    print("[PIPELINE] encoding train and test features with DAE")
    encoded_train = {
        fault_id: extract_encoded_features(dae_model, values, device=device, mode=args.dae_feature_mode)
        for fault_id, values in bundle.train_standardized.items()
    }
    encoded_test = {
        fault_id: extract_encoded_features(dae_model, values, device=device, mode=args.dae_feature_mode)
        for fault_id, values in bundle.test_standardized.items()
    }

    # 第四步：对编码特征再做一次标准化，统计量仍然只来自 d00。
    # 这一步对应原始工作流中的“二次标准化”。
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

    # 第五步：把各故障的训练特征拼成一个统一的分类训练矩阵。
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

    # 第六步：逐故障评估测试集准确率，并收集整体预测结果用于热力图。
    print("[PIPELINE] evaluating classifier on test faults")
    per_fault_accuracy: dict[str, float] = {}
    y_true_all = []
    y_pred_all = []
    for fault_id in TRAIN_FAULT_IDS:
        # 这里使用特殊样本数规则，是为了严格对齐 MATLAB 口径。
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

    # 将逐类结果拼成完整标签序列，后续统一计算平均准确率和混淆矩阵。
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    mean_accuracy = float(np.mean(list(per_fault_accuracy.values())))
    heat = build_confusion(y_true, y_pred, labels=list(range(1, len(TRAIN_FAULT_IDS) + 1)))
    fault_labels = [f"F{fault_id}" for fault_id in TRAIN_FAULT_IDS]

    # 第七步：保存模型、标准化统计量和实验指标，方便后续复现与对比。
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

    # 第八步：生成报告所需图像。
    # 这些图并不都参与训练本身，但对于交付和结果解释很重要。
    print("[PIPELINE] rendering figures")
    plot_matrix_lines(bundle.train_raw[0], "Figure 1.2 Unstandardized d00", paths["figures"] / "figure_4_2.png")
    plot_matrix_lines(bundle.train_standardized[0], "Figure 1.3 Standardized d00", paths["figures"] / "figure_4_3.png")
    plot_matrix_lines(bundle.train_standardized[0], "Figure 1.4 d00 Standardized Data", paths["figures"] / "figure_4_4.png")
    plot_matrix_lines(bundle.train_standardized[1], "Figure 1.5 d01 Standardized Data", paths["figures"] / "figure_4_5.png")
    plot_matrix_lines(bundle.train_standardized[2], "Figure 1.6 d02 Standardized Data", paths["figures"] / "figure_4_6.png")
    plot_matrix_lines(clf_train_x, "Figure 1.7 Training Feature Standardization", paths["figures"] / "figure_4_7.png")
    plot_matrix_lines(
        np.vstack(
            [bundle.classifier_test_features[fault_id][: min(200, bundle.classifier_test_features[fault_id].shape[0])] for fault_id in TRAIN_FAULT_IDS]
        ),
        "Figure 1.8 Testing Feature Standardization",
        paths["figures"] / "figure_4_8.png",
        max_rows=200,
    )
    plot_label_sequence(y_true, "Figure 1.9 Data Labels", paths["figures"] / "figure_4_9.png")
    plot_network_diagram(paths["figures"] / "figure_4_10.png")
    plot_training_history(clf_history.losses, clf_history.accuracies, "Figure 1.11 Training Process", paths["figures"] / "figure_4_11.png")
    plot_heatmap(heat, fault_labels, "Figure 1.12 Heatmap", paths["figures"] / "figure_4_12.png")
    plot_training_history(dae_history.losses, dae_history.accuracies, "DAE Training", paths["figures"] / "dae_training.png")

    print(f"[RESULT] mean_accuracy={mean_accuracy:.4f}")
    print(f"[RESULT] saved outputs to {paths['root']}")
    return {
        "mean_accuracy": mean_accuracy,
        "per_fault_accuracy": per_fault_accuracy,
        "output_dir": str(paths["root"]),
    }


def main() -> None:
    """命令行入口函数。

    保持这个入口很薄，只负责：
    1. 解析参数
    2. 调用主流程
    """
    args = _parse_args()
    run_experiment(args)


__all__ = ["main", "run_experiment"]


if __name__ == "__main__":
    main()
