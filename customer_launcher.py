from __future__ import annotations

"""客户版一键启动入口。

这个入口专门给打包后的客户程序使用：
- 启动时弹出中文输入框，让客户修改 DAE 训练轮数
- 只暴露一个入口，不让客户接触内部源码结构
- 自动读取程序同目录下的 `data567.mat`
"""

import argparse
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog

from te_dae.DAE_main import run_experiment


DEFAULT_DAE_EPOCHS = 2500


def _program_root() -> Path:
    """返回当前启动程序所在目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _ask_dae_epochs() -> int | None:
    """弹窗询问 DAE 训练轮数。"""
    root = tk.Tk()
    root.withdraw()
    root.update()

    while True:
        value = simpledialog.askstring(
            "DAE 参数设置",
            "请输入 DAE 训练轮数，默认 2500 次：",
            initialvalue=str(DEFAULT_DAE_EPOCHS),
            parent=root,
        )
        if value is None:
            root.destroy()
            return None
        value = value.strip()
        if not value:
            root.destroy()
            return DEFAULT_DAE_EPOCHS
        if value.isdigit() and int(value) > 0:
            root.destroy()
            return int(value)
        messagebox.showerror("输入错误", "DAE 训练轮数必须是正整数，请重新输入。", parent=root)


def _show_info(title: str, message: str) -> None:
    """显示中文提示框。"""
    root = tk.Tk()
    root.withdraw()
    root.update()
    messagebox.showinfo(title, message, parent=root)
    root.destroy()


def main() -> None:
    """客户版程序入口。"""
    dae_epochs = _ask_dae_epochs()
    if dae_epochs is None:
        return

    program_root = _program_root()
    data_path = program_root / "data567.mat"
    if not data_path.exists():
        _show_info(
            "缺少数据文件",
            "未找到 data567.mat。\n请把 data567.mat 放在程序所在目录后再运行。",
        )
        return

    _show_info(
        "开始运行",
        "程序将开始完整运行。\n"
        "所有图片、指标和模型都会自动生成到 outputs 文件夹中。\n"
        "运行时间较长，请耐心等待。",
    )

    args = argparse.Namespace(
        dae_epochs=dae_epochs,
        clf_epochs=300,
        dae_lr=0.0001,
        clf_lr=0.0001,
        wuc=0.01,
        seed=42,
        device="auto",
        output_dir="outputs",
        dae_feature_mode="bottleneck_relu",
        data_path=str(data_path),
    )

    try:
        result = run_experiment(args)
    except Exception as exc:
        _show_info("运行失败", f"程序运行失败：\n{exc}")
        raise

    _show_info(
        "运行完成",
        "程序已运行完成。\n"
        f"平均准确率：{result['mean_accuracy']:.4f}\n"
        f"结果目录：{result['output_dir']}",
    )


if __name__ == "__main__":
    main()
