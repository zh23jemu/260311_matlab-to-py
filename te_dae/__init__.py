"""TE DAE Python 包入口。

当前包内部已经按职责拆分为更直观的模块，例如：
- `run_experiment`：总流程
- `load_data`：数据读取与预处理
- `train_dae`：DAE 训练
- `train_classifier`：分类器训练
- `plot_results`：结果出图

这里对外导出最常用的主入口函数，方便包级调用。
"""

from te_dae.run_experiment import main, run_experiment

__all__ = ["main", "run_experiment"]
