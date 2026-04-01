from __future__ import annotations

"""旧绘图模块名的兼容包装层。

真正的绘图实现已经集中到 `te_dae.plot_results`。
这里保留旧模块名，只负责重新导出相同函数，避免旧导入路径失效。
"""

from te_dae.plot_results import (
    plot_heatmap,
    plot_label_sequence,
    plot_matrix_lines,
    plot_network_diagram,
    plot_training_history,
)

__all__ = [
    "plot_heatmap",
    "plot_label_sequence",
    "plot_matrix_lines",
    "plot_network_diagram",
    "plot_training_history",
]
