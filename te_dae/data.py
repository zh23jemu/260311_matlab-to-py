from __future__ import annotations

"""旧数据模块名的兼容包装层。

项目重组后，真正的数据处理实现已经移动到 `te_dae.load_data`。
保留这个文件的原因是避免旧代码或旧文档中的导入路径立刻失效。

也就是说：
- 新代码建议直接使用 `te_dae.load_data`
- 旧代码继续 `import te_dae.data` 也仍然可以工作
"""

from te_dae.constants import ALL_IDS, DROP_COLS, TRAIN_FAULT_IDS
from te_dae.load_data import DatasetBundle, add_noise, load_te_dataset

__all__ = [
    "ALL_IDS",
    "DROP_COLS",
    "TRAIN_FAULT_IDS",
    "DatasetBundle",
    "add_noise",
    "load_te_dataset",
]
