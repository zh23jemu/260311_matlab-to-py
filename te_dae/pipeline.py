from __future__ import annotations

"""旧主流程模块名的兼容包装层。

当前真正的主流程已经迁移到 `te_dae.run_experiment`。
保留这个文件是为了兼容以前的调用方式，例如：
- `from te_dae.pipeline import main`
- 旧文档或脚本中直接引用 `te_dae.pipeline`

本文件本身不再承载业务逻辑。
"""

from te_dae.run_experiment import main, run_experiment

__all__ = ["main", "run_experiment"]


if __name__ == "__main__":
    main()
