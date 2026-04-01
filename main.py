"""项目总入口文件。

这个文件故意保持得非常薄，只负责把命令行运行请求转发给
`te_dae.run_experiment.main()`。

这样做的目的有两个：
1. 用户在项目根目录里一眼就能找到真正的启动入口。
2. 入口保持稳定，即使内部模块继续调整，`python main.py` 的用法也不需要变化。
"""

from te_dae.run_experiment import main


if __name__ == "__main__":
    # 这里不直接写业务逻辑，避免入口文件越来越重。
    # 所有真正的实验流程都在 te_dae.run_experiment.main 中统一维护。
    main()
