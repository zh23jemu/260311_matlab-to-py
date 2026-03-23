"""项目总入口。"""

from te_dae.pipeline import main


if __name__ == "__main__":
    # 所有训练、评估、画图、导出逻辑都集中在 pipeline.main 中。
    main()
