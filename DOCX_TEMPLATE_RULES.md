# DOCX Template Rules

## Purpose

`去噪自编码器py编程.docx` 是本项目后续生成 Word 报告时的母版模板。

后续所有新的 Python 报告，都应当在保留该文档整体结构、章节顺序、图题风格和正文语气的前提下，只替换与当前 Python 实现和实验结果相关的内容。

本文件用于规定：

- 哪些内容固定保留
- 哪些内容需要按当前实验替换
- 图片、代码和结果指标分别对应哪些项目文件

## Fixed Structure

以下内容应尽量保持与 `去噪自编码器py编程.docx` 一致：

1. 章节层级
   - `4 基于 Python 的降噪自编码器故障诊断仿真实现`
   - `4.1 TE 数据集说明`
   - `4.2 仿真实现`
   - `4.2.1 数据标准化处理`
   - `4.2.2 添加噪声`
   - `4.2.3 DAE 模型和特征提取`
   - `4.2.4 定义分类标签和训练神经网络`
   - `4.3 Python 仿真结果`
   - `4.4 本章小结`

2. 正文写法
   - 优先使用论文式表述
   - 常用句式包括：
     - “在本章中……”
     - “本文采用……”
     - “程序如下：”
     - “结果如下图所示：”
     - “由此可见……”
     - “综上所述……”

3. 图题风格
   - 统一保留“图 4.x    名称”这一风格
   - 图题应位于图片前一行或与图片配套出现

4. 章节内容顺序
   - 先叙述原理或步骤
   - 再给出程序
   - 再给出图或结果说明

## Replaceable Sections

以下内容应根据当前 Python 项目实际情况替换：

1. 数据来源描述
   - 固定写 `CNN/data567.mat`
   - 固定写训练集使用 `_6`
   - 固定写测试集由 `_5` 和 `_7` 合并

2. 代码片段
   - 不再引用 MATLAB 代码
   - 改为引用当前 Python 代码中的等价实现

3. 模型参数
   - DAE 学习率、轮数、批次大小
   - 分类器学习率、轮数、批次大小
   - 特征提取方式
   - 噪声强度

4. 仿真结果
   - 使用当前选定实验的 `metrics_*.json`
   - 使用当前选定实验的热值图
   - 如果过程图来自本地一次完整导出，则默认使用 `outputs/figures/`

5. 小结
   - 需要与当前最终配置、当前最终准确率保持一致

## Canonical Code Sources

生成报告时，代码引用优先来自以下文件：

- [main.py](/C:/Coding/260311_matlab-to-py/main.py)
- [data.py](/C:/Coding/260311_matlab-to-py/te_dae/data.py)
- [models.py](/C:/Coding/260311_matlab-to-py/te_dae/models.py)
- [train.py](/C:/Coding/260311_matlab-to-py/te_dae/train.py)
- [pipeline.py](/C:/Coding/260311_matlab-to-py/te_dae/pipeline.py)
- [plotting.py](/C:/Coding/260311_matlab-to-py/te_dae/plotting.py)

其中推荐的章节映射如下：

- `4.1` 使用 [data.py](/C:/Coding/260311_matlab-to-py/te_dae/data.py)
- `4.2.1` 使用 [data.py](/C:/Coding/260311_matlab-to-py/te_dae/data.py)
- `4.2.2` 使用 [data.py](/C:/Coding/260311_matlab-to-py/te_dae/data.py)
- `4.2.3` 使用 [models.py](/C:/Coding/260311_matlab-to-py/te_dae/models.py) 和 [pipeline.py](/C:/Coding/260311_matlab-to-py/te_dae/pipeline.py)
- `4.2.4` 使用 [data.py](/C:/Coding/260311_matlab-to-py/te_dae/data.py)、[models.py](/C:/Coding/260311_matlab-to-py/te_dae/models.py)、[pipeline.py](/C:/Coding/260311_matlab-to-py/te_dae/pipeline.py)
- `4.3` 使用 `report_assets/` 中当前最终选定结果

## Figure Mapping

如果当前 `outputs/figures/` 目录存在完整导出的过程图，则按如下方式映射：

- 图 4.2 -> `outputs/figures/figure_4_2.png`
- 图 4.3 -> `outputs/figures/figure_4_3.png`
- 图 4.4 -> `outputs/figures/figure_4_4.png`
- 图 4.5 -> `outputs/figures/figure_4_5.png`
- 图 4.6 -> `outputs/figures/figure_4_6.png`
- 图 4.7 -> `outputs/figures/figure_4_7.png`
- 图 4.8 -> `outputs/figures/figure_4_8.png`
- 图 4.9 -> `outputs/figures/figure_4_9.png`
- 图 4.10 -> `outputs/figures/figure_4_10.png`
- 图 4.11 -> `outputs/figures/figure_4_11.png`
- 图 4.12 -> `report_assets/` 中最终选定热值图

默认最终热值图使用当前最佳实验：

- [figure_4_12_bottleneck_s42_w0025_clr2e4_ce1000.png](/C:/Coding/260311_matlab-to-py/report_assets/figure_4_12_bottleneck_s42_w0025_clr2e4_ce1000.png)

## Result Source Rules

默认以当前最佳结果作为报告中的正式结果来源：

- [metrics_bottleneck_s42_w0025_clr2e4_ce1000.json](/C:/Coding/260311_matlab-to-py/report_assets/metrics_bottleneck_s42_w0025_clr2e4_ce1000.json)

当前默认正式结果参数为：

- 特征提取方式：瓶颈层激活特征
- 随机种子：`42`
- 噪声强度：`0.025`
- DAE 学习率：`0.0001`
- 分类器学习率：`0.0002`
- 分类器训练轮数：`1000`

当前默认正式结果指标为：

- `mean_accuracy = 0.9494705882352942`
- `F8 = 0.8670`
- `F11 = 0.8730`
- `F13 = 0.7830`

## Replacement Checklist

每次基于模板生成新报告时，至少检查以下内容：

1. 标题是否仍保持原文档风格
2. 章节编号是否连续且与原模板一致
3. MATLAB 字样是否已改成 Python
4. 代码片段是否全部来自当前 Python 项目
5. 图 4.2 到图 4.11 是否对应本地导出过程图
6. 图 4.12 是否对应本次选定实验结果
7. 指标数值是否与 `metrics_*.json` 一致
8. 小结中的最终配置是否与当前正式结果一致

## Recommended Workflow

后续生成新 Word 报告时，推荐遵循以下流程：

1. 以 `去噪自编码器py编程.docx` 作为母版模板
2. 先生成或更新 `PYTHON_DAE_REPORT.md`
3. 按本文件规则核对章节、代码、图片和指标
4. 再将内容写入新的 `.docx`
5. 最后进行一次人工排版检查
