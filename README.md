# TE DAE Python

本项目用于复现 `去噪自编码器py编程.docx` 中描述的工作流程，并给出对应的 Python 实现版本。

## Python 版本

本项目建议使用 `Python 3.11`。

当前项目虚拟环境实际使用的版本为：

```text
Python 3.11.0
```

为避免依赖兼容性问题，不建议使用过低或差异过大的 Python 版本。

## 环境安装

在项目根目录下执行以下命令安装依赖：

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 快速冒烟测试

如果只想确认程序是否能够正常运行，可以执行一个很短的测试：

```powershell
venv\Scripts\python.exe main.py --dae-epochs 2 --clf-epochs 2 --device cpu
```

该命令不会进行完整训练，只用于检查数据读取、模型构建、训练流程和结果导出是否正常。

## 完整训练

如果要在本地执行完整训练，可以使用以下命令：

```powershell
venv\Scripts\python.exe main.py --dae-epochs 2500 --clf-epochs 300 --device cpu
```

如果本机支持 CUDA，也可以使用 GPU：

```powershell
venv\Scripts\python.exe main.py --dae-epochs 2500 --clf-epochs 300 --device cuda
```

## 程序执行步骤

如果需要从头运行本项目，建议按照以下步骤执行：

### 第 1 步：进入项目目录

确保当前命令行所在目录为项目根目录：

```powershell
cd C:\Coding\260311_matlab-to-py
```

### 第 2 步：确认 Python 虚拟环境可用

如果项目中的虚拟环境已经创建完成，可以直接使用：

```powershell
venv\Scripts\python.exe --version
```

如果能够正确输出 Python 版本号，说明当前虚拟环境可正常使用。

### 第 3 步：安装依赖

第一次运行项目时，需要先安装依赖：

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 第 4 步：确认数据文件存在

程序默认读取以下数据文件：

```text
CNN/data567.mat
```

因此在运行前需要确认该文件已经存在于项目目录中，否则程序将无法完成数据加载。

### 第 5 步：先执行一次快速测试

建议先运行一个短周期测试，确认程序流程正常：

```powershell
venv\Scripts\python.exe main.py --dae-epochs 2 --clf-epochs 2 --device cpu
```

这一步主要用于检查以下内容是否正常：

- 数据是否能够正确读取
- 模型是否能够正常构建
- 训练与评估流程是否能够完整跑通
- `outputs/` 目录是否能够正常生成结果

### 第 6 步：执行正式训练

如果快速测试没有问题，再执行正式训练：

```powershell
venv\Scripts\python.exe main.py --dae-epochs 2500 --clf-epochs 300 --device cpu
```

如果你希望使用当前项目中已经验证效果更好的参数组合，也可以显式指定参数运行，例如：

```powershell
venv\Scripts\python.exe main.py --dae-epochs 2500 --clf-epochs 1000 --dae-lr 0.0001 --clf-lr 0.0002 --wuc 0.025 --seed 42 --dae-feature-mode bottleneck_relu --device cpu --output-dir outputs
```

该组参数对应当前项目中已经得到的最佳结果配置。

### 第 7 步：查看终端输出信息

程序运行过程中会在终端输出日志，用于帮助判断训练进度和结果情况，主要包括：

- `[PIPELINE]`
  表示当前流程阶段，例如读取数据、编码特征、保存结果等。

- `[DAE]`
  表示降噪自编码器训练过程中的损失变化。

- `[CLF]`
  表示分类器训练过程中的损失和准确率变化。

- `[EVAL]`
  表示各故障类别在测试集上的准确率。

- `[RESULT]`
  表示最终平均准确率以及输出目录位置。

### 第 8 步：查看输出文件

训练完成后，可以到 `outputs/` 目录中查看结果文件：

- `outputs/models/`
  保存训练后的模型参数。

- `outputs/metrics/metrics.json`
  保存本次训练的指标结果。

- `outputs/figures/`
  保存报告中使用的图片，包括热值图、训练过程图和数据图。

### 第 9 步：整理正式交付结果

如果需要整理正式结果用于提交或同步到 Git，建议从 `outputs/` 中挑选主要结果复制到 `report_assets/`，通常至少包括：

- `metrics.json`
- 最终热值图

本项目当前推荐的正式结果文件为：

- `report_assets/metrics_bottleneck_s42_w0025_clr2e4_ce1000.json`
- `report_assets/figure_4_12_bottleneck_s42_w0025_clr2e4_ce1000.png`

### 第 10 步：查看最终报告

如果需要查看最终整理好的报告，可直接打开：

```text
PYTHON_DAE_REPORT.docx
```

该文件包含 Python 实现流程、关键代码说明和最终实验结果。

## 输出结果

程序运行完成后，默认会在 `outputs/` 目录下生成以下内容：

- `outputs/models/dae.pt`
  训练完成后的降噪自编码器模型参数。

- `outputs/models/classifier.pt`
  训练完成后的分类器模型参数。

- `outputs/metrics/metrics.json`
  当前实验的指标结果，包括平均准确率、各故障类别准确率以及训练参数。

- `outputs/figures/figure_4_2.png` 到 `outputs/figures/figure_4_12.png`
  论文式报告中使用的主要图像，包括数据标准化图、训练过程图、网络结构图和热值图。

- `outputs/figures/dae_training.png`
  DAE 训练过程图。

## 交付给客户的文件说明

如果需要将 Python 版本整理后交付给客户，建议交付以下文件，并说明各文件用途。

- `main.py`
  项目总入口文件。运行该文件即可启动完整流程，并调用 `te_dae/` 目录中的主流程代码。

- `te_dae/data.py`
  用于读取 `CNN/data567.mat`，删除第 46 列和第 50 列，完成第一次 z-score 标准化，并为 DAE 训练添加高斯噪声。

- `te_dae/models.py`
  定义降噪自编码器和分类神经网络的模型结构。

- `te_dae/train.py`
  包含模型训练循环、特征提取、分类预测以及混淆矩阵构造等核心训练逻辑。

- `te_dae/pipeline.py`
  负责组织完整流程，包括数据读取、DAE 训练、编码特征提取、分类器训练、结果评估以及文件导出。

- `te_dae/plotting.py`
  用于生成报告中的各类图像，包括数据曲线图、训练过程图、网络结构图和热值图。

- `requirements.txt`
  项目运行所需的 Python 依赖列表。

- `README.md`
  项目说明文档，包含环境安装、运行方法、输出说明以及交付文件说明。

- `PYTHON_DAE_REPORT.docx`
  最终提交给客户的 Word 报告，内容包括 Python 实现过程、关键代码说明和实验结果展示。

- `report_assets/metrics_bottleneck_s42_w0025_clr2e4_ce1000.json`
  当前选定的最终指标文件，记录最佳配置下的平均准确率以及各故障类别准确率。

- `report_assets/figure_4_12_bottleneck_s42_w0025_clr2e4_ce1000.png`
  当前选定的最终热值图，对应最佳实验结果。

以下文件根据客户实际需求决定是否交付：

- `run_te_dae.slurm`
  仅在客户需要通过 Slurm 集群运行训练任务时才需要交付。

- `CNN/data567.mat`
  仅在客户本地没有 TE 数据集文件时才需要一起交付。

## 说明

本项目中的 `outputs/` 目录主要用于本地运行和中间结果保存，通常不建议作为正式交付内容直接打包给客户。正式交付时建议以代码、最终报告和最终结果文件为主。
