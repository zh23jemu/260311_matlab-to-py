# 1 基于 Python 的降噪自编码器故障诊断仿真实现

## 1.1 TE 数据集说明

在本章中，仿照 `去噪自编码器py编程.docx` 中的实现过程，利用当前项目中的 Python 程序对田纳西-伊斯曼（TE）过程故障诊断进行仿真分析。本文所采用的数据集为 `CNN/data567.mat`，其中包含训练数据与测试数据。训练集样本选取 `d00_6` 至 `d20_6`，测试集样本则由 `d00_5` 至 `d20_5` 与 `d00_7` 至 `d20_7` 两组数据合并而成。

由于部分变量对本次仿真分析无意义，因此参照原文档中的处理方式，将原始 52 维数据中的第 46 列和第 50 列删除，从而得到 50 维输入变量。其 Python 实现见 [data.py](/C:/Coding/260311_matlab-to-py/te_dae/data.py)，程序如下：

```python
TRAIN_FAULT_IDS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
ALL_IDS = list(range(21))
DROP_COLS = [45, 49]


def _select_features(array: np.ndarray) -> np.ndarray:
    keep = [idx for idx in range(array.shape[1]) if idx not in DROP_COLS]
    return array[:, keep].astype(np.float32)
```

训练集和测试集的选取方式如下：

```python
train_raw = {
    fault_id: _select_features(data[_fault_key("d", fault_id, "6")]) for fault_id in ALL_IDS
}

test_raw = {}
for fault_id in ALL_IDS:
    part_5 = _select_features(data[_fault_key("d", fault_id, "5")])
    part_7 = _select_features(data[_fault_key("d", fault_id, "7")])
    test_raw[fault_id] = np.vstack([part_5, part_7]).astype(np.float32)
```

由此可见，Python 版本在数据来源、样本划分方式以及输入维度处理方面，与原文档描述的实现思路是一致的。

## 1.2 仿真实现

在 Python 仿真过程中，本文按照原文档中的技术路线展开，即首先对 TE 数据进行标准化处理，然后添加噪声训练 DAE 模型，接着提取编码特征并再次标准化，最后将提取后的特征输入神经网络进行分类识别。整体实现步骤与 `去噪自编码器py编程.docx` 中给出的 MATLAB 仿真过程相对应。

### 1.2.1 数据标准化处理

在 TE 过程中，数据标准化是不可或缺的一部分。由于不同过程变量具有不同的量纲和数值范围，若直接将原始数据输入网络，容易使某些变量在训练过程中占据过大的影响权重，从而降低模型稳定性与识别效果。因此，本文与原文档一致，采用 z-score 标准化方法对数据进行预处理。

通常采取的标准化手段是 z-score 标准化，也称为标准差标准化。经过标准化后的数据，其均值接近于零，标准差接近于一，从而能够有效减小不同变量尺度差异对模型训练的不利影响。本文在标准化过程中，以正常工况 `d00` 的训练样本作为基准，先计算均值与标准差，再将其作用于全部训练数据和测试数据。该部分 Python 实现见 [data.py](/C:/Coding/260311_matlab-to-py/te_dae/data.py)，程序如下：

```python
base_mean = train_raw[0].mean(axis=0)
base_std = train_raw[0].std(axis=0, ddof=1)
base_std[base_std == 0] = 1.0

train_standardized = {
    fault_id: ((values - base_mean) / base_std).astype(np.float32)
    for fault_id, values in train_raw.items()
}
test_standardized = {
    fault_id: ((values - base_mean) / base_std).astype(np.float32)
    for fault_id, values in test_raw.items()
}
```

图 1.2 为未标准化处理的数据：

![图 1.2 未标准化处理的数据](outputs/figures/figure_4_2.png)

图 1.3 为标准化处理过的数据：

![图 1.3 标准化处理后的数据](outputs/figures/figure_4_3.png)

标准化处理完数据后，得到 `d00`、`d01`、`d02` 的标准化结果如下图所示。

图 1.4 `d00` 标准化数据：

![图 1.4 d00 标准化数据](outputs/figures/figure_4_4.png)

图 1.5 `d01` 标准化数据：

![图 1.5 d01 标准化数据](outputs/figures/figure_4_5.png)

图 1.6 `d02` 标准化数据：

![图 1.6 d02 标准化数据](outputs/figures/figure_4_6.png)

综上所述，可以看出数据经过标准化后基本处于零值附近波动，说明标准化处理有效减小了原始数据之间的尺度差异。

### 1.2.2 添加噪声

在 DAE 模型训练过程中，通过向输入样本中加入噪声，可以提高数据的鲁棒性。所谓鲁棒性，是指模型对于噪声扰动的抵抗能力。DAE 的思想在于：通过学习带噪输入到无噪输出的映射关系，使网络能够更好地学习到数据本质特征，而不是过度依赖输入中的局部细节。

原文档采用 `sqrt(wuc) * randn(...)` 的形式向样本添加高斯噪声。当前 Python 程序对这一过程进行了等价实现，程序见 [data.py](/C:/Coding/260311_matlab-to-py/te_dae/data.py)，如下所示：

```python
def add_noise(data: np.ndarray, wuc: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(wuc), size=data.shape)
    return (data + noise).astype(np.float32)
```

在本文最终实验中，噪声强度参数取 `wuc = 0.025`。实验表明，该噪声强度在整体准确率以及难分类故障识别效果之间取得了较好的平衡。

### 1.2.3 DAE 模型和特征提取

一、训练样本与正常样本建立网络模型并训练网络。

在完成数据标准化和噪声添加之后，本文进一步利用降噪自编码器对 TE 过程数据进行特征提取。降噪自编码器的基本思想是：将带噪输入样本送入网络，通过编码器压缩到低维特征空间，再由解码器将其重建回原始输入，从而使网络在去噪重建的过程中学习到更具代表性的深层特征。与原文档中的实现思路一致，本文选取 50 维标准化数据作为网络输入，并采用六层全连接结构完成编码与解码。

本文学习率设置为 `0.0001`，迭代次数设置为 `2500` 次，批次尺寸设置为 `32`，隐含层第一层设置为 `50`，第二层设置为 `45`，第三层设置为 `40`。其中，编码器部分将输入特征逐步压缩至 40 维瓶颈层，解码器部分再按对称结构恢复原始输入维度。其 Python 实现见 [models.py](/C:/Coding/260311_matlab-to-py/te_dae/models.py)，程序部分如下：

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 50) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 45)
        self.fc3 = nn.Linear(45, 40)
        self.fc4 = nn.Linear(40, 45)
        self.fc5 = nn.Linear(45, 50)
        self.fc6 = nn.Linear(50, input_dim)
        self.act = nn.LeakyReLU()
```

由上式可以看出，该网络的整体结构可概括为 `50-45-40-45-50`，并在各层之间引入 `LeakyReLU` 激活函数。虽然 Python 中的层定义与原文档中的 MATLAB 写法不同，但二者在网络层数、隐含层神经元设置以及编码解码的总体思想上是一致的。

（1）参数随机化，程序如下：

在网络训练过程中，为了避免样本按故障类别顺序排列对训练过程造成影响，需要在每个训练阶段对样本进行随机打乱。当前 Python 程序通过 `DataLoader` 的 `shuffle=True` 和固定随机种子生成器来完成这一过程，其实现见 [train.py](/C:/Coding/260311_matlab-to-py/te_dae/train.py)，程序如下：

```python
def _build_loader(features: np.ndarray, targets: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    if targets is None:
        dataset = TensorDataset(feature_tensor, feature_tensor)
    else:
        target_tensor = torch.from_numpy(targets)
        dataset = TensorDataset(feature_tensor, target_tensor)

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
```

由此可见，当前 Python 实现与原文档中“先随机化样本，再进入网络训练”的处理方式是一致的，并且在固定随机种子的条件下能够保证实验可复现。

（2）设置网络层参数，6 层全连接层程序如下：

按照原文档的写法，需要分别设置输入层尺寸、各层全连接层神经元个数以及输出维度。对于当前 Python 版本来说，这部分参数已经直接体现在模型定义中，即输入层维度为 50，第一隐含层为 50，第二隐含层为 45，第三隐含层为 40，解码层对称恢复为 45 和 50，最后输出层恢复到输入维度。对应关系如下：

```python
self.fc1 = nn.Linear(input_dim, 50)
self.fc2 = nn.Linear(50, 45)
self.fc3 = nn.Linear(45, 40)
self.fc4 = nn.Linear(40, 45)
self.fc5 = nn.Linear(45, 50)
self.fc6 = nn.Linear(50, input_dim)
```

其中，`input_dim` 表示输入数据维度，在本实验中固定为 50。由此可得，Python 版在网络层参数设置方面与原文档所描述的 6 层全连接网络是一一对应的。

（3）网络层代码，主要包括全连接层，网络层结构 `50-45-40-45-50`，激活函数选择 `LeakyReLU`，程序如下：

```python
def encode(self, x: torch.Tensor) -> torch.Tensor:
    x = self.act(self.fc1(x))
    x = self.act(self.fc2(x))
    x = self.act(self.fc3(x))
    return x


def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encode(x)
    x = self.act(self.fc4(x))
    x = self.act(self.fc5(x))
    x = self.fc6(x)
    return x
```

由上述程序可以看出，输入样本首先经过三层编码结构提取出瓶颈特征，然后再经三层解码结构重建原始输入，从而完成去噪自编码过程。该网络既能够保留输入数据的主要信息，又能够在特征压缩过程中学习到更具判别能力的隐藏表示。

（4）网络参数设置和训练网络程序如下：

本文选用随机梯度下降法对 DAE 网络进行训练。训练时，输入为带噪样本，输出目标为未加噪的标准化样本，因此损失函数采用均方误差损失。该部分主程序见 [pipeline.py](/C:/Coding/260311_matlab-to-py/te_dae/pipeline.py)，训练子程序见 [train.py](/C:/Coding/260311_matlab-to-py/te_dae/train.py)，程序如下：

```python
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
```

```python
def train_autoencoder(
    model: nn.Module,
    noisy_features: np.ndarray,
    clean_features: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_interval: int = 100,
) -> TrainHistory:
    loader = _build_loader(noisy_features, clean_features, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.MSELoss()
```

从中可以看出，Python 版本在训练过程中同样采用了带噪输入与干净输出配对训练的方式，并通过随机梯度下降和动量项更新网络参数。这与原文档中利用 `sgdm` 训练 DAE 网络的思路相一致。

（5）查看特征代码如下：

DAE 训练完成后，需要将训练数据输入网络，提取编码特征。原文档中通过查看第 3 层特征来完成特征提取；在当前 Python 程序中，则通过 `extract_features` 函数统一封装两种特征提取方式，以便与原文档中的分层查看过程进行对照。相关程序见 [models.py](/C:/Coding/260311_matlab-to-py/te_dae/models.py)，如下所示：

```python
def extract_features(self, x: torch.Tensor, mode: str = "fc3_linear") -> torch.Tensor:
    if mode == "fc3_linear":
        return self.encode_fc3_linear(x)
    if mode == "bottleneck_relu":
        return self.encode(x)
    raise ValueError(f"Unsupported feature extraction mode: {mode}")
```

经过多轮实验对比，本文最终选取瓶颈层激活输出作为特征提取方式，因为该方式在综合准确率和关键故障识别效果方面更优。该过程与原文档中“训练完成后查看编码层特征”的目标是一致的。

二、训练数据放入网络模型进行预测。

将得到的训练数据输入训练好的 DAE 网络后，即可得到对应的编码特征。该部分程序见 [train.py](/C:/Coding/260311_matlab-to-py/te_dae/train.py)，程序如下：

```python
def encode_features(model: nn.Module, features: np.ndarray, device: torch.device, mode: str = "fc3_linear") -> np.ndarray:
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        encoded = model.extract_features(tensor, mode=mode).cpu().numpy().astype(np.float32)
    return encoded
```

在主流程中，程序会依次对全部训练工况和测试工况进行编码，具体程序如下：

```python
encoded_train = {
    fault_id: encode_features(dae_model, values, device=device, mode=args.dae_feature_mode)
    for fault_id, values in bundle.train_standardized.items()
}
encoded_test = {
    fault_id: encode_features(dae_model, values, device=device, mode=args.dae_feature_mode)
    for fault_id, values in bundle.test_standardized.items()
}
```

三、训练数据编码特征标准化。

与原文档一致，在得到编码特征之后，还需要对编码特征再次进行标准化处理。本文选取正常工况 `d00` 的编码特征作为标准化基准，重新计算其均值和标准差，再将这些统计量应用到所有训练故障特征和测试故障特征上。程序如下：

```python
d00_features = encoded_train[0]
feature_mean = d00_features.mean(axis=0)
feature_std = d00_features.std(axis=0, ddof=1)
feature_std[feature_std == 0] = 1.0
```

```python
bundle.classifier_train_features = {
    fault_id: ((encoded_train[fault_id] - feature_mean) / feature_std).astype(np.float32)
    for fault_id in TRAIN_FAULT_IDS
}
```

训练数据编码特征标准化结果如下图 1.7 所示：

![图 1.7 训练数据编码特征标准化](outputs/figures/figure_4_7.png)

由图可见，经过第二次标准化处理之后，各类训练故障特征在数值范围上进一步趋于一致，从而为后续神经网络分类提供了更加稳定的输入。

四、将测试数据放入网络模型进行预测。

在完成训练数据编码之后，还需要将测试数据输入已经训练好的 DAE 网络，以获取测试样本的编码特征。该过程与训练数据编码过程一致，只是输入数据由训练样本替换为测试样本。测试数据编码特征标准化结果如下图 1.8 所示：

![图 1.8 测试数据编码特征标准化](outputs/figures/figure_4_8.png)

由图可见，测试数据在经过 DAE 编码和标准化处理后，其特征分布与训练数据特征分布保持了较好的统一性，这为后续分类网络的故障识别提供了基础。

五、将提取的训练数据和测试数据编码特征保存，并将保存的文件放入神经网络进行分类。

经过上述处理后，训练故障特征和测试故障特征均已由原始输入转换为低维编码特征。随后，这些标准化后的编码特征将作为分类神经网络的输入，用于后续故障标签定义、分类网络训练以及测试结果预测。

### 1.2.4 定义分类标签和训练神经网络

一、定义分类标签。

根据原文档中的分类设置，本文共定义 17 个故障类别，分别对应 `F1、F2、F4、F5、F6、F7、F8、F10、F11、F12、F13、F14、F16、F17、F18、F19、F20`。在 Python 程序中，故障编号与分类标签的映射关系见 [data.py](/C:/Coding/260311_matlab-to-py/te_dae/data.py)，程序如下：

```python
classifier_labels = {fault_id: index + 1 for index, fault_id in enumerate(TRAIN_FAULT_IDS)}
```

其中，第一个故障对应标签 1，第二个故障对应标签 2，第四个故障对应标签 3，依此类推，直到第二十个故障对应标签 17。也就是说，17 类故障样本都具有一一对应的分类标签。为了便于后续分类网络训练，程序会按照固定顺序将各故障特征拼接起来，并生成与之对应的标签序列。数据标签可视化结果如下图 1.9 所示：

![图 1.9 数据对应标签](outputs/figures/figure_4_9.png)

二、训练神经网络。

将从 DAE 特征提取的训练数据输入神经网络进行训练。本文在 Python 版本中同样采用全连接神经网络对编码特征进行分类识别，其总体思路与原文档中的 MATLAB 分类网络保持一致。分类器输入为 40 维编码特征，输出为 17 个故障类别。主程序如下：

```python
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
```

其中，`train_classifier` 属于分类神经网络的训练子程序，`features` 表示训练数据，`labels` 表示标签，`learning_rate` 表示学习率，`epochs` 表示循环次数。其作用与原文档中的 `NN_layer` 子程序相对应。

分类器的 Python 实现见 [models.py](/C:/Coding/260311_matlab-to-py/te_dae/models.py)，程序如下：

```python
class ClassifierNet(nn.Module):
    def __init__(self, input_dim: int = 40, num_classes: int = 17) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.Tanh(),
            nn.Linear(400, 250),
            nn.Tanh(),
            nn.Linear(250, num_classes),
        )
```

由上式可以看出，分类神经网络由三层全连接层组成，第一层设置 400 个神经元，第二层设置 250 个神经元，第三层输出固定为 17 类；激活函数采用 `Tanh` 函数。这与原文档中“400-250-17”的网络设置是一致的。

（1）训练标签随机化。

将标签分类好之后，每一类数据都在一起，这样训练时容易影响网络学习效果。为了避免这种情况，需要在训练开始前将数据和标签同时随机打乱。当前 Python 程序延续了这一思想，在构造分类器训练数据加载器时设置 `shuffle=True`，使样本在每轮训练中按随机顺序进入网络。其实现见 [train.py](/C:/Coding/260311_matlab-to-py/te_dae/train.py)，程序如下：

```python
loader = _build_loader(features, labels.astype(np.int64), batch_size=batch_size, shuffle=True)
```

```python
def _build_loader(features: np.ndarray, targets: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    if targets is None:
        dataset = TensorDataset(feature_tensor, feature_tensor)
    else:
        target_tensor = torch.from_numpy(targets)
        dataset = TensorDataset(feature_tensor, target_tensor)

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
```

由此可见，Python 版本在训练分类器时同样完成了“训练标签随机化”的处理，并且能够在固定随机种子的条件下保证实验可复现。

（2）设置网络层参数，3 层全连接层程序如下：

按照原文档的写法，分类网络需要设置输入尺寸、第一层神经元个数、第二层神经元个数以及输出类别数。对于当前 Python 版本来说，这些参数已经直接写入网络定义中，对应关系如下：

```python
nn.Linear(input_dim, 400)
nn.Linear(400, 250)
nn.Linear(250, num_classes)
```

其中，`input_dim` 在本实验中固定为 40，表示 DAE 输出的编码特征维度；`num_classes` 固定为 17，表示故障类别数。因此，当前 Python 分类器与原文档中的三层全连接分类网络在结构上是一一对应的。

（3）网络层代码，主要包括全连接层和激活函数，激活函数选取 `Tanh` 函数，程序如下：

```python
self.net = nn.Sequential(
    nn.Linear(input_dim, 400),
    nn.Tanh(),
    nn.Linear(400, 250),
    nn.Tanh(),
    nn.Linear(250, num_classes),
)
```

从上述程序可以看出，编码特征首先经过第一层全连接层映射到高维空间，再通过第二层全连接层进一步提取判别信息，最后输出 17 个类别的分类结果。网络中两次使用 `Tanh` 激活函数，以增强模型的非线性表达能力。

（4）设置网络参数和训练网络程序如下：

根据上文可知，本文采用随机梯度下降法训练分类器。在当前最终实验中，分类网络学习率设置为 `0.0002`，训练轮数设置为 `1000`，批次大小设置为 `300`。训练子程序见 [train.py](/C:/Coding/260311_matlab-to-py/te_dae/train.py)，程序如下：

```python
def train_classifier(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_interval: int = 20,
) -> TrainHistory:
    loader = _build_loader(features, labels.astype(np.int64), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
```

训练过程中，程序会在每个批次中计算前向输出、交叉熵损失和预测类别，并通过梯度下降不断更新网络参数。对应程序如下：

```python
for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()
    predictions = logits.argmax(dim=1)
```

由此可见，Python 版本已经完整实现了与原文档中 `NN_layer` 相对应的分类训练流程，只是在具体实现上采用了 PyTorch 的训练接口。

开始训练网络模型后，程序会同步记录损失与准确率，并导出训练过程图。网络结构图和训练过程图如下所示。

图 1.10 网络结构图：

![图 1.10 网络结构图](outputs/figures/figure_4_10.png)

图 1.11 训练过程：

![图 1.11 训练过程](outputs/figures/figure_4_11.png)

三、测试网络。

将测试数据对应的编码特征输入已经训练好的分类网络后，即可得到故障类别预测结果。该部分程序见 [train.py](/C:/Coding/260311_matlab-to-py/te_dae/train.py)，程序如下：

```python
def predict_classes(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = model(tensor)
        predictions = logits.argmax(dim=1).cpu().numpy() + 1
    return predictions.astype(np.int64)
```

在主流程中，程序依次对各故障测试集进行预测，并统计每一类故障的分类准确率。核心程序如下：

```python
for fault_id in TRAIN_FAULT_IDS:
    count = min(_test_slice_count(fault_id), bundle.classifier_test_features[fault_id].shape[0])
    features = bundle.classifier_test_features[fault_id][:count]
    preds = predict_classes(classifier, features, device=device)
    label = bundle.classifier_labels[fault_id]
    fault_acc = float((preds == label).mean())
    per_fault_accuracy[f"F{fault_id}"] = fault_acc
```

四、计算准确率和热值图。

在得到所有测试类别的预测结果之后，程序会进一步统计各故障准确率，并构造按行归一化的混淆矩阵，用于绘制热值图。其核心思想与原文档中的 MATLAB 程序一致，即分别计算每一类测试样本被正确识别的比例，再将全部类别的识别结果汇总成二维热值矩阵，从而对分类结果进行可视化展示。

在原文档中，测试集通常按类别分别统计，例如用 `accuracy_t1`、`accuracy_t2`、`accuracy_t4` 等变量表示各故障类别的准确率，其本质就是“预测正确的样本数除以该类测试样本总数”。对于当前 Python 实现来说，这一过程已经统一封装在逐类评估循环中。程序如下：

```python
for fault_id in TRAIN_FAULT_IDS:
    count = min(_test_slice_count(fault_id), bundle.classifier_test_features[fault_id].shape[0])
    features = bundle.classifier_test_features[fault_id][:count]
    preds = predict_classes(classifier, features, device=device)
    label = bundle.classifier_labels[fault_id]
    truth = np.full(count, label, dtype=np.int64)
    y_true_all.append(truth)
    y_pred_all.append(preds)
    fault_acc = float((preds == label).mean())
    per_fault_accuracy[f"F{fault_id}"] = fault_acc
```

由上式可见，`fault_acc = float((preds == label).mean())` 的作用就对应于原文档中的 `accuracy_t1 = sum(output_real1 == 1)/2000` 这一类表达式。也就是说，当前 Python 程序虽然没有显式写成 `accuracy_t1`、`accuracy_t2` 的形式，但它对每个故障类别分别完成了相同的准确率统计过程，并将结果保存到 `per_fault_accuracy` 中。

在本文当前实现中，除 `F6` 按 MATLAB 习惯取 `247` 个测试样本外，其余故障类别均使用 `2000` 个测试样本进行准确率计算，因此整体统计口径与原文档保持一致。

在得到所有故障类别的预测结果之后，程序会进一步拼接真实标签和预测标签，并计算平均准确率与热值矩阵。相关程序如下：

```python
y_true = np.concatenate(y_true_all)
y_pred = np.concatenate(y_pred_all)
mean_accuracy = float(np.mean(list(per_fault_accuracy.values())))
heat = build_confusion(y_true, y_pred, labels=list(range(1, len(TRAIN_FAULT_IDS) + 1)))
```

其中，`mean_accuracy` 表示 17 类故障平均识别准确率；`heat` 表示按行归一化后的混淆矩阵，其作用与原文档中 `Heat(i,j)` 矩阵是一致的。原文档通过循环逐项计算 `Heat(1,i)`、`Heat(2,i)` 等值，而当前 Python 程序则将这一过程封装到 [train.py](/C:/Coding/260311_matlab-to-py/te_dae/train.py) 中的 `build_confusion` 函数里，程序如下：

```python
def build_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums
```

由此可见，Python 版本虽然在代码形式上采用了混淆矩阵函数进行统一计算，但在逻辑上与原文档逐项统计热值图矩阵的方法是一致的，都是为了反映“某一类真实故障被预测为各类别的比例”。

热值图绘制程序见 [plotting.py](/C:/Coding/260311_matlab-to-py/te_dae/plotting.py)，其主要实现如下：

```python
def plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)
```

通过上述程序，可以将每一类故障的分类结果直观地显示在热值图中。热值图的作用在于结果可视化，可以更加清晰地看到预测结果以及不同故障类别之间的混淆关系，因此它是评价故障诊断结果的重要图形表达方式。

## 1.3 Python 仿真结果

运行程序后，可以得到网络结构图、训练过程图以及热值图。经过多轮参数搜索与结果对比，本文最终选取瓶颈层激活特征、随机种子 `42`、噪声强度 `0.025`、DAE 学习率 `0.0001`、分类器学习率 `0.0002`、分类器训练轮数 `1000` 作为当前 Python 版本的最优仿真配置。

对应指标文件为 [metrics_bottleneck_s42_w0025_clr2e4_ce1000.json](/C:/Coding/260311_matlab-to-py/report_assets/metrics_bottleneck_s42_w0025_clr2e4_ce1000.json)，热值图文件为 [figure_4_12_bottleneck_s42_w0025_clr2e4_ce1000.png](/C:/Coding/260311_matlab-to-py/report_assets/figure_4_12_bottleneck_s42_w0025_clr2e4_ce1000.png)。

根据结果文件可知，模型平均准确率为 `0.9495`。其中，`F1` 的准确率为 `0.9950`，`F2` 的准确率为 `0.9875`，`F4` 的准确率为 `0.9825`，`F5` 的准确率为 `0.9940`，`F6` 与 `F7` 的准确率均达到 `1.0000`。对于相对较难识别的故障类别，`F8` 的准确率为 `0.8670`，`F11` 的准确率为 `0.8730`，`F13` 的准确率为 `0.7830`。此外，`F14`、`F16` 和 `F20` 的准确率分别达到 `0.9525`、`0.9485` 和 `0.9705`。

从结果可以看出，随着分类训练轮数增加，模型准确率逐步提高，损失逐步降低。在本文最终配置下，多数故障类别已经取得较高识别精度，其中 `F1`、`F2`、`F4`、`F5`、`F6`、`F7` 等类别均表现较好；相对而言，`F8`、`F11` 与 `F13` 仍然属于较难识别的类别，但与项目早期实验结果相比，这三类故障的识别准确率已有明显提升。

当前最优模型得到的热值图如下图 1.12 所示：

![图 1.12 热值图](report_assets/figure_4_12_bottleneck_s42_w0025_clr2e4_ce1000.png)

从图中可以看出，热值主要集中在对角线附近，说明多数故障样本能够被正确识别；同时，部分非对角区域仍然存在一定热值分布，表明少数故障类别之间仍存在混淆现象，其中 `F8`、`F11` 和 `F13` 相关类别更为明显。

## 1.4 本章小结

在本章中，首先从 TE 过程数据中选取一部分样本作为训练集，另外两部分样本合并作为测试集，并删除第 46 列和第 50 列数据，使原始 52 维数据变为 50 维输入。随后，对训练数据和测试数据进行了 z-score 标准化处理，并在 DAE 训练阶段适度加入高斯噪声，以提高模型的鲁棒性和特征学习能力。

在此基础上，本文利用六层全连接降噪自编码器对数据进行了特征提取，并将提取后的编码特征再次标准化，随后输入三层全连接神经网络进行分类识别。最终，通过对噪声强度、随机种子、特征提取方式、分类器学习率和训练轮数等参数进行多轮实验比较，获得了当前 Python 版本的最优配置，即瓶颈层激活特征、随机种子 `42`、噪声强度 `0.025`、分类器学习率 `0.0002`、分类器训练轮数 `1000`。

在该配置下，模型平均准确率达到 `0.9495`，说明当前 Python 版本已经较好地复现了原文档中的主要实现过程，并在 TE 过程故障诊断任务上取得了较好的实验效果。因此，该结果可以作为本项目基于 Python 实现的降噪自编码器故障诊断仿真报告。
