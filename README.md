# CVnotebook
具体笔记详见jupyternotebook注释

## Stage 1 Logistic Regression

### 1.赛题要求

这次的赛题旨在利用脑PET图像进行轻度认知障碍（MCI）的疾病预测。参赛者需要根据所提供的脑PET图像数据集，构建模型来进行轻度认知障碍（MCI）的预测，即对被试者进行分类，判断其是否属于MCI（轻度认知障碍）或健康（NC）。解题思路可以遵循计算机视觉分类的一般步骤，包括数据预处理、模型构建、训练和评估等阶段。在这个赛题中，可以采用深度学习模型（如卷积神经网络）等方法来从脑PET图像中提取特征，并进行分类预测，最终分类结果通过F1score指标评判。

### 2.baseline中人工提取的特征

非零像素的数量、零像素的数量、平均值、标准差、在列方向上平均值不为零的数量、在行方向上平均值不为零的数量、列方向上的最大平均值、行方向上的最大平均值

### 3.baseline的基本思路

baseline使用的机器学习方法（基于特征提取和逻辑回归）的优势和局限在于：

优势：
1.解释性强： 逻辑回归是一种线性模型，其预测结果相对容易解释，可以理解每个特征对预测的影响程度。
2.计算效率高： 逻辑回归计算速度较快，适用于大规模数据集和较大特征空间。
3.简单快速： 逻辑回归是一个相对简单的模型，不需要大量的超参数调整。

局限性：
1.这种方法需要手动设计和提取特征，依赖于领域知识，且特征的质量对模型影响很大。
2.局限于线性关系： 逻辑回归只能建模线性关系，对于非线性关系的数据表现较差。
3.泛化能力受限： 逻辑回归在面对高维和复杂数据时，泛化能力可能受到限制，可能会出现欠拟合或过拟合。

------------------------
## Stage 2 CNN

### 1.优化和上分思路

尝试使用CV交叉验证的方法以及修改ResNet模型，类似于ResNet-50、ResNet101等更深的模型进行进一步优化。1.尝试不同的折叠数目或增加epoch，来寻找更好的模型泛化性能。（同时注意过拟合）2.超参数调整：调整学习率、权重衰减等超参数，使用学习率调度器（StepLR、ReduceLROnPlateau）来动态调整学习率。

目前的方案（更新中）：
| 方案 | pp套件 | 全NC | CV交叉验证 | res101,学习率为0.001,epoch=3 |    res101,epoch=10      |  resnext    |   以F1分数为引导调整学习率、迭代次数等     |
| :---         |     :---:      |     :---:      |  :---:      |   :---:      |  :---:      |  :---:      |  ---: |
| 分数  | 0.57426     | 0.74214    |  0.72848  |  0.74684 |     0.76712	     |    一看就不对    |     0.71     |  
| 备注     | 没有继续优化       | 可能大多数是NC,还挺高     |   反而更低了？可能是随机种子的关系，炼丹常态    | 改了网络之后有了进步  |   中奖了     |     好像不适合       | 唯结果论不可取  |
|||||第98号为MIC|预测13个为MIC,有三个预测错了|Stage

| 方案 | MobileNet | ShuffleNet | ResNet152，epoch=10|res101,epoch=100|
| :---         |     :---:      |   :---: |:---: | ---: |
| 分数  | 大多数MIC     | 全NC   |  0.5  |0.73333|
| 备注     | 不适用      | 分数不错但结果不行 |深层网络不一定好|预测9个为MIC,有四个预测错了|

----------
目前想法：或许应该换模型了，resnet的极限似乎快到了

### 2.各种模型

在选择适用于本次任务的模型时，通常需要考虑任务的性质、数据集的大小、计算资源以及性能要求。以下是一些常见的模型，并根据一般情况下的适用性进行了简要说明：

1. **ResNet、ResNeXt、Wide ResNet**：这些是非常经典的深度卷积神经网络，在许多计算机视觉任务上都表现良好。ResNet 具有跳跃连接来解决梯度消失问题，ResNeXt 进一步扩展了这一思想，Wide ResNet 则引入了更宽的层来增加模型的容量。

2. **DenseNet**：具有密集连接的网络结构，可以有效地利用前面层的特征，适合在有限的数据集上训练。

3. **EfficientNet、EfficientNetV2**：这些模型在参数量和计算资源的限制下，具有较好的性能表现，可以在资源有限的情况下取得较好的效果。

4. **Inception V3**：通过多个不同大小的卷积核并行提取特征，适合用于需要捕捉多尺度特征的任务。

5. **VisionTransformer、SwinTransformer**：这些是基于自注意力机制的模型，在处理序列数据上表现优异。VisionTransformer 适用于计算资源充足的情况，而 SwinTransformer 可以在资源受限的情况下取得较好的效果。

6. **AlexNet、VGG**：这些是较早的模型，可以用于基准测试和小型任务。但相对于后续的模型，它们可能在参数量和效率方面有所不足。

7. **MobileNet V2、MobileNet V3**：这些模型旨在在移动设备上实现高效推理，适合移动端部署。

8. **SqueezeNet**：类似于 MobileNet，适合在资源受限的环境中部署。

9. **ConvNeXt、MNASNet、RegNet、ShuffleNet V2**：这些模型在一些特定任务和场景中表现出色，但可能较少被使用。

### 3.代码解析

```
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

train_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Test/*')

np.random.shuffle(train_path)
np.random.shuffle(test_path)

DATA_CACHE = {}
class XunFeiDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index]) 
            img = img.dataobj[:,:,:, 0]
            DATA_CACHE[self.img_path[index]] = img
        
        # 随机选择一些通道            
        idx = np.random.choice(range(img.shape[-1]), 50)
        img = img[:, :, idx]
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(image = img)['image']
        
        img = img.transpose([2,0,1])
        return img,torch.from_numpy(np.array(int('NC' in self.img_path[index])))
    
    def __len__(self):
        return len(self.img_path)
        
import albumentations as A
train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[:-10],
            A.Compose([
            A.RandomRotate90(),
            A.RandomCrop(120, 120),
            A.HorizontalFlip(p=0.5),
            A.RandomContrast(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
    ), batch_size=2, shuffle=True, num_workers=0, pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[-10:],
            A.Compose([
            A.RandomCrop(120, 120),
        ])
    ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False
)

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path,
            A.Compose([
            A.RandomCrop(128, 128),
            A.HorizontalFlip(p=0.5),
            A.RandomContrast(p=0.5),
        ])
    ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False
)
```

1. **导入库**：代码一开始导入了所需的各种库，包括数据处理、机器学习框架（PyTorch）、图像处理等等。

2. **设置随机种子**：这段代码设置了PyTorch的随机种子，以确保在可重复的情况下进行训练，这在实验复现和调试中很有用。

3. **定义数据路径**：通过`glob`库读取训练和测试数据的文件路径，并随机打乱它们的顺序。

4. **数据缓存**：`DATA_CACHE`是一个字典，用于缓存已加载的图像数据，以避免重复加载相同的数据。这可以提高数据加载的效率。

5. **自定义数据集类**：`XunFeiDataset`是一个自定义的数据集类，继承自PyTorch的`Dataset`。它负责加载和处理图像数据。每个数据点包括一个图像和一个与图像路径相关的标签。

   - `__init__`函数初始化数据集对象，并接受图像路径和可选的数据预处理变换。
   - `__getitem__`函数根据索引加载图像数据，进行一系列的预处理，包括通道选择、数据类型转换、数据增强（如果有的话）等，并返回预处理后的图像和标签。
   - `__len__`函数返回数据集的长度。

6. **数据加载器设置**：使用`torch.utils.data.DataLoader`来创建训练、验证和测试数据加载器，这些加载器将利用上述定义的数据集进行批量数据加载。

   - `train_loader`：训练数据加载器，用于训练模型。它使用了数据增强的数据预处理，批量大小为2。
   - `val_loader`：验证数据加载器，用于验证模型。它没有数据增强，批量大小为2。
   - `test_loader`：测试数据加载器，用于模型的最终测试。它也没有数据增强，批量大小为2。

7. **数据增强**：数据增强是通过`albumentations`库实现的。它包括随机旋转、随机裁剪、水平翻转、随机对比度和随机亮度等操作。这些操作可以增加模型的泛化能力，提高模型对于不同数据分布的适应能力。

总体而言，这段代码构建了一个用于医学图像分析的数据处理和加载框架，包括数据集类、数据加载器，以及一些数据预处理操作。
```
class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()
                
        model = models.resnet101(True)
        model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, 2)
        self.resnet = model
        
    def forward(self, img):        
        out = self.resnet(img)
        return out
        
model = XunFeiNet()
model = model.to('cuda')
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), 0.001)
```
这段代码定义了一个名为 `XunFeiNet` 的神经网络模型，这个模型基于预训练的 ResNet-101 架构，并在顶部进行了微调以适应特定任务。然后，它将模型放置到 CUDA 设备上，定义了损失函数（交叉熵损失）和优化器（AdamW）。

1. **定义 `XunFeiNet` 模型类**：这个类继承自 `nn.Module`，是一个自定义的神经网络模型。在构造函数 `__init__` 中，首先调用了父类的构造函数，然后执行以下步骤：

    - 创建一个预训练的 ResNet-101 模型并加载预训练权重（通过 `models.resnet101(True)`）。
    - 修改 ResNet-101 模型的第一个卷积层，将输入通道数从默认的 3 通道改为 50 通道（与你的数据通道数相对应）。
    - 替换模型的全局平均池化层为自适应平均池化层，以适应不同尺寸的输入图像。
    - 替换模型的全连接层（最后一层）为一个适合任务的线性层，输出维度为 2，对应于任务的类别数目。

2. **`forward` 方法**：这个方法定义了数据从模型的输入到输出的流程。在这个情况下，输入是图像数据，通过 ResNet-101 的各层传递，最终得到模型的输出。

3. **创建模型对象**：通过实例化 `XunFeiNet` 类，你创建了一个模型对象 `model`。

4. **将模型移至 CUDA 设备**：`model.to('cuda')` 将模型移动到 CUDA 设备上，以便在 GPU 上进行计算。前提是你的计算机有可用的 CUDA 设备。

5. **定义损失函数**：使用交叉熵损失作为分类任务的损失函数。交叉熵损失适用于多类别分类问题。

6. **定义优化器**：使用 AdamW 优化器来优化模型的参数，学习率设置为 0.001。AdamW 是一种变种的 Adam 优化器，通常在深度学习中表现良好。

这段代码的主要目的是定义模型结构、损失函数和优化器，为模型的训练做好准备。
```
def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).long()  # Convert target to LongTensor

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(loss.item())

        train_loss += loss.item()

    return train_loss / len(train_loader)

def validate(val_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda().long()  # Convert target to LongTensor

            output = model(input)
            loss = criterion(output, target)

            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset)
```
这段代码定义了两个函数，`train` 和 `validate`，用于训练和验证模型。

1. **`train` 函数**：
   - `train_loader`：训练数据加载器，用于提供训练数据批次。
   - `model`：模型对象，用于训练。
   - `criterion`：损失函数，用于计算训练损失。
   - `optimizer`：优化器，用于更新模型的参数。

   这个函数执行以下操作：
   - 将模型设置为训练模式（`model.train()`）。
   - 初始化训练损失为 0。
   - 迭代训练数据批次，在每个批次中进行以下操作：
     - 将输入数据和目标标签转移到 CUDA 设备上。
     - 将输入数据传递给模型，得到模型输出。
     - 计算预测结果和目标标签之间的损失。
     - 清零优化器的梯度。
     - 反向传播计算梯度。
     - 使用优化器更新模型参数。
     - 每处理 20 个批次，打印当前批次的损失。
     - 将当前批次的损失累加到总训练损失中。
   - 返回平均训练损失。

2. **`validate` 函数**：
   - `val_loader`：验证数据加载器，用于提供验证数据批次。
   - `model`：模型对象，用于验证。
   - `criterion`：损失函数，用于计算验证损失。

   这个函数执行以下操作：
   - 将模型设置为评估模式（`model.eval()`）。
   - 初始化验证准确率为 0。
   - 使用 `torch.no_grad()` 上下文，避免在验证过程中计算梯度。
   - 迭代验证数据批次，在每个批次中进行以下操作：
     - 将输入数据和目标标签转移到 CUDA 设备上。
     - 将输入数据传递给模型，得到模型输出。
     - 计算预测结果和目标标签之间的损失。
     - 计算预测结果中的最大值索引，与目标标签比较以计算准确率。
     - 将正确预测的样本数量累加到验证准确率中。
   - 返回平均验证准确率。

这两个函数分别用于训练和验证模型，分别计算训练损失和验证准确率。在训练循环中，`train` 函数通过多次迭代和参数更新来训练模型。在验证循环中，`validate` 函数评估模型在验证数据上的性能。通常，训练过程会在多个周期（epochs）中重复执行，每个周期包括训练和验证阶段。
```
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

skf = StratifiedKFold(n_splits=10, random_state=233, shuffle=True)

labels = [int('NC' in path) for path in train_path]

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_path, labels)):
    print(f"Fold {fold_idx + 1}/{skf.get_n_splits()}")
    
    train_loader = torch.utils.data.DataLoader(
        XunFeiDataset(np.array(train_path)[train_idx],  # Use np.array to index train_path
            A.Compose([
                A.RandomRotate90(),
                A.RandomCrop(120, 120),
                A.HorizontalFlip(p=0.5),
                A.RandomContrast(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ])
        ), batch_size=2, shuffle=True, num_workers=0, pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        XunFeiDataset(np.array(train_path)[val_idx],  # Use np.array to index train_path
            A.Compose([
                A.RandomCrop(120, 120),
            ])
        ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False
    )
    
    for epoch in range(10):
        print(f"Epoch [{epoch + 1}/{10}]")
        
        train_loss = train(train_loader, model, criterion, optimizer)
        val_acc = validate(val_loader, model, criterion)
        train_acc = validate(train_loader, model, criterion)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda()
                target = target.cuda().long()

                output = model(input)
                predicted_probs.extend(output.cpu().numpy())
                predicted_labels.extend(output.argmax(1).cpu().numpy())  # Using argmax to get predicted labels
                true_labels.extend(target.cpu().numpy())
        
    # Save the trained model
    model_save_path = f'./resnet101_fold{fold_idx}.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

# Convert true labels and predicted probabilities to NumPy arrays
true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)
predicted_labels = np.array(predicted_labels)

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(true_labels, predicted_probs[:, 1])  # Assuming you have binary classification
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print("AUC:", roc_auc)

# 计算F1分数
f1 = f1_score(true_labels, predicted_labels)
print("F1 Score:", f1)
```
这段代码展示了如何使用交叉验证（Cross-Validation）来训练、验证和评估模型，并计算模型的性能指标。下面将对每个主要部分进行解释：

1. **导入库**：代码一开始导入了一些需要用到的库，包括 `train_test_split`、`StratifiedKFold`、`KFold` 用于交叉验证，以及 `roc_curve`、`auc`、`f1_score` 用于计算 ROC 曲线、AUC 和 F1 分数，还有 `plt` 用于绘制图形。

2. **设置交叉验证参数**：使用 `StratifiedKFold` 创建了一个分层交叉验证对象 `skf`，将数据分成了 10 个折叠，设置了随机种子和数据洗牌（shuffle）。

3. **生成标签**：根据文件路径中是否包含 `'NC'` 字符，将数据的标签生成为二进制标签（0 或 1）。

4. **交叉验证循环**：使用 `enumerate` 遍历每个折叠，分别得到训练索引 `train_idx` 和验证索引 `val_idx`。

   在每个折叠内，进行以下操作：
   - 创建训练数据加载器 `train_loader` 和验证数据加载器 `val_loader`，使用当前折叠的索引来从训练数据中获取对应的数据。
   - 针对每个折叠，进行 10 个周期的训练循环（可以根据实际情况调整），在每个周期内：
     - 调用 `train` 函数进行训练，并记录训练损失。
     - 调用 `validate` 函数计算验证集的准确率和训练集的准确率。
     - 打印当前周期的训练损失、训练准确率和验证准确率。
     - 在验证集上进行推断，获取预测的概率和标签，并将它们存储在相应的列表中。

   完成 10 个周期后，保存训练好的模型，并打印保存路径。

5. **计算 ROC 曲线和 AUC**：使用预测的概率和真实标签计算 ROC 曲线的假正率（FPR）和真正率（TPR），然后计算 AUC 值。

6. **绘制 ROC 曲线**：绘制 ROC 曲线，展示分类器在不同阈值下的性能。

7. **计算和打印 AUC**：打印计算得到的 AUC 值，用于评估模型在不同阈值下的性能。

8. **计算和打印 F1 分数**：使用预测的标签和真实标签计算 F1 分数，该分数综合了精确度和召回率。

总之，这段代码演示了如何使用分层交叉验证来评估模型性能，并计算 ROC 曲线、AUC 值以及 F1 分数等评价指标，以更全面地了解模型在不同数据子集上的表现。

## Stage 3

### 卷积层学习的特征（举例）：

<img width="603" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/9415b243-0720-426c-9e42-a5ed105936f6">

<img width="609" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/9be7c6a5-7370-4185-83e3-65ab29fadefb">

### loss图像

<img width="388" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/ec7ad428-0ade-48ff-b906-28a0d1552c9f">

<img width="364" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/5e283303-788c-43f3-b69c-fb430ffd5efc">

No CV

<img width="386" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/9f257ceb-23de-4655-9022-feb9a6858195">

No CV + BN

<img width="378" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/43736ea2-efc6-4593-8932-035656f670d8">

<img width="381" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/cd2bc595-cb41-4b7d-8d9b-cfe4b480eac5">

CV + No BN epoch=100*10

事实证明，100次迭代之后可能会出现过拟合，效果可能不佳。

### 遇到的困难

1.网络极度欠拟合，无论是训练集准确率还是验证集准确率都在0.5-0.6范围内。

2.由于训练集很少，我们必须从极少的信息中提取抽象的概念。

3.每次重启相同的代码，结果也会大相径庭，目前不知道原因。

### 经验

1.无论什么网格，差距都很小，也就是说结果与网络无关。

2.交叉检验没有达到预期的效果，收效甚微。

3.事实证明，增加epoch可以显著降低loss，大约100比较好。

### 交叉熵损失函数

交叉熵损失函数（Cross-Entropy Loss Function）是在机器学习和深度学习中经常使用的一种损失函数，特别是在分类问题中。它用于衡量模型的预测值与实际标签之间的差异，从而帮助优化模型的参数，使其能够更好地进行分类任务。

nn.CrossEntropyLoss的官方定义
This criterion combines LogSoftmax and NLLLoss in one single class.

```
import   torch
y = torch.LongTensor([0])
z = torch.Tensor([[0.2,0.1,-0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z,y)
print(loss)
# tensor(0.9729)
```
1.首先是softmax
```
z = torch.softmax(z,1)
print(z)
# tensor([[0.3780, 0.3420, 0.2800]])
```
2.negative log likelihood
```
print(-torch.log(torch.Tensor([0.3780])))
# tensor([0.9729])
```
https://pytorch.org/docs/stable/nn.html
https://blog.csdn.net/weixin_43918691/article/details/115582452
https://www.zhihu.com/tardis/bd/art/35709485?source_id=1001
####
注意：如果不管自己的网络怎么训练二元交叉熵的loss一直为0.6931，这是因为log(0.5)=0.6931

那么有两种可能，第一种是最后输出的二维向量值相等，因为二分类问题随便猜测的概率就为0.5，相当于没分类。

第二种是最后输出向量值太低，在计算幂函数时相差不大。

https://www.jianshu.com/p/45c2180cab17
https://blog.csdn.net/weixin_40267472/article/details/82216668
https://zhuanlan.zhihu.com/p/107687473

## Stage 4

### 1.认识nii图像

大部分医学领域导出dicom格式，但是太复杂了。很多时候，将dicom转换为nifti格式也就是nii格式。后缀名为.nii的文件格式又叫NIfTI-1，它改编自广泛使用的ANALYZE™7.5格式。一些比NIfTI-1发展早的老软件也可以兼容NIfTI-1。

一个NIFTI格式主要包含三部分：hdr,ext,img。


**hdr/header**


这部分数据长度是固定的，当然不同版本可能规定的长度不同，但是同一版本的多个nii文件是相同的。

header里包含的信息有：维度，x,y,z，单位是毫米。还有第四个维度，就是时间。这部分储存的主要是四个数字。

voxel size(体素大小)：毫米单位的x,y,z大小。

数据类型，一般是int16，这个精度不够，最好使用double类型。

Form和转换矩阵，每一个Form都对应一个转换矩阵。暂时不知道Form是什么。


**Extension**


是自己可以随意定义数据的部分，可以自己用。但是通用的软件公司都无法使用这部分。


**Image**


储存3D或者4D的图像数据

https://blog.csdn.net/weixin_42089190/article/details/116710684

ITK-SNAP是一个用于三维医学图像分割的软件应用程序。它是宾夕法尼亚大学宾州图像计算与科学实验室（PICSL）的Paul Yushkevich博士和犹他大学科学计算与成像研究所（SCI）的Guido Gerig博士合作的产物，他的愿景是创造一个工具，将致力于一个特定的功能，分割，并将易于使用和学习。ITK-SNAP是免费的、开源的、多平台的，下载地址为http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3。

ITK-SNAP加载数据集之后初始界面如下：初始化界面中三张图分别表示某一个slice所对应的横断面、矢状面、冠状面截图。横断面表示的是从上往下看时，患者组织的截图；矢状面表示从左往右看时，患者组织的截图；冠状面表示从前往后看时，患者组织的截图。

https://blog.csdn.net/qq_41776781/article/details/111992844

<img width="1055" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/b52ffca1-3d0f-4e8b-ab66-d38de8c5e8dd">


### 2.输入通道

<img width="415" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/3b70d39c-d59d-478c-af5d-4ee6e5a8638b">

---------

<img width="475" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/a40257b1-50ef-45c1-892c-0501d494a1a8">

**nii图像头文件信息**

```
NIfTI Header Information:
<class 'nibabel.nifti1.Nifti1Header'> object, endian='>'
sizeof_hdr      : 348
data_type       : b''
db_name         : b'011_S_0023'
extents         : 16384
session_error   : 0
regular         : b'r'
dim_info        : 0
dim             : [  4 128 128  63   1   0   0   0]
intent_p1       : 0.0
intent_p2       : 0.0
intent_p3       : 0.0
intent_code     : none
datatype        : uint16
bitpix          : 16
slice_start     : 0
pixdim          : [1.        2.059405  2.059405  2.4250002 0.        0.        0.
 0.       ]
vox_offset      : 0.0
scl_slope       : nan
scl_inter       : nan
slice_end       : 0
slice_code      : unknown
xyzt_units      : 2
cal_max         : 0.0
cal_min         : 0.0
slice_duration  : 0.0
toffset         : 0.0
glmax           : 32767
glmin           : 0
descrip         : b''
aux_file        : b''
qform_code      : scanner
sform_code      : unknown
quatern_b       : 0.0
quatern_c       : 0.0
quatern_d       : 0.0
qoffset_x       : 128.0
qoffset_y       : 128.0
qoffset_z       : 63.0
srow_x          : [0. 0. 0. 0.]
srow_y          : [0. 0. 0. 0.]
srow_z          : [0. 0. 0. 0.]
intent_name     : b''
magic           : b'n+1'
```
**nii图像shape信息**
```
import nibabel as nib

def get_nifti_shape(nifti_file_path):
    try:
        nifti_image = nib.load(nifti_file_path)
        image_shape = nifti_image.shape
        return image_shape
    except Exception as e:
        print("Error:", e)
        return None

nifti_file_path = "./test/86.nii"  # 替换为你的NIfTI文件的路径
image_shape = get_nifti_shape(nifti_file_path)

if image_shape is not None:
    print(f"The shape of the NIfTI image is: {image_shape}")

```
```
output:
The shape of the NIfTI image is: (128, 128, 47, 1)
The shape of the NIfTI image is: (128, 128, 63, 1)
The shape of the NIfTI image is: (256, 256, 47, 1)
The shape of the NIfTI image is: (128, 128, 768, 1)
```
这表示图像数据的形状，具体为四维（x、y、z和时间维度）的数组。在这个例子中，图像数据的尺寸为 128x128x768x1，其中768是z方向的维度。说明图像大小不一样。

最后一个维度代表图像通道数，通常情况下，通道数为1，因为医学图像通常是单通道的，灰度图像只有一个颜色通道。

在如下操作中，实际上是取img = img.dataobj[:,:,:, 0]，也就是第一个通道。idx = np.random.choice(range(img.shape[-1]), 50)并随机取z维的50个切片作为索引

然而在之前的工作中我们发现，有很大一部分图片的维度并不是50，甚至少于50，在这种随机的过程中，很有可能丢失部分信息，增加了随机性。
```
    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index]) 
            img = img.dataobj[:,:,:, 0]
            DATA_CACHE[self.img_path[index]] = img
        
        # 随机选择一些通道            
        idx = np.random.choice(range(img.shape[-1]), 50)
        img = img[:, :, idx]
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(image = img)['image']
        
        img = img.transpose([2,0,1])

        print(img.shape[-1])
        return img,torch.from_numpy(np.array(int('NC' in self.img_path[index])))
```
我们可以尝试可视化这50个切片：
```
import matplotlib.pyplot as plt

# 创建XunFeiDataset实例
transform = None  # 如果有图像变换，可以在这里添加
dataset = XunFeiDataset(train_path, transform=transform)

# 随机选择一些图像
num_images_to_display = 5
selected_indices = np.random.choice(len(dataset), num_images_to_display, replace=False)

# 遍历选定的图像
for idx in selected_indices:
    img, label = dataset[idx]
    img_path = dataset.img_path[idx]  # 获取图像路径

    # 提取图像文件名作为标题
    img_name = os.path.basename(img_path)

    # 显示50个通道的图像
    fig, axes = plt.subplots(10, 5, figsize=(15, 30))
    fig.suptitle(f"Image: {img_name} ", fontsize=16)

    for i in range(10):
        for j in range(5):
            channel_idx = i * 5 + j
            channel_img = img[channel_idx]
            axes[i, j].imshow(channel_img, cmap='gray')
            axes[i, j].set_title(f"Channel {channel_idx}")
            axes[i, j].axis('off')

    plt.show()
```

<img width="613" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/641d6fcd-62fb-49bf-ab1d-ff5b6b3306f0">

### 3.设置随机种子

## 杂谈

1.为什么要用F1score,准确率+召回率，比较均衡的评价指标。

2.库内有nii转化为png的python脚本

3.下载特定包换腾讯源-i http://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com

4.全NC的结果是0.74214，现在最佳结果为0.76712

5.41个NC,判断公式为 
The formula is: $F1score=\frac{2}{\frac{59}{x}+\frac{100-x}{x}}$.




