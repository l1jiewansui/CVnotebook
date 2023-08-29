# CV notebook         by Meng
<img width="500" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/a8bd2013-a33b-40ce-a6ee-67c544af6932">
[toc]

## Stage 1 Logistic Regression baseline 综述

### 1.赛题要求

这次的赛题旨在利用脑PET图像进行轻度认知障碍（MCI）的疾病预测。参赛者需要根据所提供的脑PET图像数据集，构建模型来进行轻度认知障碍（MCI）的预测，即对被试者进行分类，判断其是否属于MCI（轻度认知障碍）或健康（NC）。解题思路可以遵循计算机视觉分类的一般步骤，包括数据预处理、模型构建、训练和评估等阶段。在这个赛题中，可以采用深度学习模型（如卷积神经网络）等方法来从脑PET图像中提取特征，并进行分类预测，最终分类结果通过F1score指标评判。

### 2.代码解析

```
import glob                # 获取文件路径
import numpy as np
import pandas as pd
import nibabel as nib      # 处理医学图像数据
from nibabel.viewers import OrthoSlicer3D    # 图像可视化
from collections import Counter              # 计数统计

# 读取训练集文件路径
train_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Test/*')

# 打乱训练集和测试集的顺序
np.random.shuffle(train_path)
np.random.shuffle(test_path)

# 对PET文件提取特征
def extract_feature(path):
    # 加载PET图像数据
    img = nib.load(path)
    # 获取第一个通道的数据
    img = img.dataobj[:, :, :, 0]
    
    # 随机筛选其中的10个通道提取特征
    random_img = img[:, :, np.random.choice(range(img.shape[2]), 10)]
    
    # 对图片计算统计值
    feat = [
        (random_img != 0).sum(),               # 非零像素的数量
        (random_img == 0).sum(),               # 零像素的数量
        random_img.mean(),                     # 平均值
        random_img.std(),                      # 标准差
        len(np.where(random_img.mean(0))[0]),  # 在列方向上平均值不为零的数量
        len(np.where(random_img.mean(1))[0]),  # 在行方向上平均值不为零的数量
        random_img.mean(0).max(),              # 列方向上的最大平均值
        random_img.mean(1).max()               # 行方向上的最大平均值
    ]
    
    # 根据路径判断样本类别（'NC'表示正常，'MCI'表示异常）
    if 'NC' in path:
        return feat + ['NC']
    else:
        return feat + ['MCI']

# 对训练集进行30次特征提取，每次提取后的特征以及类别（'NC'表示正常，'MCI'表示异常）被添加到train_feat列表中。
train_feat = []
for _ in range(30):
    for path in train_path:
        train_feat.append(extract_feature(path))
     
# 对测试集进行30次特征提取   
test_feat = []
for _ in range(30):
    for path in test_path:
        test_feat.append(extract_feature(path))
        
# 使用训练集的特征作为输入，训练集的类别作为输出，对逻辑回归模型进行训练。
from sklearn.linear_model import LogisticRegression
m = LogisticRegression(max_iter=1000)
m.fit(
    np.array(train_feat)[:, :-1].astype(np.float32),  # 特征
    np.array(train_feat)[:, -1]                       # 类别
)

# 对测试集进行预测并进行转置操作，使得每个样本有30次预测结果。
test_pred = m.predict(np.array(test_feat)[:, :-1].astype(np.float32))
test_pred = test_pred.reshape(30, -1).T

# 对每个样本的30次预测结果进行投票，选出最多的类别作为该样本的最终预测类别，存储在test_pred_label列表中。
test_pred_label = [Counter(x).most_common(1)[0][0] for x in test_pred]

# 生成提交结果的DataFrame，其中包括样本ID和预测类别。
submit = pd.DataFrame(
    {
        'uuid': [int(x.split('/')[-1][:-4]) for x in test_path],  # 提取测试集文件名中的ID
        'label': test_pred_label                                  # 预测的类别
    }
)

# 按照ID对结果排序并保存为CSV文件
submit = submit.sort_values(by='uuid')
submit.to_csv('submit1.csv', index=None)
```

代码流程如下：

1. **导入库：** 这些语句导入了所需的库，包括用于文件处理的`glob`，用于数据处理的`numpy`和`pandas`，用于处理医学图像数据的`nibabel`，以及其他必要的库。

2. **读取文件路径：** 使用`glob.glob`函数，代码获取训练集和测试集的文件路径。这些路径将用于加载医学图像数据。

3. **打乱数据顺序：** 使用`np.random.shuffle`函数，训练集和测试集的文件路径被随机打乱，以确保数据的随机性和泛化能力。

4. **提取特征函数：** 定义了一个函数`extract_feature`，用于从医学图像数据中提取特征。特征提取的过程包括加载图像、选择通道、计算统计值，并根据文件路径判断样本类别。

5. **特征提取和标签构建：** 使用循环，代码对训练集和测试集进行30次特征提取。每次提取的特征和类别被添加到相应的列表（`train_feat`和`test_feat`）中。

6. **逻辑回归模型训练：** 导入`LogisticRegression`类后，代码初始化了一个逻辑回归模型，并使用训练集的特征和类别进行训练。

7. **测试集预测：** 使用训练好的逻辑回归模型，代码对测试集进行预测，并对预测结果进行了一些转置和整形操作，以使每个样本有30次预测结果。

8. **投票预测：** 对于每个样本，代码对其30次预测结果进行投票，选择出现次数最多的类别作为最终预测类别，并将这些最终预测类别存储在`test_pred_label`列表中。

9. **结果生成和保存：** 使用预测结果，代码生成一个包含样本ID和最终预测类别的DataFrame，并对ID进行排序。最后，将这个DataFrame保存为CSV文件（'submit1.csv'）。

### 3.baseline中人工提取的特征

非零像素的数量、零像素的数量、平均值、标准差、在列方向上平均值不为零的数量、在行方向上平均值不为零的数量、列方向上的最大平均值、行方向上的最大平均值

### 4.baseline的基本思路

baseline使用的机器学习方法（基于特征提取和逻辑回归）的优势和局限在于：

**优势：**

1.解释性强： 逻辑回归是一种线性模型，其预测结果相对容易解释，可以理解每个特征对预测的影响程度。

2.计算效率高： 逻辑回归计算速度较快，适用于大规模数据集和较大特征空间。

3.简单快速： 逻辑回归是一个相对简单的模型，不需要大量的超参数调整。

**局限性：**

1.这种方法需要手动设计和提取特征，依赖于领域知识，且特征的质量对模型影响很大。

2.局限于线性关系： 逻辑回归只能建模线性关系，对于非线性关系的数据表现较差。

3.泛化能力受限： 逻辑回归在面对高维和复杂数据时，泛化能力可能受到限制，可能会出现欠拟合或过拟合。

------------------------
## Stage 2 CNN baseline 代码基本解析

如果不读懂代码盲目上手，那么就是空中楼阁。

### 1.预处理

注意：num_workers=0

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

### 2.建立模型

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

### 3.定义训练、验证函数

注意Convert target to LongTensor

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

### 4.训练

这里已经魔改过了，baseline就是简单地迭代。

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

```
def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0
    
    test_pred = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            test_pred.append(output.data.cpu().numpy())
            
    return np.vstack(test_pred)
    
pred = None
for _ in range(20):
    if pred is None:
        pred = predict(test_loader, model, criterion)
    else:
        pred += predict(test_loader, model, criterion)
        
submit = pd.DataFrame(
    {
        'uuid': [int(x.split('\\')[-1][:-4]) for x in test_path],
        'label': pred.argmax(1)
})
submit['label'] = submit['label'].map({1:'NC', 0: 'MCI'})
submit = submit.sort_values(by='uuid')
submit.to_csv('submit1_2.csv', index=None)

```

这段代码是一个用于在测试集上进行预测并生成提交文件的过程。主要步骤如下：

1. **定义 `predict` 函数**:
   - `predict` 函数用于在给定的 `test_loader` 上进行模型预测，并返回预测结果的集合。
   - 函数首先将模型设置为评估模式（`model.eval()`），然后初始化变量 `val_acc` 用于存储验证准确率。
   - 在 `test_pred` 列表中，使用 `torch.no_grad()` 上下文管理器遍历测试集加载器中的数据。对于每个批次的输入 `input`，通过模型得到输出 `output`，并将输出数据的 NumPy 表示添加到 `test_pred` 列表中。
   - 最后，函数返回一个垂直堆叠的 NumPy 数组，其中包含所有预测结果。

2. **循环预测**:
   - 使用一个循环进行多次预测（在这里是 20 次），以增加预测结果的稳定性。
   - 在第一次循环中，直接调用 `predict` 函数得到 `pred`，并在后续循环中将其他预测结果累加到 `pred` 中。

3. **生成提交文件**:
   - 创建一个空的变量 `pred`，它将存储最终的预测结果。
   - 使用循环调用 `predict` 函数并将预测结果累加到 `pred` 中。这样，你将得到 20 次预测的累计结果。
   - 使用 `pd.DataFrame` 构造一个 DataFrame `submit`，其中包含两列：'uuid' 和 'label'。'uuid' 是测试样本的标识符，'label' 是预测的标签。
   - 将 'uuid' 列设置为测试样本的标识符，通过从文件路径中提取出相应的信息。'label' 列将被设置为预测结果的标签。
   - CNN最后，对 DataFrame `submit` 按 'uuid' 列进行排序，并将其保存为 CSV 文件（'submit1_2.csv'）。

总之，这段代码用于对测试数据集进行多次预测，累计预测结果，并将最终的预测结果生成为一个 CSV 提交文件。预测过程使用了一个循环，以获得更稳定的预测结果。最后，生成的提交文件将包含测试样本的标识符和预测标签。

------------------------



## Stage 3 CNN方案的优化思路

​        这一篇是对我一整个周期下来想到的所有优化思路的总结。因为baseline的分数不好，不能作为基础版本进行对比，并且训练集准确率和验证集准确率一直跳变，因此几乎无法可视化优化的效果，导致最终的笔记可能比较凌乱。

```
### 遇到的困难

1.网络极度欠拟合，无论是训练集准确率还是验证集准确率都在0.5-0.6范围内。

2.由于训练集很少，我们必须从极少的信息中提取抽象的概念。

3.每次重启相同的代码，结果也会大相径庭。

### 经验

1.无论什么网格，差距都很小，也就是说结果与网络无关。

2.交叉检验没有达到预期的效果，收效甚微。

3.事实证明，增加epoch可以显著降低loss，大约40比较好。
```

​       优化思路共分:1.nii图像解析、2.卷积神经网络、3.交叉熵损失函数、4.图像增强、5.设置随机种子、6.k折交叉验证

​       还有很多细微的地方，比如优化器的选择，但对最终结果影响很小，因此不再展开。

-----------------------------------



### 1.nii图像解析

​        我们知道，2020年的脑PET图像分析和疾病预测挑战赛用的是png图像，而这次比赛使用的是nii图像。nii图像我们可以理解为伪3D图像。相当于做CT时从你前脑到后脑，上脑到下脑的一套图像都提取出来了，并且所有图像的长宽比相同，也就是都是正方形，只有z轴方向大小多变，我们一般分析疾病时也用的是z轴方向的切片，也就是正方形的切片。那么问题来了，相较于2020年的数据，这次边缘的数据必定充斥着无效和脏数据，进一步加大了训练的难度

#### 1.认识nii图像

​      大部分医学领域导出dicom格式，但是太复杂了。很多时候，将dicom转换为nifti格式也就是nii格式。后缀名为.nii的文件格式又叫NIfTI-1，它改编自广泛使用的ANALYZE™7.5格式。一些比NIfTI-1发展早的老软件也可以兼容NIfTI-1。

​       一个NIFTI格式主要包含三部分：hdr,ext,img。


##### **1.hdr/header**

```
这部分数据长度是固定的，当然不同版本可能规定的长度不同，但是同一版本的多个nii文件是相同的。

header里包含的信息有：维度，x,y,z，单位是毫米。还有第四个维度，就是时间。这部分储存的主要是四个数字。

voxel size(体素大小)：毫米单位的x,y,z大小。

数据类型，一般是int16，这个精度不够，最好使用double类型。

Form和转换矩阵，每一个Form都对应一个转换矩阵。暂时不知道Form是什么。
```

###### 头文件信息举例

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

##### **2.Extension** 

```
是自己可以随意定义数据的部分，可以自己用。但是通用的软件公司都无法使用这部分。
```

##### **3.Image**

```
储存3D或者4D的图像数据
```

##### 4.输入通道

<img width="415" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/3b70d39c-d59d-478c-af5d-4ee6e5a8638b">

---------

<img width="475" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/a40257b1-50ef-45c1-892c-0501d494a1a8">

##### **5.nii图像shape信息**（和我们比赛紧密相关的）

查看代码：

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

参考网站：https://blog.csdn.net/weixin_42089190/article/details/116710684

-----------------------------------------

#### 2.相关软件

​        ITK-SNAP是一个用于三维医学图像分割的软件应用程序。它是宾夕法尼亚大学宾州图像计算与科学实验室（PICSL）的Paul Yushkevich博士和犹他大学科学计算与成像研究所（SCI）的Guido Gerig博士合作的产物，他的愿景是创造一个工具，将致力于一个特定的功能，分割，并将易于使用和学习。ITK-SNAP是免费的、开源的、多平台的，下载地址为http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3（这个网站对我们理解数据非常有用）。

​         ITK-SNAP加载数据集之后初始界面如下：初始化界面中三张图分别表示某一个slice所对应的横断面、矢状面、冠状面截图。横断面表示的是从上往下看时，患者组织的截图；矢状面表示从左往右看时，患者组织的截图；冠状面表示从前往后看时，患者组织的截图。

参考网站：https://blog.csdn.net/qq_41776781/article/details/111992844



<img width="1055" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/b52ffca1-3d0f-4e8b-ab66-d38de8c5e8dd">



-----------------------------------

#### 3.理解源代码

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

我们同时可以发现，部分图片基本不可能提取到有用信息，可能需要清除，在这里先按下不表。

----------------

#### 4.修改类函数

选取所有通道，而不是随机选取50。代码如下：

```
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
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img
        
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        img = img.transpose([2, 0, 1])
        return img, torch.from_numpy(np.array(int('NC' in self.img_path[index])))
    
    def __len__(self):
        return len(self.img_path)
```

<img width="432" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/ba955012-3383-44e8-940a-293997a69298">

我们随机选取的50个切片很容易选到没有多少有效信息的图像。

### 2.卷积神经网络

​        在深度学习中，卷积神经网络（CNN或ConvNet）是一类深度神经网络，最常用于分析视觉图像。CNN使用多层感知器的变体设计，需要最少的预处理。它们也被称为移位不变或空间不变人工神经网络（SIANN），基于它们的共享权重架构和平移不变性特征。

​        CNN的基本模块是由输入和输出层以及多个隐藏层组成，隐藏层可分为卷积层，池化层、RELU层和全连接层。与其他图像分类算法相比，CNN使用相对较少的预处理。这意味着网络学习传统算法中手工设计的过滤器。这种独立于特征设计中的先前知识和人力的努力是一个主要优点。它们可用于图像和视频识别，推荐系统，图像分类，医学图像分析和自然语言处理。

#### 1.卷积神经网络模型

​        在选择适用于本次任务的模型时，通常需要考虑任务的性质、数据集的大小、计算资源以及性能要求。以下是一些常见的模型，并根据一般情况下的适用性进行了简要说明：

1. **ResNet、ResNeXt、Wide ResNet**：这些是非常经典的深度卷积神经网络，在许多计算机视觉任务上都表现良好。ResNet 具有跳跃连接来解决梯度消失问题，ResNeXt 进一步扩展了这一思想，Wide ResNet 则引入了更宽的层来增加模型的容量。

2. **DenseNet**：具有密集连接的网络结构，可以有效地利用前面层的特征，适合在有限的数据集上训练。

3. **EfficientNet、EfficientNetV2**：这些模型在参数量和计算资源的限制下，具有较好的性能表现，可以在资源有限的情况下取得较好的效果。

4. **Inception V3**：通过多个不同大小的卷积核并行提取特征，适合用于需要捕捉多尺度特征的任务。

5. **VisionTransformer、SwinTransformer**：这些是基于自注意力机制的模型，在处理序列数据上表现优异。VisionTransformer 适用于计算资源充足的情况，而 SwinTransformer 可以在资源受限的情况下取得较好的效果。

6. **AlexNet、VGG**：这些是较早的模型，可以用于基准测试和小型任务。但相对于后续的模型，它们可能在参数量和效率方面有所不足。

7. **MobileNet V2、MobileNet V3**：这些模型旨在在移动设备上实现高效推理，适合移动端部署。

8. **SqueezeNet**：类似于 MobileNet，适合在资源受限的环境中部署。

9. **ConvNeXt、MNASNet、RegNet、ShuffleNet V2**：这些模型在一些特定任务和场景中表现出色，但可能较少被使用。

   

这是在本次任务中，我应用不同模型做出的效果：
| 方案 | pp套件 | 全NC | CV交叉验证 | res101,学习率为0.001,epoch=3 |    res101,epoch=10      |  resnext    |   以F1分数为引导调整学习率、迭代次数等     |
| :---         |     :---:      |     :---:      |  :---:      |   :---:      |  :---:      |  :---:      |  ---: |
| 分数  | 0.57426     | 0.74214    |  0.72848  |  0.74684 |     0.76712	     |    一看就不对    |     0.71     |  
| 备注     | 没有继续优化       | 可能大多数是NC,还挺高     |   反而更低了？可能是随机种子的关系，炼丹常态    | 改了网络之后有了进步  |   中奖了     |     好像不适合       | 唯结果论不可取  |
|||||第98号为MIC|预测13个为MIC,有三个预测错了|Stage

| 方案 | MobileNet | ShuffleNet | ResNet152，epoch=10|res101,epoch=100|
| :---         |     :---:      |   :---: |:---: | ---: |
| 分数  | 大多数MIC     | 全NC   |  0.5  |0.73333|
| 备注     | 不适用      | 分数不错但结果不行 |深层网络不一定好|预测9个为MIC,有四个预测错了|
|  |  |  |||

----------
结论：不管使用什么2D模型,效果相差无几，选择resnet的任意模型即可。

**notice：在应用resnet50及以上时注意修改输出层的维度，从512改为1024。**



#### 2.初始卷积层

```
model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
```

​         这个50指的是输入图像的图像通道，默认为3，实际上是因为ResNet训练的都是RGB图像。但实际上我们将切片数直接定义为图像通道数了，这样做合理性有待商榷。



###### 偏置项

​        为什么不使用偏置项？这是一个问题。若x变成x+b,代入公式，得到的y结果与先前一致，所以不需加入偏置。

参考网站：https://blog.csdn.net/weixin_43334838/article/details/124244086



#### 3.池化层

`nn.AdaptiveAvgPool2d` 是 PyTorch 中的一个层，用于在二维输入张量上执行自适应平均池化操作。通常情况下，它用于将不同尺寸的输入张量转换为固定尺寸的输出张量。

- `nn.AdaptiveAvgPool2d(output_size)` 接受一个形状为 `(batch_size, channels, height, width)` 的输入张量，计算输入张量每个通道在每个空间维度（高度和宽度）上的平均值。`output_size` 参数指定了期望的输出空间维度，而你这里使用的是 `(1, 1)`，这意味着将输入张量的长度和宽度都缩减为 1。

这通常用于确保卷积层的输出可以直接馈送到具有固定数量神经元的全连接层，而不考虑原始输入图像的尺寸，使其在传递到全连接层之前具有固定的尺寸 `(batch_size, channels, 1, 1)`。这种自适应池化操作有助于使网络对输入图像尺寸的变化更具鲁棒性，还可以减少最终全连接层中的参数数量，有助于防止**过拟合**。

#### 4.卷积层学习的特征（举例）

<img width="603" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/9415b243-0720-426c-9e42-a5ed105936f6">

<img width="609" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/9be7c6a5-7370-4185-83e3-65ab29fadefb">

#### 5.优化意见

​        一般来说，对于彩色图像，通常有红色、绿色和蓝色三个颜色通道。对于灰度图像，只有一个通道。在这种情况下，输入通道数指的是在卷积层中输入的特征图的通道数。ResNet默认为3，而魔改版为50。

​        但这里的50应该是z轴高度值，也就是切片数，在网络上其他处理方案中，这里本来应该设为1，不然就相当于少了50倍的数据。

但分开成单个slice十分困难，随机选择768可能会造成大量重复（运行时间很长），或许我们应该考虑3D卷积。

https://blog.csdn.net/cf_jack/article/details/129167837

https://blog.csdn.net/jiacong_wang/article/details/105631229

https://blog.csdn.net/qq_51143616/article/details/125463512?ops_request_misc=&request_id=&biz_id=102&utm_term=1%E9%80%9A%E9%81%93resnet%E9%A2%84%E8%AE%AD%E7%BB%83%E6%9D%83%E9%87%8D&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-125463512.142

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
 
# Define ResNet model
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=1, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
```

#### 6.解决方案（不使用3D卷积）

​         我们知道，医学影像的边缘slice可能没有关键信息或者噪声严重，我们或许可以不随机选择50层，而是从中间选取若干层（因为最小是47层图片，所以这边先选了46层）。

```
    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index]) 
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img
        
        num_channels = img.shape[-1]
        middle_start = (num_channels - 46) // 2  # 计算开始通道的索引，使得选取的 46 个通道位于中间位置
        selected_channels = slice(middle_start, middle_start + 46)  # 创建一个切片对象用于选取连续的 46 个通道
        img = img[:, :, selected_channels]  # 选取最中间的 46 个通道
        img = img.astype(np.float32)
        print(selected_channels)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        img = img.transpose([2, 0, 1])
        return img, torch.from_numpy(np.array(int('NC' in self.img_path[index])))

```

可以发现所有图片都取了中间层，降低了误差。
<img width="90" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/1b7b9b51-df10-452c-b5ea-dcfefb85f630">

分数不错，可以接收。

<img width="383" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/1c801841-adc4-4e2f-9221-3e813f87ccb9">

### 3.交叉熵损失函数

​         交叉熵损失函数（Cross-Entropy Loss Function）是在机器学习和深度学习中经常使用的一种损失函数，特别是在分类问题中。它用于衡量模型的预测值与实际标签之间的差异，从而帮助优化模型的参数，使其能够更好地进行分类任务。

**nn.CrossEntropyLoss的官方定义**
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

参考文章：

https://pytorch.org/docs/stable/nn.html
https://blog.csdn.net/weixin_43918691/article/details/115582452
https://www.zhihu.com/tardis/bd/art/35709485?source_id=1001
####
注意：如果不管自己的网络怎么训练二元交叉熵的loss一直为0.6931，这是因为log(0.5)=0.6931

那么有两种可能，第一种是最后输出的二维向量值相等，因为二分类问题随便猜测的概率就为0.5，相当于没分类。

第二种是最后输出向量值太低，在计算幂函数时相差不大，比如0.01和-0.01。

https://www.jianshu.com/p/45c2180cab17
https://blog.csdn.net/weixin_40267472/article/details/82216668
https://zhuanlan.zhihu.com/p/107687473

因为是一个二分类问题，因此我们这里输出的是一个二维矩阵，



### 





### 4.图像增强

```
    XunFeiDataset(train_path[:-10],
            A.Compose([
            A.RandomRotate90(),
            A.RandomCrop(120, 120),
            A.HorizontalFlip(p=0.5),
            A.RandomContrast(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
```

https://zhuanlan.zhihu.com/p/107399127/

albumentations 是一个给予 OpenCV的快速训练数据增强库，拥有非常简单且强大的可以用于多种任务（分割、检测）的接口，易于定制且添加其他框架非常方便。

它可以对数据集进行逐像素的转换，如模糊、下采样、高斯造点、高斯模糊、动态模糊、RGB转换、随机雾化等；也可以进行空间转换（同时也会对目标进行转换），如裁剪、翻转、随机裁剪等。

```
A.RandomRotate90()：随机将图像旋转90度（0、90、180或270度）。
A.RandomCrop(120, 120)：随机从图像中裁剪出一个120x120的区域。
A.HorizontalFlip(p=0.5)：以50%的概率水平翻转图像。
A.RandomContrast(p=0.5)：以50%的概率随机调整图像的对比度。
A.RandomBrightnessContrast(p=0.5)：以50%的概率随机调整图像的亮度和对比度。
```
```
#可视化变化代码
import matplotlib.pyplot as plt
import albumentations as A

# 选择一张图像进行可视化
sample_index = 0
sample_image, _ = train_loader.dataset[sample_index]

# 定义增强操作
augmentations = [
    A.RandomRotate90(p=1),
    A.RandomCrop(120, 120, p=1),
    A.HorizontalFlip(p=1),
    A.RandomContrast(p=1),
    A.RandomBrightnessContrast(p=1)
]

# 可视化不同增强下的图像效果
plt.figure(figsize=(15, 5))
plt.subplot(1, len(augmentations) + 1, 1)
plt.title("Original Image")
plt.imshow(sample_image[0], cmap="gray")

for idx, augmentation in enumerate(augmentations, start=2):
    augmented_image = augmentation(image=sample_image)["image"]
    plt.subplot(1, len(augmentations) + 1, idx)
    plt.title(augmentation.__class__.__name__)
    plt.imshow(augmented_image[0], cmap="gray")

plt.tight_layout()
plt.show()
```
<img width="800" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/670b2590-beb2-4264-8220-acac138511fe">

```
# 选择一个批次进行可视化


import matplotlib.pyplot as plt
import albumentations as A
sample_batch, _ = next(iter(train_loader))


# 定义增强操作

augmentations = A.Compose([
    A.RandomRotate90(),
    A.RandomCrop(120, 120),
    A.HorizontalFlip(p=0.5),
    A.RandomContrast(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])

plt.figure(figsize=(15, 5))

for i in range(sample_batch.shape[0]):
    original_image = sample_batch[i, 0].numpy()  # 原始图像
    augmented_image = augmentations(image=original_image)["image"]  # 增强后的图像

    plt.subplot(2, sample_batch.shape[0], i + 1)
    plt.title("Original")
    plt.imshow(original_image, cmap="gray")

    plt.subplot(2, sample_batch.shape[0], i + sample_batch.shape[0] + 1)
    plt.title("Augmented")
    plt.imshow(augmented_image, cmap="gray")

plt.tight_layout()
plt.show()
```

<img width="138" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/b49fd47f-0057-4657-bbdf-4378dc991546">


#### Compose

转换按照列表中的顺序依次应用。

#### OneOf

Compose组合下的变换是要挨着顺序做的，而OneOf组合里面的变换是系统自动选择其中一个来做，而这里的概率参数p是指选定后的变换被做的概率。

Albumentations库提供了许多图像增强变换，用于数据增强、数据预处理和数据转换。以下是Albumentations库中常用的一些变换操作，它们可以用于增强训练数据集以及进行数据预处理：

1. **颜色和对比度变换**：
   - `RandomBrightnessContrast`: 随机调整图像的亮度和对比度。
   - `HueSaturationValue`: 随机调整图像的色调、饱和度和亮度。
   - `RGBShift`: 随机调整图像的RGB通道的值。

2. **几何变换**：
   - `HorizontalFlip`: 水平翻转图像。
   - `VerticalFlip`: 垂直翻转图像。
   - `RandomRotate90`: 随机旋转图像90度。
   - `Rotate`: 旋转图像给定角度。

3. **尺寸调整和裁剪**：
   - `Resize`: 调整图像大小。
   - `RandomResizedCrop`: 随机裁剪和调整大小。
   - `CenterCrop`: 居中裁剪图像。

4. **滤波和模糊**：
   - `Blur`: 对图像进行模糊操作。
   - `GaussianBlur`: 对图像应用高斯模糊。
   - `MedianBlur`: 对图像应用中值模糊。

5. **填充和遮罩**：
   - `PadIfNeeded`: 根据需要填充图像。
   - `Cutout`: 在图像中随机生成遮罩以删除图像区域。

6. **形态学变换**：
   - `ElasticTransform`: 弹性变换，模拟图像变形。
   - `GridDistortion`: 对图像进行网格扭曲。

7. **像素级变换**：
   - `RandomBrightness`: 随机调整图像的亮度。
   - `RandomContrast`: 随机调整图像的对比度。

8. **混合和合成**：
   - `RandomGamma`: 随机调整图像的Gamma值。
   - `RandomShadow`: 在图像上添加随机阴影效果。
   - `Compose`: 将多个变换组合在一起。

https://albumentations.ai/docs/transforms/transforms_overview/

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



### 5.设置随机种子



#### seed()函数的作用

官方解释：

> 1 seed() 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随机数都相同。
> 2 设置随机数种子不设置的话按系统根据时间作为参数 设置后可以保证我们每一个产生的随机数是一样的

什么意思？这里先来了解一下随机数的背景，随机数就是随机产生的数，既然叫做随机数，那么应当具有随机性，如果随机种子能固定随机数，那么“随机数”的说法是否是个悖论。这里就要引入一个“伪随机数”的概念。**在机器控制的算法或过程中，所产生的都是伪随机数，是通过一定规律产生的，也就是过程可控的**。这种规律叫做随机数生成器，但是规律很复杂，就像一个黑盒，我们可以近似地将其视为随机数的模拟。随机种子就可以认为是在机器模拟过程中固定随机数产生过程的船锚。

> 随机数生成器：早期使用的线性同余产生器、经典斐波那契产生器和反馈移位寄存器产生器；上世纪八九十年代发展的非线性同余产生器，多步线性递推产生器，以及进位借位运算产生器；目前常用的组合产生器（前述都属于单一随机数产生器）等等。具体会涉及到密码学以及信息安全的知识。

我们可以将产生随机数的过程视为一个固定的黑盒，把设定好的种子扔进去，黑盒将会返回所想要的随机数，以及保证生成下一个随机数的新的种子，再将这颗种子继续丢进黑盒，循环迭代，从而进入随机数生成的无限循环。因此，**只要初始随机种子不改变，我们将会得到固定的随机数序列。**

参考文章：[深度学习中的随机数种子_深度学习seed_redfish95的博客-CSDN博客](https://blog.csdn.net/redfish95/article/details/129000000)

我们在训练的时候可以记录修改seed函数，方便复现。



### 6.k折交叉验证

​        交叉验证是在机器学习建立模型和验证模型参数时常用的办法，一般被用于评估一个机器学习模型的表现。更多的情况下，我们也用交叉验证来进行模型选择(model selection)。

  交叉验证，顾名思义，就是重复的使用数据，把得到的样本数据进行切分，组合为不同的训练集和测试集，用训练集来训练模型，用测试集来评估模型预测的好坏。在此基础上可以得到多组不同的训练集和测试集，某次训练集中的某样本在下次可能成为测试集中的样本，即所谓“交叉”。

  那么什么时候才需要交叉验证呢？交叉验证用在数据不是很充足的时候。如果数据样本量小于一万条，我们就会采用交叉验证来训练优化选择模型。如果样本大于一万条的话，我们一般随机的把数据分成三份，一份为训练集（Training Set），一份为验证集（Validation Set），最后一份为测试集（Test Set）。用训练集来训练模型，用验证集来评估模型预测的好坏和选择模型及其对应的参数。把最终得到的模型再用于测试集，最终决定使用哪个模型以及对应参数。

一、基本方法
  回到交叉验证，根据切分的方法不同，交叉验证分为下面三种：

  第一种是简单交叉验证，所谓的简单，是和其他交叉验证方法相对而言的。首先，我们随机的将样本数据分为两部分（比如： 70%的训练集，30%的测试集），然后用训练集来训练模型，在测试集上验证模型及参数。接着，我们再把样本打乱，重新选择训练集和测试集，继续训练数据和检验模型。最后我们选择损失函数评估最优的模型和参数。

  第二种是 S折交叉验证（ S-Folder Cross Validation），也是经常会用到的。和第一种方法不同， S折交叉验证先将数据集 D随机划分为 S个大小相同的互斥子集，即 ，每次随机的选择 S-1份作为训练集，剩下的1份做测试集。当这一轮完成后，重新随机选择 S份来训练数据。若干轮（小于 S ）之后，选择损失函数评估最优的模型和参数。注意，交叉验证法评估结果的稳定性和保真性在很大程度上取决于 S取值。

​       第三种是留一交叉验证（Leave-one-out Cross Validation），它是第二种情况的特例，此时 S等于样本数 N，这样对于 N个样本，每次选择 N-1个样本来训练数据，留一个样本来验证模型预测的好坏。此方法主要用于样本量非常少的情况，比如对于普通适中问题， N小于50时，一般采用留一交叉验证。**（若训练的是nii，用留一验证或许会更好）**

​        通过反复的交叉验证，用损失函数来度量得到的模型的好坏，最终我们可以得到一个较好的模型。那这三种情况，到底我们应该选择哪一种方法呢？一句话总结，如果我们只是对数据做一个初步的模型建立，不是要做深入分析的话，简单交叉验证就可以了。否则就用S折交叉验证。在样本量少的时候，使用S折交叉验证的特例留一交叉验证。



参考文章：https://blog.csdn.net/weixin_42691585/article/details/113971857

---------------------

## Stage 4 Image preprocessing + CNN + k-fold cross validation 方案

​          这是我最终选择的方案，参考了2020优秀案例，实际上就是将nii转成了png，使得准确率和损失函数的数量级达到一个可观的程度

### 1.预处理

参考pre.ipynb，基本思路是取所有z轴切片的中间1/3而不是随机选50，这样就可以初步筛选有效样品。最后再通过自适应裁剪，使得处理后的图片能尽可能贴合大脑外壳，从而增大其ROI的区域。

#### 1.查看图片维度

```
import os
import nibabel as nib

# 文件夹路径
folder_path = '脑PET图像分析和疾病预测挑战赛公开数据/test/'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.nii.gz') or filename.endswith('.nii'):
        nifti_file_path = os.path.join(folder_path, filename)

        # 读取NIfTI文件
        nifti_img = nib.load(nifti_file_path)

        # 获取图像数据的形状（长、宽维度）
        nifti_shape = nifti_img.shape[:2]  # 获取前两个维度（长、宽）

        # 打印文件名和形状
        print(f"File: {filename}, Shape: {nifti_shape}")
```

output:

Shape: (128, 128)、Shape: (168, 168)、Shape: (256, 256)、Shape: (400, 400)

test、train\MCI文件夹中图片是相同的（Shape: (128, 128)），train\NC文件夹中的图像非常混乱

```
File: 1.nii, Shape: (128, 128)
File: 10.nii, Shape: (128, 128)
File: 11.nii, Shape: (128, 128)
File: 12.nii, Shape: (128, 128)
File: 13.nii, Shape: (128, 128)
File: 14.nii, Shape: (128, 128)
File: 15.nii, Shape: (128, 128)
File: 16.nii, Shape: (128, 128)
File: 17.nii, Shape: (168, 168)
File: 18.nii, Shape: (256, 256)
File: 19.nii, Shape: (168, 168)
File: 2.nii, Shape: (128, 128)
File: 20.nii, Shape: (168, 168)
File: 21.nii, Shape: (128, 128)
File: 22.nii, Shape: (256, 256)
File: 23.nii, Shape: (256, 256)
File: 24.nii, Shape: (256, 256)
File: 25.nii, Shape: (400, 400)
File: 3.nii, Shape: (128, 128)
File: 4.nii, Shape: (128, 128)
File: 5.nii, Shape: (128, 128)
File: 6.nii, Shape: (128, 128)
File: 7.nii, Shape: (128, 128)
File: 8.nii, Shape: (128, 128)
File: 9.nii, Shape: (128, 128)
```

<img width="420" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/887f8781-5c26-4a81-9421-300e48a5e1eb">


对于某些过大的图像，在128*128区域内有效数据其实已经够用。

#### 2.nii三维图像转成png二维切片（以NC为例，后续还要加上批处理）

```
import os
import nibabel as nib
from PIL import Image
import numpy as np

# 文件夹路径
folder_path = '脑PET图像分析和疾病预测挑战赛公开数据/train/NC/'

# 创建保存切片的输出文件夹
output_folder = '脑PET图像分析和疾病预测挑战赛公开数据/NC/'
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.nii.gz') or filename.endswith('.nii'):
        nifti_file_path = os.path.join(folder_path, filename)

        # 读取NIfTI文件
        nifti_img = nib.load(nifti_file_path)

        # 获取图像数据和形状
        nifti_data = nifti_img.get_fdata()
        nifti_shape = nifti_data.shape

        # 获取切片数量
        num_slices = nifti_shape[2]

        # 遍历切片并进行处理
        for slice_index in range(num_slices):
            # 提取单个切片
            single_slice = nifti_data[:, :, slice_index, 0]  # 注意这里索引为 [:, :, slice_index, 0]

            # 将切片数据规范化到0-255范围
            normalized_slice = ((single_slice - np.min(single_slice)) / (np.max(single_slice) - np.min(single_slice)) * 255).astype(np.uint8)

            # 将切片转换为PIL图像
            pil_image = Image.fromarray(normalized_slice)

            # 构造切片文件路径
            slice_filename = f"{filename.split('.')[0]}_slice_{slice_index}.png"
            slice_filepath = os.path.join(output_folder, slice_filename)

            # 保存切片为图像文件
            pil_image.save(slice_filepath)

            # 输出切片信息
            print(f"File: {filename}, Slice: {slice_index}, Shape: {single_slice.shape}")
```

<img width="414" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/de3cb72b-16d9-4eb6-82ac-bb57675916e3">



我们的数据和2020年的数据还是很不一样的，很多切片是脏数据，需要清洗；二维直接拿过来训练效果必定不佳。可以考虑三维卷积神经网络。

#### 3.查看像素值

这一步是为了检验2020方案的合理性，我们可以发现，2020方案将大于50像素的坐标作为边界是合理的。

```
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 图像文件路径
image_path = 'output_slices/1_slice_25.png'  # 替换为实际的图像文件路径

# 打开图像
image = Image.open(image_path)

# 将图像转换为NumPy数组
image_array = np.array(image)

# 获取图像的宽度和高度
width, height = image_array.shape

# 创建网格图
plt.figure(figsize=(width / 3, height / 3))  # 调整图像大小

# 绘制网格图
plt.imshow(image_array, cmap='gray', aspect='auto')

# 添加网格线
plt.grid(True, which='both', color='black', linewidth=1)

# 显示像素值
for y in range(height):
    for x in range(width):
        plt.text(x, y, str(image_array[y, x]), color='red', ha='center', va='center')

# 显示坐标轴
plt.xticks(np.arange(-0.5, width, 1), [])
plt.yticks(np.arange(-0.5, height, 1), [])

# 显示网格图
plt.show()
```

<img width="1112" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/e2927973-b2f9-407f-a1fa-3bbf378698be">


我们也可以查看裁剪效果：

```
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageProcessor:
    def crop_2(self, img):
        dowm = img.shape[0]
        up = img.shape[1]
        max1 = max(dowm, up)
        dowm = (max1 - dowm) // 2
        up = (max1 - up) // 2
        dowm_zuo, dowm_you = dowm, dowm
        up_zuo, up_you = up, up
        if (max1 - img.shape[0]) % 2 != 0:
            dowm_zuo = dowm_zuo + 1
        if (max1 - img.shape[1]) % 2 != 0:
            up_zuo = up_zuo + 1
        matrix_pad = np.pad(img, pad_width=((dowm_zuo, dowm_you),
                                            (up_zuo, up_you))
                            , mode="constant",
                            constant_values=(0, 0))
        img = matrix_pad
        return img

    def crop_1(self, img_path):
        img = Image.open(img_path)
        img = np.array(img)
        index = np.where(img > 50)
        x = index[0]
        y = index[1]
        max_x = max(x)
        min_x = min(x)
        max_y = max(y)
        min_y = min(y)
        max_x = max_x + 10
        min_x = min_x - 10
        max_y = max_y + 10
        min_y = min_y - 10
        if max_x > img.shape[0]:
            max_x = img.shape[0]
        if min_x < 0:
            min_x = 0
        if max_y > img.shape[1]:
            max_y = img.shape[1]
        if min_y < 0:
            min_y = 0
        img = img[min_x:max_x, min_y:max_y]
        return self.crop_2(img)

# 创建ImageProcessor实例
processor = ImageProcessor()

# 图像文件路径
img_path = 'E:/PETtrain/test/1_slice_30.png'  # 替换为实际的图像文件路径

# 打开原始图像
original_img = Image.open(img_path)

# 将原始图像转换为灰度图
gray_original_img = original_img.convert('L')

# 第一个裁剪步骤
cropped_img_1 = processor.crop_1(img_path)

# 第二个裁剪步骤
final_cropped_img = processor.crop_2(cropped_img_1)

# 创建画布
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 显示灰度原始图像
axes[0].imshow(gray_original_img, cmap='gray')
axes[0].set_title("Gray Original Image")
axes[0].axis('off')  # 不显示坐标轴

# 显示裁剪图像
axes[1].imshow(final_cropped_img, cmap='gray')
axes[1].set_title("Cropped Image")
axes[1].axis('off')  # 不显示坐标轴

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()
```

<img width="620" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/3dcf77d1-9513-4ee5-8a21-7c130c3843a1">


效果不错！

#### 4.批处理

这一步将两个裁剪处理融合，直接进行批处理。

```
import numpy as np
from PIL import Image

class ImageProcessor:
    def crop_2(self, img):
        dowm = img.shape[0]
        up = img.shape[1]
        max1 = max(dowm, up)
        dowm = (max1 - dowm) // 2
        up = (max1 - up) // 2
        dowm_zuo, dowm_you = dowm, dowm
        up_zuo, up_you = up, up
        if (max1 - img.shape[0]) % 2 != 0:
            dowm_zuo = dowm_zuo + 1
        if (max1 - img.shape[1]) % 2 != 0:
            up_zuo = up_zuo + 1
        matrix_pad = np.pad(img, pad_width=((dowm_zuo, dowm_you),
                                            (up_zuo, up_you))
                            , mode="constant",
                            constant_values=(0, 0))
        img = matrix_pad
        return img

    def crop_1(self, img_path):
        img = Image.open(img_path)
        img = np.array(img)
        index = np.where(img > 50)
        x = index[0]
        y = index[1]
        max_x = max(x)
        min_x = min(x)
        max_y = max(y)
        min_y = min(y)
        max_x = max_x + 10
        min_x = min_x - 10
        max_y = max_y + 10
        min_y = min_y - 10
        if max_x > img.shape[0]:
            max_x = img.shape[0]
        if min_x < 0:
            min_x = 0
        if max_y > img.shape[1]:
            max_y = img.shape[1]
        if min_y < 0:
            min_y = 0
        img = img[min_x:max_x, min_y:max_y]
        return self.crop_2(img)

# 创建ImageProcessor实例
processor = ImageProcessor()

# 图像文件夹路径
folder_path = '脑PET图像分析和疾病预测挑战赛公开数据/MCI'  # 替换为实际的图像文件夹路径

# 创建保存结果的文件夹
output_folder = '脑PET图像分析和疾病预测挑战赛公开数据/Train/MCI'
os.makedirs(output_folder, exist_ok=True)

# 遍历图像文件夹中的所有图像
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.png') or filename.lower().endswith('.jpg'):
        img_path = os.path.join(folder_path, filename)

        # 打开原始图像
        original_img = Image.open(img_path)

        # 第一个裁剪步骤
        cropped_img_1 = processor.crop_1(img_path)

        # 第二个裁剪步骤
        final_cropped_img = processor.crop_2(cropped_img_1)

        # 保存裁剪图像，保持原始的文件名
        output_path = os.path.join(output_folder, filename)  # 这里使用原始的filename
        final_cropped_pil_img = Image.fromarray(final_cropped_img)
        final_cropped_pil_img.save(output_path)
        print(f"Final cropped image saved to {output_path}")
```

### 2.模型训练

这次的模型训练和baseline最大的不同就是输入维度=1，适当调整了batch size和epoch，显著地增大了准确率，降低了loss。

具体修改，详见cnn.ipynb。

<img width="547" alt="image" src="https://github.com/l1jiewansui/CVnotebook/assets/134419371/57c6e1e6-c173-4bb3-9874-8c8e3ce46639">

准确率依然存在跳变，说明方案依然存在改进空间。（已经通过减小学习率解决，学习率需要1e-4以降）

### 3.测试集

测试集的处理也是非常重要的，首先需要对nii文件进行相同的处理，取1/3的切片，然后对所有切片进行模型判断NC/MCI，最后根据nii文件所有切片投票的多寡来判断最终类别。

当然，这可能会对某些模糊的对象产生误判，因为NC/MCI投票无法体现图像偏向NC/MCI的程度，但直接累加结果的二维向量（正如baseline一样），可能无法保证数量级的一致，因为不同的切片结果向量的数量级是不一样的。因此本方案依然采用投票法。

```
from collections import Counter
from tqdm import tqdm

def predict_majority(test_loader, model):
    model.eval()
    test_pred = []

    with torch.no_grad():
        for input, _ in test_loader:
            input = input.cuda()
            output = model(input)
            test_pred.append(output.cpu())  # No need for .data.numpy() here

    return torch.cat(test_pred, dim=0)  # Concatenate along batch dimension

# Get model predictions
pred = predict_majority(test_loader, model)

# Calculate predicted labels
predicted_labels = pred.argmax(dim=1)
predicted_labels = predicted_labels.tolist()

# Create a dictionary to store NC and MCI counts for each slice
slice_counts = {}
for i, path in enumerate(test_path):
    slice_name = path.split('/')[-1]
    slice_num = slice_name.split('_')[1]
    label = predicted_labels[i]

    if slice_num not in slice_counts:
        slice_counts[slice_num] = Counter()

    slice_counts[slice_num][label] += 1

# Create a dictionary to store predictions for each sample
sample_predictions = {}

# Fill in sample_predictions with individual predictions
for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
    input = input.cuda()
    output = model(input)
    predicted_scores = output.cpu().tolist()  # Get prediction scores instead of labels
    
    slice_names = [path.split('\\')[-1] for path in test_path[i * 32 : (i + 1) * 32]]
    sample_ids = [name.split('_')[0] for name in slice_names]
    
    for j, sample_id in enumerate(sample_ids):
        if sample_id not in sample_predictions:
            sample_predictions[sample_id] = []
        sample_predictions[sample_id].append(predicted_scores[j])

# Determine majority label for each sample
final_predictions = []
sorted_sample_predictions = sorted(sample_predictions.items(), key=lambda x: int(x[0]))

for sample_id, scores_list in sorted_sample_predictions:
    avg_scores = np.mean(scores_list, axis=0)  # Calculate average scores across slices
    predicted_label = np.argmax(avg_scores)  # Choose the class with the highest average score
    
    # Assuming predicted_label 0 corresponds to "NC" and predicted_label 1 corresponds to "MCI"
    if predicted_label == 0:
        final_predictions.append('NC')
    else:
        final_predictions.append('MCI')

# Create submission DataFrame
submit = pd.DataFrame(
    {
        'uuid': [int(sample_id) for sample_id, _ in sorted_sample_predictions],
        'label': final_predictions
    }
)
submit.to_csv('submits.csv', index=None)
```
21:39最后一次提交，成绩为0.86364	

## 杂谈和勘误

1.为什么要用F1score,准确率+召回率，比较均衡的评价指标。

2.库内有nii转化为png的python脚本

3.下载特定包换腾讯源-i http://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com

4.全NC的结果是0.74214。

5.41个NC,判断公式为 

The formula is: 

```
$$
F1score=\frac{2}{\frac{59}{x}+\frac{100-x}{x}}
$$
```

6.baseline里train_loader和val_loader可能打错了，裁剪应该是128不是120



Thank you for watching!
