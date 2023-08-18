# CVnotebook
具体笔记详见jupyternotebook注释

## Phase 1

### 1.赛题要求

这次的赛题旨在利用脑PET图像进行轻度认知障碍（MCI）的疾病预测。参赛者需要根据所提供的脑PET图像数据集，构建模型来进行轻度认知障碍（MCI）的预测，即对被试者进行分类，判断其是否属于MCI（轻度认知障碍）或健康（NC）。解题思路可以遵循计算机视觉分类的一般步骤，包括数据预处理、模型构建、训练和评估等阶段。在这个赛题中，可以采用深度学习模型（如卷积神经网络）等方法来从脑PET图像中提取特征，并进行分类预测，最终分类结果通过F1score指标评判。

### 2.baseline中人工提取的特征是哪些？

非零像素的数量、零像素的数量、平均值、标准差、在列方向上平均值不为零的数量、在行方向上平均值不为零的数量、列方向上的最大平均值、行方向上的最大平均值

### 3.你对baseline还有些什么样的认知？

baseline使用的机器学习方法（基于特征提取和逻辑回归）的优势和局限在于：

优势：
1.解释性强： 逻辑回归是一种线性模型，其预测结果相对容易解释，可以理解每个特征对预测的影响程度。
2.计算效率高： 逻辑回归计算速度较快，适用于大规模数据集和较大特征空间。
3.简单快速： 逻辑回归是一个相对简单的模型，不需要大量的超参数调整。

局限性：
1.这种方法需要手动设计和提取特征，依赖于领域知识，且特征的质量对模型影响很大。
2.局限于线性关系： 逻辑回归只能建模线性关系，对于非线性关系的数据表现较差。
3.泛化能力受限： 逻辑回归在面对高维和复杂数据时，泛化能力可能受到限制，可能会出现欠拟合或过拟合。

### 4.你还有些什么优化和上分思路？

目前在尝试CNN方法，尝试使用CV交叉验证的方法以及修改ResNet模型，类似于ResNet-50、ResNet101等更深的模型进行进一步优化。下一步计划是1.尝试不同的折叠数目或增加epoch，来寻找更好的模型泛化性能。（同时注意过拟合）2.超参数调整：调整学习率、权重衰减等超参数，使用学习率调度器（StepLR、ReduceLROnPlateau）来动态调整学习率。

### 5.各种模型

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

### 6.代码解析

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

## 杂谈

1.为什么要用F1score,准确率+召回率，比较均衡的评价指标。

2.库内有nii转化为png的python脚本

3.下载特定包换腾讯源-i http://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com

4.全NC的结果是0.74214，现在最佳结果为0.76712
