## Import the necessary libraries.
import torch
import torchvision
import torchvision.transforms as transforms


## Load the MNIST Dataset
## You can easily load the MNIST dataset using PyTorch's torchvision library.

# 定义图像变换操作，将原始图像转换为模型可接受的格式
transform = transforms.Compose([
    transforms.ToTensor(),                  # 将图像转换为 PyTorch 的 Tensor 格式
    transforms.Normalize((0.5,), (0.5,))   # 对图像进行标准化，将像素值缩放到范围[-1, 1]
])

# 从 torchvision 中加载 MNIST 数据集的训练集
# - root='./data': 数据集将被下载并保存在当前工作目录下的 'data' 子目录中
# - train=True: 加载训练集
# - download=True: 如果数据集不存在，则下载数据集
# - transform=transform: 对加载的图像应用上面定义的图像变换操作
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 创建一个 DataLoader 对象，用于对数据进行批量加载和处理
# - trainset: 要加载的数据集
# - batch_size=4: 每个批次包含的图像样本数量
# - shuffle=True: 打乱数据，以便在每个 epoch 中随机访问样本
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# 定义类标签，这里是用字符串表示数字 0 到 9
classes = tuple(str(i) for i in range(10))