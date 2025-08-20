import os
from PIL import Image
import sys
sys.path.append('..')
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10, resnet56_cifar10
from sanity_check.backends.resnet20_128 import resnet34_MSTAR, resnet50_MSTAR, resnet18_MSTAR
from sanity_check.backends.densenet import densenet121, densenet161, densenet169, densenet201

class MSTAR(Dataset):
    """MSTAR数据集加载器"""

    def __init__(self, root, train=True, transform=None):
        """
        初始化MSTAR数据集
        :param root: 数据集根目录
        :param train: 是否为训练集
        :param transform: 图像变换
        """
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        # 确定数据集文件夹
        self.data_dir = os.path.join(self.root, 'train' if train else 'test')

        # 获取所有类别（子文件夹名称）
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 加载图像路径和对应的标签
        self.images = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            # 获取该类别下所有BMP图像
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith('.bmp'):
                    img_path = os.path.join(cls_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __getitem__(self, index):
        """
        获取指定索引的样本
        :param index: 样本索引
        :return: 图像和标签的元组
        """
        img_path = self.images[index]
        label = self.labels[index]

        # 打开图像（MSTAR数据集通常为灰度图，这里转换为RGB以便与预训练模型兼容）
        img = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """返回数据集大小"""
        return len(self.images)


# 数据变换 - 可以根据需要调整
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(128, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载数据集
def get_mstar_loaders(root='mstar', batch_size=64, num_workers=4):
    """
    获取MSTAR数据集的数据加载器
    :param root: 数据集根目录
    :param batch_size: 批处理大小
    :param num_workers: 加载数据的进程数
    :return: 训练集和测试集的数据加载器
    """
    trainset = MSTAR(
        root=root,
        train=True,
        transform=train_transform
    )

    testset = MSTAR(
        root=root,
        train=False,
        transform=test_transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return trainloader, testloader


def check_model_dimensions(model, test_loader):
    """检查模型各层输入输出维度是否匹配"""
    # 获取一个批次的测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    print(f"输入图像批次形状: {images.shape}")  # 应为 [batch_size, 3, H, W]

    # 跟踪每一层的输出维度
    x = images  # 初始输入

    # 记录每一层的输出维度
    layer_outputs = []

    # 第一层：conv1 + bn1 + relu
    x = model.conv1(x)
    layer_outputs.append(("conv1", x.shape))
    x = model.bn1(x)
    x = F.relu(x)
    layer_outputs.append(("conv1 + bn + relu", x.shape))

    # 第二层：layer1
    x = model.layer1(x)
    layer_outputs.append(("layer1", x.shape))

    # 第三层：layer2
    x = model.layer2(x)
    layer_outputs.append(("layer2", x.shape))

    # 第四层：layer3
    x = model.layer3(x)
    layer_outputs.append(("layer3", x.shape))

    # 平均池化层
    x = model.avg_pool2d(x)
    layer_outputs.append(("avg_pool2d", x.shape))

    # 展平操作
    x = x.view(x.size(0), -1)
    layer_outputs.append(("展平后", x.shape))

    # 全连接层
    try:
        x = model.linear(x)
        layer_outputs.append(("全连接层", x.shape))
        print("所有层维度匹配正常!")
    except RuntimeError as e:
        print(f"维度匹配错误: {e}")

    # 打印各层输出维度
    print("\n各层输出维度:")
    for layer_name, shape in layer_outputs:
        print(f"{layer_name}: {shape}")

    return layer_outputs
# 使用示例
if __name__ == '__main__':
    # 获取数据加载器
    # trainloader, testloader = get_mstar_loaders(root='../datasets/MSTAR')
    model = resnet50_MSTAR()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # for X, y in trainloader:
    #     X = X
    #     y = y
    #     y_pred = model.forward(X)

    # # 打印数据集信息
    # print(f"训练集样本数: {len(trainloader.dataset)}")
    # print(f"测试集样本数: {len(testloader.dataset)}")
    # print(f"类别: {trainloader.dataset.classes}")



