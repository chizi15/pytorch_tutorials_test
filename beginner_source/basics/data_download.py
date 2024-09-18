import os
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


cwd = os.path.dirname(__file__)
try: # 检测数据集是否已下载
    training_data = datasets.FashionMNIST(
        root=os.path.join(cwd, "data"),
        train=True,
        download=False,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=os.path.join(cwd, "data"),
        train=False,
        download=False,
        transform=ToTensor()
    )

    ds = datasets.FashionMNIST(
        root=os.path.join(cwd, "data_transform"),
        train=True,
        download=False,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

except Exception as e: # 下载数据集
    print(e)
    training_data = datasets.FashionMNIST(
        root=os.path.join(cwd, "data"),
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=os.path.join(cwd, "data"),
        train=False,
        download=True,
        transform=ToTensor()
    )

    ds = datasets.FashionMNIST(
        root=os.path.join(cwd, "data_transform"),
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )


def estimate_dataset_memory_usage(dataset):
    # 假设数据集中的每个样本大小都相似
    sample_size = 0
    calc_size = 1000
    for i in range(calc_size):  # 取前calc_size个样本作为样本大小的估计
        img, label = dataset[i]
        sample_size += img.element_size() * img.nelement()
    
    total_samples = len(dataset)
    estimated_size = (sample_size / calc_size) * total_samples  # 估计整个数据集的大小
    return estimated_size, calc_size
