from torch import tensor
from torch.utils.data import dataset
import torchvision
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.transforms import GaussianBlur

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),    # normalize
    # transforms.CenterCrop(30),  # ? 中心裁减，这个感觉在MNIST上用处不大
    # transforms.RandomErasing(),  # ? 随机擦除对MNIST可能使用效果不太好
    transforms.RandomPerspective(distortion_scale=0.2),  # * 角度在20°以内对于MNIST的label应该是安全的
    transforms.GaussianBlur(kernel_size=(7, 7)),
    # transforms.Lambda(lambd=(lambda x:twist(x))),
    transforms.Lambda(lambd=lambda x:pepperNoise(x, prob=0.1)),
])


def twist(img: torch.Tensor, alpha: float = 0.5):
    """
    Args: @param img    图像原文本，Tensor
          @param alpha  扭曲的程度，在0-1之间
    """
    return img


def pepperNoise(img: torch.Tensor, prob: float = 0):
    noise_tensor = torch.rand(img.shape)
    salt = torch.max(img)
    pepper = torch.min(img)
    img[noise_tensor < prob/2] += salt
    img[noise_tensor > prob/2] += pepper
    return img+noise_tensor


def img_show(img: torch.Tensor, label: torch.Tensor):
    print(np.array(label).reshape(1, 1))
    img = torchvision.utils.make_grid(image).numpy().transpose(1, 2, 0)  # *std + mean
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    data_train = datasets.MNIST(root="Data", transform=transform, train=True)
    data_test = datasets.MNIST(root="Data", train=False, transform=transform)
    data_loader_train = DataLoader(dataset=data_train, batch_size=1, shuffle=True)
    data_loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=True)

    for i, (image, label) in enumerate(data_loader_train):
        img_show(image, label)
        break
