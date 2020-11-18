import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms.transforms import RandomErasing
from .loss import _gram


class ImageDataset(Dataset):
    def __init__(self, path, image_shape, device):
        super(ImageDataset, self).__init__()
        self.rgb_mean = np.asarray([0.485, 0.456, 0.406])
        self.rgb_std = np.asarray([0.229, 0.224, 0.225])
        tensor_normalizer = transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        width, height, = image_shape
        self.device = device
        self.data_transform = transforms.Compose([
            transforms.RandomResizedCrop((width, height), ratio=(1, 1)),
            transforms.ToTensor(),
            transforms.RandomErasing(),
            tensor_normalizer
        ])
        path = Path(path)
        self.images = []
        for file in os.listdir(path):
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                image = Image.open(path / file)
                self.images.append(image)
        self.images = [self._preprocess(ele, image_shape) for ele in self.images]
        self.images = torch.cat(self.images, dim=0)
        self.images = self.images.to(device)

    def _preprocess(self, images, image_shape):
        process = transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        return process(images).unsqueeze(dim=0)  # (batch_size, 3, H, W)

    @staticmethod
    def postprocess(img_tensor):
        rgb_mean = np.asarray([0.485, 0.456, 0.406])
        rgb_std = np.asarray([0.229, 0.224, 0.225])
        inv_normalize = transforms.Normalize(
            mean=-rgb_mean/rgb_std,
            std=1/rgb_std
        )
        to_pil_image = transforms.ToPILImage()
        return to_pil_image(inv_normalize(img_tensor.cpu()).clamp(0, 1))

    def __getitem__(self, index: int):
        return self.images[index]

    def __len__(self) -> int:
        return len(self.images)

    def get(self, extractor):
        contents, styles = extractor(self.images)
        styles_y_gram = [_gram(y) for y in styles]
        return self.images, contents, styles, styles_y_gram

    @staticmethod
    def save(path, images):
        path = Path(path)
        if not os.path.exists(path):
            os.makedirs(path)
        for i, image in enumerate(images):
            plt.imshow(image)
            plt.axis('off')
            plt.show()
            plt.savefig(path / f'{i}.jpg')
