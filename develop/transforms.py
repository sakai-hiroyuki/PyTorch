import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose
from torchvision.utils import save_image, make_grid


class Cutout(object):
    '''
    Randomly mask out one or more patches from an image.
    
    Attributes
    ----------
    n_holes: int
        Number of patches to cut out of each image.

    length: int
        The length (in pixels) of each square patch.

    References
    ----------
    - T DeVries, GW Taylor. "Improved regularization of convolutional
      neural networks with cutout." arXiv preprint arXiv:1708.04552 (2017).
    - https://github.com/uoguelph-mlrg/Cutout
    '''
    def __init__(self, n_holes: int, length: int):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor):
        '''
        Parameters
        ----------
        img: torch.Tensor
            Tensor image of size (C, H, W).
        
        Returns
        -------
        torch.Tensor
            Image with n_holes of dimension length x length cut out of it.
        '''
        h: int = img.size(1)
        w: int = img.size(2)

        mask: np.ndarray = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


train_transform = Compose([
    ToTensor(),
    Cutout(1, 16)
])
test_transform = Compose([
    ToTensor()
])

train_data : Dataset = CIFAR10('./data', train=True, download=False, transform=train_transform)
test_data: Dataset = CIFAR10('./data', train=False, download=False, transform=test_transform)

train_loader: DataLoader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader: DataLoader = DataLoader(test_data, batch_size=64, shuffle=False)



for inputs, labels in train_loader:
    imgs = inputs
    grid = make_grid(imgs, nrow=8, padding=2)
    save_image(grid, 'test.jpg')

    break
