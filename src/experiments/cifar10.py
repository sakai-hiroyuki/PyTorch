from typing import Optional
from time import time
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torchvision.transforms import ToTensor, Compose
from torchvision.datasets import CIFAR10

from experiments import Experiment
from utils import Cutout

__all__ = ['ExperimentCIFAR10']


class ExperimentCIFAR10(Experiment):
    def __init__(
        self,
        model     : nn.Module,
        optimizer : Optimizer,
        max_epoch : int,
        batch_size: int,
        data_dir  : str = './data',
        csv_dir   : str = './results/csv/cifar10',
        csv_name  : Optional[str] = None,
        prm_dir   : str = './results/prm/cifar10',
        prm_name  : Optional[str] = None,
        token     : Optional[str] = None,
        download  : bool=True,
        cutout    : bool=False
    ) -> None:

        super().__init__(
            model,
            optimizer,
            max_epoch=max_epoch,
            batch_size=batch_size,
            data_dir=data_dir,
            csv_dir=csv_dir,
            csv_name=csv_name,
            prm_dir=prm_dir,
            prm_name=prm_name,
            token=token
        )
        self.download = download
        self.cutout = cutout
    
    def prepare_data(self) -> tuple[DataLoader, DataLoader]:
        train_transforms = [ToTensor()]
        if self.cutout:
            train_transforms.append(Cutout(1, 16))
        train_data: Dataset = CIFAR10(
            self.data_dir,
            train=True,
            download=self.download,
            transform=Compose(train_transforms)
        )

        test_transforms = [ToTensor()]
        test_data: Dataset = CIFAR10(
            self.data_dir,
            train=False,
            download=False,
            transform=Compose(test_transforms)
        )

        train_loader: DataLoader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader : DataLoader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def _train(self, train_loader: DataLoader) -> list[float]:
        model = self.model
        optimizer = self.optimizer

        criterion = nn.CrossEntropyLoss()
        i           : int   = 0       # ステップ数
        total       : int   = 0       # 全ての訓練データの数
        correct     : int   = 0       # 正しく分類された訓練データ数
        running_loss: float = 0.0     # 経験損失の合計
        tic         : float = time()  # 訓練開始の時間 (UNIX時間)

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs: torch.Tensor = model(inputs)
            loss   : torch.Tensor = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i += 1

        train_loss: float = running_loss / i
        train_acc : float = correct / total
        train_time: float = time() - tic
        return [train_loss, train_acc, train_time]
    
    def _eval(self, test_loader: DataLoader) -> list[float]:
        model = self.model

        criterion = nn.CrossEntropyLoss()
        i           : int   = 0    # ステップ数
        total       : int   = 0    # 全てのテストデータの数
        correct     : int   = 0    # 正しく分類されたテストデータの数
        running_loss: float = 0.0  # 予測損失の合計

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs: torch.Tensor = model(inputs)
                loss   : torch.Tensor = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                i += 1

        test_loss: float = running_loss / i
        test_acc : float = correct / total
        return [test_loss, test_acc]
