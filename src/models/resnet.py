from __future__ import annotations
from typing import Optional

import torch
from torch import nn


__all__ = [
    'resnet20_cifar10',
    'resnet32_cifar10',
    'resnet44_cifar10',
    'resnet56_cifar10',
    'resnet110_cifar10',
]


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes  : int,
        planes    : int,
        stride    : int=1,
        downsample: Optional[nn.Module]=None
    ) -> None:

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCIFAR10(nn.Module):
    def __init__(
        self,
        layers     : list[int],
        num_classes: int=10
    ) -> None:

        super(ResNetCIFAR10, self).__init__()
        self.inplanes = 32

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(16, layers[0])
        self.conv3 = self._make_layer(32, layers[1], stride=2)
        self.conv4 = self._make_layer(64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int=1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(
            BasicBlock(self.inplanes, planes, stride, downsample)
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(self.inplanes, planes)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def resnet20_cifar10() -> ResNetCIFAR10:
    return ResNetCIFAR10([3, 3, 3])


def resnet32_cifar10() -> ResNetCIFAR10:
    return ResNetCIFAR10([5, 5, 5])


def resnet44_cifar10() -> ResNetCIFAR10:
    return ResNetCIFAR10([7, 7, 7])


def resnet56_cifar10() -> ResNetCIFAR10:
    return ResNetCIFAR10([9, 9, 9])


def resnet110_cifar10() -> ResNetCIFAR10:
    return ResNetCIFAR10([18, 18, 18])
