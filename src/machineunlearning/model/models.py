"""
Source: https://github.com/weiaicunzai/pytorch-cifar100
"""

import torch
import torch.nn.functional as F
from torch import nn

from machineunlearning.model.resnet import BasicBlock, BottleNeck, ResNet


def ResNet18(num_classes, input_channels):
    """return a ResNet 18 object"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channel=input_channels)


def ResNet34(num_classes, input_channels):
    """return a ResNet 34 object"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, input_channel=input_channels)


def ResNet50(num_classes, input_channels):
    """return a ResNet 50 object"""
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, input_channel=input_channels)


def ResNet101(num_classes, input_channels):
    """return a ResNet 101 object"""
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, input_channel=input_channels)


def ResNet152(num_classes, input_channels):
    """return a ResNet 152 object"""
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes, input_channel=input_channels)


class MLP(nn.Module):
    def __init__(self, num_classes, input_channels, image_size=32, hidden=(256, 128)):
        super().__init__()
        in_dim = input_channels * image_size * image_size
        self.f_connected1 = nn.Linear(in_dim, hidden[0])
        self.f_connected2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        return self.out(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_channels, image_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        feat = image_size // 8
        self.fc1 = nn.Linear(64 * feat * feat, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LRTorchNet(nn.Module):
    def __init__(self, num_classes, input_channels, image_size=32):
        super().__init__()
        self.linear = nn.Linear(input_channels * image_size * image_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.linear(x))
