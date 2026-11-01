"""Network architectures: LeNet and ResNet18."""
import torch
import torch.nn as nn
import torchvision


class LeNet(nn.Module):
    """Conv(5x5)->Pool(2x2)->Conv(5x5)->Pool(2x2)->Dense(120)->Dense(82)->Dense(num_classes)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, input_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            flat_dim = self._features(dummy).flatten(1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 120)
        self.fc2 = nn.Linear(120, 82)
        self.fc3 = nn.Linear(82, num_classes)

    def _features(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._features(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def resnet18(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """Standard torchvision ResNet18, with the first convolution adapted to the input channels."""
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def resnet18_cifar(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """ResNet18 with the small-image stem used in the adversarial-training literature
    (Madry et al., 2018; Rice et al., 2020): a 3x3, stride-1 first convolution and no max-pooling,
    so that 32x32 inputs reach the first residual stage at full resolution."""
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(name: str, dataset: str) -> nn.Module:
    from .data import NUM_CLASSES
    in_channels = 1 if dataset == "mnist" else 3
    input_size = 28 if dataset == "mnist" else (64 if dataset == "tinyimagenet" else 32)
    num_classes = NUM_CLASSES[dataset]
    if name == "lenet":
        return LeNet(in_channels=in_channels, num_classes=num_classes, input_size=input_size)
    elif name == "resnet18":
        return resnet18(in_channels=in_channels, num_classes=num_classes)
    elif name == "resnet18-cifar":
        return resnet18_cifar(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
