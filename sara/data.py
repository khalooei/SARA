"""Data loaders for MNIST, CIFAR10, CIFAR100, SVHN and Tiny-ImageNet-200."""
import os

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .tiny_imagenet import TinyImageNet

# Datasets are stored in ./data by default; set SARA_DATA_ROOT to use another location.
DATA_ROOT = os.environ.get(
    "SARA_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)

# Images are kept in [0, 1] without per-channel normalization, so perturbation
# budgets and the pixel-range projection are expressed directly in pixel space.


def get_datasets(name: str, augment: bool = True, augmix: bool = False):
    name = name.lower()
    if name == "mnist":
        train_tf = [T.ToTensor()]
        test_tf = [T.ToTensor()]
        train = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=T.Compose(train_tf))
        test = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=T.Compose(test_tf))
    elif name == "cifar10":
        train_tf = [T.ToTensor()]
        if augmix:
            # AugMix (Hendrycks et al., 2020) operates on uint8 PIL images and
            # must therefore precede ToTensor().
            train_tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.AugMix(), T.ToTensor()]
        elif augment:
            train_tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]
        test_tf = [T.ToTensor()]
        train = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=T.Compose(train_tf))
        test = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=T.Compose(test_tf))
    elif name == "cifar100":
        train_tf = [T.ToTensor()]
        if augment:
            train_tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]
        test_tf = [T.ToTensor()]
        train = torchvision.datasets.CIFAR100(DATA_ROOT, train=True, download=True, transform=T.Compose(train_tf))
        test = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=T.Compose(test_tf))
    elif name == "tinyimagenet":
        root = os.path.join(DATA_ROOT, "tiny-imagenet-200")
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"{root} not found. Download "
                "https://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into "
                f"{DATA_ROOT} (Tiny-ImageNet is not downloaded automatically)."
            )
        train_tf = [T.ToTensor()]
        if augment:
            train_tf = [T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor()]
        test_tf = [T.ToTensor()]
        train = TinyImageNet(root, split="train", transform=T.Compose(train_tf))
        test = TinyImageNet(root, split="val", transform=T.Compose(test_tf))
    elif name == "svhn":
        train_tf = [T.ToTensor()]
        test_tf = [T.ToTensor()]
        train = torchvision.datasets.SVHN(DATA_ROOT, split="train", download=True, transform=T.Compose(train_tf))
        test = torchvision.datasets.SVHN(DATA_ROOT, split="test", download=True, transform=T.Compose(test_tf))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test


def get_loaders(name: str, batch_size: int = 128, augment: bool = True, num_workers: int = 8,
                 train_subset: int = None, test_subset: int = None, augmix: bool = False):
    train, test = get_datasets(name, augment=augment, augmix=augmix)
    if train_subset is not None:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(train), generator=g)[:train_subset]
        train = torch.utils.data.Subset(train, idx.tolist())
    if test_subset is not None:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(test), generator=g)[:test_subset]
        test = torch.utils.data.Subset(test, idx.tolist())
    # Persistent workers and prefetching keep data loading from becoming the
    # bottleneck for these small models.
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                   persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, **common)
    return train_loader, test_loader


DEFAULT_EPSILON = {
    "mnist": 0.2,
    "cifar10": 0.03,
    "cifar100": 0.03,
    "svhn": 0.03,
    "tinyimagenet": 0.03,
}

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "tinyimagenet": 200,
}
