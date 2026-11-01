"""Data loaders for MNIST, CIFAR10, CIFAR100, SVHN and Tiny-ImageNet-200."""
import os

import torch
import torch.nn.functional as F
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


AUGMENT_CROP_PADDING = {"cifar10": 4, "cifar100": 4, "tinyimagenet": 8}


def _tensor_cache(name: str, split: str):
    """uint8 image tensor (N, C, H, W) and int64 labels of a dataset split, cached in DATA_ROOT.

    The cache is built from the torchvision dataset on first use; it can also be created
    directly (for example from a mirror) as a dict {"x": uint8 tensor, "y": int64 tensor}.
    """
    path = os.path.join(DATA_ROOT, f"{name}_{split}.pt")
    if os.path.exists(path):
        d = torch.load(path)
        return d["x"], d["y"]
    train, test = get_datasets(name, augment=False)
    ds = train if split == "train" else test
    xs, ys = [], []
    # Single process: worker processes pass batches through /dev/shm, which is small in containers.
    for x, y in DataLoader(ds, batch_size=1000, num_workers=0):
        xs.append((x * 255).round().to(torch.uint8))
        ys.append(torch.as_tensor(y))
    x, y = torch.cat(xs), torch.cat(ys).long()
    torch.save({"x": x, "y": y}, path)
    return x, y


class GPUTensorLoader:
    """Mini-batch iterator over a dataset held on the GPU as uint8 tensors.

    Random cropping (zero padding) and horizontal flipping are applied on the GPU, with
    the same semantics as torchvision's RandomCrop(padding) and RandomHorizontalFlip.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, drop_last: bool,
                 crop_padding: int = 0, device="cuda", seed: int = None):
        self.x = x.to(device)
        self.y = y.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pad = crop_padding
        self.dataset = self.y  # len(loader.dataset) == number of samples
        # With a seed, shuffling and augmentation draw from a dedicated generator, so that runs with the same seed see
        # the same mini-batches whatever random numbers the training step consumes (paired comparisons).
        self.generator = torch.Generator(device=self.y.device).manual_seed(seed) if seed is not None else None

    def __len__(self):
        n = self.y.shape[0]
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _augment(self, xb):
        B, C, H, W = xb.shape
        p = self.pad
        xp = F.pad(xb, (p, p, p, p))
        oy = torch.randint(0, 2 * p + 1, (B,), device=xb.device, generator=self.generator)
        ox = torch.randint(0, 2 * p + 1, (B,), device=xb.device, generator=self.generator)
        rows = (oy[:, None] + torch.arange(H, device=xb.device)[None]).view(B, 1, H, 1)
        cols = (ox[:, None] + torch.arange(W, device=xb.device)[None]).view(B, 1, 1, W)
        xp = torch.gather(xp, 2, rows.expand(B, C, H, W + 2 * p))
        xb = torch.gather(xp, 3, cols.expand(B, C, H, W))
        flip = torch.rand(B, device=xb.device, generator=self.generator) < 0.5
        return torch.where(flip.view(B, 1, 1, 1), xb.flip(3), xb)

    def __iter__(self):
        n = self.y.shape[0]
        if self.shuffle:
            idx = torch.randperm(n, device=self.y.device, generator=self.generator)
        else:
            idx = torch.arange(n, device=self.y.device)
        stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for i in range(0, stop, self.batch_size):
            b = idx[i:i + self.batch_size]
            xb = self.x[b].float().div_(255.0)
            if self.pad > 0:
                xb = self._augment(xb)
            yield xb, self.y[b]


def get_gpu_loaders(name: str, batch_size: int = 128, augment: bool = True, val_size: int = 0,
                    train_subset: int = None, test_subset: int = None, eval_batch_size: int = 500,
                    device="cuda", seed: int = None):
    """GPU-resident train / validation / test loaders.

    With val_size > 0, a fixed random subset of the training set (independent of the
    training seed) is held out as a validation set and excluded from training; model
    selection then never uses the test set.
    """
    x_tr, y_tr = _tensor_cache(name, "train")
    x_te, y_te = _tensor_cache(name, "test")
    perm = torch.randperm(x_tr.shape[0], generator=torch.Generator().manual_seed(0))
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    if train_subset is not None:
        tr_idx = tr_idx[:train_subset]
    if test_subset is not None:
        te_idx = torch.randperm(x_te.shape[0], generator=torch.Generator().manual_seed(0))[:test_subset]
        x_te, y_te = x_te[te_idx], y_te[te_idx]
    pad = AUGMENT_CROP_PADDING.get(name, 0) if augment else 0
    train_loader = GPUTensorLoader(x_tr[tr_idx], y_tr[tr_idx], batch_size, shuffle=True, drop_last=True,
                                   crop_padding=pad, device=device, seed=seed)
    test_loader = GPUTensorLoader(x_te, y_te, eval_batch_size, shuffle=False, drop_last=False, device=device)
    val_loader = None
    if val_size > 0:
        val_loader = GPUTensorLoader(x_tr[val_idx], y_tr[val_idx], eval_batch_size, shuffle=False,
                                     drop_last=False, device=device)
    return train_loader, val_loader, test_loader


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
