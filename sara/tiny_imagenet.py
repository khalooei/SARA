"""Tiny-ImageNet-200 dataset (200 classes, 64x64, 500 train / 50 val images per
class). It is not included in torchvision; this Dataset reads the standard directory
layout obtained by unzipping
https://cs231n.stanford.edu/tiny-imagenet-200.zip:

    <root>/train/<wnid>/images/<wnid>_<n>.JPEG
    <root>/val/images/val_<n>.JPEG
    <root>/val/val_annotations.txt      (val_<n>.JPEG \t <wnid> \t ...)
    <root>/wnids.txt                    (200 class ids, one per line)
"""
import os

from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        assert split in ("train", "val")
        self.root = root
        self.transform = transform

        with open(os.path.join(root, "wnids.txt")) as f:
            self.classes = sorted(line.strip() for line in f if line.strip())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        if split == "train":
            for wnid in self.classes:
                img_dir = os.path.join(root, "train", wnid, "images")
                for fname in sorted(os.listdir(img_dir)):
                    self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[wnid]))
        else:
            ann_path = os.path.join(root, "val", "val_annotations.txt")
            img_dir = os.path.join(root, "val", "images")
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    fname, wnid = parts[0], parts[1]
                    if wnid in self.class_to_idx:
                        self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
