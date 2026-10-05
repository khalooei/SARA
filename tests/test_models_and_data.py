import os

import pytest
import torch
from PIL import Image

from sara.data import DEFAULT_EPSILON, NUM_CLASSES
from sara.models import build_model
from sara.tiny_imagenet import TinyImageNet


@pytest.mark.parametrize(
    "model, dataset, shape",
    [
        ("lenet", "mnist", (2, 1, 28, 28)),
        ("resnet18", "cifar10", (2, 3, 32, 32)),
        ("resnet18", "cifar100", (2, 3, 32, 32)),
        ("resnet18", "tinyimagenet", (2, 3, 64, 64)),
    ],
)
def test_model_output_shapes(model, dataset, shape):
    net = build_model(model, dataset).eval()
    with torch.no_grad():
        out = net(torch.rand(*shape))
    assert out.shape == (shape[0], NUM_CLASSES[dataset])


def test_dataset_constants():
    assert set(DEFAULT_EPSILON) == set(NUM_CLASSES)
    assert DEFAULT_EPSILON["mnist"] == 0.2
    assert NUM_CLASSES["tinyimagenet"] == 200


def test_tiny_imagenet_reader(tmp_path):
    wnids = ["n01", "n02"]
    (tmp_path / "wnids.txt").write_text("\n".join(wnids) + "\n")
    for wnid in wnids:
        d = tmp_path / "train" / wnid / "images"
        d.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (64, 64)).save(d / f"{wnid}_{i}.JPEG")
    val = tmp_path / "val" / "images"
    val.mkdir(parents=True)
    lines = []
    for i, wnid in enumerate(wnids):
        Image.new("RGB", (64, 64)).save(val / f"val_{i}.JPEG")
        lines.append(f"val_{i}.JPEG\t{wnid}\t0\t0\t63\t63")
    (tmp_path / "val" / "val_annotations.txt").write_text("\n".join(lines) + "\n")

    train = TinyImageNet(os.fspath(tmp_path), split="train")
    test = TinyImageNet(os.fspath(tmp_path), split="val")
    assert len(train) == 6 and len(test) == 2
    assert [label for _, label in test.samples] == [0, 1]
