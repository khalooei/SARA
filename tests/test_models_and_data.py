import os

import pytest
import torch
from PIL import Image

from sara.data import DEFAULT_EPSILON, NUM_CLASSES, GPUTensorLoader
from sara.models import build_model
from sara.tiny_imagenet import TinyImageNet


@pytest.mark.parametrize(
    "model, dataset, shape",
    [
        ("lenet", "mnist", (2, 1, 28, 28)),
        ("resnet18", "cifar10", (2, 3, 32, 32)),
        ("resnet18", "cifar100", (2, 3, 32, 32)),
        ("resnet18", "tinyimagenet", (2, 3, 64, 64)),
        ("resnet18-cifar", "cifar10", (2, 3, 32, 32)),
        ("resnet18-cifar", "svhn", (2, 3, 32, 32)),
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


def test_gpu_tensor_loader_crop_and_batches():
    import torch.nn.functional as F

    from sara.data import GPUTensorLoader
    x = torch.randint(0, 256, (10, 3, 8, 8), dtype=torch.uint8)
    y = torch.arange(10)
    loader = GPUTensorLoader(x, y, batch_size=4, shuffle=True, drop_last=True, crop_padding=2, device="cpu")
    assert len(loader) == 2
    seen = []
    for xb, yb in loader:
        assert xb.shape == (4, 3, 8, 8) and xb.max() <= 1.0
        padded = F.pad(x[yb].float() / 255, (2, 2, 2, 2))
        for i in range(4):
            crops = [padded[i, :, a:a + 8, b:b + 8] for a in range(5) for b in range(5)]
            assert any(torch.equal(c, xb[i]) or torch.equal(c.flip(2), xb[i]) for c in crops)
        seen += yb.tolist()
    assert len(set(seen)) == 8


def test_seeded_loader_order_is_independent_of_other_random_draws():
    x = torch.randint(0, 256, (64, 3, 8, 8), dtype=torch.uint8)
    y = torch.arange(64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = GPUTensorLoader(x, y, 16, shuffle=True, drop_last=True, crop_padding=2, device=device, seed=3)
    b = GPUTensorLoader(x, y, 16, shuffle=True, drop_last=True, crop_padding=2, device=device, seed=3)
    batches_a = [(xb.clone(), yb.clone()) for xb, yb in a]
    batches_b = []
    for xb, yb in b:
        torch.rand(1000, device=device)  # random numbers drawn by the training step in between
        batches_b.append((xb, yb))
    for (xa, ya), (xb, yb) in zip(batches_a, batches_b):
        assert torch.equal(ya, yb) and torch.equal(xa, xb)
