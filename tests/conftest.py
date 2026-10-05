import pytest
import torch
import torch.nn as nn


class TinyNet(nn.Module):
    """Small fully connected classifier on 1x8x8 inputs, used for fast CPU tests."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def tiny_batch():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(16, 1, 8, 8, generator=g)
    y = torch.randint(0, 3, (16,), generator=g)
    return x, y


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return TinyNet(num_classes=3)


@pytest.fixture
def fitted_model(tiny_batch):
    """TinyNet fitted briefly to the test batch, so that challenging samples exist and can be corrected."""
    torch.manual_seed(0)
    model = TinyNet(num_classes=3)
    x, y = tiny_batch
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(8):  # partially fitted: some samples remain challenging but correctable
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return model
