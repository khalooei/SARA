import math

import pytest
import torch

from sara.attacks import pgd_attack
from sara.baselines import fat_training_step, gairat_training_step, ohem_training_step
from sara.confidence_guard import SARAConfig
from sara.evaluate import evaluate
from sara.lipschitz import local_lipschitz_estimate


def test_pgd_stays_in_threat_model(tiny_model, tiny_batch):
    x, y = tiny_batch
    eps = 0.1
    x_adv = pgd_attack(tiny_model, x, y, eps, alpha=eps / 4, steps=5)
    assert (x_adv - x).abs().max() <= eps + 1e-6
    assert torch.all((x_adv >= 0) & (x_adv <= 1))


@pytest.mark.parametrize("step", [ohem_training_step, fat_training_step, gairat_training_step])
def test_baseline_steps(tiny_model, tiny_batch, step):
    x, y = tiny_batch
    opt = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
    stats = step(tiny_model, opt, x, y, None, SARAConfig(epsilon=0.1))
    assert math.isfinite(stats["loss_total"])


def test_evaluate(tiny_model, tiny_batch):
    loader = [tiny_batch]
    out = evaluate(tiny_model, loader, torch.device("cpu"), epsilon=0.1, pgd_steps=3, run_apgd=False)
    assert out["n"] == tiny_batch[0].shape[0]
    assert 0.0 <= out["pgd_acc"] <= out["clean_acc"] + 1e-9 <= 100.0 + 1e-9


def test_local_lipschitz_estimate_is_nonnegative(tiny_model, tiny_batch):
    assert local_lipschitz_estimate(tiny_model, tiny_batch[0], radius=0.05, n_dirs=3) >= 0.0
