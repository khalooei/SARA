import math

import pytest
import torch

from sara.confidence_guard import SARAConfig
from sara.losses import build_base_loss
from sara.trainer import sara_training_step


def _step(model, x, y, cfg, base="ce", use_sara=True, **kwargs):
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr = 0 keeps the parameters fixed
    kw = {} if base == "ce" else {"pgd_steps": 2}
    return sara_training_step(model, opt, x, y, build_base_loss(base), cfg, base_loss_kwargs=kw,
                              use_sara=use_sara, **kwargs)


def test_training_step_reports_statistics(fitted_model, tiny_batch):
    x, y = tiny_batch
    stats = _step(fitted_model, x, y, SARAConfig(epsilon=0.5, kappa=20, alpha=0.5))
    for key in ("loss_base", "loss_cg", "loss_total", "n_challenging", "n_corrected", "success_rate"):
        assert key in stats
    assert stats["loss_total"] == pytest.approx(stats["loss_base"] + stats["loss_cg"], rel=1e-5)
    assert stats["batch_size_effective"] == x.shape[0] + stats["n_corrected"]


def test_without_sara_reduces_to_base_loss(tiny_model, tiny_batch):
    x, y = tiny_batch
    stats = _step(tiny_model, x, y, SARAConfig(), use_sara=False)
    assert stats["loss_cg"] == 0.0
    assert stats["n_corrected"] == 0


def test_warmup_disables_correction(tiny_model, tiny_batch):
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, kappa=20, alpha=0.5, sara_warmup_epochs=2)
    assert _step(tiny_model, x, y, cfg, epoch=1)["loss_cg"] == 0.0


def test_confidence_cap_makes_cg_term_constant(fitted_model, tiny_batch):
    # Remark 1: with the cap, an accepted sample scored under the parameters that accepted it
    # contributes the constant -log(xi), and the hinge term is zero.
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, tau=0.05, xi=0.5, kappa=50, alpha=0.5, cg_confidence_cap=True)
    stats = _step(fitted_model, x, y, cfg)
    assert stats["n_corrected"] > 0
    assert stats["loss_cg"] == pytest.approx(-math.log(cfg.xi), abs=1e-5)


@pytest.mark.parametrize("base", ["ce", "pgd", "trades", "mart"])
def test_all_base_losses(tiny_model, tiny_batch, base):
    x, y = tiny_batch
    stats = _step(tiny_model, x, y, SARAConfig(epsilon=0.1, kappa=3), base=base)
    assert math.isfinite(stats["loss_total"])


def test_gradient_norm_diagnostics(tiny_model, tiny_batch):
    x, y = tiny_batch
    stats = _step(tiny_model, x, y, SARAConfig(epsilon=0.5, kappa=20, alpha=0.5), diag_grad_norms=True)
    assert stats["grad_norm_base"] > 0
