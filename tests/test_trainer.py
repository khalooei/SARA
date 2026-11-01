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


def test_correction_memory_ring_buffer():
    from sara.confidence_guard import CorrectionMemory
    mem = CorrectionMemory(5)
    assert mem.sample(3) == (None, None)
    mem.push(torch.arange(4.0).view(4, 1), torch.arange(4))
    mem.push(torch.arange(4.0, 7.0).view(3, 1), torch.arange(4, 7))
    assert len(mem) == 5
    assert set(mem.y.tolist()) == {2, 3, 4, 5, 6}  # the two oldest entries were overwritten
    xs, ys = mem.sample(10)
    assert xs.shape[0] == 5 and torch.equal(xs.view(-1).long(), ys)


def test_revisiting_is_zero_inside_acceptance_region(fitted_model, tiny_batch):
    # Eq. 13: a stored correction still inside the acceptance region contributes no loss.
    from sara.confidence_guard import CorrectionMemory
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, tau=0.05, xi=0.5, kappa=50, alpha=0.5, cg_confidence_cap=True,
                     revisit_memory=256, revisit_batch=256)
    mem = CorrectionMemory(cfg.revisit_memory)
    first = _step(fitted_model, x, y, cfg, memory=mem)  # fills the memory; nothing to revisit yet
    assert first["n_revisited"] == 0 and len(mem) == first["n_corrected"] > 0
    second = _step(fitted_model, x, y, cfg, memory=mem)  # lr = 0: stored samples are still accepted
    assert second["n_revisited"] > 0
    assert second["revisit_violation_rate"] == 0.0
    assert second["loss_revisit"] == pytest.approx(0.0, abs=1e-5)


def test_robust_correction_extends_base_batch(fitted_model, tiny_batch):
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, kappa=20, alpha=0.5, robust_correction=True)
    stats = _step(fitted_model, x, y, cfg, base="pgd")
    assert stats["n_corrected"] > 0
    assert math.isfinite(stats["loss_total"])


@pytest.mark.parametrize("base", ["pgd", "trades", "mart", "fat"])
def test_alignment_term(fitted_model, tiny_batch, base):
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, kappa=20, alpha=0.5, cg_confidence_cap=True, align_weight=1.0)
    stats = _step(fitted_model, x, y, cfg, base=base)
    assert stats["n_corrected"] > 0
    assert stats["loss_align"] >= 0.0
    assert stats["loss_total"] == pytest.approx(
        stats["loss_base"] + stats["loss_cg"] + stats["loss_revisit"] + stats["loss_align"], rel=1e-5)


@pytest.mark.parametrize("base", ["ce", "pgd"])
def test_natural_alignment_term(fitted_model, tiny_batch, base):
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, kappa=20, alpha=0.5, cg_confidence_cap=True, align_natural_weight=1.0)
    stats = _step(fitted_model, x, y, cfg, base=base)
    assert stats["n_corrected"] > 0
    assert stats["loss_align"] > 0.0


def test_natural_ce_term(fitted_model, tiny_batch):
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, kappa=20, alpha=0.5, cg_confidence_cap=True, nat_weight=1.0)
    stats = _step(fitted_model, x, y, cfg, base="pgd")
    assert stats["loss_nat"] > 0.0
    assert stats["loss_total"] == pytest.approx(
        stats["loss_base"] + stats["loss_cg"] + stats["loss_revisit"] + stats["loss_align"] + stats["loss_nat"],
        rel=1e-5)


def test_alignment_term_in_evaluation_mode(fitted_model, tiny_batch):
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, kappa=20, alpha=0.5, cg_confidence_cap=True, align_weight=1.0,
                     align_eval_mode=True)
    stats = _step(fitted_model, x, y, cfg, base="pgd")
    assert stats["n_corrected"] > 0
    assert stats["loss_align"] >= 0.0
    assert fitted_model.training


def test_capped_cg_term_adds_no_gradient(fitted_model, tiny_batch):
    # Remark 1: with the cap, accepted samples contribute a constant, so the parameter gradient of a SARA step
    # without alignment equals that of the base loss alone.
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, tau=0.05, xi=0.5, kappa=50, alpha=0.5, cg_confidence_cap=True)
    stats = _step(fitted_model, x, y, cfg)
    assert stats["n_corrected"] > 0
    g_sara = [p.grad.clone() for p in fitted_model.parameters() if p.grad is not None]
    _step(fitted_model, x, y, cfg, use_sara=False)
    g_base = [p.grad.clone() for p in fitted_model.parameters() if p.grad is not None]
    assert len(g_sara) == len(g_base)
    for a, b in zip(g_sara, g_base):
        assert torch.allclose(a, b, atol=1e-6)


def test_capped_cg_term_ignores_rescoring_drift(fitted_model, tiny_batch, monkeypatch):
    # On a GPU, re-scoring accepted samples in another forward pass (bfloat16, another batch composition) can
    # move them just outside the acceptance region. The capped term must still contribute no gradient.
    import sara.trainer as trainer
    original = trainer._eval_mode_probs

    def drifted(model, x):
        p = original(model, x)
        y_hat = p.argmax(1, keepdim=True)
        return (p * 0.8).scatter_add(1, y_hat, torch.zeros_like(y_hat, dtype=p.dtype)) + 0.2 / p.shape[1]

    monkeypatch.setattr(trainer, "_eval_mode_probs", drifted)
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, tau=0.05, xi=0.5, kappa=50, alpha=0.5, cg_confidence_cap=True)
    stats = _step(fitted_model, x, y, cfg)
    assert stats["n_corrected"] > 0
    g_sara = [p.grad.clone() for p in fitted_model.parameters() if p.grad is not None]
    _step(fitted_model, x, y, cfg, use_sara=False)
    g_base = [p.grad.clone() for p in fitted_model.parameters() if p.grad is not None]
    for a, b in zip(g_sara, g_base):
        assert torch.allclose(a, b, atol=1e-6)


def test_model_ema_averages_parameters_and_batchnorm_statistics():
    from sara.utils import ModelEMA
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    ema = ModelEMA(model, decay=0.5)
    w0 = model[0].weight.detach().clone()
    with torch.no_grad():
        model[0].weight.add_(1.0)
        model[1].running_mean.add_(2.0)
    ema.update(model)  # first update: d = min(0.5, 2 / 11)
    d = 2 / 11
    assert torch.allclose(ema.module[0].weight, d * w0 + (1 - d) * (w0 + 1.0), atol=1e-6)
    assert torch.allclose(ema.module[1].running_mean, torch.full((4,), (1 - d) * 2.0), atol=1e-6)
    assert ema.module[1].num_batches_tracked == model[1].num_batches_tracked
    assert not any(p.requires_grad for p in ema.module.parameters())
