import math

import pytest
import torch
import torch.nn.functional as F

from sara.confidence_guard import (
    SARAConfig,
    cg_damping_weight,
    continuous_confidence_weight,
    correct_challenging_samples,
    identify_challenging_and_confidence,
    margin,
    sample_level_cg_loss,
    scale_xi_for_num_classes,
)


def test_margin_matches_definition():
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    y = torch.tensor([0, 1])
    h = margin(probs, y)
    # h(x, y) = max_{j != y} f_j(x) - f_y(x)
    assert torch.allclose(h, torch.tensor([0.2 - 0.7, 0.6 - 0.3]))


@pytest.mark.parametrize("num_classes, expected", [(10, 0.5), (100, 0.05), (200, 0.025)])
def test_class_count_relative_threshold(num_classes, expected):
    assert scale_xi_for_num_classes(0.5, num_classes) == pytest.approx(expected)


def test_cg_loss_bounds_misclassification():
    # Proposition 2(i): 1[argmax_j f_j(x) != y] <= min{CE / log 2, ReLU(h + tau) / tau}.
    torch.manual_seed(0)
    tau = 0.05
    probs = F.softmax(torch.randn(256, 5) * 3, dim=1)
    y = torch.randint(0, 5, (256,))
    wrong = (probs.argmax(dim=1) != y).float()
    ce = -torch.log(probs.gather(1, y.view(-1, 1)).squeeze(1))
    hinge = F.relu(margin(probs, y) + tau)
    assert torch.all(wrong <= ce / math.log(2) + 1e-6)
    assert torch.all(wrong <= hinge / tau + 1e-6)
    assert torch.all(sample_level_cg_loss(probs, y, tau, 1.0) >= 0)


def test_hinge_vanishes_once_margin_condition_holds():
    probs = torch.tensor([[0.9, 0.05, 0.05]])
    y = torch.tensor([0])
    loss = sample_level_cg_loss(probs, y, tau=0.05, lambda_margin=1.0)
    assert loss.item() == pytest.approx(-math.log(0.9), abs=1e-6)


def test_identify_challenging(tiny_model, tiny_batch):
    x, y = tiny_batch
    mask, py = identify_challenging_and_confidence(tiny_model, x, y, xi=0.5)
    with torch.no_grad():
        probs = F.softmax(tiny_model(x), dim=1)
    expected = (probs.argmax(dim=1) != y) | (probs.gather(1, y.view(-1, 1)).squeeze(1) < 0.5)
    assert torch.equal(mask, expected)
    assert py.shape == y.shape
    assert tiny_model.training


def test_correction_respects_constraints_and_acceptance(fitted_model, tiny_batch):
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.5, tau=0.05, xi=0.5, kappa=50, alpha=0.5)
    fitted_model.train()
    x_star, y_star, stats = correct_challenging_samples(fitted_model, x, y, cfg)

    assert fitted_model.training, "the training mode must be restored after the search"
    assert stats["n_corrected"] == x_star.shape[0] > 0
    assert not x_star.requires_grad

    # Every corrected sample lies in the admissible set: the eps-ball and the pixel range.
    accepted = []
    for xs, ys in zip(x_star, y_star):
        dist = (x - xs).flatten(1).abs().amax(dim=1)
        idx = torch.nonzero((dist <= cfg.epsilon + 1e-6) & (y == ys)).flatten()
        assert idx.numel() > 0
        accepted.append(idx[0])
    assert torch.all((x_star >= 0) & (x_star <= 1))

    # ... and satisfies the acceptance criterion of Eq. 7 under the current parameters.
    fitted_model.eval()
    with torch.no_grad():
        probs = F.softmax(fitted_model(x_star), dim=1)
    assert torch.all(probs.gather(1, y_star.view(-1, 1)).squeeze(1) >= cfg.xi - 1e-6)
    assert torch.all(margin(probs, y_star) <= -cfg.tau + 1e-6)


def test_correction_budget(tiny_model, tiny_batch):
    x, y = tiny_batch
    cfg = SARAConfig(epsilon=0.3, kappa=3, max_batch_correct=4)
    _, _, stats = correct_challenging_samples(tiny_model, x, y, cfg)
    assert stats["n_challenging"] == x.shape[0]
    assert stats["n_processed"] == 4


def test_empty_challenging_set(tiny_model):
    x = torch.zeros(0, 1, 8, 8)
    y = torch.zeros(0, dtype=torch.long)
    x_star, y_star, stats = correct_challenging_samples(tiny_model, x, y, SARAConfig())
    assert x_star.shape[0] == y_star.shape[0] == 0
    assert stats["success_rate"] == 0.0


def test_default_step_size():
    assert SARAConfig(epsilon=0.2, kappa=10).step_size() == pytest.approx(0.04)
    assert SARAConfig(epsilon=0.2, kappa=10, alpha=0.01).step_size() == pytest.approx(0.01)


def test_reweighting_and_damping():
    assert continuous_confidence_weight(torch.rand(8), 0.5, 0.0) is None
    w = continuous_confidence_weight(torch.tensor([0.1, 0.4, 0.9, 1.0]), 0.5, 2.0)
    assert w.mean().item() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2] == w[3]
    assert cg_damping_weight(0.25) == pytest.approx(0.75)
    assert cg_damping_weight(1.5) == 0.0
