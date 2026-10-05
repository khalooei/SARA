"""SARA: Self-Adaptive Revisiting Awareness for robust and generalizable classification.

The method revisits misclassified and low-confidence samples during training, moves each
of them to a nearby point that the current model classifies correctly with sufficient
confidence and margin (Confidence Guard loss), and adds the corrected samples to the
mini-batch.
"""
from .confidence_guard import (
    SARAConfig,
    correct_challenging_samples,
    identify_challenging,
    sample_level_cg_loss,
    scale_xi_for_num_classes,
)
from .losses import build_base_loss
from .trainer import sara_training_step

__version__ = "1.0.0"

__all__ = [
    "SARAConfig",
    "build_base_loss",
    "correct_challenging_samples",
    "identify_challenging",
    "sample_level_cg_loss",
    "sara_training_step",
    "scale_xi_for_num_classes",
]
