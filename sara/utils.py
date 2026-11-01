"""Training utilities: mixed-precision context, seeding, metric averaging, JSONL logging and weight averaging."""
import contextlib
import copy
import json
import os
import random
import time

import numpy as np
import torch


def amp_ctx(enabled: bool = True):
    """bfloat16 autocast for forward passes. bfloat16 has the exponent range of fp32 and
    therefore needs no loss scaling, which keeps input gradients in the inner SARA
    search numerically safe.
    """
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def set_seed(seed: int, deterministic: bool = False):
    """Seed Python, NumPy and PyTorch; optionally request deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Deterministic cuDNN algorithms, for exact run-to-run reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # cuDNN autotuning stays off: the challenging and corrected subsets change size
        # at every step, and autotuning each new shape is far slower than the default
        # heuristics. TF32 matrix multiplication is allowed on supported GPUs.
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class AverageMeter:
    """Running weighted average of a scalar."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


class JsonlLogger:
    """Appends one JSON object per line to a log file."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._f = open(path, "a")

    def log(self, **kwargs):
        kwargs["_t"] = time.time()
        self._f.write(json.dumps(kwargs) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def gpu_mem_mb():
    """Peak allocated GPU memory in MiB (0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


class ModelEMA:
    """Exponential moving average of the weights of a model (weight averaging).

    Floating-point parameters and buffers (including the batch-normalization statistics) are averaged with
    decay min(decay, (1 + n) / (10 + n)) after the n-th update, so that the average is not dominated by the
    random initialization early in training; integer buffers are copied.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()
        for name, avg in self.module.state_dict().items():
            new = current[name].detach()
            if avg.dtype.is_floating_point:
                avg.mul_(d).add_(new, alpha=1 - d)
            else:
                avg.copy_(new)
