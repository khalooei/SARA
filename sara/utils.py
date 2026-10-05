"""Training utilities: mixed-precision context, seeding, metric averaging and JSONL logging."""
import contextlib
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
        # Fixed input sizes: let cuDNN select the fastest algorithms, and allow
        # TF32 matrix multiplication on supported GPUs.
        torch.backends.cudnn.benchmark = True
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
