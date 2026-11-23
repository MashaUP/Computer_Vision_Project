"""Seeding for the 3-seed protocol: Python, NumPy, PyTorch (CPU and CUDA) and DataLoader workers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG. ``deterministic=True`` also forces deterministic cuDNN kernels (slower)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - DataLoader signature
    """``worker_init_fn`` for DataLoader: derive each worker's NumPy/random seed from torch's."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
