"""Reproducibility helpers — device-agnostic (CPU · CUDA · TPU/XLA)."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs.

    CUDA-specific calls (``torch.cuda.manual_seed_all``,
    ``torch.backends.cudnn.*``) are only made when CUDA is actually available,
    so this function is safe to call on CPU and TPU environments.

    When ``torch_xla`` is installed, the XLA device RNG is also seeded.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── CUDA (only when a GPU is present) ────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── XLA / TPU (only when torch_xla is installed) ─────────────────────────
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import]
        xm.set_rng_state(seed)
    except (ImportError, AttributeError):
        pass  # torch_xla not installed — no-op on CPU/CUDA
