"""Shared pieces of the neural-network experiment (torch is imported lazily)."""

from __future__ import annotations

import numpy as np


def select_device(use_gpu: bool) -> str:
    import torch

    if use_gpu and torch.cuda.is_available():
        return "cuda"
    if use_gpu and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def signed_pid_relrep(values_i: np.ndarray, values_j: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Signed ``h(x) - h(r)`` block used by the pairwise NN experiment (legacy convention)."""
    big = np.concatenate([np.asarray(values_i), np.asarray(values_j)], axis=0)
    ref = big[np.asarray(anchors, dtype=np.int64)]
    return big[:, None] - ref[None, :]
