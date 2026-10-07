"""Statistical structure of M3R manifolds (paper Sec. 4.2): PCA alignment and variance transfer."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from m3r.blocks import BlockStore
from m3r.classes import KIN_METRICS
from m3r.relrep import build_relrep, intraclass_anchors

METRIC_PAIRS: tuple[tuple[str, str], ...] = (("w2", "sw"), ("w2", "ps"), ("sw", "ps"))


def class_relreps(
    store: BlockStore,
    class_idx: int,
    anchors: Sequence[int],
    metrics: Sequence[str] = KIN_METRICS,
    *,
    dtype: type = np.float64,
) -> dict[str, np.ndarray]:
    """Intraclass M3R coordinates ``D[:, anchors]`` (distances) for every metric."""
    out: dict[str, np.ndarray] = {}
    for metric in metrics:
        block = store.load(metric, class_idx, class_idx, as_distance=True)
        out[metric] = build_relrep(block, anchors, dtype=dtype)
        del block
    return out


def structure_anchors(n_events: int, n_anchors: int, seed: int) -> np.ndarray:
    """The shared reference set of Figs. 4-5 (``RandomState(seed).choice(n, k)``)."""
    return intraclass_anchors(n_events, n_anchors, seed)


def raw_centered_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, components)`` of the raw-centred covariance, largest first."""
    centered = X - X.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    return singular_values ** 2 / (centered.shape[0] - 1), components


def pc_alignment(components_a: np.ndarray, components_b: np.ndarray, n_pcs: int = 5) -> np.ndarray:
    """``|v_i^(a) . v_i^(b)|`` for the leading ``n_pcs`` principal directions (AbsCosSim)."""
    signed = components_a[:n_pcs] @ components_b[:n_pcs].T
    return np.abs(np.diag(signed))


def covariance(X: np.ndarray) -> np.ndarray:
    centered = X - X.mean(axis=0, keepdims=True)
    return centered.T @ centered / (centered.shape[0] - 1)


def pc1_transfer(pc1_source: np.ndarray, cov_target: np.ndarray) -> float:
    """Fraction of target variance explained by the source PC1 direction (eq. for EV in the paper)."""
    return float(pc1_source @ cov_target @ pc1_source / np.trace(cov_target))


def alignment_table(relreps: Mapping[str, np.ndarray], n_pcs: int = 5) -> dict[tuple[str, str], np.ndarray]:
    comps = {m: raw_centered_pca(X)[1] for m, X in relreps.items()}
    return {(a, b): pc_alignment(comps[a], comps[b], n_pcs) for a, b in METRIC_PAIRS if a in comps and b in comps}


def transfer_table(relreps: Mapping[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """All directed PC1 transfers ``EV(source^(1) -> target)`` including self-transfers."""
    pc1 = {m: raw_centered_pca(X)[1][0] for m, X in relreps.items()}
    cov = {m: covariance(X) for m, X in relreps.items()}
    return {(s, t): pc1_transfer(pc1[s], cov[t]) for s in relreps for t in relreps}


def bootstrap_pca(
    relreps: Mapping[str, np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
    n_alignment_pcs: int = 5,
) -> dict:
    """Bootstrap stability of the spectra, directions and cross-metric alignment."""
    metrics = list(relreps)
    n_events = next(iter(relreps.values())).shape[0]
    full = {m: raw_centered_pca(X) for m, X in relreps.items()}
    n_pcs = next(iter(relreps.values())).shape[1]
    boot_ratio = {m: np.empty((n_bootstrap, n_pcs)) for m in metrics}
    boot_self_cos = {m: np.empty((n_bootstrap, n_pcs)) for m in metrics}
    pairs = [(a, b) for a, b in METRIC_PAIRS if a in relreps and b in relreps]
    boot_pair_cos = {pair: np.empty((n_bootstrap, n_alignment_pcs)) for pair in pairs}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_events, size=n_events)
        comps = {}
        for m in metrics:
            eig, comp = raw_centered_pca(relreps[m][idx])
            boot_ratio[m][b] = eig / eig.sum()
            boot_self_cos[m][b] = np.abs(np.sum(comp * full[m][1], axis=1))
            comps[m] = comp
        for a, c in pairs:
            boot_pair_cos[(a, c)][b] = np.abs(
                np.sum(comps[a][:n_alignment_pcs] * comps[c][:n_alignment_pcs], axis=1)
            )
    return {
        "ratio": {m: full[m][0] / full[m][0].sum() for m in metrics},
        "components": {m: full[m][1] for m in metrics},
        "boot_ratio": boot_ratio,
        "boot_self_cos": boot_self_cos,
        "pair_cos": {(a, c): pc_alignment(full[a][1], full[c][1], n_alignment_pcs) for a, c in pairs},
        "boot_pair_cos": boot_pair_cos,
    }
