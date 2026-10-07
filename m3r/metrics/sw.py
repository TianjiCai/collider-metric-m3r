"""Spectral 2-Wasserstein (SW2) distance via the SPECTER package (JAX).

The heavy imports are deferred to ``make_specter`` so that ``import m3r``
works without JAX installed. ``specterpy`` 1.1.0 still imports the removed
``jax.tree_map``; ``make_specter`` installs a one-line shim before importing it.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import numpy as np


def build_sw_events(kinematic_events: np.ndarray) -> np.ndarray:
    sw_events = np.array(kinematic_events[..., :3], dtype=np.float32, copy=True)
    pt_sums = sw_events[..., 0].sum(axis=1, keepdims=True)
    np.divide(
        sw_events[..., 0],
        pt_sums,
        out=sw_events[..., 0],
        where=pt_sums > 0.0,
    )
    return sw_events



def spectral_measure(jet: np.ndarray, omega_power: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Discrete spectral function ``s(omega) = sum_ij z_i z_j delta(omega - dR_ij)`` of one jet.

    Returns ``(omega, weight)`` with one entry per constituent pair (self pairs
    at ``omega = 0``). ``jet`` columns are ``(pt, y, phi, ...)``; zero-``pt``
    rows are dropped.
    """
    jet = np.asarray(jet, dtype=np.float64)
    jet = jet[jet[:, 0] > 0]
    z = jet[:, 0] / jet[:, 0].sum()
    y, phi = jet[:, 1], jet[:, 2]
    dphi = np.abs(phi[:, None] - phi[None, :])
    dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
    dr = np.sqrt((y[:, None] - y[None, :]) ** 2 + dphi ** 2)
    return dr.ravel() ** omega_power, np.outer(z, z).ravel()


def sw2_exact(jet_a: np.ndarray, jet_b: np.ndarray) -> float:
    """Reference SW2 between two jets from 1-D optimal transport of their spectral functions.

    Independent of SPECTER; useful for tests and for checking block conventions.
    """
    import ot

    xa, wa = spectral_measure(jet_a)
    xb, wb = spectral_measure(jet_b)
    return float(np.sqrt(max(ot.emd2_1d(xa, xb, wa, wb, metric="sqeuclidean"), 0.0)))


def force_jax_cpu() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


def make_specter(*, force_cpu: bool = True, compile: bool = True, cache_dir: str | Path | None = None) -> Any:
    """Instantiate ``pyspecter.SPECTER`` with the compatibility shim applied."""
    if force_cpu:
        force_jax_cpu()
    if cache_dir is not None:
        os.environ.setdefault("SPECTER_CACHE_DIR", str(cache_dir))
    import jax

    if not hasattr(jax, "tree_map"):  # removed in JAX 0.6; specterpy 1.1.0 still imports it
        jax.tree_map = jax.tree.map  # type: ignore[attr-defined]
    try:
        from pyspecter.SPECTER import SPECTER
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SW2 distances need the `pyspecter` package: pip install 'm3r[sw]' "
            "(pins jax<0.6 and specterpy)."
        ) from exc
    return SPECTER(compile=compile)


def spectral_emd_batch(specter: Any, events_a: np.ndarray, events_b: np.ndarray) -> np.ndarray:
    """``SW2**2`` (SPECTER's ``ds2``) for every row pair of two aligned event batches."""
    return np.asarray(specter.spectralEMD(events_a, events_b), dtype=np.float64)


def sw2_matrix(
    specter: Any,
    events_a: np.ndarray,
    events_b: np.ndarray | None = None,
    *,
    batch_size: int = 4096,
    squared: bool = False,
    progress: bool = False,
) -> np.ndarray:
    """Pairwise SW2 matrix between two event sets (or within one set when ``events_b`` is None).

    Streams index pairs in batches so a 10k x 10k job never materialises the
    full index grid. Returns distances unless ``squared=True``.
    """
    symmetric = events_b is None
    if symmetric:
        events_b = events_a
    n_a, n_b = len(events_a), len(events_b)
    out = np.zeros((n_a, n_b), dtype=np.float64)
    if symmetric:
        total = n_a * (n_a + 1) // 2
        pairs = itertools.combinations_with_replacement(range(n_a), 2)
    else:
        total = n_a * n_b
        pairs = itertools.product(range(n_a), range(n_b))
    iterator = range(0, total, batch_size)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="SW2 batches", leave=False)
    for _ in iterator:
        chunk = list(itertools.islice(pairs, batch_size))
        if not chunk:
            break
        ii = np.fromiter((p[0] for p in chunk), dtype=np.int64, count=len(chunk))
        jj = np.fromiter((p[1] for p in chunk), dtype=np.int64, count=len(chunk))
        ds2 = spectral_emd_batch(specter, events_a[ii], events_b[jj])
        values = ds2 if squared else np.sqrt(np.clip(ds2, 0.0, None))
        out[ii, jj] = values
        if symmetric:
            out[jj, ii] = values
    return out

