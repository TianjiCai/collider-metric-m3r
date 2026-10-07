"""Multi-reference relative representations (M3R).

An event ``x`` is mapped to ``(d(x, r_1), ..., d(x, r_N))`` for a reference
set sampled from the dataset. ``sample_anchors`` reproduces the random draws
used for every published result bit-for-bit (legacy ``np.random.seed`` +
per-class ``np.random.choice`` semantics), so re-running a stage yields the
same references.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from m3r.blocks import BlockPair, BlockStore, to_distance


def pair_labels(n0: int, n1: int | None = None) -> np.ndarray:
    """Binary labels for a class pair: ``n0`` zeros followed by ``n1`` ones."""
    return np.repeat([0, 1], [n0, n0 if n1 is None else n1])


def sample_anchors(
    labels: np.ndarray,
    anchors_per_class: int,
    seed: int,
    n_classes: int | None = None,
) -> np.ndarray:
    """Draw ``anchors_per_class`` reference events from every class.

    Bit-exact port of the legacy ``relative_rep`` anchor draw: a fresh
    ``RandomState(seed)`` and one ``choice(bucket, k, replace=False)`` per
    class, in class order. Because ``RandomState.choice(..., replace=False)``
    is ``permutation(bucket)[:k]``, the anchors for ``k`` references are a
    prefix of those for any larger ``k``.
    """
    labels = np.asarray(labels)
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    rng = np.random.RandomState(seed)
    anchors = np.zeros(anchors_per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        bucket = np.where(labels == c)[0]
        anchors[c * anchors_per_class:(c + 1) * anchors_per_class] = rng.choice(
            bucket, size=anchors_per_class, replace=False
        )
    return anchors


def intraclass_anchors(n_events: int, anchors: int, seed: int) -> np.ndarray:
    """Anchors for a single-class (intraclass) representation; same RNG as ``sample_anchors``."""
    return sample_anchors(np.zeros(n_events, dtype=np.int64), anchors, seed, n_classes=1)


def mado_anchors(n_events: int, max_anchors: int, seed: int) -> np.ndarray:
    """Nested anchor set used by the MADO sweep (``default_rng`` convention)."""
    return np.random.default_rng(seed).choice(n_events, size=max_anchors, replace=False)


def build_relrep(distance_matrix: np.ndarray, anchors: Sequence[int], dtype: type = np.float64) -> np.ndarray:
    """``D[:, anchors]`` as a contiguous float array (the M3R coordinates)."""
    anchors = np.asarray(anchors, dtype=np.int64)
    return np.ascontiguousarray(np.asarray(distance_matrix)[:, anchors], dtype=dtype)


def pair_relrep_from_blocks(
    pair: BlockPair,
    anchors: Sequence[int],
    *,
    as_distance: bool = True,
) -> np.ndarray:
    """M3R coordinates of every event of a class pair without materialising the full matrix."""
    anchors = np.asarray(anchors, dtype=np.int64)
    out = np.empty((pair.n, anchors.size), dtype=np.float64)
    for col, anchor in enumerate(anchors):
        out[:, col] = pair.column(int(anchor), as_distance=False)
    if as_distance:
        to_distance(out, pair.stored_power, out=out)
    return out


def pair_relrep(
    store: BlockStore,
    metric: str,
    i: int,
    j: int,
    anchors_per_class: int,
    seed: int,
    *,
    as_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, anchors)`` for class pair ``(i, j)``.

    ``X`` has shape ``(n_i + n_j, 2 * anchors_per_class)``; anchors are drawn
    with ``sample_anchors`` on the pair labels (first class first).
    """
    pair = store.pair(metric, i, j)
    labels = pair.labels()
    anchors = sample_anchors(labels, anchors_per_class, seed, n_classes=2)
    X = pair_relrep_from_blocks(pair, anchors, as_distance=as_distance)
    return X, labels, anchors


def pid_values(part_ids: np.ndarray) -> np.ndarray:
    """Event-level particle-content scalar: the sum over constituents (multiplicity or total charge)."""
    return np.sum(np.asarray(part_ids), axis=1)


def pid_relrep(values_i: np.ndarray, values_j: np.ndarray, anchors: Sequence[int]) -> np.ndarray:
    """Non-kinematic M3R block ``|h(x) - h(r)|`` for one PID feature.

    ``values_*`` are per-event scalars (``pid_values`` of the stored per-particle
    arrays); ``anchors`` index the concatenated pair. Matches the legacy
    ``compute_part_ids_rel_rep`` (same anchors as the kinematic blocks).
    """
    big = np.concatenate([np.asarray(values_i), np.asarray(values_j)], axis=0)
    ref = big[np.asarray(anchors, dtype=np.int64)]
    return np.abs(big[:, None] - ref[None, :])
