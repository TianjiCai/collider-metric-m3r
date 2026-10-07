"""Reconstruction fidelity of a single event manifold under M3R (paper Sec. 4.1).

Two measures, both comparing the original metric geometry with the Euclidean
geometry of the relative representation at increasing reference count:

* **JAUC** - area under the mean k-nearest-neighbour overlap (modified Jaccard)
  curve, random baseline 1/2 (Appendix A of the paper);
* **MADO** - modified average decision ordering, the fraction of
  (event pair, probe) triples whose closer/farther ordering is preserved,
  estimated by Monte Carlo over sampled pairs and probes.

The functions are ports of ``intraclass_jaccard_anchors.py`` and
``intraclass_decision_ordering_anchors.py`` with identical RNG conventions, so
the published numbers are reproduced exactly when fed the same distances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import simpson
from scipy.spatial.distance import pdist, squareform

from m3r.relrep import build_relrep, intraclass_anchors, mado_anchors

# --------------------------------------------------------------------------- #
# Modified Jaccard / JAUC
# --------------------------------------------------------------------------- #


def neighbor_grid(n_events: int, step: int) -> np.ndarray:
    """Neighbourhood sizes ``k`` at which the overlap is evaluated."""
    return np.arange(1, n_events - 1, step)


def rank_neighbors(distance_matrix: np.ndarray) -> np.ndarray:
    """Per-row neighbour ranking, excluding the event itself (``argsort``, ties by index)."""
    return np.argsort(distance_matrix, axis=1)[:, 1:]


def jaccard_curve(
    original_rank: np.ndarray,
    relrep_rank: np.ndarray,
    neighbors: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Mean modified Jaccard overlap ``|KNN_k ∩ KNN'_k| / k`` for every ``k`` in ``neighbors``."""
    n = original_rank.shape[0]
    sims = np.empty(len(neighbors))
    for pos, k in enumerate(neighbors):
        total = 0.0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            buf = np.empty((end - start, 2 * k), dtype=original_rank.dtype)
            buf[:, :k] = original_rank[start:end, :k]
            buf[:, k:] = relrep_rank[start:end, :k]
            buf.sort(axis=1)
            total += np.sum(buf[:, :-1] == buf[:, 1:])
        sims[pos] = total / (n * k)
    return sims


def jaccard_auc(curve: np.ndarray, neighbors: np.ndarray, n_events: int) -> float:
    """Simpson-rule area of the overlap curve against the neighbour fraction ``k / N``."""
    return float(simpson(y=curve, x=neighbors / n_events))


@dataclass
class JaucResult:
    anchors: list[int]
    neighbors: np.ndarray
    curves: np.ndarray            # (len(anchors), len(neighbors))
    aucs: np.ndarray              # (len(anchors),)
    mean_jaccard: np.ndarray      # (len(anchors),)
    n_events: int
    seed: int
    anchor_indices: dict[int, np.ndarray] = field(default_factory=dict)


def jauc_sweep(
    distance_matrix: np.ndarray,
    anchor_values: list[int],
    seed: int,
    *,
    neighbors_step: int = 100,
    chunk_size: int = 1024,
    progress: bool = False,
) -> JaucResult:
    """JAUC at every reference count in ``anchor_values`` for one class.

    ``distance_matrix`` must hold distances (not squared distances) if the
    relative representation is to follow the paper's definition.
    """
    D = np.asarray(distance_matrix)
    n = D.shape[0]
    anchor_values = sorted(set(int(a) for a in anchor_values))
    if anchor_values[-1] > n:
        raise ValueError(f"Maximum anchor count {anchor_values[-1]} exceeds available events ({n}).")
    neighbors = neighbor_grid(n, neighbors_step)
    if neighbors.size == 0:
        raise ValueError(f"No neighbour counts for n={n}, step={neighbors_step}.")
    original_rank = rank_neighbors(D)
    curves = np.zeros((len(anchor_values), len(neighbors)))
    anchor_sets: dict[int, np.ndarray] = {}
    iterator = anchor_values
    if progress:
        from tqdm import tqdm
        iterator = tqdm(anchor_values, desc="JAUC anchors", leave=False)
    for pos, k in enumerate(iterator):
        anchors = intraclass_anchors(n, k, seed)
        anchor_sets[k] = anchors
        rel = build_relrep(D, anchors)
        relrep_rank = rank_neighbors(squareform(pdist(rel, metric="euclidean")))
        curves[pos] = jaccard_curve(original_rank, relrep_rank, neighbors, chunk_size)
        del relrep_rank
    aucs = np.array([jaccard_auc(curves[pos], neighbors, n) for pos in range(len(anchor_values))])
    return JaucResult(
        anchors=anchor_values,
        neighbors=neighbors,
        curves=curves,
        aucs=aucs,
        mean_jaccard=curves.mean(axis=1),
        n_events=n,
        seed=seed,
        anchor_indices=anchor_sets,
    )


# --------------------------------------------------------------------------- #
# Modified average decision ordering (MADO)
# --------------------------------------------------------------------------- #


def condensed_to_square_indices(condensed: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Map condensed (``pdist``) indices to ``(i, j)`` with ``i < j``."""
    condensed = np.asarray(condensed, dtype=np.int64)
    i = n - 2 - np.floor(np.sqrt(-8 * condensed + 4 * n * (n - 1) - 7) / 2.0 - 0.5).astype(np.int64)
    j = condensed + i + 1 - n * (n - 1) // 2 + (n - i) * (n - i - 1) // 2
    return i, j


def sample_pairs(n: int, sample_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    total = n * (n - 1) // 2
    condensed = rng.choice(total, size=min(sample_size, total), replace=False)
    return condensed_to_square_indices(condensed, n)


def sample_references(n: int, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if sample_size >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=sample_size, replace=False))


def sampled_decision_ordering(
    original_ref: np.ndarray,
    approx_ref: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    chunk_size: int = 4096,
) -> float:
    """Fraction of (pair, probe) comparisons with the same closer/farther decision (ties count 1/2)."""
    total = 0.0
    count = 0
    for start in range(0, pair_i.size, chunk_size):
        stop = min(start + chunk_size, pair_i.size)
        ic, jc = pair_i[start:stop], pair_j[start:stop]
        product = (original_ref[ic] - original_ref[jc]) * (approx_ref[ic] - approx_ref[jc])
        total += float(np.count_nonzero(product > 0.0))
        total += 0.5 * float(np.count_nonzero(product == 0.0))
        count += int(product.size)
    return total / count


def mado_sweep_sampled(
    distance_matrix: np.ndarray,
    anchor_indices: np.ndarray,
    reference_indices: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    anchor_values: list[int],
    *,
    pair_chunk_size: int = 4096,
    ref_chunk_size: int = 128,
    progress: bool = False,
) -> np.ndarray:
    """MADO for each nested anchor count, accumulating squared relative distances in float32."""
    D = np.asarray(distance_matrix, dtype=np.float32)
    n_refs = reference_indices.size
    original_ref = np.asarray(D[:, reference_indices], dtype=np.float32)
    approx_sq = np.zeros((D.shape[0], n_refs), dtype=np.float32)
    approx_ref = np.empty_like(approx_sq)
    scores = np.empty(len(anchor_values), dtype=np.float64)
    iterator = anchor_values
    if progress:
        from tqdm import tqdm
        iterator = tqdm(anchor_values, desc="MADO anchors", leave=False)
    next_pos = 0
    for pos, k in enumerate(iterator):
        while next_pos < k:
            a = anchor_indices[next_pos]
            anchor_d = D[:, a]
            ref_anchor_d = D[reference_indices, a]
            for rs in range(0, n_refs, ref_chunk_size):
                re = min(rs + ref_chunk_size, n_refs)
                diff = anchor_d[:, None] - ref_anchor_d[rs:re][None, :]
                approx_sq[:, rs:re] += diff * diff
            next_pos += 1
        np.sqrt(approx_sq, out=approx_ref)
        scores[pos] = sampled_decision_ordering(original_ref, approx_ref, pair_i, pair_j, pair_chunk_size)
    return scores


@dataclass
class MadoResult:
    anchors: list[int]
    scores: np.ndarray
    n_events: int
    sample_pairs: int
    sample_references: int
    seed: int
    exact: bool


def mado_sweep(
    distance_matrix: np.ndarray,
    anchor_values: list[int],
    class_seed: int,
    *,
    sample_pairs_n: int = 20_000,
    sample_references_n: int = 128,
    pair_chunk_size: int = 4096,
    ref_chunk_size: int = 128,
    exact: bool = False,
    progress: bool = False,
) -> MadoResult:
    """MADO sweep for one class.

    Seeds follow the published convention: anchors from ``default_rng(class_seed)``,
    pairs from ``class_seed + 10_000`` and probes from ``class_seed + 20_000``,
    where ``class_seed = base_seed + class_index``.
    """
    D = np.asarray(distance_matrix, dtype=np.float32)
    n = D.shape[0]
    anchor_values = sorted(set(int(a) for a in anchor_values))
    if anchor_values[-1] > n:
        raise ValueError(f"Maximum anchor count {anchor_values[-1]} exceeds available events ({n}).")
    anchors = mado_anchors(n, anchor_values[-1], class_seed)
    if exact:
        scores = np.array([exact_mado(D, anchors[:k]) for k in anchor_values])
        used_pairs, used_refs = n * (n - 1) // 2, n
    else:
        pair_i, pair_j = sample_pairs(n, sample_pairs_n, np.random.default_rng(class_seed + 10_000))
        refs = sample_references(n, sample_references_n, np.random.default_rng(class_seed + 20_000))
        scores = mado_sweep_sampled(
            D, anchors, refs, pair_i, pair_j, anchor_values,
            pair_chunk_size=pair_chunk_size, ref_chunk_size=ref_chunk_size, progress=progress,
        )
        used_pairs, used_refs = int(pair_i.size), int(refs.size)
    return MadoResult(anchor_values, scores, n, used_pairs, used_refs, class_seed, exact)


# --------------------------------------------------------------------------- #
# Exact (small-N) reference implementations, kept for tests
# --------------------------------------------------------------------------- #


def strict_jaccard_similarity(D1: np.ndarray, D2: np.ndarray, query: int, k: int) -> float:
    n1 = np.argsort(D1[query])[1:k + 1]
    n2 = np.argsort(D2[query])[1:k + 1]
    return len(np.intersect1d(n1, n2, assume_unique=True)) / k


def average_strict_jaccard(D1: np.ndarray, D2: np.ndarray, k: int) -> float:
    return float(np.mean([strict_jaccard_similarity(D1, D2, q, k) for q in range(D1.shape[0])]))


def decision_ordering(D1: np.ndarray, D2: np.ndarray, a: int, b: int) -> float:
    return float(np.mean(np.heaviside((D1[a] - D1[b]) * (D2[a] - D2[b]), 0.5)))


def decision_ordering_sim(D1: np.ndarray, D2: np.ndarray) -> float:
    """Exact MADO over every unordered pair and every probe (O(N^3))."""
    n = D1.shape[0]
    i_ind, j_ind = np.triu_indices(n, k=1)
    return float(sum(decision_ordering(D1, D2, i, j) for i, j in zip(i_ind, j_ind)) / len(i_ind))


def exact_mado(distance_matrix: np.ndarray, anchor_indices: np.ndarray) -> float:
    rel = np.asarray(distance_matrix[:, anchor_indices], dtype=np.float64)
    return decision_ordering_sim(np.asarray(distance_matrix), squareform(pdist(rel, metric="euclidean")))
