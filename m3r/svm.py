"""Support-vector-machine probes of inter-manifold decision boundaries (paper Sec. 5).

* ``relrep_svm_auc`` - RBF SVC on M3R coordinates, fixed hyperparameters (Fig. 6 curves);
* ``abs_svm_grid`` - RBF SVC in the *original* metric space through a precomputed kernel
  built from the stored block values, with validation-selected ``C``/``gamma`` (Fig. 6 "Abs");
* ``relrep_svm_grid`` - RBF SVC on M3R coordinates with an optional validation grid and
  support-vector bookkeeping (Fig. 7).

All splits are ports of the published code paths and reproduce them exactly.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from m3r.blocks import BlockPair


@dataclass(frozen=True)
class Splits:
    train: np.ndarray
    val: np.ndarray
    trainval: np.ndarray
    test: np.ndarray


def stratified_splits(labels: np.ndarray, seed: int, test_fraction: float = 0.2,
                      validation_fraction: float = 0.2) -> Splits:
    """Stratified train / validation / test split (64 / 16 / 20 with the defaults)."""
    all_idx = np.arange(labels.size)
    trainval, test = train_test_split(all_idx, test_size=test_fraction, stratify=labels, random_state=seed)
    train, val = train_test_split(trainval, test_size=validation_fraction, stratify=labels[trainval], random_state=seed)
    return Splits(train=train, val=val, trainval=trainval, test=test)


def permutation_split(n: int, seed: int, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Unstratified permutation split used by the Fig. 6 relative-space sweep."""
    indices = np.random.default_rng(seed).permutation(n)
    n_train = int(train_fraction * n)
    return indices[:n_train], indices[n_train:]


def relrep_svm_auc(X: np.ndarray, y: np.ndarray, seed: int, *, C: float = 1.0, gamma: float | str = "scale",
                   kernel: str = "rbf", train_fraction: float = 0.8) -> float:
    """Test ROC AUC of an SVC trained on M3R coordinates (port of ``RelRepPairwiseSVM.svm_with_features``)."""
    train_idx, test_idx = permutation_split(len(X), seed, train_fraction)
    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X[train_idx], y[train_idx])
    return float(roc_auc_score(y[test_idx], clf.decision_function(X[test_idx])))


def kernel_from_stored(S: np.ndarray, gamma: float, normalization: float, stored_power: int,
                       *, symmetric: bool = False) -> np.ndarray:
    """RBF kernel ``exp(-gamma (d / d_med)^2)`` from a slice of *stored* block values.

    With ``stored_power == 2`` the stored values are already ``d**2`` so no squaring is
    applied; the normalisation is then the median of the stored (squared) values, as in
    the published runs.
    """
    if normalization <= 0:
        raise ValueError("Kernel normalization must be positive.")
    K = np.array(S, dtype=np.float64, copy=True)
    K *= 1.0 / normalization
    if stored_power == 1:
        K = K ** 2
    K *= -gamma
    np.exp(K, out=K)
    if symmetric:
        np.fill_diagonal(K, 1.0)
    return K


@dataclass
class SvmResult:
    test_auc: float
    best_C: float
    best_gamma: float | str
    best_val_auc: float
    sv_count: int
    sv_fraction: float
    n_fit: int
    test_scores: np.ndarray | None = None
    test_labels: np.ndarray | None = None
    val_grid: dict = field(default_factory=dict)


def abs_svm_grid(pair: BlockPair, labels: np.ndarray, splits: Splits, gamma_grid: Sequence[float],
                 C_grid: Sequence[float], normalization: float, *, keep_scores: bool = False) -> SvmResult:
    """Precomputed-kernel SVC in the original metric space with validation-selected hyperparameters.

    Port of ``AbsPairwiseSVM.svm_with_precomputed_kernel``: scan ``gamma`` then ``C``,
    keep the first strictly-best validation AUC, refit on train+val, report the test AUC
    and the support-vector fraction of the refit model.
    """
    y_train, y_val = labels[splits.train], labels[splits.val]
    y_trainval, y_test = labels[splits.trainval], labels[splits.test]
    best_val_auc, best_gamma, best_C = -np.inf, None, None
    val_grid: dict = {}
    for gamma in gamma_grid:
        K_train = kernel_from_stored(pair.slice(splits.train, splits.train), gamma, normalization,
                                     pair.stored_power, symmetric=True)
        K_val = kernel_from_stored(pair.slice(splits.val, splits.train), gamma, normalization, pair.stored_power)
        for C in C_grid:
            svc = SVC(kernel="precomputed", C=C)
            svc.fit(K_train, y_train)
            val_auc = float(roc_auc_score(y_val, svc.decision_function(K_val)))
            val_grid[(float(gamma), float(C))] = val_auc
            if val_auc > best_val_auc:
                best_val_auc, best_gamma, best_C = val_auc, gamma, C
        del K_train, K_val, svc
        gc.collect()
    K_trainval = kernel_from_stored(pair.slice(splits.trainval, splits.trainval), best_gamma, normalization,
                                    pair.stored_power, symmetric=True)
    K_test = kernel_from_stored(pair.slice(splits.test, splits.trainval), best_gamma, normalization, pair.stored_power)
    final = SVC(kernel="precomputed", C=best_C)
    final.fit(K_trainval, y_trainval)
    test_scores = final.decision_function(K_test)
    test_auc = float(roc_auc_score(y_test, test_scores))
    support_mask = np.zeros(K_trainval.shape[0], dtype=bool)
    support_mask[final.support_] = True
    result = SvmResult(
        test_auc=test_auc, best_C=float(best_C), best_gamma=float(best_gamma), best_val_auc=float(best_val_auc),
        sv_count=int(support_mask.sum()), sv_fraction=float(support_mask.mean()), n_fit=int(K_trainval.shape[0]),
        test_scores=test_scores if keep_scores else None, test_labels=y_test if keep_scores else None,
        val_grid=val_grid,
    )
    del K_trainval, K_test, final
    gc.collect()
    return result


def relrep_svm_grid(X: np.ndarray, y: np.ndarray, splits: Splits, C_grid: Sequence[float],
                    gamma_grid: Sequence[float | str] = ("scale",), *, keep_scores: bool = False) -> SvmResult:
    """RBF SVC on M3R coordinates with validation-selected ``C``/``gamma`` and support-vector counts.

    A grid with a single entry gives the fixed-hyperparameter setting used for
    the paper's Fig. 7 (``C = 10``, ``gamma = "scale"``).
    """
    y_train, y_val = y[splits.train], y[splits.val]
    y_trainval, y_test = y[splits.trainval], y[splits.test]
    best_val_auc, best_gamma, best_C = -np.inf, None, None
    val_grid: dict = {}
    for gamma in gamma_grid:
        for C in C_grid:
            svc = SVC(kernel="rbf", gamma=gamma, C=C)
            svc.fit(X[splits.train], y_train)
            val_auc = float(roc_auc_score(y_val, svc.decision_function(X[splits.val])))
            val_grid[(gamma, float(C))] = val_auc
            if val_auc > best_val_auc:
                best_val_auc, best_gamma, best_C = val_auc, gamma, C
    final = SVC(kernel="rbf", gamma=best_gamma, C=best_C)
    final.fit(X[splits.trainval], y_trainval)
    test_scores = final.decision_function(X[splits.test])
    support_mask = np.zeros(splits.trainval.size, dtype=bool)
    support_mask[final.support_] = True
    return SvmResult(
        test_auc=float(roc_auc_score(y_test, test_scores)), best_C=float(best_C), best_gamma=best_gamma,
        best_val_auc=float(best_val_auc), sv_count=int(support_mask.sum()), sv_fraction=float(support_mask.mean()),
        n_fit=int(splits.trainval.size), test_scores=test_scores if keep_scores else None,
        test_labels=y_test if keep_scores else None, val_grid=val_grid,
    )
