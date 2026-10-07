"""Boosted-decision-tree classification on unified M3R representations (paper Sec. 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from m3r.blocks import BlockStore
from m3r.classes import KIN_METRICS, PID_FEATURES
from m3r.relrep import pair_relrep, pid_relrep, pid_values

PID_ALIAS = "pid"


def expand_features(features: Sequence[str]) -> list[str]:
    """Expand the ``pid`` alias into the six particle-content features."""
    out: list[str] = []
    for name in features:
        if name == PID_ALIAS:
            out.extend(PID_FEATURES)
        else:
            out.append(name)
    return out


def canonical_set_key(features: Sequence[str]) -> str:
    """Order-independent name of a feature set: kinematic metrics in PS, SW2, W2 order, then ``pid``."""
    expanded = expand_features(features)
    kin = [m for m in ("ps", "sw", "w2") if m in expanded]
    pid = set(f for f in expanded if f not in KIN_METRICS)
    key = "+".join(kin)
    if pid == set(PID_FEATURES):
        key = f"{key}+pid" if key else "pid"
    elif pid:
        key = "+".join([key] if key else []) + ("+" if key else "") + "+".join(sorted(pid))
    return key


def split_features(features: Sequence[str]) -> tuple[list[str], list[str]]:
    """Kinematic metrics first (listed order), then PID features (listed order), as in the published runs."""
    expanded = expand_features(features)
    kin = [m for m in ("ps", "sw", "w2") if m in expanded]
    pid = [f for f in expanded if f not in KIN_METRICS]
    unknown = [f for f in pid if f not in PID_FEATURES]
    if unknown:
        raise KeyError(f"Unknown features {unknown}; expected metrics {KIN_METRICS} or PID {PID_FEATURES}.")
    return kin, pid


def feature_matrix(
    store: BlockStore,
    pid_loader: Callable[[int, str], np.ndarray],
    i: int,
    j: int,
    features: Sequence[str],
    anchors_per_class: int,
    seed: int,
    *,
    as_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified M3R feature matrix for class pair ``(i, j)``.

    ``pid_loader(class_idx, feature)`` returns the per-particle array for one class
    (``(n_events, n_particles)``); it is summed to the event-level scalar inside.
    Returns ``(X, y, anchors)`` with the kinematic blocks first, then PID blocks.
    """
    kin, pid = split_features(features)
    blocks: list[np.ndarray] = []
    labels = anchors = None
    for metric in kin:
        X_m, labels, anchors = pair_relrep(store, metric, i, j, anchors_per_class, seed, as_distance=as_distance)
        blocks.append(X_m)
    if pid:
        if anchors is None:
            from m3r.relrep import pair_labels, sample_anchors
            n = store.n_per_class
            labels = pair_labels(n)
            anchors = sample_anchors(labels, anchors_per_class, seed, n_classes=2)
        for feature in pid:
            vi = pid_values(pid_loader(i, feature))
            vj = pid_values(pid_loader(j, feature))
            blocks.append(pid_relrep(vi, vj, anchors))
    if not blocks:
        raise ValueError("No features selected.")
    return np.concatenate(blocks, axis=1), labels, anchors


def background_rejection_at_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    valid = tpr >= target_tpr
    if not np.any(valid):
        return float("nan")
    fpr_at = float(np.min(fpr[valid]))
    return float("inf") if fpr_at <= 0.0 else 1.0 / fpr_at


@dataclass
class BdtResult:
    test_auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    background_rejections: dict[float, float]
    n_iter: int


def bdt_auc(X: np.ndarray, y: np.ndarray, params: dict, *, split_seed: int = 42, test_size: float = 0.2,
            target_tprs: Sequence[float] = (0.3, 0.5, 0.995)) -> BdtResult:
    """Stratified hold-out ROC AUC of a ``HistGradientBoostingClassifier`` (port of ``RelRepPairwiseBDT``)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=split_seed)
    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    return BdtResult(
        test_auc=float(roc_auc_score(y_test, scores)), fpr=fpr, tpr=tpr, thresholds=thresholds,
        background_rejections={float(t): background_rejection_at_tpr(fpr, tpr, t) for t in target_tprs},
        n_iter=int(getattr(clf, "n_iter_", 0)),
    )
