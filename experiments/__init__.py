"""Hydra-driven tagging experiments used in the paper (see main.py).

Experiment classes are imported lazily so that an SVM run does not import torch.
``getattr(experiments, "RelRepPairwiseSVM")`` triggers the import.
"""

from __future__ import annotations

import importlib

_REGISTRY = {
    "BaseExperiment": "experiments.base_experiment",
    "RelRepPairwiseSVM": "experiments.rel_rep_pairwise_svm",   # Figs. 6 and 7
    "AbsPairwiseSVM": "experiments.abs_pairwise_svm",           # Fig. 6 "Abs" benchmark
    "RelRepPairwiseBDT": "experiments.rel_rep_pairwise_bdt",   # Fig. 8 and Table 1
    "RelRepPairwiseNN": "experiments.rel_rep_pairwise_nn",     # multi-tower MLP robustness check (Sec. 6 footnote)
}

__all__ = list(_REGISTRY)


def __getattr__(name: str):
    if name in _REGISTRY:
        module = importlib.import_module(_REGISTRY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'experiments' has no attribute {name!r}")
