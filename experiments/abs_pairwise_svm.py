"""Pairwise RBF-SVM classification in the original metric space (the "Abs" baseline of Fig. 6).

The kernel ``exp(-gamma (d / d_med)^2)`` is evaluated from the stored distance blocks
through a precomputed Gram matrix; ``C`` and ``gamma`` are selected on a validation split.
"""

from __future__ import annotations

import gc

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from experiments._plots import upper_triangle_heatmap
from experiments.base_experiment import BaseExperiment
from m3r.svm import abs_svm_grid, stratified_splits


class AbsPairwiseSVM(BaseExperiment):
    def __init__(self, config: DictConfig, exp_dir: str):
        super().__init__(config, exp_dir)
        self.seed = int(config.seed)
        self.plot_only = bool(config.plot_only)
        self.test_fraction = float(config.test_fraction)
        self.validation_fraction = float(config.validation_fraction)
        self.gamma_grid = [float(g) for g in config.svm.gamma_grid]
        c_by_metric = OmegaConf.to_container(config.svm.C_grid_by_metric, resolve=True)
        default_c = OmegaConf.to_container(config.svm.default_C_grid, resolve=True)
        self.C_grid = [float(c) for c in c_by_metric.get(self.metric, default_c)]
        self.save_margins = bool(config.get("save_margins", False))

    @property
    def result_name(self) -> str:
        return f"svm_abs_{self.dataset_size}_{self.metric}.npz"

    def run(self) -> None:
        if self.plot_only:
            payload = np.load(self.results_dir / self.result_name)
            self.plot(payload["test_aucs"], payload["sv_fractions"])
            return
        n = self.n_classes
        arrays = {k: np.full((n, n), np.nan) for k in ("test_aucs", "val_aucs", "best_Cs", "best_gammas", "sv_fractions")}
        arrays["sv_counts"] = np.zeros((n, n), dtype=int)
        margins: dict[str, np.ndarray] = {}
        for task in tqdm(self.tasks, total=len(self.tasks)):
            i, j = task.classes
            pair = self.store.pair(self.metric, i, j)
            labels = pair.labels()
            splits = stratified_splits(labels, self.seed, self.test_fraction, self.validation_fraction)
            normalization = pair.upper_triangle_median()
            res = abs_svm_grid(pair, labels, splits, self.gamma_grid, self.C_grid, normalization,
                               keep_scores=self.save_margins)
            for key, value in (("test_aucs", res.test_auc), ("val_aucs", res.best_val_auc), ("best_Cs", res.best_C),
                               ("best_gammas", res.best_gamma), ("sv_fractions", res.sv_fraction), ("sv_counts", res.sv_count)):
                arrays[key][i, j] = arrays[key][j, i] = value
            if res.test_scores is not None:
                margins[f"margin_{i}_{j}"] = res.test_scores
                margins[f"margin_labels_{i}_{j}"] = res.test_labels
            del pair
            gc.collect()
        if margins:
            from m3r.results_io import save_npz
            save_npz(self.results_dir / self.result_name.replace(".npz", "_margins.npz"), **margins)
        self.save_results(self.result_name, C_grid=np.asarray(self.C_grid), gamma_grid=np.asarray(self.gamma_grid),
                          test_fraction=self.test_fraction, validation_fraction=self.validation_fraction,
                          normalization="median_upper_triangle_stored", **arrays)
        self.plot(arrays["test_aucs"], arrays["sv_fractions"])

    def plot(self, test_aucs: np.ndarray, sv_fractions: np.ndarray | None = None) -> None:
        upper_triangle_heatmap(test_aucs, f"Pairwise SVM AUC ({self.metric} absolute space)", "AUC",
                               self.plots_dir / f"pairwise_svm_abs_{self.dataset_size}_{self.metric}_aucs.pdf",
                               labels=self.get_class_names())
        if sv_fractions is not None:
            upper_triangle_heatmap(sv_fractions, f"Support-vector fraction ({self.metric} absolute space)",
                                   "Support vector fraction",
                                   self.plots_dir / f"pairwise_svm_abs_{self.dataset_size}_{self.metric}_sv_fractions.pdf",
                                   vmin=0.0, vmax=1.0, labels=self.get_class_names())
