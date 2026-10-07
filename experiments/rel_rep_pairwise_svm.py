"""Pairwise RBF-SVM classification on M3R coordinates (paper Sec. 5, Figs. 6-7).

Two modes:

* ``mode=fixed``  - fixed ``C``/``gamma`` and an 80/20 permutation split; reports the test AUC
  per class pair (the relative-space curves of Fig. 6);
* ``mode=study``  - stratified train/validation/test split, optional validation grid over
  ``C``/``gamma``, and support-vector bookkeeping (Fig. 7). A single-entry grid is the
  fixed-hyperparameter setting of the paper (``C=10``, ``gamma=scale``).
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from experiments._plots import upper_triangle_heatmap
from experiments.base_experiment import BaseExperiment
from m3r.relrep import pair_relrep
from m3r.svm import relrep_svm_auc, relrep_svm_grid, stratified_splits


class RelRepPairwiseSVM(BaseExperiment):
    def __init__(self, config: DictConfig, exp_dir: str):
        super().__init__(config, exp_dir)
        self.anchor_per_class = int(config.anchor_per_class)
        self.seed = int(config.seed)
        self.plot_only = bool(config.plot_only)
        self.mode = str(config.get("mode", "fixed"))
        self.svm_params = OmegaConf.to_container(config.svm, resolve=True)
        self.study = OmegaConf.to_container(config.get("study", {}), resolve=True) or {}
        sw_as_distance = bool(self.paper["metrics"].get("sw_as_distance", True))
        self.as_distance = sw_as_distance if self.metric == "sw" else True
        if config.get("sw_coordinates", None) is not None:
            self.as_distance = str(config.sw_coordinates) == "distance"

    @property
    def result_name(self) -> str:
        return f"svm_rel_rep_{self.dataset_size}_{self.metric}_{self.anchor_per_class}.npz"

    def run(self) -> None:
        if self.plot_only:
            payload = np.load(self.results_dir / self.result_name)
            self.plot(payload["test_aucs"], payload["sv_fractions"] if "sv_fractions" in payload.files else None)
            return
        n = self.n_classes
        test_aucs = np.full((n, n), np.nan)
        extras: dict[str, np.ndarray] = {}
        if self.mode == "study":
            extras = {k: np.full((n, n), np.nan) for k in ("val_aucs", "best_C", "best_gamma", "sv_fractions")}
            extras["sv_counts"] = np.zeros((n, n), dtype=int)
            margins: dict[str, np.ndarray] = {}
        for task in tqdm(self.tasks, total=len(self.tasks)):
            i, j = task.classes
            X, y, anchors = pair_relrep(self.store, self.metric, i, j, self.anchor_per_class, self.seed,
                                        as_distance=self.as_distance)
            if self.mode == "fixed":
                test_aucs[i, j] = test_aucs[j, i] = relrep_svm_auc(
                    X, y, self.seed, C=float(self.svm_params["C"]), gamma=self.svm_params["gamma"],
                    kernel=self.svm_params.get("kernel", "rbf"),
                    train_fraction=float(self.config.get("train_fraction", 0.8)))
            else:
                splits = stratified_splits(y, self.seed, float(self.study.get("test_fraction", 0.2)),
                                           float(self.study.get("validation_fraction", 0.2)))
                res = relrep_svm_grid(X, y, splits, [float(c) for c in self.study.get("C_grid", [10.0])],
                                      list(self.study.get("gamma_grid", ["scale"])),
                                      keep_scores=bool(self.study.get("save_margins", False)))
                test_aucs[i, j] = test_aucs[j, i] = res.test_auc
                extras["val_aucs"][i, j] = extras["val_aucs"][j, i] = res.best_val_auc
                extras["best_C"][i, j] = extras["best_C"][j, i] = res.best_C
                g = np.nan if isinstance(res.best_gamma, str) else float(res.best_gamma)
                extras["best_gamma"][i, j] = extras["best_gamma"][j, i] = g
                extras["sv_fractions"][i, j] = extras["sv_fractions"][j, i] = res.sv_fraction
                extras["sv_counts"][i, j] = extras["sv_counts"][j, i] = res.sv_count
                if res.test_scores is not None:
                    margins[f"margin_{i}_{j}"] = res.test_scores
                    margins[f"margin_labels_{i}_{j}"] = res.test_labels
            del X
        payload = dict(test_aucs=test_aucs, anchor_per_class=self.anchor_per_class, seed=self.seed, mode=self.mode,
                       sw_coordinates="distance" if self.as_distance else "stored", **self.svm_params)
        if self.mode == "study":
            payload.update(extras)
            payload["C_grid"] = np.asarray(self.study.get("C_grid", [10.0]), dtype=float)
            payload["gamma_grid"] = np.asarray([str(g) for g in self.study.get("gamma_grid", ["scale"])])
            payload["test_fraction"] = float(self.study.get("test_fraction", 0.2))
            payload["validation_fraction"] = float(self.study.get("validation_fraction", 0.2))
            if margins:
                from m3r.results_io import save_npz
                save_npz(self.results_dir / self.result_name.replace(".npz", "_margins.npz"), **margins)
        self.save_results(self.result_name, **payload)
        self.plot(test_aucs, extras.get("sv_fractions"))

    def plot(self, test_aucs: np.ndarray, sv_fractions: np.ndarray | None = None) -> None:
        tag = f"{self.metric} relative space, {2 * self.anchor_per_class} references"
        upper_triangle_heatmap(test_aucs, f"Pairwise SVM AUC ({tag})", "AUC",
                               self.plots_dir / f"pairwise_svm_{self.dataset_size}_{self.metric}_aucs_{self.anchor_per_class}.pdf",
                               labels=self.get_class_names())
        if sv_fractions is not None:
            upper_triangle_heatmap(sv_fractions, f"Support-vector fraction ({tag})", "Support vector fraction",
                                   self.plots_dir / f"pairwise_svm_{self.dataset_size}_{self.metric}_sv_fractions_{self.anchor_per_class}.pdf",
                                   vmin=0.0, vmax=1.0, labels=self.get_class_names())
