"""Pairwise BDT classification on unified M3R representations (paper Sec. 6, Fig. 8 and Table 1).

The feature set is either a named set from ``configs/paper.yaml`` (``feature_set=ps+sw+w2+pid``)
or an explicit list (``metrics=[ps,sw,part_charge]``). Kinematic blocks come first (PS, SW2, W2
order), then the particle-content blocks.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from experiments._plots import upper_triangle_heatmap
from experiments.base_experiment import BaseExperiment
from m3r.bdt import bdt_auc, canonical_set_key, expand_features, feature_matrix


class RelRepPairwiseBDT(BaseExperiment):
    def __init__(self, config: DictConfig, exp_dir: str):
        super().__init__(config, exp_dir)
        self.anchor_per_class = int(config.anchor_per_class)
        self.seed = int(config.seed)
        self.plot_only = bool(config.plot_only)
        self.val_size = float(config.val_size)
        self.save_roc = bool(config.get("save_roc", True))
        self.bdt_params = OmegaConf.to_container(config.bdt, resolve=True)
        self.target_tprs = [float(t) for t in config.get("target_tprs", [0.3, 0.5, 0.995])]
        explicit = config.get("metrics", None)
        if explicit:
            self.features = list(OmegaConf.to_container(explicit, resolve=True))
        else:
            sets = self.paper["bdt"]["feature_sets"]
            key = str(config.feature_set)
            if key not in sets:
                raise KeyError(f"Unknown feature_set {key!r}; known: {sorted(sets)}")
            self.features = list(sets[key])
        self.feature_key = canonical_set_key(expand_features(self.features))
        sw_as_distance = bool(self.paper["metrics"].get("sw_as_distance", True))
        self.as_distance = sw_as_distance

    @property
    def result_name(self) -> str:
        return f"bdt_rel_rep_{self.dataset_size}_{self.anchor_per_class}.npz"

    def run(self) -> None:
        if self.plot_only:
            payload = np.load(self.results_dir / self.result_name, allow_pickle=True)
            self.plot(payload["test_aucs"])
            return
        n = self.n_classes
        test_aucs = np.full((n, n), np.nan)
        rejections = np.full((len(self.target_tprs), n, n), np.nan)
        n_iters = np.zeros((n, n), dtype=int)
        roc: dict[str, np.ndarray] = {}
        for task in tqdm(self.tasks, total=len(self.tasks)):
            i, j = task.classes
            X, y, _ = feature_matrix(self.store, self.part_ids, i, j, self.features, self.anchor_per_class,
                                     self.seed, as_distance=self.as_distance)
            res = bdt_auc(X, y, self.bdt_params, split_seed=self.seed, test_size=self.val_size,
                          target_tprs=self.target_tprs)
            test_aucs[i, j] = test_aucs[j, i] = res.test_auc
            n_iters[i, j] = n_iters[j, i] = res.n_iter
            for k, t in enumerate(self.target_tprs):
                rejections[k, i, j] = rejections[k, j, i] = res.background_rejections[t]
            if self.save_roc:
                roc[f"fpr_{i}_{j}"] = res.fpr; roc[f"tpr_{i}_{j}"] = res.tpr; roc[f"thresholds_{i}_{j}"] = res.thresholds
            del X
        if roc:
            from m3r.results_io import save_npz
            save_npz(self.results_dir / self.result_name.replace(".npz", "_roc.npz"), **roc)
        self.save_results(self.result_name, test_aucs=test_aucs, background_rejections=rejections,
                          target_tprs=np.asarray(self.target_tprs), n_iter=n_iters,
                          features=np.asarray(expand_features(self.features)), feature_key=self.feature_key,
                          anchor_per_class=self.anchor_per_class, seed=self.seed,
                          sw_coordinates="distance" if self.as_distance else "stored",
                          **{f"bdt_{k}": v for k, v in self.bdt_params.items()})
        self.plot(test_aucs)

    def plot(self, test_aucs: np.ndarray) -> None:
        upper_triangle_heatmap(test_aucs, f"BDT AUC ({self.feature_key}, {self.anchor_per_class} refs/class)", "AUC",
                               self.plots_dir / f"pairwise_bdt_{self.dataset_size}_{self.feature_key}_aucs_{self.anchor_per_class}.pdf",
                               labels=self.get_class_names())
