"""Pairwise multi-tower MLP on M3R feature blocks (robustness check mentioned in Sec. 6).

Each tower of ``network.tower_specs`` is named after a kinematic metric or a PID
feature; the matching M3R block (``2 * anchor_per_class`` columns) feeds that tower.
PID blocks use the signed difference ``h(x) - h(r)`` as in the published runs.
"""

from __future__ import annotations

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from experiments._nn_common import select_device, signed_pid_relrep
from experiments._plots import upper_triangle_heatmap
from experiments.base_experiment import BaseExperiment
from m3r.classes import KIN_METRICS
from m3r.relrep import pair_labels, pair_relrep, pid_values, sample_anchors


class RelRepPairwiseNN(BaseExperiment):
    def __init__(self, config: DictConfig, exp_dir: str):
        super().__init__(config, exp_dir)
        self.device = select_device(bool(config.use_gpu))
        self.anchor_per_class = int(config.anchor_per_class)
        self.seed = int(config.seed)
        self.plot_only = bool(config.plot_only)
        self.train_params = config.training
        self.network = config.network
        self.do_train = bool(config.train)
        features = config.get("features", None)
        if features:
            # Build one tower per requested feature ("pid" expands to the six PID features).
            from omegaconf import OmegaConf

            from m3r.bdt import expand_features
            kin_hidden = list(config.get("kin_hidden_dims", [256, 128]))
            pid_hidden = list(config.get("pid_hidden_dims", [64, 32]))
            specs = [{"name": f, "in_dim": self.anchor_per_class * 2,
                      "hidden_dims": kin_hidden if f in KIN_METRICS else pid_hidden}
                     for f in expand_features(list(features))]
            self.network.tower_specs = OmegaConf.create(specs)
        self.kin_metrics, self.part_id_metrics = [], []
        for i, spec in enumerate(self.network.tower_specs):
            (self.kin_metrics if spec.name in KIN_METRICS else self.part_id_metrics).append(str(spec.name))
            self.network.tower_specs[i].in_dim = self.anchor_per_class * 2
        self.network.output_dim = 2

    @property
    def result_name(self) -> str:
        return f"mlp_rel_rep_{self.dataset_size}_{self.anchor_per_class}.npz"

    @property
    def feature_key(self) -> str:
        from m3r.bdt import canonical_set_key
        return canonical_set_key(self.kin_metrics + self.part_id_metrics)

    def run(self) -> None:
        if self.plot_only:
            self.plot(np.load(self.results_dir / self.result_name)["test_aucs"])
            return
        from m3r.nn import evaluate_network_torch, train_network_torch

        n = self.n_classes
        test_aucs = np.full((n, n), np.nan)
        for task in tqdm(self.tasks, total=len(self.tasks)):
            i, j = task.classes
            blocks, labels, anchors = [], pair_labels(self.events_per_class), None
            for metric in self.kin_metrics:
                X_m, labels, anchors = pair_relrep(self.store, metric, i, j, self.anchor_per_class, self.seed)
                blocks.append(X_m)
            if anchors is None:
                anchors = sample_anchors(labels, self.anchor_per_class, self.seed, n_classes=2)
            for feature in self.part_id_metrics:
                blocks.append(signed_pid_relrep(pid_values(self.part_ids(i, feature)), pid_values(self.part_ids(j, feature)), anchors))
            X = np.concatenate(blocks, axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=self.train_params.val_size,
                                                                stratify=labels, random_state=42)
            model = instantiate(self.network)
            if self.do_train:
                model, _, _ = train_network_torch(model, X_train, y_train, val_size=self.train_params.val_size,
                                                  epochs=self.train_params.epochs, patience=self.train_params.patience,
                                                  batch_size=self.train_params.batch_size,
                                                  learning_rate=self.train_params.learning_rate, device=self.device)
            _, _, probs = evaluate_network_torch(model, X_test, y_test, device=self.device)
            test_aucs[i, j] = test_aucs[j, i] = roc_auc_score(y_test, probs[:, 1])
        self.save_results(self.result_name, test_aucs=test_aucs, anchor_per_class=self.anchor_per_class,
                          kin_metrics=np.asarray(self.kin_metrics), part_id_metrics=np.asarray(self.part_id_metrics))
        self.plot(test_aucs)

    def plot(self, test_aucs: np.ndarray) -> None:
        upper_triangle_heatmap(test_aucs, f"MLP AUC ({'+'.join(self.kin_metrics + self.part_id_metrics)}, {self.anchor_per_class} refs/class)",
                               "AUC", self.plots_dir / f"pairwise_mlp_{self.dataset_size}_aucs_{self.anchor_per_class}.pdf",
                               labels=self.get_class_names())
