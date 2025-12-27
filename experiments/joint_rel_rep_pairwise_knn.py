from experiments.base_experiment import BaseExperiment
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
import utils.ot_utils
import utils.comparison_utils
import os


class JointRelRepPairwiseKNN(BaseExperiment):
    def __init__(self, config: DictConfig, exp_dir: str):
        super().__init__(config, exp_dir)
        self.anchor_per_class = config.anchor_per_class
        self.data_path = config.data_path
        self.seed = config.seed
        self.plot_only = config.plot_only
        self.metric_1 = config.metric_1
        self.metric_2 = config.metric_2
        self.metric_2_scaling = config.metric_2_scaling
        self.knn_sim_metric = config.knn_sim_metric
        # self.log.info("Value of plot_only:", self.plot_only)

    def run(self):
        if self.plot_only:
            file = np.load(
                f"{self.exp_dir}/results/knn_rel_rep_{self.dataset_size}_{self.metric_1}_{self.metric_2}_{self.anchor_per_class}.npz"
            )
            aucs = file["test_aucs"]
            self.plot(aucs)
            return
        test_aucs = np.zeros((8, 8))
        best_ks = np.zeros((8, 8))

        for i, j in tqdm(
            combinations(range(8), 2), total=len(list(combinations(range(8), 2)))
        ):
            self_matrix_1_file = (
                self.project_path
                + f"{self.data_path}{self.metric_1}/{self.dataset_size}_{self.metric_1}/{self.metric_1}_block_{i}_{i}.npy"
            )
            self_matrix_2_file = (
                self.project_path
                + f"{self.data_path}{self.metric_1}/{self.dataset_size}_{self.metric_1}/{self.metric_1}_block_{j}_{j}.npy"
            )
            off_diag_matrix_file = (
                self.project_path
                + f"{self.data_path}{self.metric_1}/{self.dataset_size}_{self.metric_1}/{self.metric_1}_block_{i}_{j}.npy"
            )

            big_matrix_1 = utils.ot_utils.combine_distance_blocks(
                [self_matrix_1_file, off_diag_matrix_file, self_matrix_2_file]
            )

            self_matrix_1_file = (
                self.project_path
                + f"{self.data_path}{self.metric_2}/{self.dataset_size}_{self.metric_2}/{self.metric_2}_block_{i}_{i}.npy"
            )
            self_matrix_2_file = (
                self.project_path
                + f"{self.data_path}{self.metric_2}/{self.dataset_size}_{self.metric_2}/{self.metric_2}_block_{j}_{j}.npy"
            )
            off_diag_matrix_file = (
                self.project_path
                + f"{self.data_path}{self.metric_2}/{self.dataset_size}_{self.metric_2}/{self.metric_2}_block_{i}_{j}.npy"
            )

            big_matrix_2 = utils.ot_utils.combine_distance_blocks(
                [self_matrix_1_file, off_diag_matrix_file, self_matrix_2_file]
            )

            labels = np.repeat([0, 1], self.events_per_class)

            rel_reps_1 = utils.comparison_utils.relative_rep(
                big_matrix_1,
                labels,
                num_classes=2,
                anchors_per_class=self.anchor_per_class,
                seed=42,
            )

            rel_reps_2 = utils.comparison_utils.relative_rep(
                big_matrix_2,
                labels,
                num_classes=2,
                anchors_per_class=self.anchor_per_class,
                seed=42,
            )

            rel_reps_2 *= self.metric_2_scaling

            rel_reps = np.concatenate((rel_reps_1, rel_reps_2), axis=1)
            # print(rel_reps.shape)

            neighbor_list = list(
                range(1, int(np.sqrt(self.events_per_class * 2) * 2), 2)
            )

            test_auc, best_k = utils.comparison_utils.knn_with_features(
                rel_reps, labels, neighbor_list, metric=self.knn_sim_metric
            )

            test_aucs[i, j] = test_auc
            best_ks[i, j] = best_k

        test_aucs = test_aucs + test_aucs.T
        best_ks = best_ks + best_ks.T

        os.makedirs(os.path.join(self.exp_dir, "results"), exist_ok=True)
        output_file_path = f"{self.exp_dir}/results/knn_rel_rep_{self.dataset_size}_{self.metric_1}_{self.metric_2}_{self.anchor_per_class}.npz"
        np.savez(file=output_file_path, test_aucs=test_aucs, best_ks=best_ks)
        self.plot(test_aucs)

    def plot(self, test_aucs):
        self.log.info("Making plots...")
        mask = np.triu(np.ones_like(test_aucs, dtype=bool), k=1)
        plot_data = np.where(mask, test_aucs, np.nan)

        vals = plot_data[~np.isnan(plot_data)]
        nonzero = vals[vals != 0]
        vmin, vmax = float(nonzero.min()), float(nonzero.max())

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color=(1, 1, 1, 0))  # transparent for masked cells

        im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks(range(self.n_classes))
        ax.set_yticks(range(self.n_classes))
        ax.set_xticklabels(
            [f"{i + 1}" for i in range(self.n_classes)], rotation=45, ha="right"
        )
        ax.set_yticklabels([f"{i + 1}" for i in range(self.n_classes)])
        ax.set_title(
            f"Pairwise kNN AUCs, {self.metric_1}+{self.metric_2} Relative Space, {2 * self.anchor_per_class} Anchors"
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Value")

        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                ax.text(
                    j,
                    i,
                    f"{test_aucs[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="w",
                )

        plt.tight_layout()
        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)
        plt.savefig(
            f"{self.exp_dir}/plots/pairwise_knn_{self.dataset_size}_{self.metric_1}_{self.metric_2}_aucs_{self.anchor_per_class}.png",
            dpi=300,
        )
