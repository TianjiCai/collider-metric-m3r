from experiments.base_experiment import BaseExperiment
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
import utils.ot_utils
import utils.comparison_utils
import os
import utils.nn_utils
from sklearn.model_selection import train_test_split
from hydra.utils import instantiate
import torch
from sklearn.metrics import roc_auc_score


class RelRepPairwiseNN(BaseExperiment):
    def __init__(self, config: DictConfig, exp_dir: str):
        super().__init__(config, exp_dir)
        self.device = (
            "cuda"
            if config.use_gpu and torch.cuda.is_available()
            else "mps"
            if config.use_gpu and torch.backends.mps.is_available()
            else "cpu"
        )
        self.anchor_per_class = config.anchor_per_class
        self.data_path = config.data_path
        self.seed = config.seed
        self.plot_only = config.plot_only
        self.train_params = config.training
        self.network = config.network
        self.kin_metrics = []
        self.part_id_metrics = []
        all_kin_metrics = ["w2", "sw", "ps"]
        for i, tower_spec in enumerate(self.network.tower_specs):
            spec_name = tower_spec.name
            if spec_name in all_kin_metrics:
                self.kin_metrics.append(spec_name)
            else:
                self.part_id_metrics.append(spec_name)
            self.network.tower_specs[i].in_dim = self.anchor_per_class * 2
        self.network.output_dim = 2
        self.train = config.train
        self.part_ids_path = config.part_ids_path
        # self.log.info("Value of plot_only:", self.plot_only)

    def run(self):
        if self.plot_only:
            file = np.load(
                f"{self.exp_dir}/results/mlp_rel_rep_{self.dataset_size}_{self.anchor_per_class}.npz"
            )
            aucs = file["test_aucs"]
            self.plot(aucs)
            return
        test_aucs = np.zeros((self.n_classes, self.n_classes))
        class_pairs = self.get_class_pairs()

        for i, j in tqdm(class_pairs, total=len(class_pairs)):
            rel_rep_blocks = []
            labels = np.repeat([0, 1], self.events_per_class)
            for kin_metric in self.kin_metrics:
                self_matrix_1_file = (
                    self.project_path
                    + f"{self.data_path}{kin_metric}/{self.dataset_size}_{kin_metric}/{kin_metric}_block_{i}_{i}.npy"
                )
                self_matrix_2_file = (
                    self.project_path
                    + f"{self.data_path}{kin_metric}/{self.dataset_size}_{kin_metric}/{kin_metric}_block_{j}_{j}.npy"
                )
                off_diag_matrix_file = (
                    self.project_path
                    + f"{self.data_path}{kin_metric}/{self.dataset_size}_{kin_metric}/{kin_metric}_block_{i}_{j}.npy"
                )

                big_matrix = utils.ot_utils.combine_distance_blocks(
                    [self_matrix_1_file, off_diag_matrix_file, self_matrix_2_file]
                )

                rel_rep = utils.comparison_utils.relative_rep(
                    big_matrix,
                    labels,
                    num_classes=2,
                    anchors_per_class=self.anchor_per_class,
                    seed=42,
                )
                
                # get the overall mean and std of the rel_rep
                # mean = np.mean(rel_rep)
                # std = np.std(rel_rep)
                # print(f"Mean: {mean}, Std: {std}")
                
                # rel_rep = (rel_rep - mean) / std

                rel_rep_blocks.append(
                    rel_rep
                )

                del big_matrix

            for part_id_metric in self.part_id_metrics:
                part_id_rel_rep = self.compute_part_ids_rel_rep(
                    i, j, part_id_metric, self.seed
                )
                # get the overall mean and std of the part_id_rel_rep
                # mean = np.mean(part_id_rel_rep)
                # std = np.std(part_id_rel_rep)
                # print(f"Mean: {mean}, Std: {std}")
                
                # part_id_rel_rep = (part_id_rel_rep - mean) / std
                
                rel_rep_blocks.append(part_id_rel_rep)

            if not rel_rep_blocks:
                raise ValueError("No metrics selected.")

            final_rel_reps = np.concatenate(rel_rep_blocks, axis=1)
            del rel_rep_blocks

            X_train, X_test, y_train, y_test = train_test_split(
                final_rel_reps,
                labels,
                test_size=self.train_params.val_size,
                stratify=labels,
                random_state=42,
            )

            model = instantiate(self.network)
            # print("Model:", model)

            if self.train:
                self.log.info("Training...")
                trained_model, train_loss_history, val_loss_history = (
                    utils.nn_utils.train_network_torch(
                        model,
                        X_train,
                        y_train,
                        val_size=self.train_params.val_size,
                        epochs=self.train_params.epochs,
                        patience=self.train_params.patience,
                        batch_size=self.train_params.batch_size,
                        learning_rate=self.train_params.learning_rate,
                        device=self.device,
                    )
                )
                # # save the trained model
                # torch.save(
                #     trained_model.state_dict(), self.exp_dir + "/trained_model.pth"
                # )
                self.train_loss_history = train_loss_history
                self.val_loss_history = val_loss_history
            # model_state_dict = torch.load(
            #     self.exp_dir + "/trained_model.pth", map_location=self.device
            # )
            # model.load_state_dict(model_state_dict)

            self.log.info("Evaluating...")
            _, self.preds, self.pred_probs = utils.nn_utils.evaluate_network_torch(
                trained_model, X_test, y_test, device=self.device
            )
            # np.save(f"{self.exp_dir}/predictions.npy", self.preds)
            # np.save(f"{self.exp_dir}/pred_probabilities.npy", self.pred_probs)
            test_aucs[i, j] = roc_auc_score(y_test, self.pred_probs[:, 1])

        os.makedirs(os.path.join(self.exp_dir, "results"), exist_ok=True)
        output_file_path = f"{self.exp_dir}/results/mlp_rel_rep_{self.dataset_size}_{self.anchor_per_class}.npz"
        np.savez(file=output_file_path, test_aucs=test_aucs)
        self.plot(test_aucs)

    def plot(self, test_aucs):
        self.log.info("Making plots...")
        mask = np.triu(np.ones_like(test_aucs, dtype=bool), k=1)
        plot_data = np.where(mask, test_aucs, np.nan)

        vmin, vmax = 0.5, 1.0

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color=(1, 1, 1, 0))  # transparent for masked cells

        im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)

        class_names = self.get_class_names()
        ax.set_xticks(range(self.n_classes))
        ax.set_yticks(range(self.n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        kin_label = "All Kin" if len(self.kin_metrics) >= 3 else self.kin_metrics
        pid_label = "All PIDs" if len(self.part_id_metrics) >= 6 else self.part_id_metrics
        ax.set_title(
            f"MLP AUCs, {kin_label} {pid_label} Rel, {self.anchor_per_class} Anc/Class"
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
            f"{self.exp_dir}/plots/pairwise_mlp_{self.dataset_size}_{self.metric}_aucs_{self.anchor_per_class}.png",
            dpi=300,
        )

    def compute_part_ids_rel_rep(self, i, j, part_id_metric, seed):
        class_i = self.get_class_name(i)
        class_j = self.get_class_name(j)

        part_ids_i = np.load(
            self.part_ids_path + f"{class_i}_part_ids_{self.dataset_size_tag}.npz"
        )[part_id_metric]
        part_ids_j = np.load(
            self.part_ids_path + f"{class_j}_part_ids_{self.dataset_size_tag}.npz"
        )[part_id_metric]

        part_ids_i_sum = np.sum(part_ids_i, axis=1)
        part_ids_j_sum = np.sum(part_ids_j, axis=1)

        np.random.seed(seed)
        labels = np.repeat([0, 1], self.events_per_class)
        anchors_indices = np.zeros(self.anchor_per_class * 2, dtype=int)
        labels_buckets = [np.where(labels == i)[0] for i in range(2)]

        for i in range(2):
            class_anchors = np.random.choice(
                labels_buckets[i], size=self.anchor_per_class, replace=False
            )
            anchors_indices[
                i * self.anchor_per_class : (i + 1) * self.anchor_per_class
            ] = class_anchors

        big = np.concatenate([part_ids_i_sum, part_ids_j_sum], axis=0)
        anchors = big[anchors_indices]

        diff = big[:, None] - anchors[None, :]
        # print("diff.shape:", diff.shape)

        return diff
