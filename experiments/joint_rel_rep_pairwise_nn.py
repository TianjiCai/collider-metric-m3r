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


class JointRelRepPairwiseNN(BaseExperiment):
    def __init__(self, config: DictConfig, exp_dir: str):
        super().__init__(config, exp_dir)
        self.device = (
            "cuda"
            if config.use_gpu and torch.cuda.is_available()
            else "mps"
            if config.use_gpu
            and torch.backends.mps.is_available()
            else "cpu"
        )
        self.anchor_per_class = config.anchor_per_class
        self.data_path = config.data_path
        self.seed = config.seed
        self.plot_only = config.plot_only
        self.metric_1 = config.metric_1
        self.metric_2 = config.metric_2
        self.metric_3 = config.get("metric_3", None)
        self.metric_2_scaling = config.metric_2_scaling
        self.train_params = config.training
        self.network = config.network
        self.train = config.train
        self.include_kin = config.include_kin
        self.include_part_ids = config.include_part_ids
        self.part_ids_path = config.part_ids_path
        # self.log.info("Value of plot_only:", self.plot_only)

    def run(self):
        if self.plot_only:
            file = np.load(
                f"{self.exp_dir}/results/mlp_rel_rep_{self.dataset_size}_{self.metric_1}_{self.metric_2}_{self.anchor_per_class}.npz"
            )
            aucs = file["test_aucs"]
            self.plot(aucs)
            return
        test_aucs = np.zeros((8, 8))

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
            
            rel_rep_blocks = []
            rel_reps_kin = None
            part_ids_rel_rep = None
            
            if self.include_kin:
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
                
                rel_reps_kin = np.concatenate((rel_reps_1, rel_reps_2), axis=1)
                rel_rep_blocks.append(rel_reps_kin)
                del rel_reps_1, rel_reps_2
                
            del big_matrix_1, big_matrix_2
            
            if self.include_part_ids:
                part_ids_rel_rep = self.compute_part_ids_rel_rep(i, j, self.seed)
                rel_rep_blocks.append(part_ids_rel_rep)

            rel_reps = np.concatenate(rel_rep_blocks, axis=1)
            del rel_rep_blocks, rel_reps_kin, part_ids_rel_rep

            X_train, X_test, y_train, y_test = train_test_split(
                rel_reps,
                labels,
                test_size=self.train_params.val_size,
                stratify=labels,
                random_state=42,
            )

            del rel_reps

            model = instantiate(self.network)

            if self.train:
                # print("Training...")
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
            np.save(f"{self.exp_dir}/pred_probabilities.npy", self.pred_probs)
            test_aucs[i, j] = roc_auc_score(
                y_test, self.pred_probs[:, 1]
            )

        os.makedirs(os.path.join(self.exp_dir, "results"), exist_ok=True)
        output_file_path = f"{self.exp_dir}/results/mlp_rel_rep_{self.dataset_size}_{self.metric_1}_{self.metric_2}_{self.anchor_per_class}.npz"
        np.savez(file=output_file_path, test_aucs=test_aucs)
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
            f"Pairwise MLP AUCs, {self.metric_1}+{self.metric_2} Relative Space, {2 * self.anchor_per_class} Anchors"
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
            f"{self.exp_dir}/plots/pairwise_mlp_{self.dataset_size}_{self.metric_1}_{self.metric_2}_aucs_{self.anchor_per_class}.png",
            dpi=300,
        )
    
    def compute_part_ids_rel_rep(self, i, j, seed):
        class_labels = np.load(self.part_ids_path + "class_labels.npz")
        class_i = next(k for k, v in class_labels.items() if v == i)
        class_j = next(k for k, v in class_labels.items() if v == j)

        dataset_size = ""
        if self.events_per_class == 10000:
            dataset_size = "10k"
        elif self.events_per_class == 100000:
            dataset_size = "100k"
        elif self.events_per_class == 4000:
            dataset_size = "4k"
        elif self.events_per_class == 1000:
            dataset_size = "1k"
        else:
            raise ValueError("Invalid dataset size")

        # shape = (dataset_size, 150, 6)
        part_ids_i = np.load(
            self.part_ids_path + f"{class_i}_part_ids_{dataset_size}.npy"
        )
        # print("Shape of part_ids_i:", part_ids_i.shape)
        part_ids_j = np.load(
            self.part_ids_path + f"{class_j}_part_ids_{dataset_size}.npy"
        )
        # print("Shape of part_ids_j:", part_ids_j.shape)
        # self.log.info("Shape of part_ids_i:", part_ids_i.shape)
        # self.log.info("Shape of part_ids_j:", part_ids_j.shape)

        # shape (dataset_size, 6)
        part_ids_i_sum = np.sum(part_ids_i, axis=1)
        part_ids_j_sum = np.sum(part_ids_j, axis=1)
        # self.log.info("Shape of part_ids_i_sum:", part_ids_i_sum.shape)
        # self.log.info("Shape of part_ids_j_sum:", part_ids_j_sum.shape)

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

        diff = big[:, None, :] - anchors[None, :, :]  # (2*dataset_size, n_anchors, 6)
        final = diff.transpose(0, 2, 1).reshape(
            big.shape[0], -1
        )  # (2*dataset_size, n_anchors*6)

        return final
