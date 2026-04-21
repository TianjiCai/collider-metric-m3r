from experiments.base_experiment import BaseExperiment
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from itertools import combinations
import utils.ot_utils
import utils.comparison_utils
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve


class RelRepPairwiseBDT(BaseExperiment):
    def __init__(self, config: DictConfig, exp_dir: str):
        super().__init__(config, exp_dir)
        self.anchor_per_class = config.anchor_per_class
        self.data_path = config.data_path
        self.seed = config.seed
        self.plot_only = config.plot_only
        self.val_size = config.val_size
        self.bdt_params = OmegaConf.to_container(config.bdt, resolve=True)
        self.part_ids_path = config.part_ids_path
        self.target_tprs = np.array([0.3, 0.5, 0.995], dtype=float)

        all_kin_metrics = {"w2", "sw", "ps"}
        self.kin_metrics = []
        self.part_id_metrics = []
        for name in config.metrics:
            if name in all_kin_metrics:
                self.kin_metrics.append(name)
            else:
                self.part_id_metrics.append(name)

    def run(self):
        if self.plot_only:
            file = np.load(
                f"{self.exp_dir}/results/bdt_rel_rep_{self.dataset_size}_{self.anchor_per_class}.npz",
                allow_pickle=True,
            )
            aucs = file["test_aucs"]
            background_rejections = (
                file["background_rejections"]
                if "background_rejections" in file.files
                else None
            )
            target_tprs = (
                file["target_tprs"] if "target_tprs" in file.files else self.target_tprs
            )
            roc_fprs = file["roc_fprs"] if "roc_fprs" in file.files else None
            roc_tprs = file["roc_tprs"] if "roc_tprs" in file.files else None
            roc_thresholds = (
                file["roc_thresholds"] if "roc_thresholds" in file.files else None
            )
            self.plot(
                aucs,
                background_rejections,
                target_tprs,
                roc_fprs,
                roc_tprs,
                roc_thresholds,
            )
            return
        test_aucs = np.zeros((self.n_classes, self.n_classes))
        background_rejections = np.full(
            (len(self.target_tprs), self.n_classes, self.n_classes), np.nan
        )
        roc_fprs = np.empty((self.n_classes, self.n_classes), dtype=object)
        roc_tprs = np.empty((self.n_classes, self.n_classes), dtype=object)
        roc_thresholds = np.empty((self.n_classes, self.n_classes), dtype=object)
        roc_fprs[:] = None
        roc_tprs[:] = None
        roc_thresholds[:] = None
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

                rel_rep_blocks.append(rel_rep)
                del big_matrix

            for part_id_metric in self.part_id_metrics:
                part_id_rel_rep = self.compute_part_ids_rel_rep(
                    i, j, part_id_metric, self.seed
                )
                rel_rep_blocks.append(part_id_rel_rep)

            if not rel_rep_blocks:
                raise ValueError("No metrics selected.")

            final_rel_reps = np.concatenate(rel_rep_blocks, axis=1)
            del rel_rep_blocks

            X_train, X_test, y_train, y_test = train_test_split(
                final_rel_reps,
                labels,
                test_size=self.val_size,
                stratify=labels,
                random_state=42,
            )

            clf = HistGradientBoostingClassifier(**self.bdt_params)
            clf.fit(X_train, y_train)

            y_score = clf.predict_proba(X_test)[:, 1]
            test_aucs[i, j] = roc_auc_score(y_test, y_score)

            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            roc_fprs[i, j] = fpr
            roc_tprs[i, j] = tpr
            roc_thresholds[i, j] = thresholds
            for idx, target_tpr in enumerate(self.target_tprs):
                background_rejections[idx, i, j] = self.background_rejection_at_tpr(
                    fpr, tpr, target_tpr
                )

        os.makedirs(os.path.join(self.exp_dir, "results"), exist_ok=True)
        output_file_path = f"{self.exp_dir}/results/bdt_rel_rep_{self.dataset_size}_{self.anchor_per_class}.npz"
        np.savez(
            file=output_file_path,
            test_aucs=test_aucs,
            background_rejections=background_rejections,
            target_tprs=self.target_tprs,
            roc_fprs=roc_fprs,
            roc_tprs=roc_tprs,
            roc_thresholds=roc_thresholds,
        )
        self.plot(
            test_aucs,
            background_rejections,
            self.target_tprs,
            roc_fprs,
            roc_tprs,
            roc_thresholds,
        )

    @staticmethod
    def background_rejection_at_tpr(fpr, tpr, target_tpr):
        valid = tpr >= target_tpr
        if not np.any(valid):
            return np.nan

        fpr_at_target = np.min(fpr[valid])
        if fpr_at_target <= 0.0:
            return np.inf
        return 1.0 / fpr_at_target

    @staticmethod
    def target_tpr_labels(target_tpr):
        percent = 100.0 * float(target_tpr)
        percent_label = f"{percent:g}"
        file_tag = percent_label.replace(".", "p")
        return percent_label, file_tag

    def get_class_names(self):
        return super().get_class_names()

    def plot_pairwise_curve_grid(self, roc_fprs, roc_tprs, curve_kind, kin_label, pid_label):
        pairs = self.get_class_pairs()
        class_names = self.get_class_names()
        n_cols = 4
        n_rows = int(np.ceil(len(pairs) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.5 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )

        for ax in axes.ravel():
            ax.set_visible(False)

        for ax, (i, j) in zip(axes.ravel(), pairs):
            ax.set_visible(True)
            fpr = roc_fprs[i, j]
            tpr = roc_tprs[i, j]

            if fpr is None or tpr is None:
                ax.text(0.5, 0.5, "Missing curve", ha="center", va="center")
                ax.set_axis_off()
                continue

            if curve_kind == "roc":
                ax.plot(fpr, tpr, color="tab:blue", linewidth=1.5)
                ax.plot(
                    [0.0, 1.0],
                    [0.0, 1.0],
                    linestyle="--",
                    color="gray",
                    linewidth=1,
                )
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
            else:
                valid = np.isfinite(fpr) & np.isfinite(tpr) & (fpr > 0.0)
                if not np.any(valid):
                    ax.text(0.5, 0.5, "No finite SIC", ha="center", va="center")
                    ax.set_axis_off()
                    continue

                sic = tpr[valid] / np.sqrt(fpr[valid])
                finite = np.isfinite(sic)
                if not np.any(finite):
                    ax.text(0.5, 0.5, "No finite SIC", ha="center", va="center")
                    ax.set_axis_off()
                    continue

                ax.plot(tpr[valid][finite], sic[finite], color="tab:orange", linewidth=1.5)
                ax.set_xlabel("TPR")
                ax.set_ylabel("TPR / sqrt(FPR)")
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(bottom=0.0)

            ax.set_title(f"{class_names[i]} vs {class_names[j]}", fontsize=10)
            ax.grid(alpha=0.2)

        curve_title = "ROC Curves" if curve_kind == "roc" else "SIC Curves"
        curve_suffix = "roc_curves" if curve_kind == "roc" else "sic_curves"
        fig.suptitle(
            f"BDT Pairwise {curve_title}, {kin_label} {pid_label} Rel, "
            f"{self.anchor_per_class} Anc/Class",
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(
            f"{self.exp_dir}/plots/pairwise_bdt_{self.dataset_size}_{self.metric}_{curve_suffix}_{self.anchor_per_class}.png",
            dpi=300,
        )
        plt.close(fig)

    def plot(
        self,
        test_aucs,
        background_rejections=None,
        target_tprs=None,
        roc_fprs=None,
        roc_tprs=None,
        roc_thresholds=None,
    ):
        self.log.info("Making plots...")
        if target_tprs is None:
            target_tprs = self.target_tprs

        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)
        kin_label = "All Kin" if len(self.kin_metrics) >= 3 else self.kin_metrics
        pid_label = "All PIDs" if len(self.part_id_metrics) >= 6 else self.part_id_metrics

        mask = np.triu(np.ones_like(test_aucs, dtype=bool), k=1)
        plot_data = np.where(mask, test_aucs, np.nan)

        vmin, vmax = 0.5, 1.0

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color=(1, 1, 1, 0))

        im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)

        class_names = self.get_class_names()
        ax.set_xticks(range(self.n_classes))
        ax.set_yticks(range(self.n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_title(
            f"BDT AUCs, {kin_label} {pid_label} Rel, {self.anchor_per_class} Anc/Class"
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
        plt.savefig(
            f"{self.exp_dir}/plots/pairwise_bdt_{self.dataset_size}_{self.metric}_aucs_{self.anchor_per_class}.png",
            dpi=300,
        )
        plt.close(fig)

        if background_rejections is None:
            self.log.warning(
                "No background rejection data found in results file; skipping those plots."
            )
        else:
            for idx, target_tpr in enumerate(target_tprs):
                percent_label, file_tag = self.target_tpr_labels(target_tpr)
                rej_data = background_rejections[idx]
                rej_plot_data = np.where(mask, rej_data, np.nan)
                finite_values = rej_plot_data[
                    np.isfinite(rej_plot_data) & (rej_plot_data > 0)
                ]

                display_data = rej_plot_data.copy()
                if finite_values.size == 0:
                    if not np.isinf(rej_plot_data).any():
                        continue
                    vmin, vmax = 1.0, 10.0
                else:
                    vmax = finite_values.max()
                    vmin = finite_values.min()
                    if np.isclose(vmin, vmax):
                        vmin = max(vmin / 10.0, 1.0)
                        vmax = max(vmax, vmin * 10.0)
                display_data[np.isinf(display_data)] = vmax

                fig, ax = plt.subplots(figsize=(6, 5))
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad(color=(1, 1, 1, 0))
                im = ax.imshow(
                    display_data,
                    cmap=cmap,
                    norm=LogNorm(vmin=max(vmin, 1e-12), vmax=vmax),
                )

                ax.set_xticks(range(self.n_classes))
                ax.set_yticks(range(self.n_classes))
                ax.set_xticklabels(class_names, rotation=45, ha="right")
                ax.set_yticklabels(class_names)
                ax.set_title(
                    "BDT Background Rejection @ "
                    f"{percent_label}% TPR, "
                    f"{kin_label} {pid_label} Rel, {self.anchor_per_class} Anc/Class"
                )

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("1 / FPR")

                for i in range(self.n_classes):
                    for j in range(i + 1, self.n_classes):
                        value = rej_data[i, j]
                        if np.isnan(value):
                            continue
                        if np.isinf(value):
                            label = "inf"
                        elif value >= 100:
                            label = f"{value:.0f}"
                        else:
                            label = f"{value:.1f}"
                        ax.text(
                            j,
                            i,
                            label,
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="w",
                        )

                plt.tight_layout()
                plt.savefig(
                    f"{self.exp_dir}/plots/pairwise_bdt_{self.dataset_size}_{self.metric}_bg_rej_tpr{file_tag}_{self.anchor_per_class}.png",
                    dpi=300,
                )
                plt.close(fig)

        if roc_fprs is None or roc_tprs is None:
            self.log.warning("No ROC curve data found in results file; skipping ROC/SIC plots.")
            return

        self.plot_pairwise_curve_grid(roc_fprs, roc_tprs, "roc", kin_label, pid_label)
        self.plot_pairwise_curve_grid(roc_fprs, roc_tprs, "sic", kin_label, pid_label)

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

        return np.abs(diff)
