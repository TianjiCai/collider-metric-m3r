from experiments.base_experiment import BaseExperiment
from omegaconf import DictConfig
import utils.nn_utils
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class RelRepMulticlassNN(BaseExperiment):
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
        self.data_path = config.data_path
        self.train = config.train
        self.anchor_per_class = config.anchor_per_class
        self.train_params = config.training
        self.network = config.network
        self.center = config.center
        self.include_kin = config.include_kin
        self.include_part_ids = config.include_part_ids

    def run(self):
        # model.to(self.device)
        # print(model)

        # assert False
        rel_reps_blocks = []
        if self.include_kin:
            kin_rel_reps = np.load(
                f"{self.project_path}{self.data_path}{self.dataset_size}/{self.metric}_{self.dataset_size}_rel_rep_{self.anchor_per_class}.npy"
            )
            rel_reps_blocks.append(kin_rel_reps)
        if self.include_part_ids:
            part_id_rel_reps = np.load(f"{self.project_path}{self.data_path}{self.dataset_size}/part_ids_{self.dataset_size}_rel_rep_{self.anchor_per_class}.npy")
            rel_reps_blocks.append(part_id_rel_reps)
            
        rel_reps = np.concatenate(rel_reps_blocks, axis=1)
        self.log.info(f"Relative representation shape: {rel_reps.shape}")

        if self.center:
            col_means = np.mean(rel_reps, axis=0)
            rel_reps -= col_means
            assert np.allclose(np.mean(rel_reps, axis=0), 0)

        labels = np.repeat(np.arange(self.n_classes), self.events_per_class)
        if rel_reps.shape[0] == labels.shape[0] - 1:
            labels = labels[1:]
        elif rel_reps.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Expected {labels.shape[0]} or {labels.shape[0] - 1} rel-rep rows, "
                f"got {rel_reps.shape[0]}."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            rel_reps,
            labels,
            test_size=self.train_params.val_size,
            stratify=labels,
            random_state=42,
        )

        self.y_test = y_test
        del rel_reps

        # print(self.network)
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
            # save the trained model
            torch.save(trained_model.state_dict(), self.exp_dir + "/trained_model.pth")
            self.train_loss_history = train_loss_history
            self.val_loss_history = val_loss_history
        # else:
        #     self.log.info("Evaluating without training...")
        #     self.log.info(
        #         "Loading the model from the run folder"
        #     )
        #     # model_state_dict = torch.load(
        #     #     f"{self.project_path}{self.data_path}trained_model.pth",
        #     #     map_location=self.device,
        #     # )
        #     model_state_dict = torch.load(
        #         self.exp_dir + "/trained_model.pth",
        #         map_location=self.device
        #     )
        #     model.load_state_dict(model_state_dict)
        #     self.log.info("Evaluating...")
        #     _, preds = utils.nn_utils.evaluate_network_torch(
        #         model, X_test, y_test, device=self.device
        #     )
        #     np.save(f"{self.exp_dir}/predictions.npy", preds)
        #     self.preds = preds
        model_state_dict = torch.load(
            self.exp_dir + "/trained_model.pth", map_location=self.device
        )
        model.load_state_dict(model_state_dict)

        self.log.info("Evaluating...")
        _, self.preds, self.pred_probs = utils.nn_utils.evaluate_network_torch(
            model, X_test, y_test, device=self.device
        )
        np.save(f"{self.exp_dir}/predictions.npy", self.preds)
        np.save(f"{self.exp_dir}/pred_probabilities.npy", self.pred_probs)
        self.plot()

    def plot_loss_history(self, train_loss_history, val_loss_history=None):
        self.log.info("Plotting loss curves...")
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label="Training Loss")
        if val_loss_history is not None:
            plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.xscale("log")
        plt.title("Training and Validation Loss over Epochs")
        plt.legend()

        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)
        plt.savefig(f"{self.exp_dir}/plots/loss_curves.png", dpi=300)

    def plot_cm(self, y_preds, y_truth):
        class_names = self.get_class_names()
        self.log.info("Plotting the confusion matrix...")
        cm = confusion_matrix(y_truth, y_preds, normalize='true')

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", values_format=".3f")

        plt.title("Confusion Matrix")
        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)
        plt.savefig(
            f"{self.exp_dir}/plots/confusion_matrix.png", dpi=300, bbox_inches="tight"
        )

    def plot(self):
        if self.train:
            self.plot_loss_history(self.train_loss_history, self.val_loss_history)
        self.plot_cm(self.preds, self.y_test)
