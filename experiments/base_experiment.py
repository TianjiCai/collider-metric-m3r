from omegaconf import DictConfig
from hydra.utils import call, instantiate
from abc import abstractmethod
import logging
from pathlib import Path

from utils import jetclass_utils


class BaseExperiment:
    def __init__(self, config: DictConfig, exp_dir: str):
        self.log = logging.getLogger("Experiment")
        self.log.info(f"Initializing {self.__class__.__name__}")
        self.exp_dir = exp_dir
        self.metric = config.metric
        self.dataset_size = config.dataset_size
        self.n_classes = config.n_classes
        self.project_root = Path(str(config.project_path)).resolve()
        self.project_path = f"{self.project_root}/"
        self.events_per_class = jetclass_utils.dataset_size_to_count(self.dataset_size)
        self.dataset_dir = jetclass_utils.dataset_dir(self.project_root, self.dataset_size)
        self.labels_path = self.dataset_dir / "class_labels.npz"
        self.dataset_size_tag = jetclass_utils.dataset_size_tag(self.events_per_class)

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot(self):
        "Create and save evaluation plots. To be implemented by the child class"
        pass

    def get_class_names(self) -> list[str]:
        if self.labels_path.exists():
            return jetclass_utils.ordered_class_names_from_labels(
                self.labels_path,
                expected_n_classes=self.n_classes,
            )
        return jetclass_utils.canonical_class_names(self.n_classes)

    def get_class_name(self, class_idx: int) -> str:
        class_names = self.get_class_names()
        return class_names[class_idx]

    def get_class_pairs(self) -> list[tuple[int, int]]:
        return jetclass_utils.pairwise_class_pairs(self.n_classes)
