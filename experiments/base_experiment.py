from omegaconf import DictConfig
from hydra.utils import call, instantiate
from abc import abstractmethod
import logging


class BaseExperiment:
    def __init__(self, config: DictConfig, exp_dir: str):
        self.log = logging.getLogger("Experiment")
        self.log.info(f"Initializing {self.__class__.__name__}")
        self.exp_dir = exp_dir
        self.metric = config.metric
        self.dataset_size = config.dataset_size
        self.n_classes = config.n_classes
        self.project_path = config.project_path
        self.events_per_class = (
            100000
            if self.dataset_size == "full"
            else 4000
            if self.dataset_size == "medium"
            else 1000
        )

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot(self):
        "Create and save evaluation plots. To be implemented by the child class"
        pass
