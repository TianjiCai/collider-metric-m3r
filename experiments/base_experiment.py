"""Shared scaffolding for the Hydra experiments."""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from m3r import load_config
from m3r.blocks import BlockStore
from m3r.classes import CLASS_NAMES, Task, pairwise_class_pairs, resolve_tasks
from m3r.results_io import save_npz, write_provenance

DATASET_SIZE_TO_COUNT = {"mini": 1_000, "medium": 4_000, "large": 10_000, "full": 100_000}


class BaseExperiment:
    def __init__(self, config: DictConfig, exp_dir: str):
        self.log = logging.getLogger("Experiment")
        self.log.info(f"Initializing {self.__class__.__name__}")
        self.config = config
        self.exp_dir = Path(exp_dir)
        self.metric: str = str(config.metric)
        self.dataset_size: str = str(config.dataset_size)
        self.n_classes: int = int(config.n_classes)
        self.events_per_class = DATASET_SIZE_TO_COUNT[self.dataset_size]
        self.dataset_size_tag = f"{self.events_per_class // 1000}k"
        self.paper = load_config(config.get("paper_config", None))
        self.root = self.paper.root
        self.store = BlockStore.from_config(self.paper, size=self.dataset_size, n_per_class=self.events_per_class)
        self.tasks: list[Task] = self._resolve_tasks(config.get("tasks", "all"))
        self.results_dir = self.exp_dir / "results"
        self.plots_dir = self.exp_dir / "plots"

    # Tasks / classes ---------------------------------------------------------
    def _resolve_tasks(self, spec: Any) -> list[Task]:
        if spec is None:
            spec = "all"
        if not isinstance(spec, str):
            spec = list(OmegaConf.to_container(spec, resolve=True)) if OmegaConf.is_config(spec) else list(spec)
        tasks = resolve_tasks(spec)
        if spec == "all" or (isinstance(spec, list) and spec == ["all"]):
            tasks = [Task.from_classes(i, j) for i, j in pairwise_class_pairs(self.n_classes)]
        return tasks

    def get_class_names(self) -> list[str]:
        return list(CLASS_NAMES[: self.n_classes])

    def get_class_name(self, class_idx: int) -> str:
        return CLASS_NAMES[class_idx]

    def get_class_pairs(self) -> list[tuple[int, int]]:
        return [task.classes for task in self.tasks]

    def computed_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_classes, self.n_classes), dtype=bool)
        for i, j in self.get_class_pairs():
            mask[i, j] = mask[j, i] = True
        return mask

    # Data helpers ------------------------------------------------------------
    def part_ids(self, class_idx: int, feature: str) -> np.ndarray:
        path = self.paper.part_ids_path(CLASS_NAMES[class_idx])
        if self.dataset_size != "large":
            path = path.with_name(path.name.replace("10k", self.dataset_size_tag))
        return np.load(path)[feature]

    # Output helpers ----------------------------------------------------------
    def save_results(self, filename: str, **arrays: Any) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        out = save_npz(self.results_dir / filename, computed_mask=self.computed_mask(), **arrays)
        write_provenance(out, config_hash=self.paper.hash(), experiment=self.__class__.__name__,
                         hydra_config=OmegaConf.to_container(self.config, resolve=True),
                         tasks=[t.slug for t in self.tasks])
        self.log.info(f"Saved {out}")
        return out

    @abstractmethod
    def run(self) -> None:
        ...

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Optional diagnostic plots; experiments override as needed."""
