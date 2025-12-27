from experiments.base_experiment import BaseExperiment
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
import utils.ot_utils
import utils.comparison_utils
import os
from scipy.spatial.distance import pdist, squareform


class JaccardAnchors(BaseExperiment):
    def __init__(self, config: DictConfig, exp_dir: str):
        super().__init__(config, exp_dir)
        self.anchor_per_class = config.anchor_per_class
        self.data_path = config.data_path
        self.seed = config.seed
        self.plot_only = config.plot_only
        # self.log.info("Value of plot_only:", self.plot_only)

    def run(self):
        for i in range(self.n_classes):
            abs_dist_matrix = np.load(
                f"{self.data_path}{self.metric}/{self.dataset_size}_{self.metric}/{self.metric}_block_{i}_{i}.npy"
            )
            labels = np.zeros(abs_dist_matrix.shape[0])
            
            # need to change anchors to be a range of anchors 
            rel_rep = utils.comparison_utils.relative_rep(abs_dist_matrix, labels, num_classes=1, anchors_per_class=self.anchor_per_class)
            
            neighbor_list = np.arange(1, self.events_per_class, self.events_per_class // 200)
            
            rel_rep_dist_mat = squareform(pdist(rel_rep, metric="euclidean"))
            
            