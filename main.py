"""Hydra entry point for the machine-learning tagging experiments.

    python main.py -cn rel_rep_pairwise_svm metric=w2 anchor_per_class=20 tasks=paper
    python main.py -cn abs_pairwise_svm metric=ps tasks=paper
    python main.py -cn rel_rep_pairwise_bdt feature_set=ps+sw+w2+pid
    python main.py -cn rel_rep_pairwise_nn 'features=[ps,sw,w2,pid]'

Configs live in configs/experiments/ (one file per experiment); shared paper settings
(data locations, tasks, feature sets) come from configs/paper.yaml through ``m3r``.
Outputs go to deterministic directories under results/experiments/ (no timestamps),
so figure scripts and ``paper/*/collect_*.py`` can find them without run discovery.
"""

from __future__ import annotations

import importlib
import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from m3r.paths import repo_root

log = logging.getLogger("main")


def _feature_key(feature_set, metrics=None) -> str:
    """Directory name for a BDT feature set: explicit list -> canonical key, else the set name."""
    from m3r.bdt import canonical_set_key

    if metrics:
        return canonical_set_key(list(metrics))
    return str(feature_set)


def _list_key(items) -> str:
    return "+".join(str(x) for x in items)


OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
OmegaConf.register_new_resolver("add", lambda a, b: int(a) + int(b), replace=True)
OmegaConf.register_new_resolver("m3r_root", lambda: str(repo_root()), replace=True)
OmegaConf.register_new_resolver("feature_key", _feature_key, replace=True)
OmegaConf.register_new_resolver("join", _list_key, replace=True)


@hydra.main(config_path="configs/experiments", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    hcfg = HydraConfig.get()
    exp_dir = cfg.prev_exp_dir or hcfg.runtime.output_dir
    if cfg.prev_exp_dir:
        from m3r.hydra_utils import update_config_from_prev

        cfg = update_config_from_prev(cfg, hcfg, exp_dir)
    experiments = importlib.import_module("experiments")
    exp_cls = getattr(experiments, cfg.experiment)
    exp_cls(cfg, exp_dir).run()


if __name__ == "__main__":
    main()
