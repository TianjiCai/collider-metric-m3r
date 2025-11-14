import hydra
import importlib
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from config_utils import update_config_from_prev

log = logging.getLogger("main")
OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b))

@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # # read hydra config
    # hcfg = HydraConfig.get()

    # # determine experiment directory, put in the run directory to rerun a previous experiment
    # exp_dir = hcfg.runtime.output_dir
    # print("project_path =", cfg.project_path)

    # # select and run experiment
    # experiments = importlib.import_module("experiments")
    # # select experiment class
    # exp_cls = getattr(experiments, cfg.experiment)
    # experiment = exp_cls(cfg, exp_dir)
    # experiment.run()

    # read hydra config
    hcfg = HydraConfig.get()

    # determine experiment directory, put in the run directory to rerun a previous experiment
    exp_dir = cfg.prev_exp_dir or hcfg.runtime.output_dir

    # resolve config if loading previous experiment
    if cfg.prev_exp_dir:
        cfg = update_config_from_prev(cfg, hcfg, exp_dir)

    # submit this script to a cluster
    # if cfg.submit:
    #     submit(cfg, hcfg, exp_dir, log)
    #     return

    # select and run experiment
    experiments = importlib.import_module("experiments")
    # select experiment class
    exp_cls = getattr(experiments, cfg.experiment)
    experiment = exp_cls(cfg, exp_dir)
    experiment.run()


if __name__ == "__main__":
    main()