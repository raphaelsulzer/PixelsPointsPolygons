import os
import torch
import hydra
from omegaconf import OmegaConf
from pixelspointspolygons.train import FFLTrainer, HiSupTrainer, Pix2PolyTrainer

from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)
    
    if cfg.experiment.model.name == "ffl":
        trainer = FFLTrainer(cfg, local_rank, world_size)
    elif cfg.experiment.model.name == "hisup":
        trainer = HiSupTrainer(cfg, local_rank, world_size)
    elif cfg.experiment.model.name == "pix2poly":
        trainer = Pix2PolyTrainer(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")
    trainer.train()

if __name__ == "__main__":
    main()