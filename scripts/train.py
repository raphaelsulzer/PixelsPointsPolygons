import os
import torch
import hydra
from omegaconf import OmegaConf
from pixelspointspolygons.train import FFLTrainer, HiSupTrainer, Pix2PolyTrainer

from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_omegaconf

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_omegaconf(cfg)
    local_rank, world_size = setup_ddp(cfg)
    
    if cfg.model.name == "ffl":
        trainer = FFLTrainer(cfg, local_rank, world_size)
    elif cfg.model.name == "hisup":
        trainer = HiSupTrainer(cfg, local_rank, world_size)
    elif cfg.model.name == "pix2poly":
        trainer = Pix2PolyTrainer(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    trainer.train()

if __name__ == "__main__":
    main()