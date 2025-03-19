import os
import torch
import torch.multiprocessing as mp
import hydra
from omegaconf import OmegaConf
from pixelspointspolygons.train import Trainer, spawn_worker


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):

    OmegaConf.resolve(cfg)
    # world_size = torch.cuda.device_count()
    # mp.spawn(spawn_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    spawn_worker(None,None, cfg)

if __name__ == "__main__":
    main()

