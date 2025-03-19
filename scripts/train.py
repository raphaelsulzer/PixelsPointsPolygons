import os
import torch
import torch.multiprocessing as mp
import hydra
from omegaconf import OmegaConf
from pixelspointspolygons.train import Trainer


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):

    OmegaConf.resolve(cfg)
    
    if cfg.multi_gpu:
        world_size = torch.cuda.device_count()
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        world_size = 1
        local_rank = 0
    
    trainer = Trainer(cfg, local_rank, world_size)
    trainer.train()

if __name__ == "__main__":
    main()