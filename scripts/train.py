import os
import torch
import hydra
from omegaconf import OmegaConf
from pixelspointspolygons.train import FFLTrainer, HiSupTrainer, Pix2PolyTrainer

# torch.autograd.set_detect_anomaly(True)
import warnings
warnings.filterwarnings("error")

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):

    OmegaConf.resolve(cfg)
    
    if cfg.multi_gpu:
        world_size = torch.cuda.device_count()
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        world_size = 1
        local_rank = 0
    
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