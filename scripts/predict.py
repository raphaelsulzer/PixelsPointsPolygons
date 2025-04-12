import hydra
import torch
import os

from omegaconf import OmegaConf

from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)
    
    if cfg.multi_gpu:
        world_size = torch.cuda.device_count()
        local_rank = int(os.environ['LOCAL_RANK'])        
        setup_ddp(world_size, local_rank)
    else:
        world_size = 1
        local_rank = 0
    
    
    if cfg.model.name == "ffl":
        predictor = FFLPredictor(cfg, local_rank, world_size)
    elif cfg.model.name == "hisup":
        predictor = HiSupPredictor(cfg, local_rank, world_size)
    elif cfg.model.name == "pix2poly":
        predictor = Pix2PolyPredictor(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    predictor.predict_dataset()
    
        
if __name__ == "__main__":
    main()
