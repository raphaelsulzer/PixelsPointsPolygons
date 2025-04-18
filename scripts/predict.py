import hydra
import torch
import os


from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_omegaconf

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_omegaconf(cfg)
    local_rank, world_size = setup_ddp(cfg)
    
    
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
