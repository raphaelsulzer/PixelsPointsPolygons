import hydra
import torch
import os
from omegaconf import OmegaConf
from hydra import initialize, compose

from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf, get_experiment_type


def predict_all():
    
    experiments = ["lidar_density_ablation4", "lidar_density_ablation16"]

    setup_hydraconf()

    res_dict = {}

    with initialize(config_path="../config", version_base="1.3"):
        for exp in experiments:
            cfg = compose(config_name="config", overrides=[f"experiment={exp}","checkpoint=best_val_iou"])
            OmegaConf.resolve(cfg)
            print(f"Running: {cfg.experiment.name}")            
            
            
            local_rank, world_size = setup_ddp(cfg)
            
            print(cfg.experiment.encoder.max_points_per_voxel)
                        
            if cfg.experiment.model.name == "ffl":
                predictor = FFLPredictor(cfg, local_rank, world_size)
            elif cfg.experiment.model.name == "hisup":
                predictor = HiSupPredictor(cfg, local_rank, world_size)
            elif cfg.experiment.model.name == "pix2poly":
                predictor = Pix2PolyPredictor(cfg, local_rank, world_size)
            else:
                raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")
            
            predictor.predict_dataset(split="val")
    
    






# @hydra.main(config_path="../config", config_name="config", version_base="1.3")
# def main(cfg):
        
#     setup_omegaconf(cfg)
    
#     predict_all(cfg)

    
        
if __name__ == "__main__":
    predict_all()
