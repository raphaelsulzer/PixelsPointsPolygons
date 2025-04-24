import hydra
import torch
import os


from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_omegaconf, get_experiment_type


def predict_all(cfg):
    
    
    local_rank, world_size = setup_ddp(cfg)
    
    for item in cfg.experiments.experiments:

        for exp in item.experiment_name:
            
            exp_name, img_dim, polygonization_method = get_experiment_type(exp)

            cfg.experiment_name = exp_name
            cfg.output_dir = 
            
            if item.model == "ffl":
                predictor = FFLPredictor(cfg, local_rank, world_size)
            elif item.model == "hisup":
                predictor = HiSupPredictor(cfg, local_rank, world_size)
            elif item.model == "pix2poly":
                predictor = Pix2PolyPredictor(cfg, local_rank, world_size)
            else:
                raise ValueError(f"Unknown model name: {cfg.model.name}")
            
            predictor.predict_dataset(split="test")
    
    






@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
        
    setup_omegaconf(cfg)
    
    predict_all(cfg)

    
        
if __name__ == "__main__":
    main()
