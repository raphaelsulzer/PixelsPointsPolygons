import hydra
import os

from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)
    
    
    if cfg.experiment.model.name == "ffl":
        predictor = FFLPredictor(cfg, local_rank, world_size)
    elif cfg.experiment.model.name == "hisup":
        predictor = HiSupPredictor(cfg, local_rank, world_size)
    elif cfg.experiment.model.name == "pix2poly":
        predictor = Pix2PolyPredictor(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")
    
    
    image_file = None
    lidar_file = None
    
    if 'image_file' in cfg and not os.path.isfile(cfg.image_file):
        raise FileExistsError(f"Image file {cfg.image_file} not found.")
    else:
        image_file = cfg.image_file
    
    if 'lidar_file' in cfg and not os.path.isfile(cfg.lidar_file):
        raise FileExistsError(f"Image file {cfg.lidar_file} not found.")
    else:
        lidar_file = cfg.lidar_file
    
    if 'image_file' not in cfg and 'lidar_file' not in cfg:
        raise ValueError("Either an image_file or lidar_file must be provided.")
    
    predictor.predict_file(img_infile=image_file, lidar_infile=lidar_file)
    
    
if __name__ == "__main__":
    main()
