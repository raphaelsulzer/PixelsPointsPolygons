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
    
    
    if 'image_file' in cfg:
        if not os.path.isfile(cfg.image_file):
            raise FileExistsError(f"Image file {cfg.image_file} not found.")
        if not cfg.experiment.encoder.use_images:
            raise ValueError("Image file provided but images are not used in the encoder. Please choose and appropriate model.")
        image_file = cfg.image_file
    else:
        if cfg.experiment.encoder.use_images:
            raise ValueError("No image file provided but images are used in the encoder. Please choose and appropriate model.")
        image_file = None
    
    if 'lidar_file' in cfg:
        if not os.path.isfile(cfg.lidar_file):
            raise FileExistsError(f"Image file {cfg.lidar_file} not found.")
        if not cfg.experiment.encoder.use_lidar:
            raise ValueError("LiDAR file provided but LiDAR is not used in the encoder. Please choose and appropriate model.")
        lidar_file = cfg.lidar_file
    else:
        if cfg.experiment.encoder.use_lidar:
            raise ValueError("No LiDAR file provided but LiDAR is used in the encoder. Please choose and appropriate model.")
        lidar_file = None
        
    if 'image_file' not in cfg and 'lidar_file' not in cfg:
        raise ValueError("Either an image_file or lidar_file must be provided using +image_file=$FILE_NAME or +lidar_file=$FILE_NAME.")
    
    predictor.predict_file(img_infile=image_file, lidar_infile=lidar_file)
    
    # TODO: implement batch prediction for tiles.
    # It needs a split function and a translate function and a merge function
    
    
if __name__ == "__main__":
    main()
