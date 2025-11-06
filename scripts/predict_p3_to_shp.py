import hydra

from pixelspointspolygons.predict import Pix2PolyGeoPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)
    
    print(f"Predict {cfg.experiment.model.name}/{cfg.experiment.name} on {cfg.experiment.dataset.country}/{cfg.evaluation.split}")

    predictor = Pix2PolyGeoPredictor(cfg, local_rank, world_size)
    
    predictor.predict_dataset_to_shp(split=cfg.evaluation.split)
    
        
if __name__ == "__main__":
    main()
