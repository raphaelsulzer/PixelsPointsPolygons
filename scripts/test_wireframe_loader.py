import hydra
import logging
from pixelspointspolygons.datasets import get_val_loader
from pixelspointspolygons.models.pix2poly import Tokenizer
from pixelspointspolygons.misc import setup_ddp, setup_hydraconf, make_logger, tensor_to_shapely_polys, plot_shapely_polygons, plot_image, denormalize_image_for_visualization
from pixelspointspolygons.predict import Pix2PolyPredictor

import matplotlib.pyplot as plt

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)

    cfg.run_type.batch_size = 1
    cfg.experiment.model.batch_size = 1

    tokenizer = Tokenizer(cfg)
    logger = make_logger("Test Wireframe Loader", level=logging.INFO)
    dataloader = get_val_loader(cfg, logger=logger, tokenizer=tokenizer)
    
    predictor = Pix2PolyPredictor(cfg,local_rank=local_rank,world_size=world_size)
    predictor.setup_model()
    
    for x_image, x_lidar, y_sequence, y_perm, tile_ids in dataloader:

        gt_polygons = predictor.coord_and_perm_to_polygons(y_sequence, y_perm)

        gt_polys = tensor_to_shapely_polys(gt_polygons[0])
    
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        
        image = denormalize_image_for_visualization(x_image[0], cfg)
        
        plot_image(image, ax=ax)
        plot_shapely_polygons(gt_polys, ax=ax)
        
        plt.savefig(f"wireframe_debug/{tile_ids[0].item()}_gt.png")
        plt.close(fig)
        
        break
    
if __name__ == "__main__":
    main()