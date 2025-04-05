import os
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
from pixelspointspolygons.datasets.build_datasets import get_train_loader
from pixelspointspolygons.misc import plot_ffl
import albumentations as A


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):

    os.makedirs("augmentation_replay", exist_ok=True)
    cfg.model.batch_size = 1
    loader = get_train_loader(cfg)
    for batch in loader:
        ge = batch["augmentation_replay"][0]
        batch = next(iter(loader))
        fig = plot_ffl(batch,show=False)
        fig.savefig(f"./augmentation_replay/{ge}.png")
    
    
if __name__ == "__main__":
    
    main()