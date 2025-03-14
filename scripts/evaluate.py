import hydra
from omegaconf import OmegaConf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pixelspointspolygons.eval import evaluate


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    evaluate(cfg.eval.gt_file, cfg.eval.dt_file, modes=cfg.eval.modes, outfile=cfg.eval.outfile)
    

if __name__ == "__main__":
    main()
