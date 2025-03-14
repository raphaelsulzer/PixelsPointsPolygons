import hydra
from omegaconf import OmegaConf

from pixelspointspolygons.eval import evaluate


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    outfile = cfg.eval.pred_file.replace("/predictions/", "/eval/").replace(".json", ".csv")
    evaluate(cfg.eval.gt_file, cfg.eval.pred_file, modes=cfg.eval.modes, outfile=outfile)
    

if __name__ == "__main__":
    main()
