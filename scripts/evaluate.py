import hydra
from omegaconf import OmegaConf

from pixelspointspolygons.eval import Evaluator


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    ee = Evaluator(cfg)
    ee.evaluate_all()    
    # ee.to_latex(csv_file=cfg.eval.eval_file)
    

if __name__ == "__main__":
    main()
