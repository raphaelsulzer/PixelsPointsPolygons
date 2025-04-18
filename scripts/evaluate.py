import hydra
from omegaconf import OmegaConf

from pixelspointspolygons.eval import Evaluator
from pixelspointspolygons.misc.shared_utils import setup_omegaconf



@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    setup_omegaconf(cfg)
    
    modes_str = "_".join(cfg.eval.modes)
    cfg.eval.eval_file = f"{cfg.eval.eval_file}_{modes_str}.csv"
    
    ee = Evaluator(cfg)
    ee.evaluate_all()    
    ee.to_latex(csv_file=cfg.eval.eval_file)
    


if __name__ == "__main__":
    main()
