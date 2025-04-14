import hydra
from omegaconf import OmegaConf

from pixelspointspolygons.eval import Evaluator


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.register_new_resolver("eq", lambda a, b: str(a) == str(b))
    OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond == "True" else b)
    OmegaConf.resolve(cfg)
    
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    modes_str = "_".join(cfg.eval.modes)
    cfg.eval.eval_file = f"{cfg.eval.eval_file}_{modes_str}.csv"
    
    ee = Evaluator(cfg)
    ee.evaluate_all()    
    ee.to_latex(csv_file=cfg.eval.eval_file)
    


if __name__ == "__main__":
    main()
