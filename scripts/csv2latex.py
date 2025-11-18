import sys
import hydra

from pixelspointspolygons.misc.shared_utils import setup_hydraconf
from pixelspointspolygons.eval import Evaluator

def parse_cli_overrides():
    # Skip the script name
    return [arg for arg in sys.argv[1:] if "=" in arg]


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_hydraconf(cfg)
            
            
    ### Evaluate
    ee = Evaluator(cfg)
    ee.pbar_disable = False

    cfg.evaluation.eval_file = f"{cfg.evaluation.eval_file}_backbone_ablation_{cfg.experiment.dataset.country}_{cfg.evaluation.split}.csv"
    

    ee.to_latex(csv_file=cfg.evaluation.eval_file,type="backbone")

        
if __name__ == "__main__":
    main()
