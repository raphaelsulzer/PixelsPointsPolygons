import hydra

import pandas as pd

from pixelspointspolygons.eval import Evaluator
from pixelspointspolygons.misc.shared_utils import setup_hydraconf



@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    
    modes_str = "_".join(cfg.evaluation.modes)
    cfg.evaluation.eval_file = f"{cfg.evaluation.eval_file}_{cfg.experiment.name}_{modes_str}.csv"
    
    print(f"Evaluate {cfg.experiment.model.name}/{cfg.experiment.name} on {cfg.experiment.dataset.country}/{cfg.evaluation.split}")

    ee = Evaluator(cfg)
    ee.pbar_disable = False
    ee.load_gt(cfg.experiment.dataset.annotations[cfg.evaluation.split])
    ee.load_predictions(cfg.evaluation.pred_file)
    res=ee.evaluate()

    df = pd.DataFrame.from_dict(res, orient='index')
    
    print("\n")
    print(df)
    print("\n")
    
    print(f"Save eval file to {cfg.evaluation.eval_file}")
    df.to_csv(cfg.evaluation.eval_file, index=True, float_format="%.3g")
    


if __name__ == "__main__":
    main()
