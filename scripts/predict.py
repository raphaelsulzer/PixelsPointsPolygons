import hydra
import pandas as pd

from pixelspointspolygons.eval import Evaluator
from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)
    
    print(f"Predict {cfg.experiment.model.name}/{cfg.experiment.name} on {cfg.experiment.country}/{cfg.evaluation.split}")
    
    
    if cfg.experiment.model.name == "ffl":
        predictor = FFLPredictor(cfg, local_rank, world_size)
    elif cfg.experiment.model.name == "hisup":
        predictor = HiSupPredictor(cfg, local_rank, world_size)
    elif cfg.experiment.model.name == "pix2poly":
        predictor = Pix2PolyPredictor(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")
    
    predictor.predict_dataset(split=cfg.evaluation.split)
    
    print(f"Evaluate {cfg.experiment.model.name}/{cfg.experiment.name} on {cfg.experiment.country}/{cfg.evaluation.split}")

    ee = Evaluator(cfg)
    ee.pbar_disable = False
    ee.load_gt(cfg.dataset.annotations[cfg.evaluation.split])
    ee.load_predictions(cfg.evaluation.pred_file)
    res=ee.evaluate()
    
    # TODO: 
    # 1. run an evaluation on the new val set of NY and NZ, have to adjust to new annotation paths
    # 2. run a training on the building annotations of CH

    df = pd.DataFrame.from_dict(res, orient='index')
    
    print("\n")
    print(df)
    print("\n")
    
    print(f"Save eval file to {cfg.evaluation.eval_file}")
    df.to_csv(cfg.evaluation.eval_file, index=True, float_format="%.3g")
        
if __name__ == "__main__":
    main()
