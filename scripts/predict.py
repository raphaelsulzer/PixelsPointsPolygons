import hydra
import pandas as pd

from pixelspointspolygons.eval import Evaluator
from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)
    
    
    if cfg.experiment.model.name == "ffl":
        predictor = FFLPredictor(cfg, local_rank, world_size)
    elif cfg.experiment.model.name == "hisup":
        predictor = HiSupPredictor(cfg, local_rank, world_size)
    elif cfg.experiment.model.name == "pix2poly":
        predictor = Pix2PolyPredictor(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")
    
    split = "val"
    cfg.eval.pred_file = cfg.eval.pred_file.replace("predictions", f"predictions_{split}")
    
    predictor.predict_dataset(split=split)
    
    ee = Evaluator(cfg)
    ee.load_gt(cfg.dataset.annotations[split])
    ee.load_predictions(cfg.eval.pred_file)
    res=ee.evaluate()

    df = pd.DataFrame.from_dict(res, orient='index')
    
    print("\n")
    print(df)
    print("\n")
    
    print(f"Save eval file to {cfg.eval.eval_file}")
    df.to_csv(cfg.eval.eval_file, index=True, float_format="%.3g")
        
if __name__ == "__main__":
    main()
