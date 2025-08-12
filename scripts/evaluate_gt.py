import hydra
import os
import pandas as pd

from pixelspointspolygons.eval import Evaluator
from pixelspointspolygons.misc.shared_utils import setup_hydraconf



@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    
    cfg.evaluation.eval_file = f"evaluate_gt.csv"
    
    path = "/data/rsulzer/dataset_pixels_points_polygons/"
    countries = ["CH", "NZ", "NY"]
    
    for country in countries:
        
        print(f"Evaluate {country} dataset")
        
        gt_file = os.path.join(path, country, "annotations_fixed_DP.json")
        pred_file = os.path.join(path, country, "annotations_DP.json")
        
        ee = Evaluator(cfg)
        ee.pbar_disable = False
        ee.load_gt(gt_file)
        ee.load_predictions(pred_file)
        res=ee.evaluate()

        df = pd.DataFrame.from_dict(res, orient='index')
        
        print("\n")
        print(df)
        print("\n")
        
        outfile = os.path.join(path,f"eval_{country}.csv")
        print(f"Save eval file to {outfile}")
        df.to_csv(outfile, index=True, float_format="%.3g")
    


if __name__ == "__main__":
    main()
